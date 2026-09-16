from __future__ import annotations

import copy
import json
import sqlite3
import time
from dataclasses import dataclass, field, replace
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel

from raft._json import _to_json
from raft._text_budget import budget_summary, measure_text, validate_budget

MAX_SQL_SECONDS = 2.0
DEFAULT_BATCH_CHARS = 400_000
DEFAULT_QUERY_CHARS = 50_000


class ArtifactTooLargeError(ValueError):
    """An intact source artifact cannot fit in a worker batch."""

    def __init__(
        self, position: int, original_position: int, size: int, budget: dict[str, Any]
    ):
        self.details = {
            "artifact_position": position,
            "original_position": original_position,
            "artifact_size": size,
            "batch_budget": budget_summary(budget),
        }
        super().__init__(
            f"Artifact at position {position} (original position {original_position}) has "
            f"{size} {budget['unit']}, exceeding batch_budget.limit={budget['limit']}. "
            "Increase the limit or split the artifact during preprocessing."
        )


@dataclass
class CaseContext:
    case_id: Any
    metadata: dict[str, Any]
    connection: sqlite3.Connection
    artifact_char_counts: list[int]
    query_budget: dict[str, Any]
    final_output_type: type[BaseModel]
    _writer: sqlite3.Connection = field(repr=False)
    _database_uri: str = field(repr=False)
    _owns_writer: bool = field(default=True, repr=False)
    pending_state: Any = field(default_factory=dict)
    pending_edits: list[dict[str, Any]] = field(default_factory=list)
    pass_finished: bool = False
    is_final_batch: bool = True
    stage: Literal["worker", "reviewer"] = "worker"

    def close(self) -> None:
        self.connection.close()
        if self._owns_writer:
            self._writer.close()

    def begin_pass(self, committed_state: Any, *, is_final_batch: bool = True) -> None:
        self.pending_state = copy.deepcopy(committed_state)
        self.pending_edits = []
        self.pass_finished = False
        self.is_final_batch = is_final_batch

    def for_review(self, output: Any) -> CaseContext:
        """Create an isolated draft and reader; the case retains database ownership."""
        return replace(
            self,
            connection=_reader_connection(self._database_uri, stage="reviewer"),
            _owns_writer=False,
            metadata=copy.deepcopy(self.metadata),
            pending_state=copy.deepcopy(output),
            pending_edits=[],
            query_budget=dict(self.query_budget),
            pass_finished=False,
            is_final_batch=True,
            stage="reviewer",
        )

    def commit_revision(self, state: Any, *, pass_number: int | None) -> None:
        """Runner-only write after a successful invocation; never exposed as SQL."""
        with self._writer:
            self._writer.execute(
                "INSERT INTO state_revisions (stage, pass_number, state_json, edits_json) "
                "VALUES (?, ?, ?, ?)",
                (self.stage, pass_number, _to_json(state), _to_json(self.pending_edits)),
            )

    def revisions(self) -> list[dict[str, Any]]:
        """Export committed history for persistence, independently of tool limits."""
        return [
            {
                "revision_id": revision_id,
                "stage": stage,
                "pass_number": pass_number,
                "state": json.loads(state),
                "edits": json.loads(edits),
            }
            for revision_id, stage, pass_number, state, edits in self._writer.execute(
                "SELECT revision_id, stage, pass_number, state_json, edits_json "
                "FROM state_revisions ORDER BY revision_id"
            )
        ]

    def query(self, query: str) -> dict[str, Any]:
        query = query.strip()
        if not query:
            return {"error": "Query cannot be empty."}

        deadline = time.monotonic() + MAX_SQL_SECONDS
        self.connection.set_progress_handler(
            lambda: 1 if time.monotonic() >= deadline else 0,
            1_000,
        )
        cursor = None
        try:
            cursor = self.connection.execute(query)
            if cursor.description is None:
                return {"error": "Only queries that return rows are allowed."}

            columns = [column[0] for column in cursor.description]
            rows: list[dict[str, Any]] = []
            result = {"columns": columns, "rows": rows, "row_count": 0}
            too_large = {
                "error": "query_result_too_large",
                "query_budget": budget_summary(self.query_budget),
                "suggestion": (
                    "Select fewer columns, narrow the query, paginate with ORDER BY and "
                    "LIMIT/OFFSET, or use substr() for large fields (1-based offsets)."
                ),
            }
            # Characters are additive; tokens must count each complete candidate
            # response, including the wrapper and changing row_count digit width.
            size = measure_text(
                json.dumps(result, ensure_ascii=False, default=str), self.query_budget
            )
            if size > self.query_budget["limit"]:
                return too_large
            for values in cursor:
                row = dict(zip(columns, values, strict=True))
                count = len(rows)
                if self.query_budget["unit"] == "chars":
                    size += (
                        len(json.dumps(row, ensure_ascii=False, default=str))
                        + (2 if count else 0)
                        + len(str(count + 1)) - len(str(count))
                    )
                rows.append(row)
                result["row_count"] = count + 1
                if self.query_budget["unit"] == "tokens":
                    size = measure_text(
                        json.dumps(result, ensure_ascii=False, default=str), self.query_budget
                    )
                if size > self.query_budget["limit"]:
                    return too_large
            return result
        except sqlite3.Error as exc:
            return {"error": str(exc)}
        finally:
            if cursor is not None:
                cursor.close()
            self.connection.set_progress_handler(None, 0)


def _build_case_context(
    case: dict[str, Any],
    *,
    id_field: str,
    artifacts_field: str,
    metadata_field: str,
    artifact_sort_field: str | None,
    query_budget: dict[str, Any] | None = None,
    final_output_type: type[BaseModel] = BaseModel,
    batch_budget: dict[str, Any] | None = None,
) -> CaseContext:
    query_budget = validate_budget(
        {"unit": "chars", "limit": DEFAULT_QUERY_CHARS} if query_budget is None else query_budget,
        "query_budget",
    )
    if batch_budget is not None:
        batch_budget = validate_budget(batch_budget, "batch_budget")
    case_id = case[id_field]
    metadata = case[metadata_field]
    artifacts = list(enumerate(case[artifacts_field]))

    if artifact_sort_field is not None:
        artifacts.sort(key=lambda pair: _artifact_sort_key(pair[1], artifact_sort_field))

    artifact_rows: list[tuple[int, int, str | None, int, str]] = []
    artifact_char_counts: list[int] = []
    for position, (original_position, artifact) in enumerate(artifacts):
        artifact_json = _to_json(artifact)
        char_count = len(artifact_json)
        if batch_budget is not None:
            size = measure_text(artifact_json, batch_budget)
            if size > batch_budget["limit"]:
                raise ArtifactTooLargeError(position, original_position, size, batch_budget)
        sort_value = (
            _to_json(artifact[artifact_sort_field])
            if artifact_sort_field is not None and artifact.get(artifact_sort_field) is not None
            else None
        )
        artifact_rows.append((position, original_position, sort_value, char_count, artifact_json))
        artifact_char_counts.append(char_count)

    database_uri = f"file:raft_case_{uuid4().hex}?mode=memory&cache=shared"
    connection = sqlite3.connect(database_uri, uri=True, check_same_thread=False)
    try:
        connection.execute(
            """
            CREATE TABLE artifacts (
                position INTEGER PRIMARY KEY,
                original_position INTEGER NOT NULL,
                sort_value TEXT,
                char_count INTEGER NOT NULL,
                artifact_json TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?)",
            artifact_rows,
        )
        connection.execute(
            """
            CREATE TABLE state_revisions (
                revision_id INTEGER PRIMARY KEY,
                stage TEXT NOT NULL CHECK (stage IN ('worker', 'reviewer')),
                pass_number INTEGER,
                state_json TEXT NOT NULL,
                edits_json TEXT NOT NULL
            )
            """
        )
        connection.commit()
        connection.enable_load_extension(False)
        return CaseContext(
            case_id=case_id,
            metadata=metadata,
            connection=_reader_connection(database_uri, stage="worker"),
            artifact_char_counts=artifact_char_counts,
            query_budget=query_budget,
            final_output_type=final_output_type,
            _writer=connection,
            _database_uri=database_uri,
        )
    except Exception:
        connection.close()
        raise


def _reader_connection(database_uri: str, *, stage: str) -> sqlite3.Connection:
    connection = sqlite3.connect(database_uri, uri=True, check_same_thread=False)
    try:
        connection.enable_load_extension(False)
        connection.execute("PRAGMA query_only = ON")

        def authorize(action, argument1, argument2, database, source):
            # SQLite also authorizes reads in subqueries, joins, and COUNT(*).
            # The role is fixed for this connection, including cached statements.
            if (
                action == sqlite3.SQLITE_READ
                and (argument1 or "").lower() == "state_revisions"
                and stage != "reviewer"
            ):
                return sqlite3.SQLITE_DENY
            return _read_only_authorizer(action, argument1, argument2, database, source)

        connection.set_authorizer(authorize)
        return connection
    except Exception:
        connection.close()
        raise


def _read_only_authorizer(
    action: int,
    argument1: str | None,
    argument2: str | None,
    database: str | None,
    source: str | None,
) -> int:
    del argument1, database, source
    denied = {
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_ANALYZE,
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_INDEX,
        sqlite3.SQLITE_CREATE_TEMP_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
        sqlite3.SQLITE_CREATE_TEMP_VIEW,
        sqlite3.SQLITE_CREATE_TRIGGER,
        sqlite3.SQLITE_CREATE_VIEW,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_DROP_INDEX,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_INDEX,
        sqlite3.SQLITE_DROP_TEMP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_TRIGGER,
        sqlite3.SQLITE_DROP_TEMP_VIEW,
        sqlite3.SQLITE_DROP_TRIGGER,
        sqlite3.SQLITE_DROP_VIEW,
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_PRAGMA,
        sqlite3.SQLITE_REINDEX,
        sqlite3.SQLITE_TRANSACTION,
        sqlite3.SQLITE_UPDATE,
    }
    if action in denied:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and argument2 == "load_extension":
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _validate_case(
    case: Any,
    *,
    id_field: str,
    artifacts_field: str,
    metadata_field: str,
    duplicate: bool,
) -> None:
    if not isinstance(case, dict):
        raise TypeError("Each case must be a dictionary.")
    if id_field not in case or case[id_field] is None:
        raise ValueError(f"Missing case identifier field: {id_field}")
    if type(case[id_field]) not in (str, int):
        raise ValueError("Case identifier must be a string or integer")
    if duplicate:
        raise ValueError(f"Duplicate case identifier: {case[id_field]!r}")
    if not isinstance(case.get(artifacts_field), list):
        raise TypeError(f"{artifacts_field} must be a list")
    if not all(isinstance(item, dict) for item in case[artifacts_field]):
        raise TypeError(f"Every item in {artifacts_field} must be a dictionary")
    if not isinstance(case.get(metadata_field), dict):
        raise TypeError(f"{metadata_field} must be a dictionary")


def _artifact_sort_key(artifact: dict[str, Any], field: str) -> tuple[Any, ...]:
    value = artifact.get(field)
    if value is None:
        return (1, "", "")
    if isinstance(value, bool):
        return (0, "number", int(value))
    if isinstance(value, (int, float)):
        return (0, "number", value)
    if isinstance(value, str):
        return (0, "string", value)
    raise TypeError(f"Artifact sort field {field!r} must contain strings or numbers")
