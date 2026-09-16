"""Canonical case records: id, metadata, and a live Pydantic output model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field, InstanceOf, SerializeAsAny, StrictInt, StrictStr

StateT = TypeVar("StateT", bound=BaseModel)


class ExtractedCase(BaseModel, Generic[StateT]):
    """Successful, fully processed case; only persistence converts models to JSON."""

    id: StrictStr | StrictInt
    metadata: dict[str, Any]
    output: SerializeAsAny[StateT]
    review: SerializeAsAny[InstanceOf[BaseModel]] | dict[str, Any] | None = None
    execution: dict[str, Any] = Field(
        default_factory=lambda: {
            "usage": {},
            "tool_calls": [],
            "elapsed_seconds": 0.0,
            "passes": 0,
            "attempts": 0,
            "revisions": [],
        }
    )


def restore_case(
    case: ExtractedCase | dict[str, Any], output_type: type[BaseModel] | None = None
) -> ExtractedCase:
    """Reuse live records unchanged; reconstruct outputs only for serialized cases."""
    if isinstance(case, ExtractedCase):
        if output_type is not None and not isinstance(case.output, output_type):
            raise ValueError("Case output does not match output_type")
        return case
    if not isinstance(case, dict) or case.get("id") is None:
        raise ValueError("Each case must have a non-null id")
    if not isinstance(case.get("metadata"), dict):
        raise ValueError("Each case must have a metadata dictionary")
    output = case.get("output")
    if isinstance(output, BaseModel):
        if output_type is not None and not isinstance(output, output_type):
            raise ValueError("Case output does not match output_type")
        return ExtractedCase[type(output)](**case)
    if output_type is None:
        raise ValueError("Provide output_type to restore a saved case's Pydantic output")
    return ExtractedCase[output_type](
        **{**case, "output": output_type.model_validate(output, by_name=True)}
    )


def load_cases(path: str | Path, *, output_type: type[StateT]) -> list[ExtractedCase[StateT]]:
    """Load an extraction JSON result or a JSON case list and restore its models."""
    with Path(path).open(encoding="utf-8") as stream:
        saved = json.load(stream)
    cases = saved["extracted_cases"] if isinstance(saved, dict) else saved
    if not isinstance(cases, list):
        raise ValueError("Expected an extraction result or a list of case records")
    return [restore_case(case, output_type) for case in cases]
