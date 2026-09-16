from __future__ import annotations

import json
from typing import Any


def _to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError as exc:
        raise TypeError("Case data must be JSON serializable") from exc


def _id_key(case_id: Any) -> str:
    try:
        return json.dumps(case_id, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return repr(case_id)


def _snake_case(value: str) -> str:
    characters: list[str] = []
    for index, character in enumerate(value):
        if character.isupper() and index:
            characters.append("_")
        characters.append(character.lower())
    return "".join(characters)
