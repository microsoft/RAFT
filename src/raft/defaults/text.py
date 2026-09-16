"""Text preparation for the legacy v4 narratives in the current case state.

Neither review assessments nor worker handoff notes are retrieval evidence.
"""

from .extraction import CaseExtraction


def state_to_text(state: CaseExtraction) -> list[str]:
    """Embed each self-contained narrative directly, preserving timeline order."""
    return [entry.narrative for entry in state.timeline]


def case_to_text(state: CaseExtraction) -> str:
    """Prefer root cause and resolution; fall back to the final timeline entry."""
    text = "\n".join(
        part for part in [state.root_cause, state.resolution_steps] if part and part.strip()
    )
    if text:
        return text
    return state.timeline[-1].narrative if state.timeline else ""
