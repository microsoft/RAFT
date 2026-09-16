"""Support-case defaults adapted from the legacy v4 extraction schema.

The legacy ``RunningState`` fields now form one editable ``CaseExtraction``.
``CaseReview`` replaces the early verdict without discarding the worker state.
There are no node deltas, terminal flags, or latched/append-only merge semantics.
Applications can supply their own Pydantic models instead.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Entity(BaseModel):
    """A specific identifier that a future similar ticket could mention verbatim."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1,
        max_length=120,
        description=(
            "Verbatim error code, component/service short name, file path, registry "
            "key, or product+version pair from the case. Not commands, UI messages, "
            "generic concepts, or verbs. Aim for about ten distinct identifiers "
            "across the whole case; prioritize diagnostic value."
        ),
    )

    @field_validator("name")
    @classmethod
    def name_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Entity name must not be blank")
        return value


class TimelineEntry(BaseModel):
    """A material state transition, independently useful for semantic retrieval."""

    model_config = ConfigDict(extra="forbid")

    narrative: str = Field(
        min_length=200,
        description=(
            "One self-contained prose paragraph, both the engineer-facing writeup "
            "and the per-entry retrieval anchor. Front-load verbatim identifiers "
            "in the first sentence, with affected scope, observed symptom, and "
            "the current issue understanding at THIS stage. Continue with the "
            "question/theory/trigger, who did what (specific commands, queries, "
            "configurations, logs), observed evidence, theories considered and "
            "settled with their evidence, and how understanding shifted. Use "
            "consistent technical terms. Target 400-1500 characters when evidence "
            "supports it; do not invent details to meet a length target."
        ),
    )

    @field_validator("narrative")
    @classmethod
    def narrative_has_substance(cls, value: str) -> str:
        if len(value.strip()) < 200:
            raise ValueError("Narrative must contain at least 200 characters excluding outer space")
        return value


class CaseExtraction(BaseModel):
    """Cumulative case evidence, kept even when the final reviewer filters it."""

    model_config = ConfigDict(extra="forbid")

    entities: list[Entity] = Field(
        description="Cumulative distinct identifiers; use [] when none are supported."
    )
    timeline: list[TimelineEntry] = Field(
        description=(
            "Chronological material transitions, not one entry per message or "
            "worker batch. Preserve each stage's knowledge rather than projecting "
            "the eventual outcome backward. Use [] when no technical segment exists."
        )
    )
    root_cause: str | None = Field(
        max_length=800,
        description=(
            "Confirmed root cause, or null when unconfirmed/unknown. For RFI or "
            "guidance without a defect, the confirmed informational answer, "
            "product behavior, constraint, or recommendation rationale may belong "
            "here. Update if later evidence corrects or supersedes it."
        ),
    )
    resolution_steps: str | None = Field(
        max_length=1200,
        description=(
            "One dense paragraph of ONLY the actions that actually resolved the "
            "case, including accepted workarounds. For RFI/guidance, the recommended "
            "action or answer that closed the case. Exclude investigation, "
            "diagnostics, and theory elimination; those belong in the timeline. "
            "Use null for unknown/unconfirmed resolution, not a proposed fix."
        ),
    )
    handoff_notes: list[str] = Field(
        default_factory=list,
        description=(
            "Accumulated context, unresolved questions, and follow-up checks for "
            "later worker passes, with follow-up outcomes. Preserve earlier context. "
            "This optional orchestration field is not a retrieval narrative."
        ),
    )

    @field_validator("root_cause", "resolution_steps")
    @classmethod
    def unknown_is_null(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Use null for an unknown conclusion, not a blank string")
        return value


class CaseReview(BaseModel):
    """Final eligibility assessment, separate from the corrected extraction."""

    model_config = ConfigDict(extra="forbid")

    extractable: bool = Field(
        strict=True,
        description=(
            "True for ANY reusable troubleshooting or resolution insight: actions, "
            "theories, mitigations, partial/confirmed fixes, proposed-but-unconfirmed "
            "resolutions, or reusable RFI/guidance (recommendations, constraints, "
            "product behavior, decision rationale). False ONLY for pure noise "
            "without technical insight: empty cases, spam, diagnostic-free "
            "duplicates/misroutes, or administrative closures."
        ),
    )
    non_extractable_reasoning: str | None = Field(
        description=(
            "A one-paragraph justification citing specific source evidence when "
            "extractable=false. Must be null when extractable=true. Do not duplicate "
            "the case state here."
        ),
    )

    @model_validator(mode="after")
    def reasoning_matches_assessment(self) -> Self:
        if self.extractable:
            if self.non_extractable_reasoning is not None:
                raise ValueError("An extractable case must have null non_extractable_reasoning")
        elif (
            self.non_extractable_reasoning is None
            or not self.non_extractable_reasoning.strip()
        ):
            raise ValueError("A non-extractable case requires nonblank non_extractable_reasoning")
        return self
