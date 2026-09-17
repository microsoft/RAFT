"""Customize the output model and its prompts together.

This example adds typed entities for exact filtering and knowledge-base lookup,
while retaining the default narrative, conclusion, and worker/reviewer contracts.
Use SupportCase as RAFT's output_type, not the worker Agent's output_type.
The reviewer still returns the separate default CaseReview assessment.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from raft.defaults import REVIEWER_INSTRUCTIONS, WORKER_INSTRUCTIONS
from raft.defaults.extraction import Entity, TimelineEntry


class SupportEntity(Entity):
    """Keep the default verbatim name; add a domain-specific classification."""

    kind: Literal["component", "error_code", "path", "registry_key", "product_version"] = Field(
        description="Identifier category for filtering and selecting a lookup tool."
    )


class SupportCase(BaseModel):
    """An application-owned schema; working handoff notes remain in execution."""

    model_config = ConfigDict(extra="forbid")

    entities: list[SupportEntity] = Field(
        description="Distinct, evidence-backed identifiers with their categories; [] if absent."
    )
    timeline: list[TimelineEntry] = Field(
        description="Chronological changes in understanding, not one entry per message or batch."
    )
    root_cause: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=800)
    ] | None = Field(description="Confirmed cause, or null when unknown or unconfirmed.")
    resolution_steps: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1200)
    ] | None = Field(description="Actions that actually resolved the case, or null if unconfirmed.")


_DOMAIN_GUIDANCE = """
<typed_entities>
Each entities item has BOTH name and kind. Preserve the verbatim name and classify
kind as component, error_code, path, registry_key, or product_version according
to target_output_schema. Deduplicate by (kind, name); do not invent identifiers.
For example, {"name": "AUTH_CERT_EXPIRED", "kind": "error_code"} is valid only when
that exact code occurs in this case. The example itself is not case evidence.

When lookup_error is available, use it to interpret an unfamiliar error code.
Catalog entries are reference guidance, not proof of this case's root cause or
resolution. Verify any conclusion against the case artifacts; retain uncertainty
when a lookup has no match or the case does not confirm the suggested explanation.
</typed_entities>
"""

# Extend compatible defaults instead of accidentally dropping the tool protocol.
# Renaming timeline/conclusion fields requires rewriting the matching guidance too.
WORKER_PROMPT = WORKER_INSTRUCTIONS + _DOMAIN_GUIDANCE
REVIEWER_PROMPT = REVIEWER_INSTRUCTIONS + _DOMAIN_GUIDANCE
