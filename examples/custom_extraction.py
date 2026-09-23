"""Customize the output model and its prompts together.

The defaults use strings for entities and timeline entries. This example replaces
entity strings with categorized objects for filtering and knowledge-base lookup,
while retaining the default timeline strings and worker/reviewer contracts.
Use SupportCase as RAFT's output_type, not the worker Agent's output_type.
The reviewer writes a separate CaseReview assessment through edit_state.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from raft.defaults import REVIEWER_INSTRUCTIONS, WORKER_INSTRUCTIONS


class SupportEntity(BaseModel):
    """Wrap an identifier with a domain-specific classification."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(max_length=120)
    kind: Literal["component", "error_code", "path", "registry_key", "product_version"] = Field(
        description="Identifier category for filtering and selecting a lookup tool."
    )


class SupportCase(BaseModel):
    """An application-owned schema; working handoff notes remain in execution."""

    model_config = ConfigDict(extra="forbid")

    entities: list[SupportEntity] = Field(
        description=(
            "Important, case-defining technical identifiers with their categories; [] if absent."
        )
    )
    timeline: list[Annotated[str, Field(max_length=4800)]]
    root_cause: str | None = Field(max_length=800)
    resolution_steps: str | None = Field(max_length=1200)


_DOMAIN_GUIDANCE = """
<typed_entities>
For this custom schema, entities uses objects instead of the default strings.
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
