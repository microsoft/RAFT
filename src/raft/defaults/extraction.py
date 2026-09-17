"""Default output models for the default instructions and text functions."""

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CaseExtraction(BaseModel):

    model_config = ConfigDict(extra="forbid")

    entities: list[Annotated[str, Field(max_length=120)]] = Field(max_length=25)
    timeline: list[Annotated[str, Field(max_length=4800)]]
    root_cause: str | None = Field(max_length=4800)
    resolution_steps: str | None = Field(max_length=4800)


class CaseReview(BaseModel):
    """Final eligibility assessment."""

    model_config = ConfigDict(extra="forbid")

    extractable: bool = Field(strict=True)
    non_extractable_reasoning: str | None

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
