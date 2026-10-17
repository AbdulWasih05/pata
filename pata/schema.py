"""Pata output contract. The single definition used by rules, prompt, eval and API.

Mirrors BRIEF.md section 3. Any change here is a change to the fine-tune target.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Market(StrEnum):
    IN = "IN"
    AE = "AE"


class Payment(StrEnum):
    COD = "COD"
    PREPAID = "PREPAID"


class Decision(StrEnum):
    DELIVERABLE = "DELIVERABLE"
    CALL_FIRST = "CALL_FIRST"
    LIKELY_FAIL = "LIKELY_FAIL"


class ReasonCode(StrEnum):
    # From BRIEF.md section 3.
    NO_BUILDING_ID = "NO_BUILDING_ID"
    LANDMARK_ONLY = "LANDMARK_ONLY"
    PINCODE_MISMATCH = "PINCODE_MISMATCH"
    MIXED_SCRIPT = "MIXED_SCRIPT"
    AMBIGUOUS_AREA = "AMBIGUOUS_AREA"
    MISSING_UNIT = "MISSING_UNIT"
    COD_HIGH_RISK = "COD_HIGH_RISK"
    # Added so every rule in section 3 has its own code (see DECISIONS.md D3).
    CALL_REQUESTED = "CALL_REQUESTED"
    NO_PHONE = "NO_PHONE"
    PO_BOX_ONLY = "PO_BOX_ONLY"
    TOO_SHORT = "TOO_SHORT"
    OK = "OK"


# Fixed order so reason_codes are deterministic in training targets.
REASON_ORDER: list[ReasonCode] = list(ReasonCode)


class TriageInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    market: Market
    raw_address: str = Field(min_length=1, max_length=1000)
    phone_present: bool = True
    payment: Payment = Payment.PREPAID
    declared_pincode: str | None = None
    city_hint: str | None = None

    @field_validator("declared_pincode", "city_hint")
    @classmethod
    def _empty_to_none(cls, v: str | None) -> str | None:
        return v or None


class Normalised(BaseModel):
    model_config = ConfigDict(extra="forbid")

    country: Market
    admin1: str = ""
    city: str = ""
    area: str = ""
    street: str = ""
    building: str = ""
    unit: str = ""
    landmark: str = ""
    postcode: str | None = None
    makani: str | None = None
    po_box: str | None = None


DRIVER_NOTE_MAX = 120


class TriageOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Decision
    confidence: float = Field(ge=0.0, le=1.0)
    reason_codes: list[ReasonCode] = Field(min_length=1)
    normalised: Normalised
    driver_note: str = Field(min_length=1, max_length=DRIVER_NOTE_MAX)
    call_script: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> TriageOutput:
        if self.decision == Decision.CALL_FIRST:
            if not self.call_script:
                raise ValueError("call_script is required when decision is CALL_FIRST")
        elif self.call_script is not None:
            raise ValueError("call_script must be null unless decision is CALL_FIRST")
        if ReasonCode.OK in self.reason_codes and len(self.reason_codes) > 1:
            raise ValueError("OK cannot be combined with other reason codes")
        if self.decision != Decision.DELIVERABLE and ReasonCode.OK in self.reason_codes:
            raise ValueError("OK is only valid for DELIVERABLE")
        return self


def output_json_schema() -> dict:
    """JSON schema of the output, for prompts and constrained decoding."""
    return TriageOutput.model_json_schema()
