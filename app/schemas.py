from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FeatureMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(min_length=1, max_length=100)
    machine_id: str = Field(min_length=1, max_length=100)
    source: Literal["live", "replay"]
    schema_version: str = Field(min_length=1, max_length=30)
    features: dict[str, float]
    event_timestamp: datetime | None = None
    recording_id: str | None = Field(default=None, max_length=200)
    window_index: int | None = Field(default=None, ge=0)
    replay_session_id: str | None = Field(default=None, max_length=100)
    sequence_number: int | None = Field(default=None, ge=0)

    @field_validator("features")
    @classmethod
    def finite_feature_values(cls, values: dict[str, float]) -> dict[str, float]:
        if not values:
            raise ValueError("At least one feature is required")
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("Feature values must all be finite numbers")
        return values


class ReplayStartRequest(BaseModel):
    machine_id: str = Field(default="machine-001", min_length=1, max_length=100)
    recording_ids: list[str] | None = None
    speed: float = Field(default=1.0, gt=0, le=100)
    delay_seconds: float | None = Field(default=None, ge=0, le=3600)
    mode: Literal["local", "mqtt"] = "local"
    continuous: bool = True


class HumanReviewRequest(BaseModel):
    review_label: Literal["Normal", "BPFI", "BPFO", "Misalignment", "Unbalance"]
    reviewed_by: str = Field(min_length=1, max_length=100)
    review_note: str | None = Field(default=None, max_length=2000)
