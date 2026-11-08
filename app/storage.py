from __future__ import annotations

from datetime import datetime, timezone
import json

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import get_settings


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    machine_id: Mapped[str] = mapped_column(String(100), index=True)
    source: Mapped[str] = mapped_column(String(20), index=True)
    recording_id: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    window_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    predicted_condition: Mapped[str] = mapped_column(String(50), index=True)
    probabilities_json: Mapped[str] = mapped_column(Text)
    features_json: Mapped[str] = mapped_column(Text)
    model_version: Mapped[str] = mapped_column(String(100))
    schema_version: Mapped[str] = mapped_column(String(30))
    event_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    latency_ms: Mapped[float] = mapped_column(Float)
    reliability_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    calibrated_probabilities_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    explanation_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    decision_status: Mapped[str] = mapped_column(String(40), default="ACCEPTED", index=True)
    review_label: Mapped[str | None] = mapped_column(String(50), nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    @property
    def probabilities(self) -> dict[str, float]:
        return json.loads(self.probabilities_json)

    @property
    def features(self) -> dict[str, float]:
        return json.loads(self.features_json)

    @property
    def reliability(self) -> dict:
        return json.loads(self.reliability_json) if self.reliability_json else {}

    @property
    def calibrated_probabilities(self) -> dict[str, float]:
        return json.loads(self.calibrated_probabilities_json) if self.calibrated_probabilities_json else {}

    @property
    def explanation(self) -> dict:
        return json.loads(self.explanation_json) if self.explanation_json else {}


class AppEvent(Base):
    __tablename__ = "system_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_type: Mapped[str] = mapped_column(String(60), index=True)
    severity: Mapped[str] = mapped_column(String(20), default="info")
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    algorithm: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(30), index=True)
    artifact_path: Mapped[str] = mapped_column(Text)
    checksum: Mapped[str] = mapped_column(String(64))
    metrics_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[str] = mapped_column(String(30), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


def make_engine():
    settings = get_settings()
    options = {"pool_pre_ping": True}
    if settings.database_url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False}
    return create_engine(settings.database_url, **options)


engine = make_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def initialize_database() -> None:
    Base.metadata.create_all(bind=engine)
