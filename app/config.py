from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Industrial Predictive Maintenance and Fault Intelligence Platform"
    database_url: str = f"sqlite:///{(ROOT / 'data' / 'app.db').as_posix()}"
    dataset_path: Path = ROOT / "Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating Machine Under Varying Load Conditions for Fault Diagnosis"
    artifact_dir: Path = ROOT / "artifacts"
    report_dir: Path = ROOT / "reports"
    feature_schema_version: str = "1.0"
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_username: str | None = None
    mqtt_password: str | None = None
    mqtt_tls: bool = False
    mqtt_features_topic: str = "machine/features"
    mqtt_predictions_topic: str = "machine/predictions"
    mqtt_events_topic: str = "machine/events"
    mqtt_client_id: str = "motor-monitoring-prediction-worker"
    admin_api_key: str | None = None
    stale_after_seconds: int = 120
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080"
    selective_target_accuracy: float = Field(default=0.90, gt=0, le=1)
    selective_minimum_accepted: int = Field(default=10, ge=1)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    (ROOT / "data").mkdir(exist_ok=True)
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    return settings
