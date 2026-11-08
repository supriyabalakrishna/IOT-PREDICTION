"""Persist reliability decisions and human review metadata."""

from alembic import op
import sqlalchemy as sa

revision = "0002_reliability_metadata"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade():
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("predictions")}
    additions = (
        ("reliability_json", sa.Text(), True, None),
        ("calibrated_probabilities_json", sa.Text(), True, None),
        ("explanation_json", sa.Text(), True, None),
        ("decision_status", sa.String(length=40), False, "ACCEPTED"),
        ("review_label", sa.String(length=50), True, None),
        ("reviewed_by", sa.String(length=100), True, None),
        ("review_note", sa.Text(), True, None),
        ("reviewed_at", sa.DateTime(timezone=True), True, None),
    )
    for name, column_type, nullable, server_default in additions:
        if name not in columns:
            op.add_column(
                "predictions",
                sa.Column(name, column_type, nullable=nullable, server_default=server_default),
            )
    indexes = {index["name"] for index in inspector.get_indexes("predictions")}
    if "ix_predictions_decision_status" not in indexes:
        op.create_index("ix_predictions_decision_status", "predictions", ["decision_status"])


def downgrade():
    inspector = sa.inspect(op.get_bind())
    indexes = {index["name"] for index in inspector.get_indexes("predictions")}
    if "ix_predictions_decision_status" in indexes:
        op.drop_index("ix_predictions_decision_status", table_name="predictions")
    columns = {column["name"] for column in inspector.get_columns("predictions")}
    for name in (
        "reviewed_at",
        "review_note",
        "reviewed_by",
        "review_label",
        "decision_status",
        "explanation_json",
        "calibrated_probabilities_json",
        "reliability_json",
    ):
        if name in columns:
            op.drop_column("predictions", name)
