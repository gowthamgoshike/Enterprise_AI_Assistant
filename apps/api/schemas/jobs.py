from pydantic import BaseModel, ConfigDict, Field


class FailedReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    days: int = Field(default=30, ge=1, le=90)
