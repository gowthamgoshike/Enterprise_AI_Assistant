from pydantic import BaseModel, Field


class FailedReportRequest(BaseModel):
    days: int = Field(default=30, ge=1, le=90)
