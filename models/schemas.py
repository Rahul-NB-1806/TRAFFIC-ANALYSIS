from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class TrafficRow(BaseModel):
    timestamp: datetime
    location: str
    two_wheeler: int = Field(ge=0)
    four_wheeler: int = Field(ge=0)
    heavy_vehicle: int = Field(ge=0)
    emergency_vehicle: int = Field(ge=0)


class UploadReport(BaseModel):
    filename: str
    rows_ingested: int
    errors: list[str] = []
    locations: list[str] = []
    time_range: Optional[tuple] = None
    total_vehicles: int = 0


class AnalyticsResult(BaseModel):
    pass
