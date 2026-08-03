from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class StandardResponse(BaseModel):
    status: str = Field(
        default="success", description="Success status of the operation"
    )
    prediction: Any = Field(
        ..., description="The prediction result made by the model"
    )
    probability: Optional[Dict[str, float]] = Field(
        default=None, description="Classification probability distribution"
    )


class BankloanInput(BaseModel):
    Income: float = Field(..., description="Annual Income ($k)")
    CCAvg: float = Field(..., description="Avg. Monthly Credit Card Spend ($k)")
    Mortgage: float = Field(..., description="House Mortgage Value ($k)")
    Education: int = Field(..., description="Education Level (1, 2, 3)")
    CD_Account: int = Field(..., description="Certificate of Deposit Account (0 or 1)")


class StudentInput(BaseModel):
    study_hours: float = Field(..., description="Daily study hours")
    assignments_completed: int = Field(..., description="Number of completed assignments")
    previous_score: float = Field(..., description="Previous exam score or GPA")
    attendance: float = Field(..., description="Attendance rate percentage")
    sleep_hours: float = Field(..., description="Daily sleep hours")


class CancerInput(BaseModel):
    radius_mean: float
    texture_mean: float
    smoothness_mean: float
    compactness_mean: float
    radius_se: float
    compactness_se: float
    concave_points_se: float = Field(..., alias="concave points_se")
    smoothness_worst: float
    symmetry_worst: float
    radius_worst: float

    class Config:
        populate_by_name = True


class DrugInput(BaseModel):
    Age: int
    BP: str
    Cholesterol: str
    Na_to_K: float


class TitanicInput(BaseModel):
  Pclass: int
  Title: str  # İsim yerine doğrudan ünvanı alıyoruz
  Sex: str
  Age: float
  SibSp: int
  Parch: int
  Ticket: Optional[str] = "None"
  Fare: float
  Cabin: Optional[str] = "None"
  Embarked: str