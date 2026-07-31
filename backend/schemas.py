from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class StandardResponse(BaseModel):
    status: str = Field(
        default="success", description="İşlemin başarı durumu"
    )
    prediction: Any = Field(..., description="Modelin yaptığı tahmin sonucu")
    probability: Optional[Dict[str, float]] = Field(
        default=None, description="Sınıflandırma olasılık dağılımı"
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

class DrugInput(BaseModel):
    Age: int
    BP: str  # Örn: 'HIGH', 'LOW', 'NORMAL'
    Cholesterol: str  # Örn: 'HIGH', 'NORMAL'
    Na_to_K: float