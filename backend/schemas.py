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


class HealthCheckResponse(BaseModel):
    status: str = Field(default="healthy", description="API sağlık durumu")
    message: str = Field(
        default="Tüm ML modelleri aktif ve çalışır durumda",
        description="Bilgilendirme mesajı",
    )


# Bankloan için giriş şemasını buraya taşıyoruz
class BankloanInput(BaseModel):
    Income: float = Field(..., description="Annual Income ($k)")
    CCAvg: float = Field(..., description="Avg. Monthly Credit Card Spend ($k)")
    Mortgage: float = Field(..., description="House Mortgage Value ($k)")
    Education: int = Field(..., description="Education Level (1, 2, 3)")
    CD_Account: int = Field(..., description="Certificate of Deposit Account (0 or 1)")