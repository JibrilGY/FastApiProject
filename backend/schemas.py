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