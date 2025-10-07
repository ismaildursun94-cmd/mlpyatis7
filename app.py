import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from proje import tahmin_et  # gerçek fonksiyon

app = FastAPI(title="LOS Predictor API", version="1.0.0")

class PredictRequest(BaseModel):
    icd_list: List[str]
    bolum: Optional[str] = None
    yas_grup: Optional[str] = None

    @field_validator("icd_list")
    @classmethod
    def _not_empty(cls, v):
        if not v:
            raise ValueError("icd_list boş olamaz")
        return [s.strip() for s in v if s and s.strip()]

@app.get("/")
def health():
    return {"status": "up", "service": "los-predictor", "version": "1.0.0"}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return tahmin_et(req.icd_list, req.bolum, req.yas_grup)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hata: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=1)