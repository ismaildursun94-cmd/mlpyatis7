# app.py
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, field_validator
from proje import tahmin_et  # sadece Pred_Final_Rounded dönen fonksiyon

app = FastAPI(title="LOS Predictor API", version="1.0.0")

# -------------------------------
# 1) API Modeli (JSON istekleri için)
# -------------------------------
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


# -------------------------------
# 2) Ana Sayfa (yalın arayüz)
# -------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
    <head>
        <title>LOS Predictor</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 520px; margin: 40px auto;
                   background: #fafafa; padding: 20px; border-radius: 10px; }
            input, button { padding: 10px; margin-top: 8px; width: 100%; box-sizing: border-box; }
            button { background: #0a66ff; color: #fff; border: none; cursor: pointer; font-weight: 600; }
            button:hover { background: #084dcc; }
            #result { font-size: 28px; font-weight: 700; margin-top: 14px; }
            label { font-size: 12px; color: #333; }
        </style>
    </head>
    <body>
        <h2>LOS Tahmin Arayüzü</h2>
        <form id="predictForm">
            <label>ICD Listesi (virgülle):</label>
            <input type="text" id="icd_list" name="icd_list" value="K80.0">

            <label>Bölüm (ops.):</label>
            <input type="text" id="bolum" name="bolum" placeholder="örn. Kardiyoloji">

            <label>Yaş Grubu (ops.):</label>
            <input type="text" id="yas_grup" name="yas_grup" placeholder="örn. 0-1">

            <button type="submit">Tahmin Et</button>
        </form>

        <div id="result"></div>

        <script>
        document.getElementById("predictForm").addEventListener("submit", async (e) => {
            e.preventDefault();
            const icd = document.getElementById("icd_list").value.split(",").map(x => x.trim()).filter(x => x);
            const bolum = document.getElementById("bolum").value.trim();
            const yas   = document.getElementById("yas_grup").value.trim();

            const body = { icd_list: icd, bolum: bolum || null, yas_grup: yas || null };
            const resEl = document.getElementById("result");
            resEl.textContent = "…";

            try {
                const res = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(body)
                });
                const text = await res.text();   // sadece sayı dönüyor
                resEl.textContent = text;
            } catch (err) {
                resEl.textContent = "Hata";
            }
        });
        </script>
    </body>
    </html>
    """


# -------------------------------
# 3) Sağlık kontrolü
# -------------------------------
@app.get("/health")
def health():
    return {"status": "up", "service": "los-predictor", "version": "1.0.0"}


# -------------------------------
# 4) Tahmin (JSON → düz metin sayı)
# -------------------------------
@app.post("/predict", response_class=PlainTextResponse)
def predict(req: PredictRequest):
    try:
        out = tahmin_et(req.icd_list, req.bolum, req.yas_grup)
        val = out.get("Pred_Final_Rounded", None)
        if val is None:
            raise RuntimeError("Pred_Final_Rounded üretilemedi.")
        return str(val)  # sadece sayı
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hata: {e}")


# -------------------------------
# 5) Form-POST isteyenler için alternatif (opsiyonel)
# -------------------------------
@app.post("/tahmin", response_class=PlainTextResponse)
def tahmin_form(
    icd_text: str = Form(...),
    bolum: Optional[str] = Form(None),
    yas_grup: Optional[str] = Form(None),
):
    try:
        icds = [s.strip() for s in icd_text.split(",") if s.strip()]
        out = tahmin_et(icds, bolum, yas_grup)
        val = out.get("Pred_Final_Rounded", None)
        if val is None:
            raise RuntimeError("Pred_Final_Rounded üretilemedi.")
        return str(val)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hata: {e}")


# -------------------------------
# 6) Çalıştırıcı
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=1)