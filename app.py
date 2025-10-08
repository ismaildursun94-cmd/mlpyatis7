# app.py
import os
from typing import List, Optional, Union
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel, field_validator
import re

# proje.py'den adapter + debug amaçlı zengin JSON dönen fonksiyonları alıyoruz
from proje import tahmin_et, app_predict, app_info

app = FastAPI(title="LOS Predictor API", version="1.1.1")

# -------------------------------
# 1) API Modeli (JSON istekleri için)
# -------------------------------
class PredictRequest(BaseModel):
    # icd_list string veya liste gelebilir: "A04||K30, R51  R90.0" ya da ["A04","K30",...]
    icd_list: Union[str, List[str]]
    bolum: Optional[str] = None
    yas_grup: Optional[str] = None

    # JSON parse edilmeden ÖNCE string/listeyi normalize et
    @field_validator("icd_list", mode="before")
    @classmethod
    def _coerce_icd_list(cls, v):
        def split_any(s: str):
            s = re.sub(r"\|\|", "|", s or "")
            return [p.strip() for p in re.split(r"[,\;\|\s]+", s) if p and p.strip()]
        if isinstance(v, str):
            return split_any(v)
        elif isinstance(v, (list, tuple, set)):
            tmp = []
            for it in v:
                if isinstance(it, str):
                    tmp.extend(split_any(it))
                else:
                    tmp.append(str(it))
            return tmp
        return []

    # Boş olamaz + trimle
    @field_validator("icd_list")
    @classmethod
    def _not_empty(cls, v: List[str]):
        v = [s.strip() for s in v if s and s.strip()]
        if not v:
            raise ValueError("icd_list boş olamaz")
        return v

# -------------------------------
# 2) Ana Sayfa (yalın arayüz)
# -------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
    <head>
        <title>LOS Predictor</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; max-width: 620px; margin: 40px auto;
                   background: #2f2f2f; color: #eee; padding: 24px; border-radius: 12px; }
            input, button { padding: 12px; margin-top: 8px; width: 100%; box-sizing: border-box;
                            border-radius: 8px; border: 1px solid #555; background:#3b3b3b; color:#eee; }
            button { background: #0a66ff; color: #fff; border: none; cursor: pointer; font-weight: 700; }
            button:hover { background: #0a55d3; }
            #result { font-size: 32px; font-weight: 800; margin-top: 16px; }
            label { font-size: 12px; color: #bbb; }
            small { color:#b9a97c }
        </style>
    </head>
    <body>
        <h2>LOS Tahmin Arayüzü</h2>
        <form id="predictForm">
            <label>ICD Listesi (virgül, ||, boşluk vs. hepsi olur):</label>
            <input type="text" id="icd_list" name="icd_list" value="A00||C91.0 T81.4, Z94.8">

            <label>Bölüm (ops.):</label>
            <input type="text" id="bolum" name="bolum" placeholder="örn. Kardiyoloji">

            <label>Yaş Grubu (ops.):</label>
            <input type="text" id="yas_grup" name="yas_grup" placeholder="örn. 0-1">

            <button type="submit">Tahmin Et</button>
        </form>

        <div id="result"></div>
        <p><small>/predict sadece sayı döndürür. Tam JSON ve ANCHOR bilgileri için <code>POST /predict_json</code> kullan.</small></p>

        <script>
        function splitAny(s) {
          if (!s) return [];
          s = s.replaceAll("||","|");
          return s.split(/[,;|\\s]+/).map(x => x.trim()).filter(Boolean);
        }

        document.getElementById("predictForm").addEventListener("submit", async (e) => {
            e.preventDefault();
            const icdRaw = document.getElementById("icd_list").value;
            const icd = splitAny(icdRaw);

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
                if(!res.ok){
                  const j = await res.json().catch(()=>({}));
                  resEl.textContent = "Hata: " + (j.detail || res.status);
                  return;
                }
                const text = await res.text();   // sadece sayı dönüyor
                resEl.textContent = text;
            } catch (err) {
                resEl.textContent = "Hata: " + (err?.message || err);
            }
        });
        </script>
    </body>
    </html>
    """

# -------------------------------
# 3) Sağlık / bilgi
# -------------------------------
@app.get("/health")
def health():
    return {"status": "up", "service": "los-predictor", "version": app.version}

@app.get("/info", response_class=JSONResponse)
def info():
    try:
        return app_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hata: {e}")

# -------------------------------
# 4) Tahmin (JSON → düz metin sayı)
# -------------------------------
@app.post("/predict", response_class=PlainTextResponse)
def predict(req: PredictRequest):
    """
    Sadece sayısal değer döndürür (Pred_Final_Rounded).
    """
    try:
        out = tahmin_et(req.icd_list, req.bolum, req.yas_grup)
        val = out.get("Pred_Final_Rounded", None)
        if val is None:
            raise RuntimeError("Pred_Final_Rounded üretilemedi.")
        return str(val)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hata: {e}")

# -------------------------------
# 5) Tam JSON isteyenler için
# -------------------------------
@app.post("/predict_json", response_class=JSONResponse)
def predict_json(req: PredictRequest):
    """
    Tüm detayları (ANCHOR_SRC, ANCHOR_P50, model skorları, final harman) döndürür.
    """
    try:
        out = app_predict(req.yas_grup or "", req.bolum or "", req.icd_list)
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hata: {e}")

# -------------------------------
# 6) Form-POST (opsiyonel)
# -------------------------------
@app.post("/tahmin", response_class=PlainTextResponse)
def tahmin_form(
    icd_text: str = Form(...),
    bolum: Optional[str] = Form(None),
    yas_grup: Optional[str] = Form(None),
):
    try:
        # Formdan gelen serbest metni de aynı kuralla parçala
        s = re.sub(r"\|\|", "|", icd_text or "")
        icds = [p.strip() for p in re.split(r"[,\;\|\s]+", s) if p.strip()]
        out = tahmin_et(icds, bolum, yas_grup)
        val = out.get("Pred_Final_Rounded", None)
        if val is None:
            raise RuntimeError("Pred_Final_Rounded üretilemedi.")
        return str(val)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hata: {e}")

# -------------------------------
# 7) Çalıştırıcı
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=1)