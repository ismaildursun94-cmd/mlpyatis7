mport os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator
from proje import tahmin_et  # gerçek fonksiyon

app = FastAPI(title="LOS Predictor API", version="1.0.0")

# -------------------------------
# 1️⃣ API Modeli
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
# 2️⃣ Ana Sayfa (HTML Arayüz)
# -------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
    <head>
        <title>LOS Predictor</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; background: #fafafa; padding: 20px; border-radius: 10px; }
            input, button { padding: 8px; margin-top: 8px; width: 100%; }
            button { background: #007bff; color: white; border: none; cursor: pointer; font-weight: bold; }
            button:hover { background: #0056b3; }
            pre { background: #eee; padding: 10px; border-radius: 6px; }
        </style>
    </head>
    <body>
        <h2>LOS Tahmin Arayüzü</h2>
        <form id="predictForm">
            <label>ICD Listesi (virgülle):</label>
            <input type="text" id="icd_list" name="icd_list" value="K11, A00.1">

            <label>Bölüm:</label>
            <input type="text" id="bolum" name="bolum" value="Kardiyoloji">

            <label>Yaş Grubu:</label>
            <input type="text" id="yas_grup" name="yas_grup" value="0-1">

            <button type="submit">Tahmin Et</button>
        </form>
        <h3>Sonuç:</h3>
        <pre id="result">Henüz sorgulanmadı...</pre>

        <script>
        document.getElementById("predictForm").addEventListener("submit", async (e) => {
            e.preventDefault();
            const icd = document.getElementById("icd_list").value.split(",").map(x => x.trim());
            const bolum = document.getElementById("bolum").value.trim();
            const yas = document.getElementById("yas_grup").value.trim();

            const body = { icd_list: icd, bolum: bolum || null, yas_grup: yas || null };
            document.getElementById("result").innerText = "Tahmin ediliyor...";

            try {
                const res = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(body)
                });
                const data = await res.json();
                document.getElementById("result").innerText = JSON.stringify(data, null, 2);
            } catch (err) {
                document.getElementById("result").innerText = "Hata: " + err;
            }
        });
        </script>
    </body>
    </html>
    """


# -------------------------------
# 3️⃣ Sağlık kontrolü (Render test)
# -------------------------------
@app.get("/health")
def health():
    return {"status": "up", "service": "los-predictor", "version": "1.0.0"}


# -------------------------------
# 4️⃣ Tahmin Endpoint
# -------------------------------
@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return tahmin_et(req.icd_list, req.bolum, req.yas_grup)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hata: {e}")


# -------------------------------
# 5️⃣ Çalıştırıcı
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=1)