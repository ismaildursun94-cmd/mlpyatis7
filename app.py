# app.py
import os
import re
import threading
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import (
    HTMLResponse,
    PlainTextResponse,
    JSONResponse,
    FileResponse,
)
from pydantic import BaseModel, field_validator

# proje.py'den tahmin fonksiyonları ve dosya adları
from proje import tahmin_et, app_predict, app_info, run_training_pipeline

# Bu değişkenler proje.py'de global; import başarısızsa varsayılanlara düşeceğiz


@app.head("/", response_class=PlainTextResponse)
def index_head():
    return ""  # 200 OK; Render'ın HEAD health-check'i 405 görmesin

try:
    from proje import (
        LOOKUP_XLSX,
        PRED_LOS_XLSX,
        VALID_PRED_XLSX,
        YENI_VAKALAR_XLSX,
        MODEL_DIR,
    )
except Exception:
    LOOKUP_XLSX = "LOS_Lookup_All.xlsx"
    PRED_LOS_XLSX = "PRED_LOS.xlsx"
    VALID_PRED_XLSX = "VALID_PREDICTIONS.xlsx"
    YENI_VAKALAR_XLSX = "YeniVakalar.xlsx"
    MODEL_DIR = "model_out"

app = FastAPI(title="LOS Predictor API", version="1.2.0")

# ---- Isınma durumu (global bayrak) ----
MODEL_READY: bool = False
MODEL_MODE: str = os.environ.get("MODE", "train").lower()  # "train" | "load" (ileride load_from_artifacts eklenebilir)
MODEL_ERROR: Optional[str] = None


def _background_warmup():
    """
    Ağır işlemi (eğitim/ısıtma) ana thread'i bloklamadan çalıştır.
    Render gibi platformlarda port taraması böylece zaman aşımına düşmez.
    """
    global MODEL_READY, MODEL_ERROR
    try:
        if MODEL_MODE == "train":
            print("[WARMUP] Training pipeline starting...")
            run_training_pipeline()
            print("[WARMUP] Training pipeline done.")
        else:
            # Buraya ileride load_from_artifacts(LOOKUP_XLSX, MODEL_DIR) koyabiliriz
            print("[WARMUP] Load mode selected, but loader not implemented yet; running training as fallback.")
            run_training_pipeline()
        MODEL_READY = True
        MODEL_ERROR = None
    except Exception as e:
        MODEL_ERROR = f"{type(e).__name__}: {e}"
        MODEL_READY = False
        print("[WARMUP][ERROR]", MODEL_ERROR)


# Sunucu ayağa kalkarken arka planda ısınma başlat
@app.on_event("startup")
async def _warmup():
    threading.Thread(target=_background_warmup, daemon=True).start()


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
            a.dl { color:#9ad; text-decoration:none; }
            .muted { color:#aaa; font-size:12px; }
        </style>
    </head>
    <body>
        <h2>LOS Tahmin Arayüzü</h2>
        <div class="muted">Sunucu durumu: <span id="ready">kontrol ediliyor…</span></div>
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

        <p>
          <small>
            Hızlı indirme: 
            <a class="dl" href="/download/lookup">Lookup XLSX</a> ·
            <a class="dl" href="/download/pred_los">PRED_LOS.xlsx</a> ·
            <a class="dl" href="/download/valid">VALID_PREDICTIONS.xlsx</a> ·
            <a class="dl" href="/download/yeni">YeniVakalar.xlsx</a>
          </small>
        </p>

        <script>
        async function pingReady(){
          try{
            const r = await fetch("/ready");
            const j = await r.json();
            document.getElementById("ready").textContent = j.ready ? "hazır" : (j.error ? "hata: "+j.error : "ısınma");
          }catch(e){
            document.getElementById("ready").textContent = "bilinmiyor";
          }
        }
        pingReady(); setInterval(pingReady, 2000);

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

@app.get("/ready", response_class=JSONResponse)
def ready():
    return {"ready": MODEL_READY, "mode": MODEL_MODE, "error": MODEL_ERROR}


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
    """Sadece sayısal değer döndürür (Pred_Final_Rounded)."""
    if not MODEL_READY:
        # Eğitim bitmeden tahmin isteme → 503
        msg = "Model hazır değil (ısınma sürüyor)." + (f" Hata: {MODEL_ERROR}" if MODEL_ERROR else "")
        raise HTTPException(status_code=503, detail=msg)

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
    """Tüm detayları (ANCHOR_SRC, ANCHOR_P50, model skorları, final harman) döndürür."""
    if not MODEL_READY:
        msg = "Model hazır değil (ısınma sürüyor)." + (f" Hata: {MODEL_ERROR}" if MODEL_ERROR else "")
        raise HTTPException(status_code=503, detail=msg)

    try:
        out = app_predict(req.yas_grup or "", req.bolum or "", req.icd_list)
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hata: {e}")


# -------------------------------
# 6) Dosya indirme uçları
# -------------------------------
def _file_or_404(path: str, download_name: Optional[str] = None) -> FileResponse:
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Dosya bulunamadı: {os.path.basename(path)}")
    return FileResponse(path, filename=download_name or os.path.basename(path))


@app.get("/download/lookup")
def download_lookup():
    return _file_or_404(LOOKUP_XLSX)


@app.get("/download/pred_los")
def download_pred_los():
    return _file_or_404(PRED_LOS_XLSX)


@app.get("/download/valid")
def download_valid():
    return _file_or_404(VALID_PRED_XLSX)


@app.get("/download/yeni")
def download_yeni():
    return _file_or_404(YENI_VAKALAR_XLSX)


@app.get("/download/model/{name}")
def download_model_file(name: str):
    # model_out klasörü içinden güvenli isimler
    safe = re.fullmatch(r"[A-Za-z0-9_.\-]+", name)
    if not safe:
        raise HTTPException(status_code=400, detail="Geçersiz dosya adı")
    path = os.path.join(MODEL_DIR, name)
    return _file_or_404(path)


# -------------------------------
# 7) Form-POST (opsiyonel)
# -------------------------------
@app.post("/tahmin", response_class=PlainTextResponse)
def tahmin_form(
    icd_text: str = Form(...),
    bolum: Optional[str] = Form(None),
    yas_grup: Optional[str] = Form(None),
):
    if not MODEL_READY:
        msg = "Model hazır değil (ısınma sürüyor)." + (f" Hata: {MODEL_ERROR}" if MODEL_ERROR else "")
        raise HTTPException(status_code=503, detail=msg)

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
# 8) Çalıştırıcı (lokal)
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=1)