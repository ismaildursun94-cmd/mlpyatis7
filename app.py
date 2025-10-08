import os
import re
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel, field_validator

# proje.py'den adapter + debug amaçlı zengin JSON dönen fonksiyonları alıyoruz
from proje import tahmin_et, app_predict, app_info

app = FastAPI(title="LOS Predictor API", version="1.2.0")

# ---- ICD listesi için kaynak Excel (proje.py'nin ürettiği) ----
ICD_XLSX = os.environ.get("LOOKUP_XLSX", "LOS_Lookup_All.xlsx")
_ICD_CACHE: Optional[List[str]] = None  # lazy cache


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
        # Güvenlik: boş/None elemanları at
        return [str(s).strip() for s in v if s and str(s).strip()]


# -------------------------------
# 2) ICD seçeneklerini ver (checkbox'lar bunu çağıracak)
# -------------------------------
@app.get("/icd_list", response_class=JSONResponse)
def icd_list_endpoint():
    global _ICD_CACHE
    if _ICD_CACHE is None:
        try:
            import pandas as pd
            df = pd.read_excel(ICD_XLSX, sheet_name="DIM_ICD")
            _ICD_CACHE = sorted(
                [str(x).strip() for x in df["ICD"].dropna().unique().tolist() if str(x).strip()]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ICD listesi okunamadı: {e}")
    return _ICD_CACHE


# -------------------------------
# 3) Ana Sayfa (checkbox + arama + metin fallback)
# -------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
    <head>
        <title>LOS Predictor</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; max-width: 960px; margin: 40px auto;
                   background: #2f2f2f; color: #eee; padding: 24px; border-radius: 12px; }
            input, button, textarea { padding: 12px; margin-top: 8px; width: 100%; box-sizing: border-box;
                            border-radius: 8px; border: 1px solid #555; background:#3b3b3b; color:#eee; }
            textarea { min-height: 64px; }
            button { background: #0a66ff; color: #fff; border: none; cursor: pointer; font-weight: 700; }
            button:hover { background: #0a55d3; }
            #result { font-size: 32px; font-weight: 800; margin-top: 16px; }
            label { font-size: 12px; color: #bbb; display:block; margin-top: 8px; }
            small { color:#b9a97c }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
            .pill { display:inline-block; background:#444; padding:4px 8px; border-radius:999px; margin:2px; font-size:12px;}
            .box { background:#262626; padding:12px; border-radius:8px; border:1px solid #444; }
            .list { max-height: 360px; overflow: auto; border:1px solid #555; border-radius:8px; padding:8px; background:#1e1e1e; }
            .row { display:flex; align-items:center; gap:8px; margin:4px 0; }
            .muted { color:#aaa; font-size:12px; }
        </style>
    </head>
    <body>
        <h2>LOS Tahmin Arayüzü</h2>

        <div class="grid">
          <div class="box">
            <h3>ICD Seçici (Arama + Checkbox)</h3>
            <input type="text" id="icd_search" placeholder="ICD ara (örn: K80)">
            <div id="icd_list_box" class="list"></div>
            <div id="picked" class="muted" style="margin-top:8px;"></div>
          </div>

          <div class="box">
            <h3>Metin ile Giriş (opsiyonel)</h3>
            <label>ICD Listesi (virgül / ; / | / boşluk ile ayırabilirsiniz):</label>
            <textarea id="icd_text">K80.0</textarea>

            <div class="grid">
                <div>
                    <label>Bölüm (ops.):</label>
                    <input type="text" id="bolum" placeholder="örn. Kardiyoloji">
                </div>
                <div>
                    <label>Yaş Grubu (ops.):</label>
                    <input type="text" id="yas_grup" placeholder="örn. 0-1">
                </div>
            </div>

            <button id="btn_predict">Tahmin Et</button>
            <div id="result"></div>
            <p class="muted">
              Not: <code>/predict</code> sadece sayıyı döndürür. Tam JSON ve ANCHOR bilgileri için
              <code>POST /predict_json</code> kullanın.
            </p>
          </div>
        </div>

        <script>
        let ICD_OPTS = [];
        let SELECTED = new Set();

        // Metni güvenli şekilde ICD listesine çevir (virgül, noktalı virgül, tek/çift pipe ve boşluklar)
        function parseIcdInput(raw) {
            return (raw || "")
              .replace(/\\|\\|/g, ",")                 // "||" -> ","
              .split(/[\\,;|\\s]+/)                    // virgül/; / | / whitespace
              .map(s => s.trim())
              .filter(s => s.length > 0);
        }

        function renderList(filter="") {
            const box = document.getElementById("icd_list_box");
            box.innerHTML = "";
            const f = filter.trim().toLowerCase();
            const items = f ? ICD_OPTS.filter(x => x.toLowerCase().includes(f)) : ICD_OPTS;
            for (const code of items) {
                const id = "chk_" + code.replace(/[^a-zA-Z0-9]/g, "_");
                const row = document.createElement("div");
                row.className = "row";
                row.innerHTML = \`
                    <input type="checkbox" id="\${id}" \${SELECTED.has(code) ? "checked": ""}>
                    <label for="\${id}" style="margin:0;">\${code}</label>
                \`;
                const cb = row.querySelector("input");
                cb.addEventListener("change", () => {
                    if (cb.checked) SELECTED.add(code); else SELECTED.delete(code);
                    renderPicked();
                });
                box.appendChild(row);
            }
        }

        function renderPicked() {
            const el = document.getElementById("picked");
            if (SELECTED.size === 0) {
                el.innerHTML = "<span class='muted'>Seçili ICD yok</span>";
                return;
            }
            el.innerHTML = "<b>Seçilenler:</b> " + Array.from(SELECTED).sort()
                .map(x => "<span class='pill'>" + x + "</span>").join(" ");
        }

        async function loadICD() {
            try {
                const res = await fetch("/icd_list");
                if (!res.ok) throw new Error("ICD listesi alınamadı");
                ICD_OPTS = await res.json();
                renderList();
                renderPicked();
            } catch (e) {
                document.getElementById("icd_list_box").innerHTML =
                    "<div class='muted'>ICD listesi yüklenemedi: " + e.message + "</div>";
            }
        }

        document.getElementById("icd_search").addEventListener("input", (e) => {
            renderList(e.target.value || "");
        });

        document.getElementById("btn_predict").addEventListener("click", async () => {
            // checkbox + metin girişini birleştir
            const fromChecks = Array.from(SELECTED);
            const fromText   = parseIcdInput(document.getElementById("icd_text").value);
            const icd = Array.from(new Set([...fromChecks, ...fromText]));

            if (icd.length === 0) {
                document.getElementById("result").textContent = "Lütfen en az bir ICD seçin / girin.";
                return;
            }

            const bolum = (document.getElementById("bolum").value || "").trim();
            const yas   = (document.getElementById("yas_grup").value || "").trim();

            const body = { icd_list: icd, bolum: bolum || null, yas_grup: yas || null };
            const resEl = document.getElementById("result");
            resEl.textContent = "…";

            try {
                const res = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(body)
                });
                if (!res.ok) {
                    const j = await res.json().catch(() => ({}));
                    throw new Error(j.detail || ("HTTP " + res.status));
                }
                const text = await res.text();   // sadece sayı dönüyor
                resEl.textContent = text;
            } catch (err) {
                resEl.textContent = "Hata: " + err.message;
            }
        });

        // sayfa açılışında ICD’leri yükle
        loadICD();
        </script>
    </body>
    </html>
    """


# -------------------------------
# 4) Sağlık / bilgi
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
# 5) Tahmin (JSON → düz metin sayı)
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
# 6) Tam JSON isteyenler için
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
# 7) Form-POST isteyenler için alternatif (opsiyonel)
# -------------------------------
@app.post("/tahmin", response_class=PlainTextResponse)
def tahmin_form(
    icd_text: str = Form(...),
    bolum: Optional[str] = Form(None),
    yas_grup: Optional[str] = Form(None),
):
    """
    Form gönderimiyle gelen tek satırlık metni robust şekilde parçalar ve tahmin döndürür.
    """
    try:
        # "A||B" ve tek |, ayrıca virgül/; ve boşlukları da ayrıştır
        icds = [s.strip() for s in re.split(r"[,\;\|\s]+", re.sub(r"\|\|", ",", icd_text or "")) if s.strip()]
        out = tahmin_et(icds, bolum, yas_grup)
        val = out.get("Pred_Final_Rounded", None)
        if val is None:
            raise RuntimeError("Pred_Final_Rounded üretilemedi.")
        return str(val)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hata: {e}")


# -------------------------------
# 8) Çalıştırıcı
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=1)