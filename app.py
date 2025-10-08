# app.py
import os
import re
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel, field_validator
import pandas as pd

from proje import tahmin_et, app_predict, app_info

LOOKUP_XLSX = os.environ.get("LOOKUP_XLSX", "LOS_Lookup_All.xlsx")

app = FastAPI(title="LOS Predictor API", version="1.3.0")

# -------------------------------
# 1) API Modeli
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
        return [str(s).strip() for s in v if s and str(s).strip()]

# -------------------------------
# 2) Ana Sayfa (Frontend)
# -------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8" />
<title>LOS Tahmin Arayüzü</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root{
    --glass: rgba(255,255,255,.85);
    --text: #1f2937;
    --muted:#6b7280;
    --accent:#2563eb;
  }
  *{box-sizing:border-box}
  body{
    margin:0; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial;
    color:var(--text);
    min-height:100vh;
    background:#0b1725;
  }
  .wrap{ max-width:1100px; margin:40px auto; padding:24px; }
  .panel{
    backdrop-filter: blur(6px);
    background: var(--glass);
    border-radius: 14px;
    padding: 24px;
    box-shadow: 0 10px 30px rgba(0,0,0,.15);
  }
  h1{ font-size:28px; margin:0 0 16px 0; color:#111827}
  .grid{ display:grid; grid-template-columns: 1fr; gap:16px; }
  @media(min-width:900px){ .grid{ grid-template-columns: 1fr 1fr; } }
  label{ font-size:13px; color:var(--muted); display:block; margin-bottom:6px; }
  select, input[type="text"]{
    width:100%; height:44px; padding:10px 12px; border-radius:10px; border:1px solid #e5e7eb; background:#fff;
    outline:none;
  }
  .btn{
    height:46px; border:none; background:var(--accent); color:#fff; font-weight:700; border-radius:10px; cursor:pointer;
  }
  .btn:disabled{ opacity:.6; cursor:not-allowed; }
  .chipbar{ display:flex; flex-wrap:wrap; gap:8px; margin-top:8px; }
  .chip{ background:#eef2ff; color:#3730a3; padding:4px 8px; border-radius:999px; font-size:12px; }
</style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <h1>LOS Tahmin Arayüzü</h1>
      <div class="grid">
        <div>
          <label>Bölüm</label>
          <select id="bolum"></select>

          <label style="margin-top:14px;">Yaş Aralığı</label>
          <select id="yas"></select>

          <label style="margin-top:14px;">ICD (çoklu seçim)</label>
          <input id="manualICD" type="text" placeholder="ICD kodu girin (örn: A04||K30||R51)" />
          <button class="btn" style="margin-top:6px; margin-bottom:8px;" id="addManual">Ekle</button>

          <div class="chipbar" id="chips"></div>

          <button class="btn" id="btn" style="margin-top:16px;">Tahmin Et</button>
        </div>

        <div>
          <div class="panel" style="background:#111827;color:#fff">
            <div style="font-size:14px; color:#9ca3af;">Tahmini Yatış (P50)</div>
            <div style="font-size:44px; font-weight:800; margin-top:4px;" id="result">-</div>
          </div>
          <div class="panel" style="margin-top:14px;">
            <div class="title" style="font-size:14px; color:#374151;">Açıklama</div>
            <div style="font-size:13px; color:#374151;">
              /predict sadece sayıyı döndürür. Tam JSON ve ANCHOR bilgileri için <code>POST /predict_json</code> kullanın.
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
const state = { icds: [] };

function addICDsFromText(txt){
  if(!txt) return;
  const parts = txt.split(/\\|\\||,|;|\\s+/).map(s=>s.trim()).filter(Boolean);
  for(const p of parts){
    if(!state.icds.includes(p)) state.icds.push(p);
  }
  renderChips();
}

function renderChips(){
  const bar = document.getElementById('chips');
  bar.innerHTML = "";
  state.icds.forEach(code=>{
    const c = document.createElement('span');
    c.className='chip';
    c.textContent = code;
    bar.appendChild(c);
  });
}

async function loadOptions(){
  const res = await fetch('/options');
  if(!res.ok) throw new Error('options yüklenemedi');
  return res.json();
}

function fillSelect(sel, items, placeholder){
  sel.innerHTML = "";
  const opt0 = document.createElement('option');
  opt0.value=""; opt0.textContent = placeholder || "Tümü";
  sel.appendChild(opt0);
  items.forEach(v=>{
    const o=document.createElement('option'); o.value=v; o.textContent=v;
    sel.appendChild(o);
  });
}

(async ()=>{
  try{
    const opts = await loadOptions();
    fillSelect(document.getElementById('bolum'), opts.bolum, 'Tümü');
    fillSelect(document.getElementById('yas'), opts.yas_grup, 'Tümü');

    document.getElementById('addManual').addEventListener('click', ()=>{
      const val = document.getElementById('manualICD').value.trim();
      addICDsFromText(val);
      document.getElementById('manualICD').value = "";
    });

    document.getElementById('btn').addEventListener('click', async ()=>{
      const bolum = document.getElementById('bolum').value || null;
      const yas = document.getElementById('yas').value || null;
      const icd_list = state.icds.slice();
      const resultEl = document.getElementById('result');

      if(icd_list.length===0){ resultEl.textContent = "ICD seçin"; return; }

      const res = await fetch('/predict', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ icd_list, bolum, yas_grup: yas })
      });
      if(!res.ok){
        const j = await res.json().catch(()=>({}));
        resultEl.textContent = "Hata: " + (j.detail || res.status);
        return;
      }
      const text = await res.text();
      resultEl.textContent = text;
    });
  }catch(err){
    document.getElementById('result').textContent = "Hata: " + err.message;
  }
})();
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
# 4) Seçenek listeleri
# -------------------------------
@app.get("/options", response_class=JSONResponse)
def options():
    try:
        if not os.path.exists(LOOKUP_XLSX):
            raise FileNotFoundError(f"{LOOKUP_XLSX} bulunamadı")
        x = pd.ExcelFile(LOOKUP_XLSX)
        yg = pd.read_excel(x, "DIM_YASGRUP")["YaşGrup"].dropna().astype(str).unique().tolist() if "DIM_YASGRUP" in x.sheet_names else []
        icd = pd.read_excel(x, "DIM_ICD")["ICD"].dropna().astype(str).tolist() if "DIM_ICD" in x.sheet_names else []
        bol = pd.read_excel(x, "LKP_2D_FULL")["Bölüm"].dropna().astype(str).unique().tolist() if "LKP_2D_FULL" in x.sheet_names else []
        yg = sorted(yg); bol = sorted(bol); icd = sorted(icd)
        return {"yas_grup": yg, "bolum": bol, "icd": icd}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"options hatası: {e}")

# -------------------------------
# 5) Tahmin
# -------------------------------
@app.post("/predict", response_class=PlainTextResponse)
def predict(req: PredictRequest):
    try:
        out = tahmin_et(req.icd_list, req.bolum, req.yas_grup)
        val = out.get("Pred_Final_Rounded", None)
        if val is None:
            raise RuntimeError("Pred_Final_Rounded üretilemedi.")
        return str(val)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hata: {e}")

@app.post("/predict_json", response_class=JSONResponse)
def predict_json(req: PredictRequest):
    try:
        return app_predict(req.yas_grup or "", req.bolum or "", req.icd_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hata: {e}")

# -------------------------------
# 6) Form
# -------------------------------
@app.post("/tahmin", response_class=PlainTextResponse)
def tahmin_form(
    icd_text: str = Form(...),
    bolum: Optional[str] = Form(None),
    yas_grup: Optional[str] = Form(None),
):
    try:
        icds = [s.strip() for s in re.split(r"[,;|\\s]+", re.sub(r"\\|\\|", "|", icd_text or "")) if s.strip()]
        out = tahmin_et(icds, bolum, yas_grup)
        val = out.get("Pred_Final_Rounded", None)
        if val is None:
            raise RuntimeError("Pred_Final_Rounded üretilemedi.")
        return str(val)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hata: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=1)