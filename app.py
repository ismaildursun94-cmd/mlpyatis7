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

# ==== MODEL / PROJE ====
from proje import tahmin_et, app_predict, run_training_pipeline

# ---- Sabitler (fallback) ----
LOOKUP_XLSX = os.environ.get("LOOKUP_XLSX", "LOS_Lookup_All.xlsx")
PRED_LOS_XLSX = os.environ.get("PRED_LOS_XLSX", "PRED_LOS.xlsx")
VALID_PRED_XLSX = os.environ.get("VALID_PRED_XLSX", "VALID_PREDICTIONS.xlsx")
YENI_VAKALAR_XLSX = os.environ.get("YENI_VAKALAR_XLSX", "YeniVakalar.xlsx")
MODEL_DIR = os.environ.get("MODEL_DIR", "model_out")

# ============================================================
#                FASTAPI (API) + MODEL ISINMA
# ============================================================
app = FastAPI(title="LOS Predictor API", version="1.4.0")

MODEL_READY: bool = False
MODEL_MODE: str = os.environ.get("MODE", "train").lower()
MODEL_ERROR: Optional[str] = None


def _background_warmup():
    """Ağır eğitimi ana thread'i bloklamadan çalıştır."""
    global MODEL_READY, MODEL_ERROR
    try:
        if MODEL_MODE == "train":
            print("[WARMUP] Training pipeline starting...")
            run_training_pipeline()
            print("[WARMUP] Training pipeline done.")
        else:
            print("[WARMUP] Load mode not implemented; running training as fallback.")
            run_training_pipeline()
        MODEL_READY = True
        MODEL_ERROR = None
    except Exception as e:
        MODEL_READY = False
        MODEL_ERROR = f"{type(e).__name__}: {e}"
        print("[WARMUP][ERROR]", MODEL_ERROR)


@app.on_event("startup")
async def _warmup():
    threading.Thread(target=_background_warmup, daemon=True).start()


# ---------------- HEAD (Render health-check) ----------------
@app.head("/",  response_class=PlainTextResponse)
def index_head():  return ""

@app.head("/ready",  response_class=PlainTextResponse)
def ready_head():  return ""

@app.head("/health", response_class=PlainTextResponse)
def health_head(): return ""


# ---------------- Tahmin request modeli ----------------
class PredictRequest(BaseModel):
    icd_list: Union[str, List[str]]
    bolum: Optional[str] = None
    yas_grup: Optional[str] = None

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

    @field_validator("icd_list")
    @classmethod
    def _not_empty(cls, v: List[str]):
        v = [s.strip() for s in v if s and s.strip()]
        if not v:
            raise ValueError("icd_list boş olamaz")
        return v


# ---------------- Ana sayfa (basit form) ----------------
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
            a.ui { display:inline-block; margin-bottom:14px; color:#9ad; }
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
        <a class="ui" href="/ui" target="_self">→ Yeni Dash Arayüzüne Git</a>
        <h2>LOS Tahmin (Basit Form)</h2>
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


# ---------------- Sağlık / bilgi ----------------
@app.get("/health")
def health():
    return {"status": "up", "service": "los-predictor", "version": app.version}

@app.get("/ready", response_class=JSONResponse)
def ready():
    return {"ready": MODEL_READY, "mode": MODEL_MODE, "error": MODEL_ERROR}

@app.get("/info", response_class=JSONResponse)
def info():
    return {"ready": MODEL_READY, "mode": MODEL_MODE, "error": MODEL_ERROR}


# ---------------- Tahmin uçları ----------------
@app.post("/predict", response_class=PlainTextResponse)
def predict(req: PredictRequest):
    if not MODEL_READY:
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

@app.post("/predict_json", response_class=JSONResponse)
def predict_json(req: PredictRequest):
    if not MODEL_READY:
        msg = "Model hazır değil (ısınma sürüyor)." + (f" Hata: {MODEL_ERROR}" if MODEL_ERROR else "")
        raise HTTPException(status_code=503, detail=msg)
    try:
        out = app_predict(req.yas_grup or "", req.bolum or "", req.icd_list)
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hata: {e}")


# ---------------- Dosya indirme ----------------
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
    safe = re.fullmatch(r"[A-Za-z0-9_.\\-]+", name)
    if not safe:
        raise HTTPException(status_code=400, detail="Geçersiz dosya adı")
    path = os.path.join(MODEL_DIR, name)
    return _file_or_404(path)


# ---------------- Form-POST (opsiyonel) ----------------
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
        s = re.sub(r"\|\|", "|", icd_text or "")
        icds = [p.strip() for p in re.split(r"[,\;\|\s]+", s) if p.strip()]
        out = tahmin_et(icds, bolum, yas_grup)
        val = out.get("Pred_Final_Rounded", None)
        if val is None:
            raise RuntimeError("Pred_Final_Rounded üretilemedi.")
        return str(val)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hata: {e}")


# ============================================================
#                LOOKUP YÜKLEME (LOS_Lookup_All.xlsx)
# ============================================================
import pandas as pd
import unicodedata

def _unique_clean(xs):
    seen, out = set(), []
    for x in xs:
        s = str(x).strip()
        if s and s not in seen:
            seen.add(s); out.append(s)
    return out

def _load_lookups_from_excel(path: str):
    """DIM_ICD ve DIM_YASGRUP'u Excel'den çek; BÖLÜM sabit listeden gelecek."""
    icd_list, yas_list = [], []
    if os.path.exists(path):
        try:
            xls = pd.ExcelFile(path)
            sheet_names = {s.lower(): s for s in xls.sheet_names}

            def read_sheet_or_column(sheet_key: str, col_key: str):
                if sheet_key.lower() in sheet_names:
                    df = pd.read_excel(xls, sheet_names[sheet_key.lower()], engine="openpyxl")
                    series = df.iloc[:, 0]
                    return _unique_clean(series.dropna().astype(str).tolist())
                # ilk sheet'te kolonu ara (fallback)
                df0 = pd.read_excel(xls, sheet_names[list(sheet_names.keys())[0]], engine="openpyxl")
                col = None
                for c in df0.columns:
                    if str(c).strip().lower() == col_key.lower():
                        col = c; break
                if col is not None:
                    return _unique_clean(df0[col].dropna().astype(str).tolist())
                return []

            icd_list = read_sheet_or_column("DIM_ICD", "DIM_ICD")
            yas_list = read_sheet_or_column("DIM_YASGRUP", "DIM_YASGRUP")
        except Exception as e:
            print("[LOOKUP][ERROR]", e)

    return icd_list, yas_list

# — ICD & Yaş Excel’den
ICD_ALL_LIST, AGE_GROUPS_LIST = _load_lookups_from_excel(LOOKUP_XLSX)

# — Bölüm SABİT (gönderdiğin liste)
BOLUM_LIST_LIST = _unique_clean([
    "Acil TIP", "Algoloji", "Anestezi ve Reanimasyon", "Anestezi ve Reanimasyon (GYB)",
    "Ağız ve Diş Sağlığı", "Beyin ve Sinir Cerrahisi", "Check Up", "Dermatoloji", "Endokrinoloji",
    "Enfeksiyon Hastalıkları", "Fizik Tedavi ve Rehabilitasyon", "Gastroenteroloji",
    "Gastroenteroloji Cerrahisi", "Genel Cerrahi", "Girişimsel Radyoloji",
    "GÖĞÜS HASTALIKLARI YOĞUN BAKIM", "Göz Hastalıkları", "Göğüs Cerrahisi", "Göğüs Hastalıkları",
    "Hematoloji Polikliniği", "Jinekoloji Onkoloji Cerrahisi", "KBB Hastalıkları", "KVC Yoğun Bakım",
    "Kadın Hastalıkları ve Doğum", "Kalp ve Damar Cerrahisi", "Kardiyoloji", "Koroner Yoğun Bakım",
    "Laboratuvar", "Meme Cerrahisi", "Nefroloji", "Neonatoloji", "Nöroloji", "Nükleer TIP",
    "Obezite Cerrahisi", "Organ Nakli", "Organ Nakli (Genel Cerrahii)", "Ortopedi ve Travmatoloji",
    "Plastik Cerrahi", "Psikiyatri", "Radyasyon Onkolojisi", "Radyoloji", "Romatoloji", "Saç Ekimi",
    "Tüp Bebek", "Tıbbi Onkoloji", "Yenidoğan Yoğun Bakım", "Çocuk Cerrahisi",
    "Çocuk Enfeksiyon Hastalıkları", "Çocuk Gastroentroloji", "Çocuk Hematolojisi",
    "Çocuk Hematolojisi ve Onkolojisi", "Çocuk Kardiyoloji", "Çocuk Nefrolojisi",
    "Çocuk Nörolojisi", "Çocuk Sağlığı ve Hastalıkları",
    "Çocuk İmmünolojisi ve Alerji Hastalıkları", "Üroloji",
    "İÇ HASTALIKLARI YOĞUN BAKIM", "İç Hastalıkları",
])

@app.get("/lookup", response_class=JSONResponse)
def lookup_all():
    return {
        "icd_all": ICD_ALL_LIST,
        "age_groups": AGE_GROUPS_LIST,
        "bolum_list": BOLUM_LIST_LIST,
    }


# ============================================================
#                    DASH UI'YI /ui ALTINA MONTAJ
# ============================================================
from dash import Dash, html, dcc, callback, Output, Input, State, no_update
import dash_bootstrap_components as dbc
from starlette.middleware.wsgi import WSGIMiddleware

def _tr_fold(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("ı", "i").replace("İ", "I")
    return s.lower()

def _filter_list(all_items, query, selected_vals):
    q = _tr_fold(query or "")
    base = [{"label": i, "value": i} for i in all_items if not q or q in _tr_fold(i)]
    sel = selected_vals if isinstance(selected_vals, list) else ([selected_vals] if selected_vals else [])
    sel_set = set(sel)
    head = [{"label": s, "value": s} for s in sel if s in all_items]
    tail = [o for o in base if o["value"] not in sel_set]
    return head + tail

def build_dash_app() -> Dash:
    dash_app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        requests_pathname_prefix="/ui/",
        update_title=None,
        serve_locally=True,               # asset'leri local servis et
        suppress_callback_exceptions=True # güvenli yanıt
    )
    dash_app.index_string = """
    <!DOCTYPE html>
    <html>
      <head>
        {%metas%}
        <title>Yatış Tahmin Aracı</title>
        {%favicon%}
        {%css%}
        <style>
          body { background:#ECEFF3; }
          .card-wrap { max-width: 550px; margin: 24px auto; }
          .pred-box { border:1px dashed #D6D9DE; background:#FAFBFC; border-radius:10px; padding:18px 20px; }
          .pred-value { font-size:26px; font-weight:700; color:#2f3b52; margin:6px 0 0 0; }
          .big-dd .Select__control { min-height: 56px; }
          .big-dd .Select__value-container { padding: 12px 12px; }
          .big-dd .Select__multi-value { margin: 6px 6px; }
          .big-dd .Select__input input { min-height: 30px; }
          .Select__menu-list { max-height: 220px; }
        </style>
      </head>
      <body>
        {%app_entry%}
        <footer>
          {%config%}
          {%scripts%}
          {%renderer%}
        </footer>
      </body>
    </html>
    """

    dash_app.layout = dbc.Container(
        dbc.Card(
            [
                dbc.CardHeader("Yatış Tahmin Aracı"),
                dbc.CardBody(
                    [
                        dbc.Label("Yaş Grubu", className="mt-2 mb-1"),
                        dcc.Dropdown(
                            options=[{"label": o, "value": o} for o in AGE_GROUPS_LIST],
                            value=None,
                            id="age",
                            placeholder="Seçiniz",
                            clearable=True,
                            className="big-dd",
                            style={"width": "100%"},
                        ),

                        dbc.Label("Bölüm", className="mt-4 mb-1"),
                        dcc.Dropdown(
                            options=[{"label": o, "value": o} for o in BOLUM_LIST_LIST],
                            value=None,
                            id="bolum",
                            placeholder="Seçiniz",
                            clearable=True,
                            className="big-dd",
                            style={"width": "100%"},
                        ),

                        dbc.Label("ICD Kodu", className="mt-4 mb-1"),
                        dcc.Dropdown(
                            options=[{"label": o, "value": o} for o in ICD_ALL_LIST],
                            value=[],
                            id="icd",
                            multi=True,
                            placeholder="Seçiniz",
                            className="big-dd",
                            style={"width": "100%"},
                        ),

                        dbc.Alert(id="limit-alert", color="warning", is_open=False, fade=True, className="mt-2"),
                        dbc.Alert(id="api-alert",   color="warning", is_open=False, fade=True, className="mt-2"),

                        html.Div(
                            [
                                html.Div("Tahmini Yatış Süresi", style={"color": "#6b7280"}),
                                html.Div(id="prediction", className="pred-value", children="—"),
                            ],
                            className="pred-box mt-4"
                        ),

                        html.Div(
                            [
                                dbc.Button("Temizle", id="reset", color="secondary", outline=True, className="me-2"),
                                dbc.Button("Tahmin Et", id="predict", color="primary"),
                            ],
                            className="mt-4 d-flex justify-content-end"
                        ),
                    ]
                ),
            ],
            body=False,
            className="card-wrap",
        ),
        fluid=True,
    )

    # ---- Arama filtreleri ----
    @dash_app.callback(
        Output("age", "options"),
        Input("age", "search_value"),
        State("age", "value"),
        prevent_initial_call=True,
    )
    def filter_age(search_value, current_value):
        return _filter_list(AGE_GROUPS_LIST, search_value, current_value)

    @dash_app.callback(
        Output("bolum", "options"),
        Input("bolum", "search_value"),
        State("bolum", "value"),
        prevent_initial_call=True,
    )
    def filter_bolum(search_value, current_value):
        return _filter_list(BOLUM_LIST_LIST, search_value, current_value)

    @dash_app.callback(
        Output("icd", "options"),
        Input("icd", "search_value"),
        State("icd", "value"),
        prevent_initial_call=True,
    )
    def filter_icd(search_value, current_values):
        return _filter_list(ICD_ALL_LIST, search_value, current_values or [])

    # ---- TEK callback: ICD limit + Reset (duplicate outputs sorunsuz)
    from dash import callback_context as ctx

    @dash_app.callback(
        Output("icd", "value"),
        Output("limit-alert", "is_open"),
        Output("limit-alert", "children"),
        Input("icd", "value"),
        Input("reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def icd_value_and_limit(values, reset_clicks):
        trigger = (ctx.triggered[0]["prop_id"] if ctx.triggered else "")  # "icd.value" | "reset.n_clicks"
        if trigger.startswith("reset"):
            return [], False, no_update
        values = values or []
        if len(values) <= 15:
            return values, False, no_update
        trimmed = values[:15]
        return trimmed, True, "En fazla 15 ICD seçebilirsiniz."

    # ---- TEK callback: Tahmin + Reset mesajı
    @dash_app.callback(
        Output("prediction", "children"),
        Output("api-alert", "is_open"),
        Output("api-alert", "children"),
        Input("predict", "n_clicks"),
        Input("reset", "n_clicks"),
        State("age", "value"),
        State("bolum", "value"),
        State("icd", "value"),
        prevent_initial_call=True,
    )
    def do_predict_or_reset(p_click, r_click, age, bolum, icds):
        trigger = (ctx.triggered[0]["prop_id"] if ctx.triggered else "")
        if trigger.startswith("reset"):
            return "—", False, no_update
        # predict:
        if not age or not bolum or not (icds and len(icds) > 0):
            return "Tüm seçimleri yapın", True, "Lütfen Yaş Grubu, Bölüm ve en az 1 ICD seçin."
        if not MODEL_READY:
            return "—", True, "Model hazır değil (ısınma sürüyor)."
        try:
            out = tahmin_et(icds, bolum, age)  # proje.tahmin_et(icd_list, bolum, yas_grup)
            val = out.get("Pred_Final_Rounded") or out.get("pred_final_rounded") \
                  or out.get("Pred_Final") or out.get("pred_final")
            if val is None:
                return "—", True, "Sunucudan beklenen yanıt alınamadı."
            return f"{int(round(float(val)))} gün", False, no_update
        except Exception as e:
            return "—", True, f"Hata: {e}"

    return dash_app


_dash = build_dash_app()
app.mount("/ui", WSGIMiddleware(_dash.server))  # Dash'ı /ui altına bağla


# ---------------- Lokal çalıştırıcı ----------------
if __name__ == "__main__":
    # Render Start Command:
    # uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1
    import uvicorn
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=1)