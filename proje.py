# -*- coding: utf-8 -*-
import os, json, re, warnings, datetime, itertools, math, random, unicodedata
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict, Counter

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from scipy import sparse
from scipy.sparse import hstack

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from xgboost import XGBRegressor
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
import joblib

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
random.seed(42)
np.random.seed(42)

# ====================== AYARLAR ======================
EXCEL_PATH = "Veri2024.xlsx"
LOOKUP_XLSX = "LOS_Lookup_All.xlsx"
MODEL_DIR = "model_out"
MAKE_YENI_VAKALAR = True
YENI_VAKALAR_XLSX = "YeniVakalar.xlsx"
PRED_LOS_XLSX = "PRED_LOS.xlsx"
VALID_PRED_XLSX = "VALID_PREDICTIONS.xlsx"
N_SAMPLES_YENI = 200
REQUIRE_ICD = True
FALLBACK_FROM_TEXT = True
RANDOM_SEED = 42
TOPK_ICD = 400
MIN_SUPPORT = 1

SPLIT_BY_COMBO = False   # False: row-split (valid'de 3D mümkün)

SATURATION_ON = True
SATURATION_K = 3.0

TOPK_NEIGHBORS = 10
RHO_J = 1.2

SHRINK_1SUPPORT_SCALE = 0.15
REMOVAL_PENALTY = 0.5
CAP_MARJ = 1.0

WINSORIZE_ON = True
WINSOR_LO = 0.00973
WINSOR_HI = 0.9785

XGB_ENS_ON = True
XGB_ALPHA_LOG = 0.20
XGB_RULE_BLEND = 0.50
XGB_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
    n_jobs=4,
    tree_method="hist"
)

STRICT_SHORT_CIRCUIT = True
PREFER_TRAIN_FOR_EXACT = True
USE_FULL_FOR_EXACT = False
# ====================================================

__all__ = ["tahmin_et","app_predict","app_predict_many","app_info"]

def stage(msg): print(f"[STAGE] {msg}", flush=True)

# ---------- yardımcılar ----------
def _norm_text_basic(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def round_half_up(x):
    if pd.isna(x): return None
    return int(Decimal(str(float(x))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

def yas_to_years(val):
    if pd.isna(val): return pd.NA
    if isinstance(val, (int, float)): return float(val)
    s = str(val).strip().lower()
    if re.fullmatch(r"\d+(?:[.,]\d+)?", s): return float(s.replace(",", "."))
    yil = re.findall(r"(\d+)\s*yıl", s); ay  = re.findall(r"(\d+)\s*ay", s); gun = re.findall(r"(\d+)\s*gün", s)
    years = 0.0
    if yil: years += sum(float(x) for x in yil)
    if ay:  years += sum(float(x) for x in ay) / 12
    if gun: years += sum(float(x) for x in gun) / 365
    if years == 0.0 and not (yil or ay or gun): return pd.NA
    return round(years, 2)

_PAREN_MAP = str.maketrans({'（':'(', '）':')', '【':'[', '】':']', '＜':'<', '＞':'>', '｛':'{', '｝':'}'})
_PREFIX_RE = re.compile(r"""^\s*[\(\[\{\<]\s*(?:ö|ö|k|a)\s*[\)\]\}\>]\s*""", re.IGNORECASE | re.VERBOSE)
_ANYWHERE_TAG_RE = re.compile(r"\(\s*(?:ö|ö|k|a)\s*\)", re.IGNORECASE)
_ICD_CODE_RE = re.compile(r"\b([A-Z][0-9]{2}(?:\.[A-Z0-9]{1,4})?)\b", re.IGNORECASE)

def clean_icd(raw) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)): return ""
    s = str(raw).strip()
    if not s: return ""
    s = s.translate(_PAREN_MAP); prev = None
    while prev != s:
        prev = s; s = _PREFIX_RE.sub("", s)
    return s.strip().upper()

def clean_text_anywhere_tags(raw) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)): return ""
    s = str(raw).strip()
    if not s: return ""
    s = s.translate(_PAREN_MAP)
    s = _ANYWHERE_TAG_RE.sub("", s)
    return re.sub(r"\s{2,}", " ", s).strip()

def split_icd_cell(s):
    if pd.isna(s): return []
    s = str(s).translate(_PAREN_MAP)
    parts = re.split(r"[;,]", s)
    parts = [clean_icd(p) for p in parts]
    return [p for p in parts if p]

def normalize_icd_set(lst):
    lst_clean = [clean_icd(x) for x in lst if str(x).strip() != ""]
    uniq = sorted(set(lst_clean), key=str)
    return uniq, "||".join(uniq)

def clean_icd_set_key(key: str) -> str:
    if key is None or (isinstance(key, float) and pd.isna(key)): return ""
    parts = [clean_icd(p.strip().upper()) for p in str(key).split("||")]
    parts = [p for p in parts if p]
    return "||".join(sorted(set(parts), key=str))

def extract_icd_from_text(text: str):
    if not isinstance(text, str) or not text.strip(): return []
    t = _ANYWHERE_TAG_RE.sub("", text.translate(_PAREN_MAP))
    return [m.upper() for m in _ICD_CODE_RE.findall(t)]

def p90(x): return x.quantile(0.9)

def yas_to_group(y):
    if pd.isna(y): return pd.NA
    y = float(y)
    if y < 0: return pd.NA
    if y <= 1:  return "0-1"
    if y <= 5:  return "2-5"
    if y <= 10: return "5-10"
    if y <= 15: return "10-15"
    if y <= 25: return "15-25"
    if y <= 35: return "25-35"
    if y <= 50: return "35-50"
    if y <= 65: return "50-65"
    return "65+"

def jaccard(a:set, b:set)->float:
    if not a and not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return inter/union if union>0 else 0.0

def as_set(key:str)->set:
    if not key: return set()
    return set([k for k in key.split("||") if k])

def as_key(s:set)->str:
    return "||".join(sorted(s))

def as_csr(x):
    return x if sparse.issparse(x) else sparse.csr_matrix(x)

# ================== 1) YÜKLE & TEMİZLE ==================
stage("Excel okunuyor")
if not os.path.exists(EXCEL_PATH):
    raise FileNotFoundError(f"Bulunamadı: {EXCEL_PATH}")

df_raw = pd.read_excel(EXCEL_PATH)
df = df_raw.copy()

# Yaş & LOS
if "Yaş" in df.columns:
    df["Yaş"] = df["Yaş"].apply(yas_to_years)
else:
    df["Yaş"] = pd.NA
df["Yatış Gün Sayısı"] = pd.to_numeric(df["Yatış Gün Sayısı"], errors="coerce")
df = df[df["Yatış Gün Sayısı"] > 0].copy()

# ICD listeleri (gerekirse metinden fallback)
if "ICD Kodu" not in df.columns:
    df["ICD Kodu"] = ""

if REQUIRE_ICD and df["ICD Kodu"].fillna("").eq("").any() and FALLBACK_FROM_TEXT:
    base_text_col = "ICD Adi Ve Kodu" if "ICD Adi Ve Kodu" in df.columns else None
    if base_text_col:
        ix = df["ICD Kodu"].fillna("").eq("")
        df.loc[ix, "ICD Kodu"] = df.loc[ix, base_text_col].fillna("").apply(
            lambda t: ",".join(extract_icd_from_text(t))
        )

df["ICD_List"] = df["ICD Kodu"].apply(split_icd_cell)
df["ICD_List_Norm"], df["ICD_Set_Key"] = zip(*df["ICD_List"].apply(normalize_icd_set))
df["ICD_Set_Key"] = df["ICD_Set_Key"].apply(clean_icd_set_key)
df["ICD_Sayısı"] = df["ICD_List_Norm"].apply(len)
df = df[~(REQUIRE_ICD & (df["ICD_Sayısı"]==0))].copy()

# Embedding metni (opsiyonel kaynak)
base_text_col = "ICD Adi Ve Kodu" if "ICD Adi Ve Kodu" in df.columns else "ICD Kodu"
df["ICD_Text_Embed"] = df[base_text_col].map(clean_text_anywhere_tags).fillna("")

# YaşGrup
df["Yaş_Yıl_Int"] = pd.to_numeric(df["Yaş"], errors="coerce").round().astype("Int64")
df["YaşGrup"] = df["Yaş"].apply(yas_to_group)

# --- Demo kanonik sözlükleri ---
YG_UNIQ = { _norm_text_basic(v): v for v in df["YaşGrup"].dropna().astype(str).unique() }
BOLUM_UNIQ = { _norm_text_basic(v): v for v in df["Bölüm"].dropna().astype(str).unique() }
def canon_demo(yas_grup: str, bolum: str) -> tuple[str,str]:
    yg = YG_UNIQ.get(_norm_text_basic(yas_grup), str(yas_grup or "").strip())
    b  = BOLUM_UNIQ.get(_norm_text_basic(bolum), str(bolum or "").strip())
    return yg, b

# ================== 2) TRAIN/VALID SPLIT ==================
stage("Train/Valid ayrımı")
df["ComboID"] = df["YaşGrup"].astype(str)+"||"+df["Bölüm"].astype(str)+"||"+df["ICD_Set_Key"].astype(str)

if SPLIT_BY_COMBO:
    stage("Kombinasyon-bazlı split (valid'de 3D bekleme)")
    unique_combos = df["ComboID"].dropna().unique()
    train_combos, valid_combos = train_test_split(unique_combos, test_size=0.2, random_state=RANDOM_SEED)
    is_train = df["ComboID"].isin(train_combos)
    train_df = df[is_train].copy()
    valid_df = df[~is_train].copy()
else:
    stage("Satır-bazlı split (valid'de 3D mümkün)")
    idx_train, idx_valid = train_test_split(df.index, test_size=0.2, random_state=RANDOM_SEED)
    train_df = df.loc[idx_train].copy()
    valid_df = df.loc[idx_valid].copy()
    _overlap_rate = (valid_df["ComboID"].isin(train_df["ComboID"].unique())).mean()
    print(f"Valid satırlarının train ile ComboID örtüşme oranı: {_overlap_rate:.2%}")

# ---- Winsorize (sadece train)
def _winsorize_series(s: pd.Series, lo: float, hi: float) -> pd.Series:
    q_lo = s.quantile(lo); q_hi = s.quantile(hi)
    return s.clip(lower=q_lo, upper=q_hi)
if WINSORIZE_ON:
    stage(f"Winsorize (train) - p{int(WINSOR_LO*100)} / p{int(WINSOR_HI*100)}")
    train_df["Yatış Gün Sayısı"] = _winsorize_series(train_df["Yatış Gün Sayısı"], WINSOR_LO, WINSOR_HI)

# ================== 3) LOOKUP TABLOLARI ==================
stage("Lookup tabloları (train + full)")

# ---- TRAIN lookuplar
lkp3 = (train_df.groupby(["YaşGrup","Bölüm","ICD_Set_Key"], as_index=False)
        .agg(N=("Yatış Gün Sayısı","count"),
             Ortalama=("Yatış Gün Sayısı", lambda x: round_half_up(x.mean())),
             P50=("Yatış Gün Sayısı","median"),
             P90=("Yatış Gün Sayısı", p90)))
lkp3["ICD_Set_Key"] = lkp3["ICD_Set_Key"].apply(clean_icd_set_key)

lkp2 = (train_df.groupby(["Bölüm","ICD_Set_Key"], as_index=False)
        .agg(N=("Yatış Gün Sayısı","count"),
             Ortalama=("Yatış Gün Sayısı", lambda x: round_half_up(x.mean())),
             P50=("Yatış Gün Sayısı","median"),
             P90=("Yatış Gün Sayısı", p90)))
lkp2["ICD_Set_Key"] = lkp2["ICD_Set_Key"].apply(clean_icd_set_key)

lkp1 = (train_df.groupby(["ICD_Set_Key"], as_index=False)
        .agg(N=("Yatış Gün Sayısı","count"),
             Ortalama=("Yatış Gün Sayısı", lambda x: round_half_up(x.mean())),
             P50=("Yatış Gün Sayısı","median"),
             P90=("Yatış Gün Sayısı", p90)))
lkp1["ICD_Set_Key"] = lkp1["ICD_Set_Key"].apply(clean_icd_set_key)

lkp0 = pd.DataFrame({
    "N": [train_df.shape[0]],
    "Ortalama": [round_half_up(train_df["Yatış Gün Sayısı"].mean())],
    "P50": [train_df["Yatış Gün Sayısı"].median()],
    "P90": [train_df["Yatış Gün Sayısı"].quantile(0.9)]
})

# ---- FULL lookuplar (birebir short-circuit garanti için)
lkp3_full = (df.groupby(["YaşGrup","Bölüm","ICD_Set_Key"], as_index=False)
             .agg(N=("Yatış Gün Sayısı","count"),
                  P50=("Yatış Gün Sayısı","median")))
lkp3_full["ICD_Set_Key"] = lkp3_full["ICD_Set_Key"].apply(clean_icd_set_key)

lkp2_full = (df.groupby(["Bölüm","ICD_Set_Key"], as_index=False)
             .agg(N=("Yatış Gün Sayısı","count"),
                  P50=("Yatış Gün Sayısı","median")))
lkp2_full["ICD_Set_Key"] = lkp2_full["ICD_Set_Key"].apply(clean_icd_set_key)

lkp1_full = (df.groupby(["ICD_Set_Key"], as_index=False)
             .agg(N=("Yatış Gün Sayısı","count"),
                  P50=("Yatış Gün Sayısı","median")))
lkp1_full["ICD_Set_Key"] = lkp1_full["ICD_Set_Key"].apply(clean_icd_set_key)

# Tekil / pair yardımcı tablolar (train)
single = train_df[train_df["ICD_Sayısı"]==1].copy()
single["ICD_Kod"] = single["ICD_List_Norm"].str[0]
LKP_ICD = single.groupby("ICD_Kod")["Yatış Gün Sayısı"].agg(N="count", P50="median").reset_index()

pairs = train_df[train_df["ICD_Sayısı"]==2].copy()
pairs["PairKey"] = pairs["ICD_List_Norm"].apply(lambda lst: "||".join(sorted(lst)))
LKP_PAIR = pairs.groupby(["YaşGrup","Bölüm","PairKey"])["Yatış Gün Sayısı"].agg(N="count", P50="median").reset_index()

DEMO_P90_MAP = train_df.groupby(["YaşGrup","Bölüm"])["Yatış Gün Sayısı"].quantile(0.9).reset_index().rename(columns={"Yatış Gün Sayısı":"P90"})
demop90_map = {(r["YaşGrup"], r["Bölüm"]): float(r["P90"]) for _, r in DEMO_P90_MAP.iterrows()}

# ================== 4) β / γ ÖĞREN ==================
stage("β/γ öğreniliyor")
icd_to_beta_samples = defaultdict(list)
ctx3 = lkp3.copy(); ctx3["Set"] = ctx3["ICD_Set_Key"].apply(as_set)
one_map = dict(zip(lkp1["ICD_Set_Key"], lkp1["P50"]))
single_map = dict(zip(LKP_ICD["ICD_Kod"], LKP_ICD["P50"]))

for _, row in ctx3.iterrows():
    key, p50 = row["ICD_Set_Key"], row["P50"]
    S = as_set(key)
    if len(S) < 2: continue
    for icd in S:
        base_candidates = []
        if icd in single_map: base_candidates.append(single_map[icd])
        if icd in one_map:    base_candidates.append(one_map.get(icd, 0.0))
        base = max(base_candidates) if base_candidates else 0.0
        delta = max(0.0, float(p50) - float(base))
        icd_to_beta_samples[icd].append(delta)

beta_icd, beta_support = {}, {}
for icd, samples in icd_to_beta_samples.items():
    if len(samples) >= MIN_SUPPORT:
        beta_icd[icd] = float(np.median(samples))
        beta_support[icd] = int(len(samples))

pair_to_gamma_samples = defaultdict(list)
for _, row in LKP_PAIR.iterrows():
    i, j = row["PairKey"].split("||"); p50 = row["P50"]
    base = max(single_map.get(i, 0.0), single_map.get(j, 0.0))
    delta = max(0.0, float(p50) - float(base))
    pair_to_gamma_samples[(i,j)].append(delta)
    pair_to_gamma_samples[(j,i)].append(delta)
gamma_pairs, gamma_support = {}, {}
for pair, samples in pair_to_gamma_samples.items():
    if len(samples) >= MIN_SUPPORT:
        gamma_pairs[pair] = float(np.median(samples))
        gamma_support[pair] = int(len(samples))

# ================== 5) LOOKUP EXCEL ==================
stage("Lookup Excel yazılıyor")
_br_base = df[["ICD_Set_Key","ICD_List_Norm"]].copy()
BR_ICDSET_MAP = _br_base.explode("ICD_List_Norm").rename(columns={"ICD_List_Norm":"ICD"}).dropna(subset=["ICD"]).drop_duplicates().reset_index(drop=True)
DIM_ICD = pd.DataFrame({"ICD": sorted({icd for lst in df["ICD_List_Norm"] for icd in lst})})
_age_order = ["0-1","2-5","5-10","10-15","15-25","25-35","35-50","50-65","65+"]
_present = [yg for yg in _age_order if yg in set(df["YaşGrup"].dropna().astype(str).unique())]
DIM_YASGRUP = pd.DataFrame({"YaşGrup": _present})

with pd.ExcelWriter(LOOKUP_XLSX, engine="xlsxwriter") as w:
    lkp3.to_excel(w, index=False, sheet_name="LKP_3D_YasGrup_TRAIN")
    lkp2.to_excel(w, index=False, sheet_name="LKP_2D_TRAIN")
    lkp1.to_excel(w, index=False, sheet_name="LKP_1D_TRAIN")
    lkp0.to_excel(w, index=False, sheet_name="LKP_0D_TRAIN")
    lkp3_full.to_excel(w, index=False, sheet_name="LKP_3D_FULL")
    lkp2_full.to_excel(w, index=False, sheet_name="LKP_2D_FULL")
    lkp1_full.to_excel(w, index=False, sheet_name="LKP_1D_FULL")
    LKP_ICD.to_excel(w, index=False, sheet_name="LKP_ICD_TRAIN")
    df[["ICD_Text_Embed"]].to_excel(w, index=False, sheet_name="TEXT_EMB_SOURCE")
    BR_ICDSET_MAP.to_excel(w, index=False, sheet_name="BR_ICDSET_MAP")
    DIM_ICD.to_excel(w, index=False, sheet_name="DIM_ICD")
    DIM_YASGRUP.to_excel(w, index=False, sheet_name="DIM_YASGRUP")
print(f"OK -> {LOOKUP_XLSX}")

# ================== 6) ANCHOR / PREDICT ==================
stage("Prediction yardımcı yapılar")

# TRAIN mapler
lkp3_map = {(r["YaşGrup"], r["Bölüm"], r["ICD_Set_Key"]):(r["P50"], r["N"]) for _,r in lkp3.iterrows()}
lkp2_map = {(r["Bölüm"], r["ICD_Set_Key"]):(r["P50"], r["N"]) for _,r in lkp2.iterrows()}
lkp1_map = {r["ICD_Set_Key"]:(r["P50"], r["N"]) for _,r in lkp1.iterrows()}
lkp0_p50 = float(lkp0["P50"].iloc[0]) if len(lkp0)>0 else 0.0
lkp0_p90 = float(lkp0["P90"].iloc[0]) if len(lkp0)>0 else 0.0

# FULL mapler (sadece exact-match short-circuit için)
lkp3_full_map = {(r["YaşGrup"], r["Bölüm"], r["ICD_Set_Key"]):(r["P50"], r["N"]) for _,r in lkp3_full.iterrows()}
lkp2_full_map = {(r["Bölüm"], r["ICD_Set_Key"]):(r["P50"], r["N"]) for _,r in lkp2_full.iterrows()}
lkp1_full_map = {r["ICD_Set_Key"]:(r["P50"], r["N"]) for _,r in lkp1_full.iterrows()}

ctx3_by_demo = defaultdict(list)
for _, r in lkp3.iterrows():
    ctx3_by_demo[(r["YaşGrup"], r["Bölüm"])].append(r["ICD_Set_Key"])

pair_floor_map = {}
for _, r in LKP_PAIR.iterrows():
    pair_floor_map[r["PairKey"]] = max(pair_floor_map.get(r["PairKey"], 0.0), float(r["P50"]))
single_floor_map = dict(zip(LKP_ICD["ICD_Kod"], LKP_ICD["P50"]))

def find_anchor(yg: str, bolum: str, key: str):
    def train_exact():
        if (yg, bolum, key) in lkp3_map: return "3D", *lkp3_map[(yg, bolum, key)], key
        if (bolum, key) in lkp2_map:     return "2D", *lkp2_map[(bolum, key)], key
        if key in lkp1_map:              return "1D", *lkp1_map[key], key
        return None

    def full_exact():
        if (yg, bolum, key) in lkp3_full_map: return "3D", *lkp3_full_map[(yg, bolum, key)], key
        if (bolum, key) in lkp2_full_map:     return "2D", *lkp2_full_map[(bolum, key)], key
        if key in lkp1_full_map:              return "1D", *lkp1_full_map[key], key
        return None

    if USE_FULL_FOR_EXACT:
        if PREFER_TRAIN_FOR_EXACT:
            return train_exact() or full_exact() or (None, None, 0, None)
        else:
            return full_exact() or train_exact() or (None, None, 0, None)
    else:
        return train_exact() or (None, None, 0, None)

def _topk_weighted_anchor(candidates, target_set:set, K:int=TOPK_NEIGHBORS, rho:float=RHO_J):
    scored = []
    for key, p50, n in candidates:
        J = jaccard(target_set, as_set(key))
        if p50 is None:
            continue
        scored.append((J, float(p50), int(n if n is not None else 0), key))

    if not scored:
        return 0.0, None, None

    scored.sort(key=lambda x: (x[0], x[2], x[1]), reverse=True)
    bestJ, bestP50, _bestN, bestKey = scored[0]

    # 🔒 Hiç örtüşme yoksa (J=0) “komşu” kabul etme
    if bestJ <= 0.0:
        return 0.0, None, None

    topk = [r for r in scored if r[0] > 0.0][:K]
    Wvals = [(((J**float(rho)) * math.log1p(max(0, n))), p50) for J, p50, n, _k in topk]
    W = sum(w for w, _ in Wvals)
    if W <= 0:
        return bestJ, bestP50, bestKey
    wmean = sum(p * w for w, p in Wvals) / W
    return bestJ, float(wmean), bestKey

def nearest_neighbor_anchor(yg:str, bolum:str, target_key:str):
    target = as_set(target_key)

    # 3D (J>0 gerek)
    cand3 = [(key, *lkp3_map.get((yg, bolum, key), (None, 0))) for key in ctx3_by_demo.get((yg, bolum), [])]
    bestJ, w_p50, bestKey = _topk_weighted_anchor(cand3, target)
    if bestKey is not None:
        return bestJ, float(w_p50 if w_p50 is not None else lkp0_p50), bestKey, "3D_DEMO"

    # 2D (J>0 gerek)
    cand2 = [(key, p50, n) for (b, key), (p50, n) in lkp2_map.items() if b == bolum]
    bestJ, w_p50, bestKey = _topk_weighted_anchor(cand2, target)
    if bestKey is not None:
        return bestJ, float(w_p50 if w_p50 is not None else lkp0_p50), bestKey, "2D"

    # 1D (J>0 varsa onu al)
    cand1 = [(key, p50, n) for key, (p50, n) in lkp1_map.items()]
    bestJ, w_p50, bestKey = _topk_weighted_anchor(cand1, target)
    if bestKey is not None:
        return bestJ, float(w_p50 if w_p50 is not None else lkp0_p50), bestKey, "1D"

    #Hiçbiri yoksa artık tahmin verme
    return 0.0, 0.0, None, "NONE"

def model_contrib(target_key:str, anchor_key:str):
    T = as_set(target_key); A = as_set(anchor_key) if anchor_key else set()
    def beta_scaled(icd):
        val = max(0.0, beta_icd.get(icd, 0.0)); sup = beta_support.get(icd, 0)
        return val * (1.0 if sup>=3 else SHRINK_1SUPPORT_SCALE)
    def gamma_scaled(i,j):
        val = max(0.0, gamma_pairs.get((i,j), gamma_pairs.get((j,i), 0.0)))
        sup = max(gamma_support.get((i,j),0), gamma_support.get((j,i),0))
        return val * (1.0 if sup>=3 else SHRINK_1SUPPORT_SCALE)
    add_single = sorted(T - A); rem_single = sorted(A - T)
    beta_sum = sum(beta_scaled(i) for i in add_single) - REMOVAL_PENALTY*sum(beta_scaled(i) for i in rem_single)
    pairs_T = set(itertools.combinations(sorted(T),2)); pairs_A = set(itertools.combinations(sorted(A),2))
    add_pairs = pairs_T - pairs_A; rem_pairs = pairs_A - pairs_T
    gamma_sum = sum(gamma_scaled(i,j) for (i,j) in add_pairs) - REMOVAL_PENALTY*sum(gamma_scaled(i,j) for (i,j) in rem_pairs)
    return beta_sum, gamma_sum, add_single

def saturation(total_add:float, k:float=SATURATION_K):
    return float(k * (1.0 - math.exp(-float(total_add)/float(k)))) if SATURATION_ON else float(total_add)

def guardrails(yg:str, bolum:str, target_key:str, pred:float):
    T = as_set(target_key); floor2 = pred
    for i,j in itertools.combinations(sorted(T),2):
        pf = max(pair_floor_map.get(f"{i}||{j}",0.0), pair_floor_map.get(f"{j}||{i}",0.0))
        floor2 = max(floor2, pf)
    floor3 = floor2
    for r in lkp3[(lkp3["YaşGrup"]==yg) & (lkp3["Bölüm"]==bolum)].itertuples():
        if as_set(r.ICD_Set_Key).issubset(T): floor3 = max(floor3, float(r.P50))
    for r in lkp2[lkp2["Bölüm"]==bolum].itertuples():
        if as_set(r.ICD_Set_Key).issubset(T): floor3 = max(floor3, float(r.P50))
    for r in lkp1.itertuples():
        if as_set(r.ICD_Set_Key).issubset(T): floor3 = max(floor3, float(r.P50))
    return floor3

def predict_one(yg:str, bolum:str, target_key:str):
    yg, bolum = canon_demo(yg, bolum)

    src, anchor_p50, n, anchor_key = find_anchor(yg, bolum, target_key)

    # 3D/2D/1D birebir eşleşme → short-circuit
    if anchor_p50 is not None and anchor_key == target_key and src in ("3D","2D","1D"):
        meta = {"ANCHOR_SRC": src, "ANCHOR_KEY": anchor_key or "", "ANCHOR_P50": float(anchor_p50),
                "ALPHA_JACCARD": 0.0, "ADDED_ICDS": "", "BETA_SUM": 0.0, "GAMMA_SUM": 0.0,
                "MODEL_PRED": float(anchor_p50), "PRED_BLEND": float(anchor_p50), "SHORT_CIRCUIT": True}
        return float(anchor_p50), meta

    # Anchor yoksa komşu
    if anchor_p50 is None:
        J, neigh_p50, neigh_key, neigh_src = nearest_neighbor_anchor(yg, bolum, target_key)
        anchor_p50, anchor_key, src, alpha = float(neigh_p50), neigh_key, f"NEIGHBOR_{neigh_src}", float(J)
    else:
        alpha = 0.0

    beta_sum, gamma_sum, added_icds = model_contrib(target_key, anchor_key)
    add_total = saturation(beta_sum + gamma_sum)
    model_pred = float(anchor_p50) + add_total
    pred_blend = (1.0 - alpha) * model_pred + alpha * float(anchor_p50)
    pred_final = float(anchor_p50) if pred_blend < 1.0 else pred_blend

    cap_ref = demop90_map.get((yg, bolum), lkp0_p90)
    cap_val = float(cap_ref) * float(CAP_MARJ) if cap_ref is not None else float("inf")
    if cap_val is not None:
        pred_final = min(float(pred_final), float(cap_val))

    meta = {"ANCHOR_SRC": src, "ANCHOR_KEY": anchor_key or "", "ANCHOR_P50": float(anchor_p50),
            "ALPHA_JACCARD": float(alpha), "ADDED_ICDS": ",".join(added_icds),
            "BETA_SUM": float(beta_sum), "GAMMA_SUM": float(gamma_sum),
            "MODEL_PRED": float(model_pred), "PRED_BLEND": float(pred_blend), "SHORT_CIRCUIT": False}
    return float(pred_final), meta

# ================== 7) MODEL ARTEFAKTLARI ==================
stage("Model dosyaları yazılıyor")
os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, "beta_icd.json"), "w", encoding="utf-8") as f:
    json.dump(beta_icd, f, ensure_ascii=False, indent=2)
with open(os.path.join(MODEL_DIR, "gamma_pairs.json"), "w", encoding="utf-8") as f:
    json.dump({f"{i}||{j}":v for (i,j), v in gamma_pairs.items()}, f, ensure_ascii=False, indent=2)
with open(os.path.join(MODEL_DIR, "config.json"), "w", encoding="utf-8") as f:
    json.dump({"created_at": datetime.datetime.now().isoformat(),
               "min_support": MIN_SUPPORT, "saturation_on": SATURATION_ON,
               "saturation_k": SATURATION_K, "notes": "Cinsiyetsiz β/γ; α=Jaccard; guardrails aktif"},
              f, ensure_ascii=False, indent=2)

# ================== 7.5) XGB ENSEMBLE ==================
if XGB_ENS_ON:
    stage("XGB (plain+log) eğitiliyor")
    icd_counts = Counter([icd for lst in train_df["ICD_List_Norm"] for icd in lst])
    XGB_TOP_ICDS = [icd for icd,_ in icd_counts.most_common(TOPK_ICD)]
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
    ohe.fit(train_df[["Bölüm","YaşGrup"]])
    mlb = MultiLabelBinarizer(classes=XGB_TOP_ICDS, sparse_output=True); mlb.fit([XGB_TOP_ICDS])
    def _pack_features(df_part: pd.DataFrame):
        X_cat = ohe.transform(df_part[["Bölüm","YaşGrup"]])
        icd_lists = df_part["ICD_List_Norm"].apply(lambda lst: [c for c in lst if c in XGB_TOP_ICDS])
        X_icd = mlb.transform(icd_lists)
        x_icd_count = np.asarray(df_part["ICD_Sayısı"]).reshape(-1,1)
        return hstack([X_cat, X_icd, x_icd_count], format="csr")
    X_train = _pack_features(train_df)
    y_train = train_df["Yatış Gün Sayısı"].astype(float).values
    y_log = np.log1p(y_train)
    xgb_plain = XGBRegressor(**XGB_PARAMS); xgb_log = XGBRegressor(**XGB_PARAMS)
    xgb_plain.fit(X_train, y_train); xgb_log.fit(X_train, y_log)
    joblib.dump(xgb_plain, os.path.join(MODEL_DIR,"xgb_plain.joblib"))
    joblib.dump(xgb_log,   os.path.join(MODEL_DIR,"xgb_log.joblib"))
    joblib.dump(ohe,       os.path.join(MODEL_DIR,"xgb_ohe.joblib"))
    joblib.dump(mlb,       os.path.join(MODEL_DIR,"xgb_mlb.joblib"))
    joblib.dump(XGB_TOP_ICDS, os.path.join(MODEL_DIR,"xgb_top_icds.joblib"))
    def xgb_predict_ens(yg, bolum, key, icd_list_norm=None):
        if icd_list_norm is None:
            icd_list_norm = key.split("||") if key else []
        df_one = pd.DataFrame({"Bölüm":[bolum],"YaşGrup":[yg],
                               "ICD_List_Norm":[icd_list_norm],"ICD_Sayısı":[len(icd_list_norm)]})
        X_one = _pack_features(df_one)
        p_plain = float(xgb_plain.predict(X_one)[0]); p_log = float(np.expm1(xgb_log.predict(X_one)[0]))
        p_ens = (1.0 - float(XGB_ALPHA_LOG)) * p_plain + float(XGB_ALPHA_LOG) * p_log
        return p_plain, p_log, p_ens
else:
    def xgb_predict_ens(yg, bolum, key, icd_list_norm=None): return np.nan, np.nan, np.nan

# ================== 8) PRED_LOS ==================
stage("PRED_LOS.xlsx üretiliyor")
uniq_combos_df = df[["YaşGrup","Bölüm","ICD_Set_Key"]].drop_duplicates().reset_index(drop=True)
pred_rows=[]
for r in tqdm(uniq_combos_df.itertuples(), total=len(uniq_combos_df), desc="PRED_LOS"):
    pred_rule, meta = predict_one(r.YaşGrup, r.Bölüm, r.ICD_Set_Key)
    icd_list_norm = r.ICD_Set_Key.split("||") if isinstance(r.ICD_Set_Key, str) and r.ICD_Set_Key else []
    p_plain, p_log, p_ens = xgb_predict_ens(r.YaşGrup, r.Bölüm, r.ICD_Set_Key, icd_list_norm)
    if STRICT_SHORT_CIRCUIT and meta.get("SHORT_CIRCUIT", False):
        pred_final_out = float(pred_rule)
    else:
        if XGB_RULE_BLEND is None or (not np.isfinite(p_ens)):
            pred_final_out = float(pred_rule)
        else:
            w = float(XGB_RULE_BLEND)
            pred_final_out = (1.0 - w) * float(pred_rule) + w * float(p_ens)
    pred_rows.append({
        "YaşGrup": r.YaşGrup, "Bölüm": r.Bölüm, "ICD_Set_Key": r.ICD_Set_Key,
        "Pred_Final": float(pred_final_out),
        "Pred_Final_Rounded": round_half_up(pred_final_out),
        "PRED_RULE": float(pred_rule),
        "PRED_XGB_PLAIN": float(p_plain) if pd.notna(p_plain) else np.nan,
        "PRED_XGB_LOG": float(p_log) if pd.notna(p_log) else np.nan,
        "PRED_XGB_ENS": float(p_ens) if pd.notna(p_ens) else np.nan, **meta
    })
pd.DataFrame(pred_rows).to_excel(PRED_LOS_XLSX, index=False); print(f"OK -> {PRED_LOS_XLSX}")

# ================== 9) YeniVakalar ==================
if MAKE_YENI_VAKALAR:
    stage("YeniVakalar.xlsx üretiliyor")
    yg_vals = df["YaşGrup"].dropna().unique().tolist()
    bolum_vals = df["Bölüm"].dropna().unique().tolist()
    icd_counts = Counter([icd for lst in df["ICD_List_Norm"] for icd in lst])
    top_icds = [icd for icd,_ in icd_counts.most_common(100)]
    def sample_icd_set():
        k = np.random.randint(1, min(5, max(2, len(top_icds))))
        return as_key(set(np.random.choice(top_icds, size=k, replace=False)))
    rows=[]
    for _ in tqdm(range(N_SAMPLES_YENI), desc="YeniVakalar"):
        yg = np.random.choice(yg_vals) if yg_vals else "35-50"
        bol = np.random.choice(bolum_vals) if bolum_vals else "Dahiliye"
        key = sample_icd_set()
        pred_rule, meta = predict_one(yg, bol, key)
        icd_list_norm = key.split("||") if key else []
        _, _, p_ens = xgb_predict_ens(yg, bol, key, icd_list_norm)
        if STRICT_SHORT_CIRCUIT and meta.get("SHORT_CIRCUIT", False):
            pred_out = float(pred_rule)
        else:
            if XGB_RULE_BLEND is None or (not np.isfinite(p_ens)):
                pred_out = float(pred_rule)
            else:
                w = float(XGB_RULE_BLEND)
                pred_out = (1.0 - w) * float(pred_rule) + w * float(p_ens)
        rows.append({"YaşGrup": yg, "Bölüm": bol, "ICD_Set_Key": key, "Pred_Final": round_half_up(pred_out), **meta})
    pd.DataFrame(rows).to_excel(YENI_VAKALAR_XLSX, index=False); print(f"OK -> {YENI_VAKALAR_XLSX}")

# ================== 10) VALID_PREDICTIONS ==================
stage("VALID_PREDICTIONS.xlsx üretiliyor (valid set)")
valid_rows=[]

def _norm_col(s: str) -> str:
    s = unicodedata.normalize("NFKD", s); s = "".join(ch for ch in s if not s or not unicodedata.combining(ch))
    s = s.lower(); s = re.sub(r"\s+", "_", s); s = re.sub(r"[^a-z0-9_]+", "_", s)
    return s.strip("_")

_norm_map = {_norm_col(c): c for c in valid_df.columns}
TRUE_LOS_COL = None
for c in ["yatis_gun_sayisi","yatis_gun_sayisi_"]:
    if c in _norm_map: TRUE_LOS_COL = _norm_map[c]; break
if TRUE_LOS_COL is None:
    if "Yatış Gün Sayısı" in valid_df.columns: TRUE_LOS_COL = "Yatış Gün Sayısı"
    elif "Yatış_Gün_Sayısı" in valid_df.columns: TRUE_LOS_COL = "Yatış_Gün_Sayısı"

n_3d_valid = 0
for r in tqdm(valid_df.itertuples(), total=len(valid_df), desc="VALID_PREDICTIONS"):
    yg, bol = canon_demo(r.YaşGrup, r.Bölüm)
    key = r.ICD_Set_Key

    if TRUE_LOS_COL is not None:
        _val = valid_df.loc[r.Index, TRUE_LOS_COL]
        try: true_los = float(_val) if pd.notna(_val) else np.nan
        except Exception: true_los = np.nan
    else:
        true_los = np.nan

    pred_rule, meta = predict_one(yg, bol, key)
    if meta.get("ANCHOR_SRC")=="3D": n_3d_valid += 1

    icd_list_norm = key.split("||") if isinstance(key, str) and key else []
    p_plain, p_log, p_ens = xgb_predict_ens(yg, bol, key, icd_list_norm)

    w = 0.5 if globals().get("XGB_RULE_BLEND") is None else float(XGB_RULE_BLEND)
    if STRICT_SHORT_CIRCUIT and meta.get("SHORT_CIRCUIT", False):
        pred_out = float(pred_rule)
    else:
        if (p_ens is None) or (not np.isfinite(p_ens)):
            pred_out = float(pred_rule)
        else:
            pred_out = (1.0 - w) * float(pred_rule) + w * float(p_ens)

    valid_rows.append({
        "YaşGrup": yg, "Bölüm": bol, "ICD_Set_Key": key,
        "True_LOS": true_los, "Pred_Final": float(pred_out),
        "Pred_Final_Rounded": round_half_up(pred_out),
        "PRED_RULE": float(pred_rule),
        "PRED_XGB_PLAIN": float(p_plain) if pd.notna(p_plain) else np.nan,
        "PRED_XGB_LOG": float(p_log) if pd.notna(p_log) else np.nan,
        "PRED_XGB_ENS": float(p_ens) if pd.notna(p_ens) else np.nan,
        **meta
    })

valid_pred_df = pd.DataFrame(valid_rows)
valid_pred_df.to_excel(VALID_PRED_XLSX, index=False)
print(f"OK -> {VALID_PRED_XLSX}")
print(f"VALID içinde 3D anchor sayısı: {n_3d_valid} satır")

# =============== ÖZET / METRİKLER ===============
stage("Özet")
if SPLIT_BY_COMBO:
    print("Split modu: KOMBINASYON (valid'de 3D bekleme).")
else:
    print("Split modu: SATIR (valid'de 3D mümkün).")
print("Train satır:", len(train_df), "Valid satır:", len(valid_df))
print("β (tekil) öğrenilen ICD:", len(beta_icd))
print("γ (ikili) öğrenilen pair:", len(gamma_pairs))

for col in ["True_LOS","Pred_Final","PRED_RULE","PRED_XGB_ENS"]:
    if col in valid_pred_df.columns:
        valid_pred_df[col] = pd.to_numeric(valid_pred_df[col], errors="coerce")

def _metrics(y_true, y_pred, tag):
    m = y_true.notna() & y_pred.notna()
    if m.any():
        yt = y_true[m].astype(float); yp = y_pred[m].astype(float)
        mae = mean_absolute_error(yt, yp); rmse = math.sqrt(mean_squared_error(yt, yp))
        print(f"{tag}  MAE: {mae:.4f}  RMSE: {rmse:.4f}")
    else:
        print(f"{tag}: Geçerli satır yok.")

_metrics(valid_pred_df["True_LOS"], valid_pred_df["Pred_Final"], "HARMAN (Rule ∘ XGB_ENS)")
if "PRED_RULE" in valid_pred_df.columns:
    _metrics(valid_pred_df["True_LOS"], valid_pred_df["PRED_RULE"], "KURAL (Rule)")
if "PRED_XGB_ENS" in valid_pred_df.columns:
    _metrics(valid_pred_df["True_LOS"], valid_pred_df["PRED_XGB_ENS"], "XGB (Plain+Log Ens)")

# ================== 11) APP KULLANIMI – FONKSİYONLAR ==================
def app_normalize_inputs(yas_grup: str, bolum: str, icd_input) -> tuple:
    yg_in = str(yas_grup or "").strip()
    b_in  = str(bolum or "").strip()
    yg, b = canon_demo(yg_in, b_in)

    if isinstance(icd_input, str):
        parts = [clean_icd(p) for p in re.split(r"[;,]", icd_input) if p.strip()]
    elif isinstance(icd_input, (list, tuple, set)):
        parts = [clean_icd(p) for p in icd_input if str(p).strip()]
    else:
        parts = []
    parts = sorted(set([p for p in parts if p]))
    key = "||".join(parts)
    return yg, b, parts, key

def app_predict(yas_grup: str, bolum: str, icd_input):
    yg, b, parts, key = app_normalize_inputs(yas_grup, bolum, icd_input)
    pred_rule, meta = predict_one(yg, b, key)

    p_plain, p_log, p_ens = xgb_predict_ens(yg, b, key, parts)
    if STRICT_SHORT_CIRCUIT and meta.get("SHORT_CIRCUIT", False):
        pred_out = float(pred_rule)
    else:
        if (p_ens is None) or (not np.isfinite(p_ens)) or (XGB_RULE_BLEND is None):
            pred_out = float(pred_rule)
        else:
            pred_out = (1.0 - float(XGB_RULE_BLEND)) * float(pred_rule) + float(XGB_RULE_BLEND) * float(p_ens)

    return {
        "yas_grup": yg,
        "bolum": b,
        "icd_list": parts,
        "icd_set_key": key,
        "pred_rule": float(pred_rule),
        "pred_xgb_plain": float(p_plain) if pd.notna(p_plain) else None,
        "pred_xgb_log": float(p_log) if pd.notna(p_log) else None,
        "pred_xgb_ens": float(p_ens) if pd.notna(p_ens) else None,
        "pred_final": float(pred_out),
        "pred_final_rounded": round_half_up(pred_out),
        "meta": meta
    }

def app_predict_many(records):
    out = []
    for rec in records:
        yg = rec.get("yas_grup", "")
        b  = rec.get("bolum", "")
        icd_in = rec.get("icd", rec.get("icd_list", ""))
        out.append(app_predict(yg, b, icd_in))
    return out

def app_info():
    return {
        "created_at": datetime.datetime.now().isoformat(),
        "winsorize_on": bool(WINSORIZE_ON),
        "xgb_on": bool(XGB_ENS_ON),
        "xgb_rule_blend": float(XGB_RULE_BLEND) if XGB_RULE_BLEND is not None else None,
        "strict_short_circuit": bool(STRICT_SHORT_CIRCUIT),
        "maps": {
            "lkp3_size": len(lkp3_map),
            "lkp2_size": len(lkp2_map),
            "lkp1_size": len(lkp1_map),
            "beta_icd": len(beta_icd),
            "gamma_pairs": len(gamma_pairs),
        }
    }

# ================== APP adapter ==================
def tahmin_et(icd_list, bolum=None, yas_grup=None):
    if isinstance(icd_list, str):
        icd_list = [s.strip() for s in icd_list.split(",") if s.strip()]

    parts = [clean_icd(x) for x in icd_list if str(x).strip()]
    parts = sorted(set([p for p in parts if p]))
    key = "||".join(parts)

    yg, b = canon_demo((yas_grup or "").strip(), (bolum or "").strip())

    pred_rule, meta = predict_one(yg, b, key)
    p_plain, p_log, p_ens = xgb_predict_ens(yg, b, key, parts)

    if STRICT_SHORT_CIRCUIT and meta.get("SHORT_CIRCUIT", False):
        pred_out = float(pred_rule)
    else:
        if (p_ens is None) or (not np.isfinite(p_ens)) or (XGB_RULE_BLEND is None):
            pred_out = float(pred_rule)
        else:
            w = float(XGB_RULE_BLEND)
            pred_out = (1.0 - w) * float(pred_rule) + w * float(p_ens)

    return {"Pred_Final": float(pred_out), "Pred_Final_Rounded": round_half_up(pred_out)}
# ================== /APP adapter ==================