# -*- coding: utf-8 -*-
# Hepsi bir arada: Lookup Excel + β/γ Katkı Modeli + Jaccard Harmanlama + Guardrails
# ŞEMA (Cinsiyetsiz): 3D(YaşGrup+Bölüm+ICD_Set) → 2D(Bölüm+ICD_Set) → 1D(ICD_Set) → 0D(genel/komşu)
# Opsiyoneller dahil "BİREBİR" uygulandı; cinsiyet tamamen çıkarıldı; eşik değerleri=1.
import os, json, re, warnings, datetime, itertools, math, random
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict, Counter

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from scipy import sparse
from scipy.sparse import hstack  # <-- EKLENDİ

# (Şimdilik kullanılmıyor ama kancalar dursun)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
#from sentence_transformers import SentenceTransformer
import joblib

# ---- YENİ: Progress bar için tqdm (yoksa zarifçe boş döngüye düş)
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x
# ---- /YENİ

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
random.seed(42)
np.random.seed(42)

# ====================== KULLANICI AYARLARI ======================
EXCEL_PATH = "Veri2024.xlsx"          # kaynak veri
LOOKUP_XLSX = "LOS_Lookup_All.xlsx"   # sabit isim
MODEL_DIR = "model_out"
MAKE_YENI_VAKALAR = True              # True => YeniVakalar + PRED_LOS + VALID_PREDICTIONS üret
YENI_VAKALAR_XLSX = "YeniVakalar.xlsx"
PRED_LOS_XLSX = "PRED_LOS.xlsx"
VALID_PRED_XLSX = "VALID_PREDICTIONS.xlsx"
N_SAMPLES_YENI = 200
REQUIRE_ICD = True                    # ICD yoksa satırı ele
FALLBACK_FROM_TEXT = True             # ICD boşsa 'ICD Adi Ve Kodu' içinden regex ile çek
RANDOM_SEED = 42
TOPK_ICD = 400                        # (ileride) one-hot için
MIN_SUPPORT = 1                       # eşik=1 (her yerde)

# ---- DEĞİŞTİ (saturation açıldı ve K küçültüldü)
SATURATION_ON = True                  # aşırı büyüme kontrolü (aktif)
SATURATION_K = 3.0                   # doygunluk eğrisi param (daha agresif doyum)

# ---- YENİ (Top-K komşu ayarları, hafif tuning)
TOPK_NEIGHBORS = 10                  # K=10
RHO_J = 1.2                          # ρ ~ 1..2 arası

# ---- YENİ (zayıf destek büzme + cap + removal yumuşatma)
SHRINK_1SUPPORT_SCALE = 0.15         # support<3 için katkı ölçeği
REMOVAL_PENALTY = 0.5                # A\T ve kaybolan çift cezalarını yumuşat
CAP_MARJ = 1.0                      # P90 üst tavan marjı

# ---- YENİ (TRUNCATE KALDIRILDI, SADECE WINSORIZE EKLENDİ) ----
WINSORIZE_ON = True                  # True: train set LOS winsorize edilir (truncate yok)
WINSOR_LO = 0.00973                  # alt yüzde (örn. 0.01 = %1)
WINSOR_HI = 0.9785                   # üst yüzde (örn. 0.99 = %99)

# ---- YENİ: XGB ENSEMBLE AYARLARI -------------------------------
XGB_ENS_ON = True                    # iki XGB modelini eğit ve kullan
XGB_ALPHA_LOG = 0.20                 # ens. ağırlığı: log-hedef payı (0..1)
XGB_RULE_BLEND = 0.50                # kural tabanlı + XGB ens yarı yarıya
XGB_PARAMS = dict(                   # makul başlangıç hiperparam.
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
    n_jobs=4,
    tree_method="hist"
)
# ===============================================================

def stage(msg): print(f"[STAGE] {msg}", flush=True)

def round_half_up(x):
    if pd.isna(x): return None
    return int(Decimal(str(float(x))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

def yas_to_years(val):
    if pd.isna(val): return pd.NA
    if isinstance(val, (int, float)): return float(val)
    s = str(val).strip().lower()
    if re.fullmatch(r"\d+(?:[.,]\d+)?", s):
        return float(s.replace(",", "."))
    yil = re.findall(r"(\d+)\s*yıl", s)
    ay  = re.findall(r"(\d+)\s*ay", s)
    gun = re.findall(r"(\d+)\s*gün", s)
    years = 0.0
    if yil: years += sum(float(x) for x in yil)
    if ay:  years += sum(float(x) for x in ay) / 12
    if gun: years += sum(float(x) for x in gun) / 365
    if years == 0.0 and not (yil or ay or gun): return pd.NA
    return round(years, 2)

# Parantez/Etiket temizliği + Regex
_PAREN_MAP = str.maketrans({'（':'(', '）':')', '【':'[', '】':']', '＜':'<', '＞':'>', '｛':'{', '｝':'}'})
_PREFIX_RE = re.compile(r"""^\s*[\(\[\{\<]\s*(?:ö|ö|k|a)\s*[\)\]\}\>]\s*""", re.IGNORECASE | re.VERBOSE)
_ANYWHERE_TAG_RE = re.compile(r"\(\s*(?:ö|ö|k|a)\s*\)", re.IGNORECASE)
_ICD_CODE_RE = re.compile(r"\b([A-Z][0-9]{2}(?:\.[A-Z0-9]{1,4})?)\b", re.IGNORECASE)

def clean_icd(raw) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)): return ""
    s = str(raw).strip()
    if not s: return ""
    s = s.translate(_PAREN_MAP)
    prev = None
    while prev != s:
        prev = s
        s = _PREFIX_RE.sub("", s)
    return s.strip().upper()

def clean_text_anywhere_tags(raw) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)): return ""
    s = str(raw).strip()
    if not s: return ""
    s = s.translate(_PAREN_MAP)
    s = _ANYWHERE_TAG_RE.sub("", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def split_icd_cell(s):
    if pd.isna(s): return []
    s = str(s).translate(_PAREN_MAP)
    parts = re.split(r"[;,]", s)
    parts = [clean_icd(p) for p in parts]
    parts = [p for p in parts if p]
    return parts

def normalize_icd_set(lst):
    lst_clean = [clean_icd(x) for x in lst if str(x).strip() != ""]
    uniq = sorted(set(lst_clean), key=str)
    return uniq, "||".join(uniq)

def clean_icd_set_key(key: str) -> str:
    if key is None or (isinstance(key, float) and pd.isna(key)): return ""
    parts = [p.strip().upper() for p in str(key).split("||")]
    parts = [clean_icd(p) for p in parts]
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
    inter = len(a & b)
    union = len(a | b)
    return inter/union if union>0 else 0.0

def as_set(key:str)->set:
    if not key: return set()
    return set([k for k in key.split("||") if k])

def as_key(s:set)->str:
    return "||".join(sorted(s))

def as_csr(x):
    return x if sparse.issparse(x) else sparse.csr_matrix(x)

# ================== 1) VERİYİ YÜKLE & TEMİZLE ==================
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

if REQUIRE_ICD and df["ICD Kodu"].fillna("").eq("") .any() and FALLBACK_FROM_TEXT:
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

# Embedding metni (kanca)
base_text_col = "ICD Adi Ve Kodu" if "ICD Adi Ve Kodu" in df.columns else "ICD Kodu"
df["ICD_Text_Embed"] = df[base_text_col].map(clean_text_anywhere_tags).fillna("")

# Yaş grup
df["Yaş_Yıl_Int"] = pd.to_numeric(df["Yaş"], errors="coerce").round().astype("Int64")
df["YaşGrup"] = df["Yaş"].apply(yas_to_group)

# ================== 2) TRAIN/VALID SPLIT ==================
stage("Train/Valid ayrımı (kombinasyon temelli)")
# Kombinasyon id: (YG,Bölüm,ICD_Set_Key)
df["ComboID"] = df["YaşGrup"].astype(str)+"||"+df["Bölüm"].astype(str)+"||"+df["ICD_Set_Key"].astype(str)
unique_combos = df["ComboID"].dropna().unique()
train_combos, valid_combos = train_test_split(unique_combos, test_size=0.2, random_state=RANDOM_SEED)
is_train = df["ComboID"].isin(train_combos)
train_df = df[is_train].copy()
valid_df = df[~is_train].copy()

# ---- YENİ: WINSORIZE (SADECE TRAIN ÜZERİNDE, TRUNCATE YOK) ----
def _winsorize_series(s: pd.Series, lo: float, hi: float) -> pd.Series:
    q_lo = s.quantile(lo)
    q_hi = s.quantile(hi)
    return s.clip(lower=q_lo, upper=q_hi)

if WINSORIZE_ON:
    stage(f"Winsorize uygulanıyor (train) - p{int(WINSOR_LO*100)} / p{int(WINSOR_HI*100)}")
    train_df["Yatış Gün Sayısı"] = _winsorize_series(train_df["Yatış Gün Sayısı"], WINSOR_LO, WINSOR_HI)
# ---- /WINSORIZE ----

# ================== 3) LOOKUP TABLOLARI (train üzerinde) ==================
stage("Lookup tabloları hesaplanıyor (train, cinsiyetsiz)")

# 3D
lkp3 = (
    train_df.groupby(["YaşGrup", "Bölüm", "ICD_Set_Key"])["Yatış Gün Sayısı"]
        .agg(N="count", Ortalama=lambda x: round_half_up(x.mean()), P50="median", P90=p90)
        .reset_index()
)
lkp3["ICD_Set_Key"] = lkp3["ICD_Set_Key"].apply(clean_icd_set_key)

# 2D
lkp2 = (
    train_df.groupby(["Bölüm", "ICD_Set_Key"])["Yatış Gün Sayısı"]
        .agg(N="count", Ortalama=lambda x: round_half_up(x.mean()), P50="median", P90=p90)
        .reset_index()
)
lkp2["ICD_Set_Key"] = lkp2["ICD_Set_Key"].apply(clean_icd_set_key)

# 1D
lkp1 = (
    train_df.groupby(["ICD_Set_Key"])["Yatış Gün Sayısı"]
        .agg(N="count", Ortalama=lambda x: round_half_up(x.mean()), P50="median", P90=p90)
        .reset_index()
)
lkp1["ICD_Set_Key"] = lkp1["ICD_Set_Key"].apply(clean_icd_set_key)

# 0D
lkp0 = pd.DataFrame({
    "N":   [train_df.shape[0]],
    "Ortalama": [round_half_up(train_df["Yatış Gün Sayısı"].mean())],
    "P50": [train_df["Yatış Gün Sayısı"].median()],
    "P90": [train_df["Yatış Gün Sayısı"].quantile(0.9)]
})

# Tekil ICD P50 (global)
single = train_df[train_df["ICD_Sayısı"]==1].copy()
single["ICD_Kod"] = single["ICD_List_Norm"].str[0]
LKP_ICD = (
    single.groupby("ICD_Kod")["Yatış Gün Sayısı"]
          .agg(N="count", P50="median").reset_index()
)

# İkili setlerin P50'si (tam iki ICD'li satırlardan) — γ öğrenimine yardımcı
pairs = train_df[train_df["ICD_Sayısı"]==2].copy()
pairs["PairKey"] = pairs["ICD_List_Norm"].apply(lambda lst: "||".join(sorted(lst)))
LKP_PAIR = (
    pairs.groupby(["YaşGrup","Bölüm","PairKey"])["Yatış Gün Sayısı"]
         .agg(N="count", P50="median").reset_index()
)

# ---- YENİ: Demografi (YG+Bölüm) bazlı P90 cap referansı
DEMO_P90_MAP = (
    train_df.groupby(["YaşGrup","Bölüm"])["Yatış Gün Sayısı"]
            .quantile(0.9).reset_index().rename(columns={"Yatış Gün Sayısı":"P90"})
)
demop90_map = {(r["YaşGrup"], r["Bölüm"]): float(r["P90"]) for _, r in DEMO_P90_MAP.iterrows()}

# ================== 4) β (tekil) ve γ (ikili) KATKILARI ÖĞREN ==================
stage("β (tekil) ve γ (ikili) katkıları öğreniliyor (robust medyan, eşik=1)")

# β_icd: medyan( P50({i, ...}) - max(P50({i}), P50(alt set)) )
icd_to_beta_samples = defaultdict(list)

# 3D bağlamda gözlenen setlerin P50'leri
ctx3 = lkp3.copy()
ctx3["Set"] = ctx3["ICD_Set_Key"].apply(as_set)

# 1D tekil P50 lookup
one_map = dict(zip(lkp1["ICD_Set_Key"], lkp1["P50"]))
single_map = dict(zip(LKP_ICD["ICD_Kod"], LKP_ICD["P50"]))

for _, row in ctx3.iterrows():
    yg, bolum, key, p50, n = row["YaşGrup"], row["Bölüm"], row["ICD_Set_Key"], row["P50"], row["N"]
    S = as_set(key)
    if len(S) < 2:
        continue
    for icd in S:
        base_candidates = []
        if icd in single_map: base_candidates.append(single_map[icd])
        if icd in as_set(next(iter([icd]))): pass  # no-op
        key_icd = icd
        if key_icd in one_map: base_candidates.append(one_map[key_icd])
        base = max(base_candidates) if base_candidates else 0.0
        delta = max(0.0, float(p50) - float(base))
        icd_to_beta_samples[icd].append(delta)

beta_icd = {}
beta_support = {}  # ---- YENİ: destek sayısı
for icd, samples in icd_to_beta_samples.items():
    if len(samples) >= MIN_SUPPORT:
        beta_icd[icd] = float(np.median(samples))
        beta_support[icd] = int(len(samples))

# γ_{i,j}: medyan( P50({i,j}) - max(P50({i}), P50({j})) )
pair_to_gamma_samples = defaultdict(list)

for _, row in LKP_PAIR.iterrows():
    yg, bolum, pairkey, p50, n = row["YaşGrup"], row["Bölüm"], row["PairKey"], row["P50"], row["N"]
    i, j = pairkey.split("||")
    base_i = single_map.get(i, 0.0)
    base_j = single_map.get(j, 0.0)
    base = max(base_i, base_j)
    delta = max(0.0, float(p50) - float(base))
    pair_to_gamma_samples[(i,j)] = pair_to_gamma_samples[(i,j)] + [delta]
    pair_to_gamma_samples[(j,i)] = pair_to_gamma_samples[(j,i)] + [delta]  # simetri

gamma_pairs = {}
gamma_support = {}  # ---- YENİ: destek sayısı
for pair, samples in pair_to_gamma_samples.items():
    if len(samples) >= MIN_SUPPORT:
        gamma_pairs[pair] = float(np.median(samples))
        gamma_support[pair] = int(len(samples))

# ================== 5) LOOKUP EXCEL'E YAZ ==================
stage("Lookup Excel yazılıyor (cinsiyetsiz)")

# ---- EKLE: Görsellik için ek sayfalar (modelden bağımsız)
_br_base = df[["ICD_Set_Key","ICD_List_Norm"]].copy()
BR_ICDSET_MAP = _br_base.explode("ICD_List_Norm").rename(columns={"ICD_List_Norm":"ICD"})
BR_ICDSET_MAP = BR_ICDSET_MAP.dropna(subset=["ICD"]).drop_duplicates().reset_index(drop=True)
BR_ICDSET_MAP = BR_ICDSET_MAP.dropna(subset=["ICD"]).reset_index(drop=True)

_all_icds = sorted({icd for lst in df["ICD_List_Norm"] for icd in lst})
DIM_ICD = pd.DataFrame({"ICD": _all_icds})

_age_order = ["0-1","2-5","5-10","10-15","15-25","25-35","35-50","50-65","65+"]
_present = [yg for yg in _age_order if yg in set(df["YaşGrup"].dropna().astype(str).unique())]
DIM_YASGRUP = pd.DataFrame({"YaşGrup": _present})
# ---- /EKLE

with pd.ExcelWriter(LOOKUP_XLSX, engine="xlsxwriter") as w:
    lkp3.to_excel(w, index=False, sheet_name="LKP_3D_YasGrup")
    lkp2.to_excel(w, index=False, sheet_name="LKP_2D")
    lkp1.to_excel(w, index=False, sheet_name="LKP_1D")
    lkp0.to_excel(w, index=False, sheet_name="LKP_0D")
    LKP_ICD.to_excel(w, index=False, sheet_name="LKP_ICD")
    df[["ICD_Text_Embed"]].to_excel(w, index=False, sheet_name="TEXT_EMB_SOURCE")
    BR_ICDSET_MAP.to_excel(w, index=False, sheet_name="BR_ICDSET_MAP")
    DIM_ICD.to_excel(w, index=False, sheet_name="DIM_ICD")
    DIM_YASGRUP.to_excel(w, index=False, sheet_name="DIM_YASGRUP")
print(f"OK -> {LOOKUP_XLSX}")

# ================== 6) ANCHOR & PREDICT YARDIMCI ==================
stage("Prediction yardımcı yapılar hazırlanıyor")

# Index map'ler
lkp3_map = {(r["YaşGrup"], r["Bölüm"], r["ICD_Set_Key"]):(r["P50"], r["N"]) for _,r in lkp3.iterrows()}
lkp2_map = {(r["Bölüm"], r["ICD_Set_Key"]):(r["P50"], r["N"]) for _,r in lkp2.iterrows()}
lkp1_map = {r["ICD_Set_Key"]:(r["P50"], r["N"]) for _,r in lkp1.iterrows()}
lkp0_p50 = float(lkp0["P50"].iloc[0]) if len(lkp0)>0 else 0.0
lkp0_p90 = float(lkp0["P90"].iloc[0]) if len(lkp0)>0 else 0.0  # ---- YENİ: global P90 fallback

# Aynı demografide tüm 3D anahtarları listeleyelim (komşu arama için)
ctx3_by_demo = defaultdict(list)
for _, r in lkp3.iterrows():
    ctx3_by_demo[(r["YaşGrup"], r["Bölüm"])].append(r["ICD_Set_Key"])

# Pair P50 floor (global, demografiye bakmadan)
pair_floor_map = {}
for _, r in LKP_PAIR.iterrows():
    pair_floor_map[r["PairKey"]] = max(pair_floor_map.get(r["PairKey"], 0.0), float(r["P50"]))

single_floor_map = dict(zip(LKP_ICD["ICD_Kod"], LKP_ICD["P50"]))

def find_anchor(yg:str, bolum:str, key:str):
    """Lookup zinciri: 3D -> 2D -> 1D -> yoksa None (komşuya geçilecek)"""
    if (yg, bolum, key) in lkp3_map:
        p50, n = lkp3_map[(yg, bolum, key)]
        return "3D", float(p50), n, key
    if (bolum, key) in lkp2_map:
        p50, n = lkp2_map[(bolum, key)]
        return "2D", float(p50), n, key
    if key in lkp1_map:
        p50, n = lkp1_map[key]
        return "1D", float(p50), n, key
    return None, None, 0, None

# ---- DEĞİŞTİ: Tek komşu yerine Top-K ağırlıklı ortalama
def _topk_weighted_anchor(candidates, target_set:set, K:int=TOPK_NEIGHBORS, rho:float=RHO_J):
    """
    candidates: iterable of (key, p50, n)
    Dönüş: (bestJ, weighted_p50, bestKey)
    Ağırlık: w = (J**rho) * log(1+N)
    """
    scored = []
    for key, p50, n in candidates:
        J = jaccard(target_set, as_set(key))
        if p50 is None:
            continue
        scored.append((J, float(p50), int(n if n is not None else 0), key))

    if not scored:
        return 0.0, None, None

    # En iyi tek komşu (tutarlılık için anchor_key bu)
    scored.sort(key=lambda x: (x[0], x[2], x[1]), reverse=True)
    bestJ, bestP50, _bestN, bestKey = scored[0]

    # Sıfır Jaccard durumunda ağırlık toplamı 0 olabilir → tek komşuya düş
    topk = [r for r in scored if r[0] > 0.0][:K]
    if not topk:
        return bestJ, bestP50, bestKey

    weights = []
    vals = []
    for J, p50, n, _k in topk:
        w = (J ** float(rho)) * math.log1p(max(0, n))
        weights.append(w)
        vals.append(p50)

    W = sum(weights)
    if W <= 0:
        return bestJ, bestP50, bestKey

    weighted_p50 = sum(v*w for v, w in zip(vals, weights)) / W
    return bestJ, float(weighted_p50), bestKey

def nearest_neighbor_anchor(yg:str, bolum:str, target_key:str):
    """
    Jaccard komşu-ankor arama SIRASI:
      1) 3D aynı demografi (YaşGrup+Bölüm)  -> ANCHOR_SRC='NEIGHBOR_3D_DEMO'
      2) 2D sadece Bölüm                    -> ANCHOR_SRC='NEIGHBOR_2D'
      3) 1D global ICD set                  -> ANCHOR_SRC='NEIGHBOR_1D'
      4) Hiç aday yoksa 0D genel            -> ANCHOR_SRC='NEIGHBOR_0D'
    Top-K: anchor_p50 = ağırlıklı ortalama; anchor_key = en iyi tek komşu.
    """
    target = as_set(target_key)

    # 1) 3D - aynı demografi
    cand3 = []
    for key in ctx3_by_demo.get((yg, bolum), []):
        p50, n = lkp3_map.get((yg, bolum, key), (None, 0))
        cand3.append((key, p50, n))
    bestJ, w_p50, bestKey = _topk_weighted_anchor(cand3, target)
    if bestKey is not None:
        return bestJ, float(w_p50 if w_p50 is not None else lkp0_p50), bestKey, "3D_DEMO"

    # 2) 2D - aynı bölüm
    cand2 = []
    for (b, key), (p50, n) in lkp2_map.items():
        if b == bolum:
            cand2.append((key, p50, n))
    bestJ, w_p50, bestKey = _topk_weighted_anchor(cand2, target)
    if bestKey is not None:
        return bestJ, float(w_p50 if w_p50 is not None else lkp0_p50), bestKey, "2D"

    # 3) 1D - global
    cand1 = []
    for key, (p50, n) in lkp1_map.items():
        cand1.append((key, p50, n))
    bestJ, w_p50, bestKey = _topk_weighted_anchor(cand1, target)
    if bestKey is not None:
        return bestJ, float(w_p50 if w_p50 is not None else lkp0_p50), bestKey, "1D"

    # 4) 0D - genel
    return 0.0, lkp0_p50, None, "0D"
# ---- /DEĞİŞTİ

def model_contrib(target_key:str, anchor_key:str):
    """
    β/γ katkıları (tek taraflı imza):
      - Eklenen tekiller (T\\A): +β
      - Çıkan tekiller (A\\T):  –β
      - Eklenen çiftler:        +γ
      - Kaybolan çiftler:       –γ
    Not: β ve γ öğrenimde ≥0; burada yalnızca 'fazlayı geri alma' amaçlı negatif işaret uygulanır.
    """
    T = as_set(target_key)
    A = as_set(anchor_key) if anchor_key else set()

    # ---- YENİ: destek tabanlı büzme yardımcıları
    def beta_scaled(icd):
        val = max(0.0, beta_icd.get(icd, 0.0))
        sup = beta_support.get(icd, 0)
        scale = 1.0 if sup >= 3 else SHRINK_1SUPPORT_SCALE
        return val * scale

    def gamma_lookup_scaled(i, j):
        val = max(0.0, gamma_pairs.get((i, j), gamma_pairs.get((j, i), 0.0)))
        sup = max(gamma_support.get((i, j), 0), gamma_support.get((j, i), 0))
        scale = 1.0 if sup >= 3 else SHRINK_1SUPPORT_SCALE
        return val * scale

    # Tekiller
    add_single = sorted(list(T - A))
    rem_single = sorted(list(A - T))
    beta_plus  = sum(beta_scaled(i) for i in add_single)
    beta_minus = sum(beta_scaled(i) for i in rem_single)
    beta_sum   = beta_plus - REMOVAL_PENALTY * beta_minus  # ---- DEĞİŞTİ

    # Çiftler
    pairs_T = set()
    for i, j in itertools.combinations(sorted(T), 2):
        pairs_T.add((i, j))

    pairs_A = set()
    for i, j in itertools.combinations(sorted(A), 2):
        pairs_A.add((i, j))

    add_pairs = pairs_T - pairs_A
    rem_pairs = pairs_A - pairs_T

    gamma_plus  = sum(gamma_lookup_scaled(i, j) for (i, j) in add_pairs)
    gamma_minus = sum(gamma_lookup_scaled(i, j) for (i, j) in rem_pairs)
    gamma_sum   = gamma_plus - REMOVAL_PENALTY * gamma_minus  # ---- DEĞİŞTİ

    return beta_sum, gamma_sum, add_single

def saturation(total_add:float, k:float=SATURATION_K):
    if not SATURATION_ON:
        return total_add
    return float(k * (1.0 - math.exp(-float(total_add)/float(k))))

def guardrails(yg:str, bolum:str, target_key:str, pred:float):
    """
    Tekil floor KALDIRILDI.
    Pair floor ve alt-küme (subset) floor'lar devam ediyor.
    """
    T = as_set(target_key)
    floor1 = pred  # tekil floor kaldırıldı, doğrudan pred

    # Pair floor
    floor2 = floor1
    for i, j in itertools.combinations(sorted(T), 2):
        pf = pair_floor_map.get(f"{i}||{j}", 0.0)
        pf = max(pf, pair_floor_map.get(f"{j}||{i}", 0.0))
        floor2 = max(floor2, pf)

    # Alt-küme floor (3D→2D→1D)
    floor3 = floor2
    for r in lkp3[(lkp3["YaşGrup"]==yg) & (lkp3["Bölüm"]==bolum)].itertuples():
        S = as_set(r.ICD_Set_Key)
        if S.issubset(T):
            floor3 = max(floor3, float(r.P50))
    for r in lkp2[lkp2["Bölüm"]==bolum].itertuples():
        S = as_set(r.ICD_Set_Key)
        if S.issubset(T):
            floor3 = max(floor3, float(r.P50))
    for r in lkp1.itertuples():
        S = as_set(r.ICD_Set_Key)
        if S.issubset(T):
            floor3 = max(floor3, float(r.P50))
    return floor3

def predict_one(yg:str, bolum:str, target_key:str):
    src, anchor_p50, n, anchor_key = find_anchor(yg, bolum, target_key)

    # === KISA DEVRE: 3D/2D/1D tam eşleşmede HİÇBİR ŞEY ekleme, GUARDRAILS DA YOK ===
    if anchor_p50 is not None and anchor_key == target_key and src in ("3D", "2D", "1D"):
        meta = {
            "ANCHOR_SRC": src,
            "ANCHOR_KEY": anchor_key if anchor_key else "",
            "ANCHOR_P50": float(anchor_p50),
            "ALPHA_JACCARD": 0.0,
            "ADDED_ICDS": "",
            "BETA_SUM": 0.0,
            "GAMMA_SUM": 0.0,
            "MODEL_PRED": float(anchor_p50),
            "PRED_BLEND": float(anchor_p50),
        }
        return float(anchor_p50), meta
    # === /KISA DEVRE ===

    if anchor_p50 is None:
        J, neigh_p50, neigh_key, neigh_src = nearest_neighbor_anchor(yg, bolum, target_key)
        anchor_p50, anchor_key = float(neigh_p50), neigh_key
        src = f"NEIGHBOR_{neigh_src}"
        alpha = float(J)  # ALPHA = en iyi tek komşu J (değişmedi)
    else:
        alpha = 0.0

    beta_sum, gamma_sum, added_icds = model_contrib(target_key, anchor_key)
    add_total = saturation(beta_sum + gamma_sum)
    model_pred = float(anchor_p50) + add_total
    pred_blend = (1.0 - alpha) * model_pred + alpha * float(anchor_p50)

    # ---- YENİ: Erken guardrail (pred_blend < 1 ise anchor P50'ye kısa devre)
    if pred_blend < 1.0:
        pred_final = float(anchor_p50)
    else:
        pred_final = pred_blend

    # ---- YENİ: P90 CAP (demografi+bölüm)
    cap_ref = demop90_map.get((yg, bolum), lkp0_p90)
    cap_val = float(cap_ref) * float(CAP_MARJ) if cap_ref is not None else float("inf")
    if cap_val is not None:
        pred_final = min(float(pred_final), float(cap_val))
    # ---- /P90 CAP

    meta = {
        "ANCHOR_SRC": src,
        "ANCHOR_KEY": anchor_key if anchor_key else "",
        "ANCHOR_P50": float(anchor_p50),
        "ALPHA_JACCARD": float(alpha),
        "ADDED_ICDS": ",".join(added_icds),
        "BETA_SUM": float(beta_sum),
        "GAMMA_SUM": float(gamma_sum),
        "MODEL_PRED": float(model_pred),
        "PRED_BLEND": float(pred_blend),
    }
    return float(pred_final), meta

# ================== 7) MODEL DOSYALARINI OLUŞTUR ==================
stage("Model dosyaları yazılıyor")
os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, "beta_icd.json"), "w", encoding="utf-8") as f:
    json.dump(beta_icd, f, ensure_ascii=False, indent=2)
with open(os.path.join(MODEL_DIR, "gamma_pairs.json"), "w", encoding="utf-8") as f:
    gamma_serial = {f"{i}||{j}":v for (i,j), v in gamma_pairs.items()}
    json.dump(gamma_serial, f, ensure_ascii=False, indent=2)
with open(os.path.join(MODEL_DIR, "config.json"), "w", encoding="utf-8") as f:
    json.dump({
        "created_at": datetime.datetime.now().isoformat(),
        "min_support": MIN_SUPPORT,
        "saturation_on": SATURATION_ON,
        "saturation_k": SATURATION_K,
        "notes": "Cinsiyetsiz β/γ; α=Jaccard; guardrails aktif"
    }, f, ensure_ascii=False, indent=2)

# ================== 7.5) XGB ENSEMBLE EĞİTİM (YENİ) ==================
if XGB_ENS_ON:
    stage("XGB (plain + log-target) özellikleri hazırlanıyor ve eğitiliyor")

    # ---- ICD top-K sınıfları (train'den)
    icd_counts = Counter([icd for lst in train_df["ICD_List_Norm"] for icd in lst])
    XGB_TOP_ICDS = [icd for icd, _ in icd_counts.most_common(TOPK_ICD)]

    # Dönüştürücüler
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
  # <<< SADECE BU SATIR DÜZELTİLDİ
    ohe.fit(train_df[["Bölüm", "YaşGrup"]])

    mlb = MultiLabelBinarizer(classes=XGB_TOP_ICDS, sparse_output=True)
    mlb.fit([XGB_TOP_ICDS])  # sınıfları sabitle

    def _pack_features(df_part: pd.DataFrame):
        # Kategorikler
        X_cat = ohe.transform(df_part[["Bölüm", "YaşGrup"]])
        # ICD multi-hot (sadece TOPK)
        icd_lists = df_part["ICD_List_Norm"].apply(lambda lst: [c for c in lst if c in XGB_TOP_ICDS])
        X_icd = mlb.transform(icd_lists)
        # Sayısal küçük özellikler (ICD sayısı)
        x_icd_count = np.asarray(df_part["ICD_Sayısı"]).reshape(-1, 1)
        X = hstack([X_cat, X_icd, x_icd_count], format="csr")
        return X

    X_train = _pack_features(train_df)
    y_train = train_df["Yatış Gün Sayısı"].astype(float).values
    y_log = np.log1p(y_train)

    xgb_plain = XGBRegressor(**XGB_PARAMS)
    xgb_log = XGBRegressor(**XGB_PARAMS)

    xgb_plain.fit(X_train, y_train)
    xgb_log.fit(X_train, y_log)

    # Artefaktları kaydet
    joblib.dump(xgb_plain, os.path.join(MODEL_DIR, "xgb_plain.joblib"))
    joblib.dump(xgb_log, os.path.join(MODEL_DIR, "xgb_log.joblib"))
    joblib.dump(ohe, os.path.join(MODEL_DIR, "xgb_ohe.joblib"))
    joblib.dump(mlb, os.path.join(MODEL_DIR, "xgb_mlb.joblib"))
    joblib.dump(XGB_TOP_ICDS, os.path.join(MODEL_DIR, "xgb_top_icds.joblib"))

    def xgb_predict_ens(yg: str, bolum: str, key: str, icd_list_norm=None):
        # tek satır özellik kur
        if icd_list_norm is None:
            icd_list_norm = key.split("||") if key else []
        df_one = pd.DataFrame({
            "Bölüm": [bolum],
            "YaşGrup": [yg],
            "ICD_List_Norm": [icd_list_norm],
            "ICD_Sayısı": [len(icd_list_norm)]
        })
        X_one = _pack_features(df_one)
        p_plain = float(xgb_plain.predict(X_one)[0])
        p_log = float(np.expm1(xgb_log.predict(X_one)[0]))
        p_ens = (1.0 - float(XGB_ALPHA_LOG)) * p_plain + float(XGB_ALPHA_LOG) * p_log
        return p_plain, p_log, p_ens
else:
    def xgb_predict_ens(yg, bolum, key, icd_list_norm=None):
        return np.nan, np.nan, np.nan

# ================== 8) PRED_LOS (tüm benzersiz kombinasyonlar) ==================
stage("PRED_LOS.xlsx üretiliyor")
uniq_combos_df = df[["YaşGrup","Bölüm","ICD_Set_Key"]].drop_duplicates().reset_index(drop=True)
pred_rows = []
for r in tqdm(uniq_combos_df.itertuples(), total=len(uniq_combos_df), desc="PRED_LOS"):
    pred_rule, meta = predict_one(r.YaşGrup, r.Bölüm, r.ICD_Set_Key)

    # XGB tahminleri (YENİ)
    icd_list_norm = r.ICD_Set_Key.split("||") if isinstance(r.ICD_Set_Key, str) and r.ICD_Set_Key else []
    p_plain, p_log, p_ens = xgb_predict_ens(r.YaşGrup, r.Bölüm, r.ICD_Set_Key, icd_list_norm)

    # Opsiyonel kural+XGB blend
    if XGB_RULE_BLEND is None or (not np.isfinite(p_ens)):
        pred_final_out = pred_rule
    else:
        w = float(XGB_RULE_BLEND)
        pred_final_out = (1.0 - w) * float(pred_rule) + w * float(p_ens)

    pred_rows.append({
        "YaşGrup": r.YaşGrup,
        "Bölüm": r.Bölüm,
        "ICD_Set_Key": r.ICD_Set_Key,
        "Pred_Final": round_half_up(pred_final_out),
        "PRED_RULE": float(pred_rule),
        "PRED_XGB_PLAIN": float(p_plain) if pd.notna(p_plain) else np.nan,
        "PRED_XGB_LOG": float(p_log) if pd.notna(p_log) else np.nan,
        "PRED_XGB_ENS": float(p_ens) if pd.notna(p_ens) else np.nan,
        **meta
    })
pred_df = pd.DataFrame(pred_rows)
pred_df.to_excel(PRED_LOS_XLSX, index=False)
print(f"OK -> {PRED_LOS_XLSX}")

# ================== 9) YeniVakalar (sentetik) ==================
if MAKE_YENI_VAKALAR:
    stage("YeniVakalar.xlsx üretiliyor (sentetik örnekler)")
    yg_vals = df["YaşGrup"].dropna().unique().tolist()
    bolum_vals = df["Bölüm"].dropna().unique().tolist()
    icd_counts = Counter([icd for lst in df["ICD_List_Norm"] for icd in lst])
    top_icds = [icd for icd, _ in icd_counts.most_common(100)]
    def sample_icd_set():
        k = np.random.randint(1, min(5, max(2, len(top_icds))))
        return as_key(set(np.random.choice(top_icds, size=k, replace=False)))

    rows = []
    for _ in tqdm(range(N_SAMPLES_YENI), desc="YeniVakalar"):
        yg = np.random.choice(yg_vals) if yg_vals else "35-50"
        bol = np.random.choice(bolum_vals) if bolum_vals else "Dahiliye"
        key = sample_icd_set()
        pred_rule, meta = predict_one(yg, bol, key)
        icd_list_norm = key.split("||") if key else []
        _, _, p_ens = xgb_predict_ens(yg, bol, key, icd_list_norm)

        if XGB_RULE_BLEND is None or (not np.isfinite(p_ens)):
            pred_out = pred_rule
        else:
            w = float(XGB_RULE_BLEND)
            pred_out = (1.0 - w) * float(pred_rule) + w * float(p_ens)

        rows.append({
            "YaşGrup": yg,
            "Bölüm": bol,
            "ICD_Set_Key": key,
            "Pred_Final": round_half_up(pred_out),
            **meta
        })
    yeni_df = pd.DataFrame(rows)
    yeni_df.to_excel(YENI_VAKALAR_XLSX, index=False)
    print(f"OK -> {YENI_VAKALAR_XLSX}")

# ================== 10) VALID_PREDICTIONS (valid split üzerinde) ==================
# ================== 10) VALID_PREDICTIONS (valid split üzerinde) ==================
stage("VALID_PREDICTIONS.xlsx üretiliyor (valid set)")
valid_rows = []

# --- DÜZELTİLEN KISIM: True_LOS sütununu sağlam biçimde tespit et ---
import unicodedata
def _norm_col(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    return s.strip("_")

_norm_map = { _norm_col(c): c for c in valid_df.columns }
_true_cands = ["yatis_gun_sayisi", "yatis_gun_sayisi_"]  # olası normalize sonuçları
TRUE_LOS_COL = None
for c in _true_cands:
    if c in _norm_map:
        TRUE_LOS_COL = _norm_map[c]
        break
if TRUE_LOS_COL is None:
    if "Yatış Gün Sayısı" in valid_df.columns:
        TRUE_LOS_COL = "Yatış Gün Sayısı"
    elif "Yatış_Gün_Sayısı" in valid_df.columns:
        TRUE_LOS_COL = "Yatış_Gün_Sayısı"
# --- /DÜZELTİLEN KISIM ---

for r in tqdm(valid_df.itertuples(), total=len(valid_df), desc="VALID_PREDICTIONS"):
    yg = r.YaşGrup
    bol = r.Bölüm
    key = r.ICD_Set_Key

    # True_LOS güvenli okuma
    if TRUE_LOS_COL is not None:
        _val = valid_df.loc[r.Index, TRUE_LOS_COL]
        try:
            true_los = float(_val) if pd.notna(_val) else np.nan
        except Exception:
            true_los = np.nan
    else:
        true_los = np.nan

    # Kural tabanlı tahmin
    pred_rule, meta = predict_one(yg, bol, key)

    # XGB ensemble (plain + log-target → ens)
    icd_list_norm = key.split("||") if isinstance(key, str) and key else []
    p_plain, p_log, p_ens = xgb_predict_ens(yg, bol, key, icd_list_norm)

    # Blend oranı (tanımlı değilse 0.5 al)
    _v = globals().get("XGB_RULE_BLEND")
    w = 0.5 if _v is None else float(_v)

    # Eğer XGB yoksa ya da NaN ise sadece kuralı kullan
    if (p_ens is None) or (not np.isfinite(p_ens)):
        pred_out = float(pred_rule)
    else:
        pred_out = (1.0 - w) * float(pred_rule) + w * float(p_ens)

    valid_rows.append({
        "YaşGrup": yg,
        "Bölüm": bol,
        "ICD_Set_Key": key,
        "True_LOS": true_los,
        "Pred_Final": float(pred_out),                     # harmanlı çıktı
        "Pred_Final_Rounded": round_half_up(pred_out),
        "PRED_RULE": float(pred_rule),                     # saf kural
        "PRED_XGB_PLAIN": float(p_plain) if pd.notna(p_plain) else np.nan,
        "PRED_XGB_LOG": float(p_log) if pd.notna(p_log) else np.nan,
        "PRED_XGB_ENS": float(p_ens) if pd.notna(p_ens) else np.nan,  # saf XGB ens
        **meta
    })

# DF'yi OLUŞTUR ve yaz
valid_pred_df = pd.DataFrame(valid_rows)
valid_pred_df.to_excel(VALID_PRED_XLSX, index=False)
print(f"OK -> {VALID_PRED_XLSX}")

# =============== Konsol özetleri ===============
stage("Özet")
print("Train kombinasyon sayısı:", len(train_combos))
print("Valid kombinasyon sayısı:", len(valid_combos))
print("β (tekil) öğrenilen ICD adedi:", len(beta_icd))
print("γ (ikili) öğrenilen pair adedi:", len(gamma_pairs))

# ---- VALID MAE / RMSE (3 farklı senaryo) ----
# Sayısal tip dönüşümü
for col in ["True_LOS", "Pred_Final", "PRED_RULE", "PRED_XGB_ENS"]:
    if col in valid_pred_df.columns:
        valid_pred_df[col] = pd.to_numeric(valid_pred_df[col], errors="coerce")

def _metrics(y_true, y_pred, tag):
    m = y_true.notna() & y_pred.notna()
    if m.any():
        yt = y_true[m].astype(float)
        yp = y_pred[m].astype(float)
        mae = mean_absolute_error(yt, yp)
        rmse = math.sqrt(mean_squared_error(yt, yp))
        print(f"{tag}  MAE: {mae:.4f}  RMSE: {rmse:.4f}")
    else:
        print(f"{tag}: Geçerli satır yok.")

# 1) Harmanlı (Pred_Final)
_metrics(valid_pred_df["True_LOS"], valid_pred_df["Pred_Final"], "HARMAN (Rule ∘ XGB_ENS)")

# 2) Sadece kural (PRED_RULE)
if "PRED_RULE" in valid_pred_df.columns:
    _metrics(valid_pred_df["True_LOS"], valid_pred_df["PRED_RULE"], "KURAL (Rule)")

# 3) Sadece XGB ensemble (PRED_XGB_ENS)
if "PRED_XGB_ENS" in valid_pred_df.columns:
    _metrics(valid_pred_df["True_LOS"], valid_pred_df["PRED_XGB_ENS"], "XGB (Plain+Log Ens)")
# ---- /VALID ----


# ================== 11) API KULLANIMI İÇİN FONKSİYON ==================
# (Sadece eklendi — yukarıdaki koda dokunulmadı)
try:
    DEFAULT_YG = train_df["YaşGrup"].mode().iloc[0]
except Exception:
    DEFAULT_YG = "35-50"

try:
    DEFAULT_BOLUM = train_df["Bölüm"].mode().iloc[0]
except Exception:
    DEFAULT_BOLUM = "Dahiliye"

def _to_icd_key(icd_list):
    if icd_list is None:
        return ""
    cleaned = [clean_icd(x) for x in icd_list if str(x).strip() != ""]
    cleaned = sorted(set(cleaned))
    return "||".join(cleaned)

def tahmin_et(icd_list, bolum=None, yas_grup=None):
    """
    Dış API'den çağrılacak tek giriş noktası.
    3D→2D→1D (0D kapalı) kural + (varsa) XGB ens ile harmanlı P50 döndürür.

    Parametreler:
      - icd_list: ["K11","A00.1", ...]
      - bolum: "Kardiyoloji" (opsiyonel; boşsa train moda)
      - yas_grup: "0-1","2-5",...,"65+" (opsiyonel; boşsa train moda)

    Dönüş:
      {"ok": True, "P50": int, "P25": None, "P75": None, "source": str, "debug": str}
    """
    if not icd_list or not isinstance(icd_list, (list, tuple, set)):
        raise ValueError("icd_list boş olamaz")

    key = _to_icd_key(icd_list)
    if not key:
        raise ValueError("Geçerli ICD bulunamadı")

    yg = yas_grup if (yas_grup and str(yas_grup).strip()) else DEFAULT_YG
    bol = bolum if (bolum and str(bolum).strip()) else DEFAULT_BOLUM

    # Kural tabanlı
    pred_rule, meta = predict_one(yg, bol, key)

    # XGB (varsa)
    icd_list_norm = key.split("||") if key else []
    p_plain, p_log, p_ens = xgb_predict_ens(yg, bol, key, icd_list_norm)

    _v = globals().get("XGB_RULE_BLEND")
    w = 0.5 if _v is None else float(_v)
    if (p_ens is None) or (not np.isfinite(p_ens)):
        pred_out = float(pred_rule)
    else:
        pred_out = (1.0 - w) * float(pred_rule) + w * float(p_ens)

    return {
        "ok": True,
        "P50": round_half_up(pred_out),
        "P25": None,
        "P75": None,
        "source": str(meta.get("ANCHOR_SRC", "")),
        "debug": json.dumps(meta, ensure_ascii=False)
    }