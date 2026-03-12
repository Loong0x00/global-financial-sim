#!/usr/bin/env python3
"""
Multi-Leader Transformer Decision Model  v5 — Two-Stage All-Leader
====================================================================
Stage 1: Economic state change prediction (every month x every active leader)
  - Input: PCA economic state 30D + leader embedding 16D + leader type 3D
           + leader country 7D + other leaders aggregate 16D + time features 6D = 78D
  - Target: next month PCA delta (30D regression, MSE loss)

Stage 2: Interest rate decision head (frozen Stage 1)
  - Input: Stage 1 predicted delta 30D + CB embedding 16D + decision inertia 5D = 51D
  - Target: rate direction cut/hold/hike (3-class classification)
  - Only trained on months with rate decisions

Evaluation:
  - Stage 1: MSE and R2 on economic change prediction
  - Stage 2: Powell LOO direction accuracy + turning point accuracy
  - Comparison with v3 (dir 75.8%, TP 48.5%) and v4 (dir 56.2%, TP 57.9%)
"""

import json
import math
import time
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("=" * 70)
print("MULTI-LEADER TRANSFORMER v5 — TWO-STAGE ALL-LEADER")
print("=" * 70)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
    print(f"[OK] PyTorch {torch.__version__}")
except ImportError:
    TORCH_OK = False
    print("[WARN] PyTorch not available")

if TORCH_OK:
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available — falling back to CPU")
        device = torch.device("cpu")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {gpu_name}  ({gpu_mem:.1f} GB VRAM)")
        print(f"[GPU] CUDA {torch.version.cuda}  device={device}")

BASE        = Path("/home/user/global-financial-sim")
DATA        = BASE / "data"
ECON_DIR    = DATA / "economic"
PROFILES    = DATA / "leaders" / "profiles"
TIMELINES   = DATA / "leaders" / "timelines"
MODELS_DIR  = BASE / "models" / "decision_functions"
OUTPUT_DIR  = BASE / "output" / "decision_function"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PCA_COMPONENTS    = 30
LEADER_EMB_DIM      = 16
N_LEADER_TYPE       = 3
N_LEADER_COUNTRY    = 7
N_TIME_FEATURES     = 6
N_CONTEXT_FEATURES  = 5

STAGE1_INPUT_DIM = N_PCA_COMPONENTS + LEADER_EMB_DIM + N_LEADER_TYPE + N_LEADER_COUNTRY + LEADER_EMB_DIM + N_TIME_FEATURES
STAGE2_INPUT_DIM = N_PCA_COMPONENTS + LEADER_EMB_DIM + N_CONTEXT_FEATURES

COMMON_CB_PARAMS = [
    "inflation_sensitivity_headline_cpi", "unemployment_sensitivity",
    "gdp_growth_sensitivity", "financial_stability_weight", "data_dependence",
    "forward_guidance_credibility", "independence_from_political_pressure",
    "risk_management_approach", "communication_transparency",
    "global_spillover_sensitivity", "quantitative_easing_willingness",
    "emergency_rate_hike_decisiveness", "dovish_bias", "hawkish_bias",
    "crisis_response_speed", "verbal_intervention_potency",
    "labor_market_dovish_bias", "deflation_spiral_fear",
    "exchange_rate_reform_commitment", "covid_pepp_response_speed",
    "flexibility_under_constraint", "consensus_building",
    "asset_bubble_tolerance", "yield_curve_control_willingness",
    "currency_stability_priority", "information_fidelity",
    "institutional_credibility_weight", "policy_reversal_willingness",
    "external_shock_responsiveness", "fiscal_monetary_coordination",
    "emerging_market_sensitivity", "macro_prudential_integration",
    "academic_rigor_in_decisions", "market_communication_skill",
    "patience_duration", "geopolitical_sensitivity",
]

COUNTRY_TO_IDX = {"US": 0, "EU": 1, "RU": 2, "CN": 3, "JP": 4, "UK": 5}
US_ELECTION_YEARS = [1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000,
                     2004, 2008, 2012, 2016, 2020, 2024, 2028]

KNOWN_PROFILE_MAP = {
    "Barack Obama": "obama", "Donald Trump": "trump",
    "George W. Bush": "george_w_bush",
    "Jerome Powell": "powell", "Janet Yellen": "yellen",
    "Ben Bernanke": "bernanke", "Alan Greenspan": "greenspan",
    "Xi Jinping": "xi_jinping", "Hu Jintao": "hu_wen",
    "Zhou Xiaochuan": "zhou_xiaochuan",
    "Vladimir Putin": "putin", "Boris Yeltsin": "yeltsin",
    "Elvira Nabiullina": "nabiullina",
    "Mario Draghi": "draghi", "Christine Lagarde": "lagarde",
    "Angela Merkel": "merkel", "Margaret Thatcher": "thatcher",
    "Nicolas Sarkozy": "sarkozy", "Tony Blair": "blair",
    "Gordon Brown": "gordon_brown",
    "Shinzo Abe": "abe_shinzo", "Junichiro Koizumi": "koizumi_junichiro",
    "Ryutaro Hashimoto": "hashimoto_ryutaro",
    "Yasuhiro Nakasone": "nakasone_yasuhiro",
    "Kakuei Tanaka": "tanaka_kakuei", "Fumio Kishida": "kishida_fumio",
    "Sanae Takaichi": "takaichi_sanae",
    "Ali Khamenei": "ali_khamenei", "Mojtaba Khamenei": "mojtaba_khamenei",
    "Vladimir Lenin": "lenin", "Joseph Stalin": "stalin",
    "Nikita Khrushchev": "khrushchev", "Leonid Brezhnev": "brezhnev",
    "Benjamin Netanyahu": "netanyahu",
}

# ─── Utilities ────────────────────────────────────────────────────────────────

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}

def _date_to_month(date_str):
    if not date_str: return None
    if re.match(r'^\d{4}-\d{2}$', date_str): return date_str
    m = re.match(r'^(\d{4})-(\d{2})-\d{2}$', date_str)
    if m: return f"{m.group(1)}-{m.group(2)}"
    m = re.match(r'^(\d{4})-Q(\d)$', date_str)
    if m:
        yr, q = int(m.group(1)), int(m.group(2))
        return f"{yr:04d}-{(q-1)*3+1:02d}"
    return None

def _all_months(start="1971-01", end="2026-02"):
    months = []
    y, m = int(start[:4]), int(start[5:])
    ey, em = int(end[:4]), int(end[5:])
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12: m = 1; y += 1
    return months

# ─── Economic Data Loader ────────────────────────────────────────────────────

class AllEconomicDataStore:
    def __init__(self):
        self._series = {}
        self._rate_data = {}
        self._bis_rates = {}
        self._all_months = _all_months("1971-01", "2026-02")
        self._month_to_idx = {m: i for i, m in enumerate(self._all_months)}
        self._load_all()

    def _add_series(self, name, data_dict):
        monthly = {}
        for date_str, val in data_dict.items():
            if val is None: continue
            month = _date_to_month(date_str)
            if month and month in self._month_to_idx:
                try: monthly[month] = float(val)
                except (ValueError, TypeError): continue
        if len(monthly) >= 6:
            self._series[name] = monthly

    def _load_macro_financial(self):
        for country in ["us", "eu", "ru", "cn", "jp", "uk"]:
            cc = country.upper()
            for ftype in ["macro", "financial"]:
                path = ECON_DIR / country / f"{ftype}.json"
                if not path.exists(): continue
                d = load_json(path)
                for ind_name, ind_data in d.get("indicators", {}).items():
                    series_data = {}
                    for item in ind_data.get("series", []):
                        if isinstance(item, dict) and "date" in item and item.get("value") is not None:
                            series_data[item["date"]] = item["value"]
                    self._add_series(f"{cc}_{ftype}_{ind_name}", series_data)
                    if ftype == "macro" and ind_name == "interest_rate":
                        monthly = {}
                        for ds, val in series_data.items():
                            mo = _date_to_month(ds)
                            if mo:
                                try: monthly[mo] = float(val)
                                except: pass
                        self._rate_data[cc] = monthly
                    if ftype == "macro" and ind_name == "fed_funds_rate" and cc == "US":
                        monthly = {}
                        for ds, val in series_data.items():
                            mo = _date_to_month(ds)
                            if mo:
                                try: monthly[mo] = float(val)
                                except: pass
                        self._rate_data["US"] = monthly

    def _load_bis(self):
        bis_dir = ECON_DIR / "bis"
        if not bis_dir.exists(): return
        cm = {"US":"US","CN":"CN","JP":"JP","GB":"UK","DE":"EU","RU":"RU","FR":"EU"}
        for fpath in bis_dir.glob("*.json"):
            if fpath.name.startswith("_"): continue
            d = load_json(fpath)
            dataset = d.get("dataset", fpath.stem)
            if dataset == "policy_rates":
                bm = {"US":"US","RU":"RU","CN":"CN","JP":"JP","GB":"UK"}
                for bc, oc in bm.items():
                    if bc in d.get("countries", {}):
                        sd = d["countries"][bc]["series"]
                        for sk, sv in sd.items():
                            monthly = {}
                            for k, v in sv["data"].items():
                                mo = _date_to_month(k)
                                if mo:
                                    try: monthly[mo] = float(v)
                                    except: pass
                            self._bis_rates[oc] = monthly
                            break
            if dataset in ("debt_securities", "locational_banking", "consolidated_banking"):
                continue
            for cc, cdata in d.get("countries", {}).items():
                oc = cm.get(cc, cc)
                for i, (sn, sv) in enumerate(cdata.get("series", {}).items()):
                    self._add_series(f"BIS_{dataset}_{oc}_{i}",
                                     {k: v for k, v in sv.get("data", {}).items()})

    def _load_commodities(self):
        cd = ECON_DIR / "commodities"
        if not cd.exists(): return
        for fpath in cd.glob("*.json"):
            if fpath.name == "manifest.json": continue
            d = load_json(fpath)
            if fpath.name in ("gold_daily.json", "silver_daily.json"):
                items = d.get("data", d) if isinstance(d, dict) else d
                if not isinstance(items, list): continue
                ma = defaultdict(list)
                for item in items:
                    if isinstance(item, dict):
                        dt, val = item.get("date",""), item.get("close") or item.get("value")
                        mo = _date_to_month(dt)
                        if mo and val is not None:
                            try: ma[mo].append(float(val))
                            except: pass
                self._add_series(f"COM_{fpath.stem.replace('_daily','')}", {m: np.mean(v) for m,v in ma.items()})
                continue
            if isinstance(d, dict):
                for sn, sv in d.items():
                    if not isinstance(sv, dict) or "data" not in sv: continue
                    items = sv["data"]
                    if not isinstance(items, list): continue
                    ma = defaultdict(list)
                    for item in items:
                        if isinstance(item, dict):
                            dt, val = item.get("date",""), item.get("value") or item.get("close")
                            mo = _date_to_month(dt)
                            if mo and val is not None:
                                try: ma[mo].append(float(val))
                                except: pass
                    self._add_series(f"COM_{sn}", {m: np.mean(v) for m,v in ma.items()})

    def _load_indices(self):
        path = ECON_DIR / "indices" / "global_indices.json"
        if not path.exists(): return
        d = load_json(path)
        for idx_name, idx_data in d.items():
            if not isinstance(idx_data, dict) or "data" not in idx_data: continue
            ma = defaultdict(list)
            for item in idx_data["data"]:
                if isinstance(item, dict):
                    dt, val = item.get("date",""), item.get("close")
                    mo = _date_to_month(dt)
                    if mo and val is not None:
                        try: ma[mo].append(float(val))
                        except: pass
            self._add_series(f"IDX_{idx_name}", {m: np.mean(v) for m,v in ma.items()})

    def _load_crypto(self):
        path = ECON_DIR / "crypto" / "crypto.json"
        if not path.exists(): return
        d = load_json(path)
        for cn, cd in d.items():
            if not isinstance(cd, dict) or "data" not in cd: continue
            ma = defaultdict(list)
            for item in cd["data"]:
                if isinstance(item, dict):
                    dt, val = item.get("date",""), item.get("close")
                    mo = _date_to_month(dt)
                    if mo and val is not None:
                        try: ma[mo].append(float(val))
                        except: pass
            self._add_series(f"CRYPTO_{cn}", {m: np.mean(v) for m,v in ma.items()})

    def _load_supplements(self):
        sd = ECON_DIR / "supplementary"
        if not sd.exists(): return
        for fpath in sd.glob("*.json"):
            if fpath.name == "manifest.json": continue
            d = load_json(fpath)
            for sn, sv in d.items():
                if not isinstance(sv, dict) or "data" not in sv: continue
                items = sv["data"]
                if not isinstance(items, list): continue
                dd = {}
                for item in items:
                    if isinstance(item, dict):
                        dt, val = item.get("date",""), item.get("value")
                        if dt and val is not None: dd[dt] = val
                self._add_series(f"SUP_{sn}", dd)

    def _load_cftc(self):
        path = ECON_DIR / "cftc" / "cot_data.json"
        if not path.exists(): return
        d = load_json(path)
        for cc, cd in d.get("contracts", {}).items():
            name = cd.get("contract_name", cc)
            monthly = {}
            for item in cd.get("data", []):
                if isinstance(item, dict):
                    dt, val = item.get("date",""), item.get("net_speculative")
                    mo = _date_to_month(dt)
                    if mo and val is not None:
                        try: monthly[mo] = float(val)
                        except: pass
            self._add_series(f"CFTC_{name.replace(' ','_')}", monthly)

    def _load_trade(self):
        path = ECON_DIR / "trade" / "bilateral_trade.json"
        if not path.exists(): return
        d = load_json(path)
        for pn, pd in d.items():
            if not isinstance(pd, dict) or "data" not in pd: continue
            items = pd["data"]
            if not isinstance(items, list): continue
            for field in ["total","trade_pct_gdp_us","trade_pct_gdp_cn","us_exports_to_cn","us_imports_from_cn"]:
                yearly = {}
                for item in items:
                    if isinstance(item, dict) and "year" in item and field in item:
                        if item[field] is not None: yearly[item["year"]] = item[field]
                if not yearly: continue
                monthly = {}
                for yr, val in yearly.items():
                    for mo in range(1, 13): monthly[f"{yr:04d}-{mo:02d}"] = val
                self._add_series(f"TRADE_{pn}_{field}", monthly)

    def _load_all(self):
        print("\n[0] Loading ALL economic data...")
        self._load_macro_financial()
        print(f"  Macro/Financial: {len(self._series)} series")
        n = len(self._series)
        self._load_bis(); print(f"  BIS: +{len(self._series)-n}"); n = len(self._series)
        self._load_commodities(); print(f"  Commodities: +{len(self._series)-n}"); n = len(self._series)
        self._load_indices(); print(f"  Indices: +{len(self._series)-n}"); n = len(self._series)
        self._load_crypto(); print(f"  Crypto: +{len(self._series)-n}"); n = len(self._series)
        self._load_supplements(); print(f"  Supplements: +{len(self._series)-n}"); n = len(self._series)
        self._load_cftc(); print(f"  CFTC: +{len(self._series)-n}"); n = len(self._series)
        self._load_trade(); print(f"  Trade: +{len(self._series)-n}")
        print(f"  TOTAL: {len(self._series)} series")
        print(f"  Rates: {list(self._rate_data.keys())}  BIS: {list(self._bis_rates.keys())}")

    def get_rate(self, country, month):
        if country in self._rate_data:
            r = self._rate_data[country].get(month)
            if r is not None: return r
        return self._bis_rates.get(country, {}).get(month)

    def build_monthly_matrix(self):
        sn = sorted(self._series.keys())
        nm = len(self._all_months)
        ns = len(sn)
        mat = np.full((nm, ns), np.nan, dtype=np.float64)
        for j, name in enumerate(sn):
            for month, val in self._series[name].items():
                if month in self._month_to_idx:
                    mat[self._month_to_idx[month], j] = val
        return mat, sn, self._all_months

# ─── PCA Pipeline ─────────────────────────────────────────────────────────────

class PCAEconomicPipeline:
    def __init__(self, n_components=N_PCA_COMPONENTS):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self._fitted = False
        self._matrix = None
        self._months = None
        self._month_to_idx = None
        self._imputed = None
        self._series_names = None

    def set_matrix(self, matrix, series_names, months):
        self._matrix = matrix
        self._series_names = series_names
        self._months = months
        self._month_to_idx = {m: i for i, m in enumerate(months)}

    def _impute_matrix(self, mat):
        r = mat.copy()
        for j in range(r.shape[1]):
            lv = np.nan
            for i in range(r.shape[0]):
                if np.isnan(r[i,j]):
                    if not np.isnan(lv): r[i,j] = lv
                else: lv = r[i,j]
        for j in range(r.shape[1]):
            lv = np.nan
            for i in range(r.shape[0]-1, -1, -1):
                if np.isnan(r[i,j]):
                    if not np.isnan(lv): r[i,j] = lv
                else: lv = r[i,j]
        return np.nan_to_num(r, nan=0.0)

    def fit(self, train_idx):
        imp = self._impute_matrix(self._matrix)
        td = imp[train_idx]
        sc = self.scaler.fit_transform(td)
        mc = min(self.n_components, sc.shape[0], sc.shape[1])
        if mc < self.n_components:
            print(f"  [PCA] Reducing {self.n_components} -> {mc}")
            self.pca = PCA(n_components=mc)
        self.pca.fit(sc)
        self._imputed = imp
        self._fitted = True
        exp = np.sum(self.pca.explained_variance_ratio_) * 100
        print(f"  [PCA] {len(train_idx)} months x {imp.shape[1]} series -> {self.pca.n_components_} PC ({exp:.1f}%)")
        return exp

    def transform_month(self, month):
        if not self._fitted:
            return np.zeros(self.n_components, dtype=np.float32)
        idx = self._month_to_idx.get(month)
        if idx is None:
            return np.zeros(self.n_components, dtype=np.float32)
        row = self._imputed[idx:idx+1]
        sc = self.scaler.transform(row)
        pf = self.pca.transform(sc)[0]
        if len(pf) < self.n_components:
            pf = np.pad(pf, (0, self.n_components - len(pf)), constant_values=0.0)
        return pf.astype(np.float32)

# ─── Leader Registry ──────────────────────────────────────────────────────────

def _normalize_name_to_profile(name):
    pm = {pf.stem.lower(): pf.stem for pf in PROFILES.glob("*.json")}
    slug = name.lower().strip().replace(" ", "_").replace(".", "")
    if slug in pm: return pm[slug]
    parts = name.lower().split()
    for k in pm:
        kp = k.split("_")
        if parts[-1] == kp[-1]: return pm[k]
    return None

def build_all_leader_registry():
    all_leaders = []
    seen = set()
    for cf in TIMELINES.glob("*.json"):
        cc = cf.stem.upper()
        d = load_json(cf)
        entries = []
        for key in ["heads_of_state", "central_bank"]:
            if key in d and isinstance(d[key], list):
                rt = "central_bank" if "bank" in key else "head_of_state"
                for e in d[key]:
                    if isinstance(e, dict) and "term_start" in e:
                        entries.append((e, rt, cc))
        if "leaders" in d and isinstance(d["leaders"], list):
            for e in d["leaders"]:
                if isinstance(e, dict) and "term_start" in e:
                    role = e.get("role", "").lower()
                    rt = "central_bank" if any(x in role for x in ["bank","ecb","boe","governor"]) else "head_of_state"
                    entries.append((e, rt, cc))
        if "roles" in d and isinstance(d["roles"], dict):
            for rn, rd in d["roles"].items():
                holders = rd.get("leaders", rd.get("holders", [])) if isinstance(rd, dict) else (rd if isinstance(rd, list) else [])
                rl = rn.lower()
                rt = "central_bank" if any(x in rl for x in ["bank","governor"]) else "head_of_state"
                for e in holders:
                    if isinstance(e, dict) and "term_start" in e:
                        entries.append((e, rt, cc))
        for e, rt, c in entries:
            name = e.get("name", "Unknown")
            start = e.get("term_start", "")
            end = e.get("term_end")
            if end in ("incumbent", "present", None, ""): end = None
            sm = _date_to_month(start)
            em = _date_to_month(end) if end else None
            if not sm: continue
            if em and em <= "1971-01": continue
            if sm < "1971-01": sm = "1971-01"
            slug = name.lower().replace(" ","_").replace(".","")
            lid = f"{slug}_{c.lower()}"
            if lid in seen: lid = f"{lid}_{sm}"
            seen.add(lid)
            pn = KNOWN_PROFILE_MAP.get(name) or _normalize_name_to_profile(name)
            if pn and not (PROFILES / f"{pn}.json").exists(): pn = None
            lc = c if c != "IR" else "other"
            all_leaders.append({"id":lid, "name":name, "country":lc, "role_type":rt,
                               "start":sm, "end":em, "profile_name":pn})
    all_leaders.sort(key=lambda x: x["start"])
    return all_leaders

# ─── Leader Embedding ─────────────────────────────────────────────────────────

def extract_leader_embedding(profile_name, dim=LEADER_EMB_DIM):
    if not profile_name: return np.zeros(dim, dtype=np.float32)
    path = PROFILES / f"{profile_name}.json"
    if not path.exists(): return np.zeros(dim, dtype=np.float32)
    try: profile = load_json(path)
    except: return np.zeros(dim, dtype=np.float32)
    lu = {}
    for v in profile.get("behavior_matrix", {}).get("vectors", []):
        if "label" in v and "value" in v and v["value"] is not None:
            lu[v["label"]] = float(v["value"])
    for g in profile.get("behavioral_parameters", []):
        for p in g.get("parameters", []):
            if "name" in p and "value" in p: lu[p["name"]] = float(p["value"])
    vals = []
    for label in COMMON_CB_PARAMS[:dim]:
        if label in lu: vals.append(lu[label])
        else:
            ms = [v for k,v in lu.items() if k.startswith(label[:10])]
            vals.append(float(np.mean(ms)) if ms else 0.0)
    arr = np.array(vals[:dim], dtype=np.float32)
    if np.count_nonzero(arr) < dim // 4 and len(lu) > 0:
        av = list(lu.values())[:dim]
        if len(av) >= dim // 2:
            arr2 = np.array(av[:dim], dtype=np.float32)
            if len(arr2) < dim: arr2 = np.pad(arr2, (0, dim-len(arr2)), constant_values=0.0)
            arr = arr2[:dim]
    if len(arr) < dim: arr = np.pad(arr, (0, dim-len(arr)), constant_values=0.0)
    return arr[:dim]

# ─── Features ─────────────────────────────────────────────────────────────────

def extract_time_features(month):
    f = np.zeros(N_TIME_FEATURES, dtype=np.float32)
    yr, mo = int(month[:4]), int(month[5:7])
    ma = yr * 12 + mo
    f[0] = (yr - 1970) / 60.0
    for ey in US_ELECTION_YEARS:
        ea = ey * 12 + 11
        if ea >= ma: f[1] = float(np.clip(1.0-(ea-ma)/48.0, 0, 1)); break
    dp = (yr + mo/12.0) / 10.0
    f[2] = math.sin(2*math.pi*dp); f[3] = math.cos(2*math.pi*dp)
    f[4] = math.sin(2*math.pi*mo/12); f[5] = math.cos(2*math.pi*mo/12)
    return f

def classify_rate_change(delta_bp):
    if delta_bp < -5: return 0
    if delta_bp > 5: return 2
    return 1

def extract_rate_decisions(econ, country, start, end=None):
    months = _all_months(start, end or "2026-02")
    decs = []; prev = None
    for month in months:
        rate = econ.get_rate(country, month)
        if rate is None: prev = None; continue
        if prev is not None:
            dbp = (rate - prev) * 100.0
            decs.append({"month":month, "rate":rate, "prev_rate":prev,
                        "delta_bp":round(dbp,1), "direction":classify_rate_change(dbp)})
        prev = rate
    return decs

def compute_decision_inertia(idx, decs):
    hs = cs = hos = 0
    for i in range(idx-1, max(-1, idx-12), -1):
        d = decs[i]["direction"]
        if d == 2: hs += 1; cs = 0; hos = 0
        elif d == 0: cs += 1; hs = 0; hos = 0
        else: hos += 1; break
    m = int(decs[idx]["month"][5:])
    return np.array([hs/12.0, cs/12.0, hos/12.0,
                     math.sin(2*math.pi*m/12), math.cos(2*math.pi*m/12)], dtype=np.float32)

def get_active_leaders(month, all_leaders):
    return [l for l in all_leaders if month >= l["start"] and (l["end"] is None or month < l["end"])]

def leader_type_oh(rt):
    v = np.zeros(N_LEADER_TYPE, dtype=np.float32)
    if rt == "central_bank": v[0] = 1.0
    elif rt == "head_of_state": v[1] = 1.0
    else: v[2] = 1.0
    return v

def leader_country_oh(c):
    v = np.zeros(N_LEADER_COUNTRY, dtype=np.float32)
    v[COUNTRY_TO_IDX.get(c, 6)] = 1.0
    return v

def build_s1_input(pca, emb, rt, cc, other_agg, tf):
    return np.concatenate([pca, emb, leader_type_oh(rt), leader_country_oh(cc), other_agg, tf]).astype(np.float32)

# ─── Stage 1 Samples ─────────────────────────────────────────────────────────

def build_stage1_samples(all_leaders, le, pca, months):
    print("\n[S1] Building Stage 1 samples...")
    m2n = {months[i]: months[i+1] for i in range(len(months)-1)}
    samples = []; skip = 0
    for month in months[:-1]:
        nm = m2n[month]
        pc = pca.transform_month(month)
        pn = pca.transform_month(nm)
        delta = pn - pc
        if np.allclose(pc, 0.0) and np.allclose(pn, 0.0): skip += 1; continue
        active = get_active_leaders(month, all_leaders)
        if not active: continue
        aembs = [le.get(l["id"], np.zeros(LEADER_EMB_DIM, dtype=np.float32)) for l in active]
        asum = np.sum(aembs, axis=0).astype(np.float32)
        na = len(active)
        tf = extract_time_features(month)
        for j, l in enumerate(active):
            emb = aembs[j]
            oa = ((asum - emb) / (na-1)).astype(np.float32) if na > 1 else np.zeros(LEADER_EMB_DIM, dtype=np.float32)
            x = build_s1_input(pc, emb, l["role_type"], l["country"], oa, tf)
            samples.append({"month":month, "lid":l["id"], "x":x, "y":delta.astype(np.float32)})
    print(f"  Samples: {len(samples)}, Skipped: {skip}")
    print(f"  Months: {len(set(s['month'] for s in samples))}, Leaders: {len(set(s['lid'] for s in samples))}")
    return samples

# ─── Models ───────────────────────────────────────────────────────────────────

if TORCH_OK:
    class Stage1Model(nn.Module):
        def __init__(self, nf=STAGE1_INPUT_DIM, dm=128, nh=8, nl=3, do=0.1, no=N_PCA_COMPONENTS):
            super().__init__()
            self.nf = nf
            self.ip = nn.Linear(1, dm)
            self.pe = nn.Embedding(nf, dm)
            el = nn.TransformerEncoderLayer(d_model=dm, nhead=nh, dim_feedforward=dm*4, dropout=do, batch_first=True)
            self.enc = nn.TransformerEncoder(el, num_layers=nl)
            self.head = nn.Sequential(nn.Linear(dm, dm), nn.GELU(), nn.Dropout(do), nn.Linear(dm, no))
            self._iw()
        def _iw(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None: nn.init.zeros_(m.bias)
        def forward(self, x):
            B, N = x.shape
            t = self.ip(x.unsqueeze(-1))
            p = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
            t = t + self.pe(p)
            return self.head(self.enc(t).mean(dim=1))
        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    class Stage2Model(nn.Module):
        def __init__(self, nd=N_PCA_COMPONENTS, ne=LEADER_EMB_DIM, ni=N_CONTEXT_FEATURES, h=64):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(nd+ne+ni, h), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(h, h), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(h, 3))
            self._iw()
        def _iw(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None: nn.init.zeros_(m.bias)
        def forward(self, d, e, i):
            return self.mlp(torch.cat([d, e, i], dim=-1))
        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ─── Training ─────────────────────────────────────────────────────────────────

def train_stage1(samples, ne=60, lr=3e-3, bs=512):
    print(f"\n[S1-TRAIN] {len(samples)} samples...")
    X = np.array([s["x"] for s in samples], dtype=np.float32)
    Y = np.array([s["y"] for s in samples], dtype=np.float32)
    nf = X.shape[1]
    print(f"  In: {nf}D, Out: {Y.shape[1]}D")
    model = Stage1Model(nf=nf); print(f"  Params: {model.count_parameters():,}")
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ne)
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    Yt = torch.tensor(Y, dtype=torch.float32).to(device)
    ns = len(X); t0 = time.time()
    for ep in range(ne):
        model.train()
        perm = torch.randperm(ns, device=device)
        el = 0.0; nb = 0
        for si in range(0, ns, bs):
            idx = perm[si:si+bs]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xt[idx]), Yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            el += loss.item(); nb += 1
        sched.step()
        if ep % 50 == 0 or ep == ne-1:
            model.eval()
            with torch.no_grad():
                pa = model(Xt)
                mse = nn.functional.mse_loss(pa, Yt).item()
                sr = ((pa-Yt)**2).sum().item()
                st = ((Yt-Yt.mean(0))**2).sum().item()
                r2 = 1.0 - sr/max(st, 1e-8)
            print(f"  ep {ep:3d}/{ne}  loss={el/max(nb,1):.6f}  MSE={mse:.6f}  R2={r2:.4f}")
    elapsed = time.time() - t0
    model.eval()
    with torch.no_grad():
        pa = model(Xt); mse = nn.functional.mse_loss(pa, Yt).item()
        sr = ((pa-Yt)**2).sum().item(); st = ((Yt-Yt.mean(0))**2).sum().item()
        r2 = 1.0 - sr/max(st, 1e-8)
    print(f"  Done {elapsed:.1f}s  MSE={mse:.6f}  R2={r2:.4f}")
    return model, {"mse": mse, "r2": r2, "elapsed": elapsed}

def build_stage2_samples(econ, all_leaders, le, pca, s1m):
    print("\n[S2] Building Stage 2 samples...")
    cbs = [l for l in all_leaders if l["role_type"] == "central_bank"]
    print(f"  CBs: {len(cbs)}")
    s1m.eval(); samples = []; stats = {}
    for cb in cbs:
        decs = extract_rate_decisions(econ, cb["country"], cb["start"], cb["end"])
        if not decs: continue
        ce = le.get(cb["id"], np.zeros(LEADER_EMB_DIM, dtype=np.float32))
        csamps = []
        for i, dec in enumerate(decs):
            month = dec["month"]
            pc = pca.transform_month(month)
            active = get_active_leaders(month, all_leaders)
            aembs = [le.get(l["id"], np.zeros(LEADER_EMB_DIM, dtype=np.float32)) for l in active]
            if aembs:
                asum = np.sum(aembs, axis=0).astype(np.float32); na = len(aembs)
            else:
                asum = np.zeros(LEADER_EMB_DIM, dtype=np.float32); na = 0
            oa = ((asum-ce)/(na-1)).astype(np.float32) if na > 1 else np.zeros(LEADER_EMB_DIM, dtype=np.float32)
            tf = extract_time_features(month)
            s1i = build_s1_input(pc, ce, cb["role_type"], cb["country"], oa, tf)
            with torch.no_grad():
                pd = s1m(torch.tensor(s1i, dtype=torch.float32).unsqueeze(0).to(device))[0].cpu().numpy()
            inert = compute_decision_inertia(i, decs)
            csamps.append({"lid":cb["id"], "cc":cb["country"], "month":month,
                          "pd":pd.astype(np.float32), "ce":ce, "inert":inert,
                          "dir":dec["direction"], "dbp":dec["delta_bp"], "s1i":s1i})
        if csamps:
            dirs = [s["dir"] for s in csamps]
            stats[cb["id"]] = {"name":cb["name"], "cc":cb["country"], "n":len(dirs),
                              "cut":dirs.count(0), "hold":dirs.count(1), "hike":dirs.count(2),
                              "prof": cb["profile_name"] is not None}
            print(f"  {cb['id']:40s} ({cb['country']}): {len(dirs):4d}  c={dirs.count(0)} h={dirs.count(1)} H={dirs.count(2)}")
            samples.extend(csamps)
    print(f"\n  Total S2: {len(samples)}")
    return samples, stats

def train_stage2(samples, ne=200, lr=3e-3, bs=256):
    print(f"\n[S2-TRAIN] {len(samples)} samples...")
    Xd = np.array([s["pd"] for s in samples], dtype=np.float32)
    Xc = np.array([s["ce"] for s in samples], dtype=np.float32)
    Xi = np.array([s["inert"] for s in samples], dtype=np.float32)
    yd = np.array([s["dir"] for s in samples], dtype=np.int64)
    model = Stage2Model(); print(f"  Params: {model.count_parameters():,}")
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ne)
    dc = np.bincount(yd, minlength=3).astype(float)
    dc = np.where(dc==0, 1, dc)
    dw = torch.tensor(1.0/dc, dtype=torch.float32).to(device); dw /= dw.sum()
    crit = nn.CrossEntropyLoss(weight=dw)
    dt = torch.tensor(Xd, dtype=torch.float32).to(device)
    ct = torch.tensor(Xc, dtype=torch.float32).to(device)
    it = torch.tensor(Xi, dtype=torch.float32).to(device)
    yt = torch.tensor(yd, dtype=torch.long).to(device)
    ns = len(samples); t0 = time.time()
    for ep in range(ne):
        model.train(); perm = torch.randperm(ns, device=device); el = 0.0; nb = 0
        for si in range(0, ns, bs):
            idx = perm[si:si+bs]; opt.zero_grad()
            loss = crit(model(dt[idx], ct[idx], it[idx]), yt[idx])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            el += loss.item(); nb += 1
        sched.step()
        if ep % 50 == 0 or ep == ne-1:
            model.eval()
            with torch.no_grad():
                pr = torch.argmax(model(dt, ct, it), dim=-1)
                acc = (pr == yt).float().mean().item()
            print(f"  ep {ep:3d}/{ne}  loss={el/max(nb,1):.4f}  acc={acc:.3f}")
    elapsed = time.time() - t0
    model.eval()
    with torch.no_grad():
        pr = torch.argmax(model(dt, ct, it), dim=-1)
        acc = (pr == yt).float().mean().item()
    print(f"  Done {elapsed:.1f}s  acc={acc:.3f}")
    return model, {"dir_acc": acc, "elapsed": elapsed}

# ─── Powell LOO ───────────────────────────────────────────────────────────────

def powell_loo(s2s, s1m, al, le, econ, pca, mat, sn, months):
    print("\n[LOO] Powell LOO...")
    pids = [s["lid"] for s in s2s if "powell" in s["lid"].lower()]
    pid = max(set(pids), key=pids.count) if pids else None
    if not pid: print("  Not found!"); return None
    ps = [s for s in s2s if s["lid"] == pid]
    os = [s for s in s2s if s["lid"] != pid]
    print(f"  Powell: {len(ps)}, Others: {len(os)}")
    if len(ps) < 5: print("  Too few"); return None

    # LOO PCA
    m2i = {m: i for i, m in enumerate(months)}
    oi = sorted([m2i[m] for m in set(s["month"] for s in os) if m in m2i])
    pca_loo = PCAEconomicPipeline(n_components=N_PCA_COMPONENTS)
    pca_loo.set_matrix(mat, sn, months)
    pca_loo.fit(oi)

    s1m.eval()
    def recomp(slist, pp):
        ds = []
        for s in slist:
            month = s["month"]; ce = s["ce"]
            pc = pp.transform_month(month)
            active = get_active_leaders(month, al)
            aembs = [le.get(l["id"], np.zeros(LEADER_EMB_DIM, dtype=np.float32)) for l in active]
            asum = np.sum(aembs, axis=0).astype(np.float32) if aembs else np.zeros(LEADER_EMB_DIM, dtype=np.float32)
            na = len(aembs)
            oa = ((asum-ce)/(na-1)).astype(np.float32) if na > 1 else np.zeros(LEADER_EMB_DIM, dtype=np.float32)
            le_ = next((l for l in al if l["id"] == s["lid"]), None)
            rt = le_["role_type"] if le_ else "central_bank"
            cc = le_["country"] if le_ else "US"
            tf = extract_time_features(month)
            s1i = build_s1_input(pc, ce, rt, cc, oa, tf)
            with torch.no_grad():
                d = s1m(torch.tensor(s1i, dtype=torch.float32).unsqueeze(0).to(device))[0].cpu().numpy()
            ds.append(d.astype(np.float32))
        return ds

    print("  Recomputing S1 deltas...")
    pd = recomp(ps, pca_loo); od = recomp(os, pca_loo)
    for i, s in enumerate(ps): s["_ld"] = pd[i]
    for i, s in enumerate(os): s["_ld"] = od[i]

    ap = []; at = []
    for ti in range(len(ps)):
        test = ps[ti]
        train = os + [s for i, s in enumerate(ps) if i != ti]
        Xd = np.array([s["_ld"] for s in train], dtype=np.float32)
        Xc = np.array([s["ce"] for s in train], dtype=np.float32)
        Xi = np.array([s["inert"] for s in train], dtype=np.float32)
        y = np.array([s["dir"] for s in train], dtype=np.int64)
        s2 = Stage2Model().to(device)
        o = optim.AdamW(s2.parameters(), lr=3e-3, weight_decay=1e-4)
        sc = optim.lr_scheduler.CosineAnnealingLR(o, T_max=100)
        dc = np.bincount(y, minlength=3).astype(float)
        dc = np.where(dc==0, 1, dc)
        dw = torch.tensor(1.0/dc, dtype=torch.float32).to(device); dw /= dw.sum()
        cr = nn.CrossEntropyLoss(weight=dw)
        dtt = torch.tensor(Xd, dtype=torch.float32).to(device)
        ctt = torch.tensor(Xc, dtype=torch.float32).to(device)
        itt = torch.tensor(Xi, dtype=torch.float32).to(device)
        ytt = torch.tensor(y, dtype=torch.long).to(device)
        n = len(train)
        for ep in range(80):
            s2.train(); perm = torch.randperm(n, device=device)
            for si in range(0, n, 256):
                idx = perm[si:si+256]; o.zero_grad()
                loss = cr(s2(dtt[idx], ctt[idx], itt[idx]), ytt[idx])
                loss.backward(); nn.utils.clip_grad_norm_(s2.parameters(), 1.0); o.step()
            sc.step()
        s2.eval()
        with torch.no_grad():
            td = torch.tensor(test["_ld"], dtype=torch.float32).unsqueeze(0).to(device)
            tc = torch.tensor(test["ce"], dtype=torch.float32).unsqueeze(0).to(device)
            ti2 = torch.tensor(test["inert"], dtype=torch.float32).unsqueeze(0).to(device)
            p = torch.argmax(s2(td, tc, ti2), dim=-1).item()
        ap.append(p); at.append(test["dir"])
        if (ti+1) % 20 == 0:
            print(f"  LOO {ti+1}/{len(ps)}  acc={np.mean(np.array(ap)==np.array(at)):.3f}")

    pa = np.array(ap); ta = np.array(at)
    da = float(np.mean(pa == ta))
    tps = [i for i in range(1, len(ps)) if ps[i-1]["dir"] != ps[i]["dir"]]
    tpa = None
    if tps:
        tpc = sum(1 for i in tps if ap[i] == at[i])
        tpa = tpc / len(tps)
    tps_str = f"{tpa:.3f} ({len(tps)} pts)" if tpa is not None else "N/A"
    print(f"\n  Powell LOO: dir={da:.3f}  TP={tps_str}  n={len(ps)}")
    return {"dir_acc":da, "tp_acc":tpa, "n":len(ps), "n_tp":len(tps), "pred":ap, "true":at}

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    if not TORCH_OK: print("[ERROR] PyTorch required"); return

    econ = AllEconomicDataStore()

    print("\n[1] Monthly matrix...")
    mat, sn, months = econ.build_monthly_matrix()
    print(f"  {mat.shape}, coverage={np.mean(~np.isnan(mat))*100:.1f}%")

    print("\n[2] Leader registry...")
    al = build_all_leader_registry()
    bt = defaultdict(int); bc = defaultdict(int); np_ = 0
    for l in al: bt[l["role_type"]] += 1; bc[l["country"]] += 1; np_ += bool(l["profile_name"])
    print(f"  {len(al)} leaders, {np_} with profiles")
    print(f"  Types: {dict(bt)}, Countries: {dict(bc)}")

    print("\n[3] Embeddings...")
    le = {}
    for l in al:
        emb = extract_leader_embedding(l["profile_name"])
        le[l["id"]] = emb
        if l["profile_name"]:
            print(f"  {l['id']:40s} prof={l['profile_name']:20s} nz={np.count_nonzero(emb)}/{LEADER_EMB_DIM}")

    print("\n[4] PCA...")
    pca = PCAEconomicPipeline(n_components=N_PCA_COMPONENTS)
    pca.set_matrix(mat, sn, months)
    exp = pca.fit(list(range(len(months))))

    s1s = build_stage1_samples(al, le, pca, months)
    s1m, s1met = train_stage1(s1s, ne=60)

    s2s, s2st = build_stage2_samples(econ, al, le, pca, s1m)
    s2m, s2met = train_stage2(s2s, ne=150)

    # Per-leader accuracy
    print("\n[7] Per-leader accuracy...")
    s2m.eval()
    dt = torch.tensor(np.array([s["pd"] for s in s2s]), dtype=torch.float32).to(device)
    ct = torch.tensor(np.array([s["ce"] for s in s2s]), dtype=torch.float32).to(device)
    it = torch.tensor(np.array([s["inert"] for s in s2s]), dtype=torch.float32).to(device)
    yd = np.array([s["dir"] for s in s2s])
    with torch.no_grad():
        pred = torch.argmax(s2m(dt, ct, it), dim=-1).cpu().numpy()
    bl = defaultdict(list)
    for i, s in enumerate(s2s): bl[s["lid"]].append(i)
    pl = {}
    print(f"\n  {'Leader':40s} {'CC':5s} {'N':5s} {'Acc':7s}")
    print("  " + "-" * 60)
    for lid, idx in sorted(bl.items()):
        a = float(np.mean(pred[idx] == yd[idx]))
        pl[lid] = {"acc":a, "n":len(idx)}
        print(f"  {lid:40s} {s2s[idx[0]]['cc']:5s} {len(idx):5d} {a*100:5.1f}%")
    oa = float(np.mean(pred == yd))
    print(f"\n  Overall: {oa:.3f}")

    ploo = powell_loo(s2s, s1m, al, le, econ, pca, mat, sn, months)

    # Save
    mp = MODELS_DIR / "multi_leader_v5_twostage.pt"
    torch.save({"s1":s1m.state_dict(), "s2":s2m.state_dict(),
                "s1_nf":STAGE1_INPUT_DIM, "s2_nf":STAGE2_INPUT_DIM,
                "n_pca":N_PCA_COMPONENTS, "n_leaders":len(al),
                "s1_n":len(s1s), "s2_n":len(s2s),
                "s1_met":s1met, "s2_met":s2met, "pca_exp":exp, "dev":str(device)}, mp)
    print(f"\n[9] Saved: {mp}")

    V3D, V3T = 0.758, 0.485; V4D, V4T = 0.562, 0.579
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY  v5 Two-Stage vs v3/v4")
    print("=" * 70)
    print(f"\n  Architecture:")
    print(f"    v3: Transformer 64D, 3L, 66D in  | v4: Transformer 128D, 4L, 86D in")
    print(f"    v5 S1: Transformer 128D, 8H, 3L, {STAGE1_INPUT_DIM}D -> {N_PCA_COMPONENTS}D ({s1m.count_parameters():,} params)")
    print(f"    v5 S2: MLP {STAGE2_INPUT_DIM}D -> 64 -> 3 ({s2m.count_parameters():,} params)")
    print(f"\n  Stage 1: {len(s1s)} samples  MSE={s1met['mse']:.6f}  R2={s1met['r2']:.4f}  {s1met['elapsed']:.1f}s")
    print(f"  Stage 2: {len(s2s)} samples  acc={oa:.3f}  {s2met['elapsed']:.1f}s")
    print(f"\n  POWELL LOO:")
    if ploo:
        d = ploo["dir_acc"]; t = ploo.get("tp_acc")
        print(f"    Dir: {d:.3f}  (v3={V3D:.3f} {(d-V3D)*100:+.1f}pp, v4={V4D:.3f} {(d-V4D)*100:+.1f}pp)")
        if t is not None:
            print(f"    TP:  {t:.3f}  (v3={V3T:.3f} {(t-V3T)*100:+.1f}pp, v4={V4T:.3f} {(t-V4T)*100:+.1f}pp)")
    else: print("    Failed")
    print(f"\n  Leaders: {len(al)} ({np_} profiled)  PCA: {N_PCA_COMPONENTS}PC/{len(sn)}series ({exp:.1f}%)")
    print(f"  Total: {time.time()-t0:.1f}s")
    print("=" * 70)

    er = {"model":"v5_twostage", "at":datetime.now().isoformat(),
          "s1":{"n":len(s1s), **{k:float(v) for k,v in s1met.items()}},
          "s2":{"n":len(s2s), **{k:float(v) for k,v in s2met.items()}},
          "pca":{"series":len(sn), "pc":N_PCA_COMPONENTS, "exp":exp},
          "leaders":{"total":len(al), "profiled":np_},
          "per_leader":pl, "overall":oa,
          "powell_loo":{"dir":ploo["dir_acc"],"tp":ploo.get("tp_acc"),"n":ploo["n"]} if ploo else None,
          "cmp":{"v3d":V3D,"v3t":V3T,"v4d":V4D,"v4t":V4T}, "dev":str(device), "t":time.time()-t0}
    ep = OUTPUT_DIR / "multi_leader_v5_evaluation.json"
    with open(ep, "w") as f: json.dump(er, f, indent=2, default=str)
    print(f"[10] Eval: {ep}")

if __name__ == "__main__":
    import os; os.chdir(BASE); main()
