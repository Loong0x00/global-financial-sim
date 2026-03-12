#!/usr/bin/env python3
"""
Multi-Leader Unified Transformer Decision Model
================================================
Trains on ALL central bank leaders' rate decisions simultaneously.

Key insight: Transformers learn shared representations.
- Shared: "central bankers hike when inflation is high"
- Individual: "Powell is more dovish than Greenspan" (via leader embeddings)

Architecture:
  Input = [economic_state ~20D] + [leader_embedding ~16D] + [context ~5D] = ~41D
  - 2-layer TransformerEncoder, d_model=32, nhead=4
  - 3 output heads: direction (cut/hold/hike), magnitude (0/25/50/75/100bp), guidance

Leaders covered:
  US Fed: Greenspan (1987-08 to 2006-01), Bernanke (2006-02 to 2014-01),
          Yellen (2014-02 to 2018-01), Powell (2018-02 to present)
  ECB:    Draghi (2011-11 to 2019-10), Lagarde (2019-11 to present)
  Russia: Nabiullina (2013-06 to present)
  China:  Zhou Xiaochuan (2002-12 to 2018-03)
  Japan:  Kuroda (2013-03 to 2023-04), Ueda (2023-04 to present)
  UK:     Multiple BOE governors
"""

import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np

print("=" * 70)
print("MULTI-LEADER TRANSFORMER DECISION MODEL  v1")
print("=" * 70)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
    print(f"[OK] PyTorch {torch.__version__}")
except ImportError:
    TORCH_OK = False
    print("[WARN] PyTorch not available — will skip model training")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path("/home/user/global-financial-sim")
DATA = BASE / "data"
PROFILES = DATA / "leaders" / "profiles"
TIMELINES = DATA / "leaders" / "timelines"
MODELS_DIR = BASE / "models" / "decision_functions"
OUTPUT_DIR = BASE / "output" / "decision_function"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Leader Registry ──────────────────────────────────────────────────────────
# Each entry: (leader_id, country, profile_filename, start_yyyymm, end_yyyymm_or_None)
LEADER_REGISTRY = [
    # US Fed Chairs
    ("greenspan",    "US", "greenspan",    "1987-08", "2006-01"),
    ("bernanke",     "US", "bernanke",     "2006-02", "2014-01"),
    ("yellen",       "US", "yellen",       "2014-02", "2018-01"),
    ("powell",       "US", "powell",       "2018-02", None),
    # ECB Presidents
    ("draghi",       "EU", "draghi",       "2011-11", "2019-10"),
    ("lagarde",      "EU", "lagarde",      "2019-11", None),
    # Russia CBR
    ("nabiullina",   "RU", "nabiullina",   "2013-06", None),
    # China PBOC
    ("zhou_xiaochuan","CN","zhou_xiaochuan","2002-12", "2018-03"),
    # Japan BOJ
    ("kuroda",       "JP", None,           "2013-03", "2023-04"),  # no profile
    ("ueda",         "JP", None,           "2023-04", None),       # no profile
    # UK BOE (use None for missing profiles)
    ("carney_boe",   "UK", None,           "2013-07", "2020-03"),  # Mark Carney
    ("bailey_boe",   "UK", None,           "2020-03", None),       # Andrew Bailey
]

# Country-to-rate-source mapping
# Priority: BIS policy_rates > macro interest_rate > supplementary
RATE_SOURCES = {
    "US": ("bis", "US"),
    "EU": ("macro", "interest_rate"),     # ECB rate from eu/macro.json
    "RU": ("bis", "RU"),
    "CN": ("bis", "CN"),
    "JP": ("bis", "JP"),
    "UK": ("bis", "GB"),
}

# ─── Economic Feature Definitions ─────────────────────────────────────────────
# Per-country economic features to extract for each decision
# Format: (indicator_source, field_name, transform)
COUNTRY_FEATURES = {
    "US": [
        ("macro",    "cpi_yoy",              "level"),
        ("macro",    "core_cpi_yoy",         "level"),
        ("macro",    "unemployment",         "level"),
        ("macro",    "gdp_growth",           "level"),
        ("macro",    "fed_funds_rate",       "level"),
        ("financial","yield_curve_10y2y",    "level"),
        ("financial","vix",                  "level"),
        ("financial","sp500",                "yoy"),
        ("macro",    "nonfarm_payrolls_change", "level"),
        ("macro",    "pce_yoy",             "level"),
    ],
    "EU": [
        ("macro",    "hicp_all_items",       "level"),
        ("macro",    "interest_rate",        "level"),
        ("macro",    "unemployment",         "level"),
        ("macro",    "gdp_growth",           "level"),
        ("macro",    "eur_usd",              "yoy"),
        ("macro",    "industrial_production","yoy"),
        ("financial","germany_10y_bond",     "level"),
        ("macro",    "m3_money_supply",      "yoy"),
    ],
    "RU": [
        ("macro",    "cpi",                  "level"),
        ("macro",    "interest_rate",        "level"),
        ("macro",    "unemployment",         "level"),
        ("macro",    "gdp",                  "level"),
        ("macro",    "exchange_rate_usd_rub","yoy"),
        ("macro",    "industrial_production","yoy"),
        ("macro",    "m2",                   "yoy"),
    ],
    "CN": [
        ("macro",    "cpi_yoy",              "level"),
        ("macro",    "discount_rate",        "level"),
        ("macro",    "gdp_growth",           "level"),
        ("macro",    "exports",              "yoy"),
        ("financial","usd_cny",              "yoy"),
        ("financial","short_rate",           "level"),
        ("macro",    "m2",                   "yoy"),
    ],
    "JP": [
        ("macro",    "cpi",                  "level"),
        ("macro",    "interest_rate",        "level"),
        ("macro",    "unemployment",         "level"),
        ("macro",    "gdp_growth",           "level"),
        ("macro",    "exchange_rate_usd_jpy","yoy"),
        ("financial","nikkei225",            "yoy"),
        ("macro",    "industrial_production","yoy"),
    ],
    "UK": [
        ("macro",    "cpi",                  "level"),
        ("macro",    "interest_rate",        "level"),
        ("macro",    "unemployment",         "level"),
        ("macro",    "gdp_growth",           "level"),
        ("macro",    "gbp_usd",              "yoy"),
        ("macro",    "industrial_production","yoy"),
        ("macro",    "m2_money_supply",      "yoy"),
    ],
}
N_ECON_FEATURES = 10   # max econ features per country (pad/truncate to this)

# ─── Leader Embedding Dimension ───────────────────────────────────────────────
LEADER_EMB_DIM = 16    # fixed, from PCA of profile parameters

# ─── Context Features ─────────────────────────────────────────────────────────
N_CONTEXT_FEATURES = 5   # hike_streak, cut_streak, hold_streak, month_sin, month_cos

TOTAL_DIM = N_ECON_FEATURES + LEADER_EMB_DIM + N_CONTEXT_FEATURES  # 31

# ─── Common Profile Parameters (for leader embedding) ─────────────────────────
# These 36 labels are expected to exist across most central banker profiles
COMMON_CB_PARAMS = [
    "inflation_sensitivity_headline_cpi",
    "unemployment_sensitivity",
    "gdp_growth_sensitivity",
    "financial_stability_weight",
    "data_dependence",
    "forward_guidance_credibility",
    "independence_from_political_pressure",
    "risk_management_approach",
    "communication_transparency",
    "global_spillover_sensitivity",
    "quantitative_easing_willingness",
    "emergency_rate_hike_decisiveness",
    "dovish_bias",
    "hawkish_bias",
    "crisis_response_speed",
    "verbal_intervention_potency",
    # Additional fallback keys
    "labor_market_dovish_bias",
    "deflation_spiral_fear",
    "exchange_rate_reform_commitment",
    "covid_pepp_response_speed",
    "flexibility_under_constraint",
    "consensus_building",
    "asset_bubble_tolerance",
    "yield_curve_control_willingness",
    "currency_stability_priority",
    "information_fidelity",
    "institutional_credibility_weight",
    "policy_reversal_willingness",
    "external_shock_responsiveness",
    "fiscal_monetary_coordination",
    "emerging_market_sensitivity",
    "macro_prudential_integration",
    "academic_rigor_in_decisions",
    "market_communication_skill",
    "patience_duration",
    "geopolitical_sensitivity",
]
assert len(COMMON_CB_PARAMS) >= LEADER_EMB_DIM


# ─── Data Loading Utilities ───────────────────────────────────────────────────

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


class EconomicDataStore:
    """Loads all economic data and provides fast lookup by (country, indicator, month)."""

    def __init__(self):
        self._data = {}     # (country, source, indicator) -> {month: value}
        self._bis_rates = {}  # country -> {month: value}
        self._load_all()

    def _load_all(self):
        for country in ["US", "EU", "RU", "CN", "JP", "UK"]:
            cc = country.lower()
            # Macro data
            macro_path = DATA / "economic" / cc / "macro.json"
            if macro_path.exists():
                md = load_json(macro_path)
                for ind_name, ind_data in md.get("indicators", {}).items():
                    series = ind_data.get("series", [])
                    self._store("macro", country, ind_name, series)

            # Financial data
            fin_path = DATA / "economic" / cc / "financial.json"
            if fin_path.exists():
                fd = load_json(fin_path)
                for ind_name, ind_data in fd.get("indicators", {}).items():
                    series = ind_data.get("series", [])
                    self._store("financial", country, ind_name, series)

        # BIS policy rates
        bis_path = DATA / "economic" / "bis" / "policy_rates.json"
        if bis_path.exists():
            bis = load_json(bis_path)
            country_map = {
                "US": "US", "EU": None,  # EU handled separately via macro
                "RU": "RU", "CN": "CN", "JP": "JP", "UK": "GB"
            }
            for our_country, bis_code in country_map.items():
                if bis_code and bis_code in bis.get("countries", {}):
                    ser_data = bis["countries"][bis_code]["series"]
                    for sk, sv in ser_data.items():
                        # Use monthly observations only (YYYY-MM format)
                        monthly = {k: v for k, v in sv["data"].items() if len(k) == 7}
                        self._bis_rates[our_country] = monthly
                        break  # only one series per country

    def _store(self, source, country, ind_name, series):
        if not series:
            return
        key = (country, source, ind_name)
        d = {}
        for item in series:
            if isinstance(item, dict) and "date" in item and item.get("value") is not None:
                d[item["date"]] = float(item["value"])
        self._data[key] = d

    def get_rate(self, country, month):
        """Get central bank policy rate for country at given month."""
        if country == "EU":
            # Use EU macro interest_rate
            d = self._data.get(("EU", "macro", "interest_rate"), {})
            return d.get(month)
        elif country in self._bis_rates:
            return self._bis_rates[country].get(month)
        return None

    def get(self, source, country, indicator, month):
        """Get indicator value at month."""
        d = self._data.get((country, source, indicator), {})
        return d.get(month)

    def get_yoy(self, source, country, indicator, month):
        """Get year-over-year change."""
        y, m = int(month[:4]), int(month[5:])
        prev_y = y - 1
        prev_month = f"{prev_y:04d}-{m:02d}"
        curr = self.get(source, country, indicator, month)
        prev = self.get(source, country, indicator, prev_month)
        if curr is not None and prev is not None and prev != 0:
            return (curr - prev) / abs(prev) * 100.0
        return None

    def get_prev_month(self, month):
        y, m = int(month[:4]), int(month[5:])
        m -= 1
        if m == 0:
            m = 12
            y -= 1
        return f"{y:04d}-{m:02d}"

    def all_months_range(self, start_yyyymm, end_yyyymm=None):
        """Generate list of YYYY-MM strings from start to end (inclusive)."""
        months = []
        y, m = int(start_yyyymm[:4]), int(start_yyyymm[5:])
        if end_yyyymm is None:
            end_yyyymm = "2026-02"
        ey, em = int(end_yyyymm[:4]), int(end_yyyymm[5:])
        while (y, m) <= (ey, em):
            months.append(f"{y:04d}-{m:02d}")
            m += 1
            if m > 12:
                m = 1
                y += 1
        return months


# ─── Rate Decision Extraction ─────────────────────────────────────────────────

def classify_rate_change(delta_bp):
    """Classify rate change in basis points into direction/magnitude."""
    if delta_bp < -5:
        direction = 0  # cut
    elif delta_bp > 5:
        direction = 2  # hike
    else:
        direction = 1  # hold

    # Round to nearest 25bp bracket
    abs_bp = abs(delta_bp)
    if abs_bp < 12.5:
        magnitude = 0
    elif abs_bp < 37.5:
        magnitude = 1   # 25bp
    elif abs_bp < 62.5:
        magnitude = 2   # 50bp
    elif abs_bp < 87.5:
        magnitude = 3   # 75bp
    else:
        magnitude = 4   # 100bp+

    return direction, magnitude


def extract_rate_decisions(econ: EconomicDataStore, leader_id: str, country: str,
                           start_yyyymm: str, end_yyyymm: str = None):
    """
    Extract monthly rate decisions for a leader from policy rate time series.

    Returns list of dicts with:
      month, rate, prev_rate, delta_bp, direction, magnitude
    """
    months = econ.all_months_range(start_yyyymm, end_yyyymm)
    decisions = []
    prev_rate = None

    for month in months:
        rate = econ.get_rate(country, month)
        if rate is None:
            prev_rate = None
            continue

        if prev_rate is not None:
            delta_bp = (rate - prev_rate) * 100.0  # convert % to bp
            direction, magnitude = classify_rate_change(delta_bp)
            decisions.append({
                "month": month,
                "rate": rate,
                "prev_rate": prev_rate,
                "delta_bp": round(delta_bp, 1),
                "direction": direction,   # 0=cut, 1=hold, 2=hike
                "magnitude": magnitude,  # 0=0bp, 1=25, 2=50, 3=75, 4=100+
            })
        prev_rate = rate

    return decisions


# ─── Economic State Vector Extraction ────────────────────────────────────────

def extract_econ_state(econ: EconomicDataStore, country: str, month: str) -> np.ndarray:
    """
    Extract N_ECON_FEATURES economic state features for country at month.
    Returns raw (un-normalized) feature vector of shape (N_ECON_FEATURES,).
    """
    features_spec = COUNTRY_FEATURES.get(country, COUNTRY_FEATURES["US"])
    vals = []
    for (src, ind, transform) in features_spec:
        if transform == "level":
            v = econ.get(src, country, ind, month)
        elif transform == "yoy":
            v = econ.get_yoy(src, country, ind, month)
        else:
            v = None
        vals.append(v)

    # Pad/truncate to N_ECON_FEATURES
    arr = np.zeros(N_ECON_FEATURES, dtype=np.float32)
    for i, v in enumerate(vals[:N_ECON_FEATURES]):
        if v is not None:
            arr[i] = float(v)
        else:
            arr[i] = np.nan
    return arr


# ─── Context Features ────────────────────────────────────────────────────────

def compute_context(decision_idx: int, decisions: list, month: str) -> np.ndarray:
    """Compute streak and time context features."""
    # Count consecutive hike/cut/hold streak
    hike_streak = 0
    cut_streak = 0
    hold_streak = 0

    for i in range(decision_idx - 1, max(-1, decision_idx - 12), -1):
        d = decisions[i]["direction"]
        if d == 2:
            hike_streak += 1
            cut_streak = 0
            hold_streak = 0
        elif d == 0:
            cut_streak += 1
            hike_streak = 0
            hold_streak = 0
        else:
            hold_streak += 1
            break  # stop at first non-streak

    m = int(month[5:])
    month_sin = math.sin(2 * math.pi * m / 12)
    month_cos = math.cos(2 * math.pi * m / 12)

    return np.array([hike_streak, cut_streak, hold_streak, month_sin, month_cos],
                    dtype=np.float32)


# ─── Leader Profile → Embedding ───────────────────────────────────────────────

def extract_leader_embedding(profile_name: str) -> np.ndarray:
    """
    Extract a fixed LEADER_EMB_DIM embedding from leader profile.

    Uses a fixed set of COMMON_CB_PARAMS labels. If a label is missing,
    uses 0.5 (neutral). Then takes first LEADER_EMB_DIM values.
    NOT trained — fixed from the behavioral profile.
    """
    if profile_name is None:
        return np.full(LEADER_EMB_DIM, 0.5, dtype=np.float32)

    path = PROFILES / f"{profile_name}.json"
    if not path.exists():
        return np.full(LEADER_EMB_DIM, 0.5, dtype=np.float32)

    try:
        profile = load_json(path)
    except Exception:
        return np.full(LEADER_EMB_DIM, 0.5, dtype=np.float32)

    # Build lookup from behavior_matrix.vectors
    lu = {}
    for v in profile.get("behavior_matrix", {}).get("vectors", []):
        if "label" in v and "value" in v and v["value"] is not None:
            lu[v["label"]] = float(v["value"])

    # Also try behavioral_parameters format
    for group in profile.get("behavioral_parameters", []):
        for param in group.get("parameters", []):
            if "name" in param and "value" in param:
                lu[param["name"]] = float(param["value"])

    # Extract values for common params
    vals = []
    for label in COMMON_CB_PARAMS[:LEADER_EMB_DIM]:
        if label in lu:
            vals.append(lu[label])
        else:
            # Try partial match
            matches = [v for k, v in lu.items() if k.startswith(label[:10])]
            vals.append(float(np.mean(matches)) if matches else 0.5)

    arr = np.array(vals[:LEADER_EMB_DIM], dtype=np.float32)
    if len(arr) < LEADER_EMB_DIM:
        arr = np.pad(arr, (0, LEADER_EMB_DIM - len(arr)), constant_values=0.5)

    return arr


# ─── Per-Country Normalizer ───────────────────────────────────────────────────

class PerCountryNormalizer:
    """
    Quantile-style normalization per country.
    Fits on training data, transforms to roughly [0, 1] range using
    robust z-score (median / IQR) then sigmoid transform.
    """

    def __init__(self):
        self.medians = {}   # country -> np.array
        self.iqrs = {}      # country -> np.array

    def fit(self, country_samples: dict):
        """
        country_samples: {country: list of raw econ vectors (N_ECON_FEATURES,)}
        """
        for country, samples in country_samples.items():
            X = np.array([s for s in samples if not np.any(np.isnan(s))])
            if len(X) < 5:
                self.medians[country] = np.zeros(N_ECON_FEATURES)
                self.iqrs[country] = np.ones(N_ECON_FEATURES)
                continue
            self.medians[country] = np.nanmedian(X, axis=0)
            q75 = np.nanpercentile(X, 75, axis=0)
            q25 = np.nanpercentile(X, 25, axis=0)
            iqr = q75 - q25
            iqr[iqr < 1e-8] = 1.0
            self.iqrs[country] = iqr

    def transform(self, country: str, vec: np.ndarray) -> np.ndarray:
        """Transform raw econ vector to normalized vector, NaN -> 0."""
        out = vec.copy()
        # Fill NaN with median
        med = self.medians.get(country, np.zeros(N_ECON_FEATURES))
        iqr = self.iqrs.get(country, np.ones(N_ECON_FEATURES))
        for i in range(len(out)):
            if np.isnan(out[i]):
                out[i] = 0.0  # neutral after normalization
            else:
                # Robust z-score
                out[i] = (out[i] - med[i]) / iqr[i]
        # Clip extreme values
        out = np.clip(out, -3, 3)
        return out

    def to_dict(self):
        return {
            "medians": {k: v.tolist() for k, v in self.medians.items()},
            "iqrs": {k: v.tolist() for k, v in self.iqrs.items()},
        }


# ─── Dataset Builder ──────────────────────────────────────────────────────────

def build_multi_leader_dataset(econ: EconomicDataStore):
    """
    Build the full dataset across all leaders.

    Returns:
      samples: list of dicts with keys:
        leader_id, country, month, econ_raw, context, direction, magnitude, guidance
      leader_embeddings: {leader_id: np.array(LEADER_EMB_DIM)}
    """
    print("\n[1] Extracting rate decisions for all leaders...")

    all_samples = []
    leader_embeddings = {}
    stats = {}

    for (lid, country, profile_name, start, end) in LEADER_REGISTRY:
        # Extract leader embedding
        emb = extract_leader_embedding(profile_name)
        leader_embeddings[lid] = emb
        emb_norm = float(np.mean(emb))  # rough check for zero embedding
        has_profile = profile_name is not None and (PROFILES / f"{profile_name}.json").exists()

        # Extract rate decisions
        decisions = extract_rate_decisions(econ, lid, country, start, end)

        if not decisions:
            print(f"  {lid} ({country}): NO RATE DATA found")
            continue

        # Build samples
        samples = []
        for i, dec in enumerate(decisions):
            month = dec["month"]

            # Economic state (raw)
            econ_raw = extract_econ_state(econ, country, month)

            # Context
            ctx = compute_context(i, decisions, month)

            # Forward guidance: Powell has explicit data, others infer from direction
            direction = dec["direction"]
            if direction == 2:
                guidance = 2  # hawkish
            elif direction == 0:
                guidance = 0  # dovish
            else:
                guidance = 1  # neutral

            # Skip months with insufficient econ data (too many NaN)
            nan_frac = np.mean(np.isnan(econ_raw))
            if nan_frac > 0.6:
                continue

            samples.append({
                "leader_id": lid,
                "country": country,
                "month": month,
                "econ_raw": econ_raw,
                "context": ctx,
                "direction": direction,
                "magnitude": dec["magnitude"],
                "guidance": guidance,
                "delta_bp": dec["delta_bp"],
            })

        # Count direction distribution
        dirs = [s["direction"] for s in samples]
        n_cut = dirs.count(0)
        n_hold = dirs.count(1)
        n_hike = dirs.count(2)

        print(f"  {lid:20s} ({country}) [{start}→{end or 'now'}]: "
              f"{len(samples):4d} samples  "
              f"cut={n_cut} hold={n_hold} hike={n_hike}  "
              f"profile={'YES' if has_profile else 'NO '}")

        stats[lid] = {
            "n_samples": len(samples),
            "n_cut": n_cut,
            "n_hold": n_hold,
            "n_hike": n_hike,
            "country": country,
            "has_profile": has_profile,
        }
        all_samples.extend(samples)

    print(f"\n  Total samples: {len(all_samples)}")
    print(f"  Leaders: {len([s for s in stats.values() if s['n_samples'] > 0])}")

    return all_samples, leader_embeddings, stats


# ─── Normalization ────────────────────────────────────────────────────────────

def fit_and_normalize(samples):
    """Fit per-country normalizer and apply to all samples."""
    print("\n[2] Fitting per-country normalizers...")

    # Collect raw econ vectors per country
    country_raw = defaultdict(list)
    for s in samples:
        country_raw[s["country"]].append(s["econ_raw"])

    norm = PerCountryNormalizer()
    norm.fit(country_raw)

    # Apply normalization
    for s in samples:
        s["econ_norm"] = norm.transform(s["country"], s["econ_raw"])

    print(f"  Normalized {len(samples)} samples across {len(country_raw)} countries")
    return norm


# ─── Feature Assembly ────────────────────────────────────────────────────────

def assemble_feature_vector(sample, leader_embeddings):
    """
    Combine normalized econ + leader_embedding + context into a single vector.
    Returns np.array of shape (TOTAL_DIM,).
    """
    econ_norm = sample["econ_norm"]                              # (N_ECON_FEATURES,)
    leader_emb = leader_embeddings[sample["leader_id"]]         # (LEADER_EMB_DIM,)
    ctx = sample["context"]                                      # (N_CONTEXT_FEATURES,)

    # Normalize context features
    ctx_norm = ctx.copy()
    ctx_norm[:3] /= 12.0  # streak values / 12
    # month_sin/cos already in [-1, 1]

    return np.concatenate([econ_norm, leader_emb, ctx_norm]).astype(np.float32)


# ─── Model Architecture ───────────────────────────────────────────────────────

if TORCH_OK:
    class MultiLeaderDecisionTransformer(nn.Module):
        """
        Unified multi-leader Transformer for central bank rate decisions.

        Each input feature is its own token (feature-as-token approach).
        Leader embedding is concatenated as fixed features (not trained here).
        """

        def __init__(self, n_features: int = TOTAL_DIM, d_model: int = 32,
                     nhead: int = 4, n_layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.n_features = n_features
            self.d_model = d_model

            # Project each scalar feature to d_model
            self.input_proj = nn.Linear(1, d_model)

            # Learnable position embeddings per feature slot
            self.pos_emb = nn.Embedding(n_features, d_model)

            # Transformer encoder
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 2,
                dropout=dropout, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

            # Output heads
            self.direction_head = nn.Linear(d_model, 3)   # cut/hold/hike
            self.magnitude_head = nn.Linear(d_model, 5)   # 0/25/50/75/100bp
            self.guidance_head  = nn.Linear(d_model, 3)   # dovish/neutral/hawkish

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x: torch.Tensor):
            """
            x: (batch, n_features) — raw feature vector
            Returns: dir_logits, mag_logits, guid_logits each (batch, n_classes)
            """
            B, N = x.shape
            # Reshape: (B, N, 1)
            x = x.unsqueeze(-1)
            # Project each feature to d_model: (B, N, d_model)
            tok = self.input_proj(x)
            # Add position embeddings
            pos_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
            tok = tok + self.pos_emb(pos_ids)
            # Transformer: (B, N, d_model)
            enc = self.encoder(tok)
            # Mean pool: (B, d_model)
            pooled = enc.mean(dim=1)

            return (
                self.direction_head(pooled),
                self.magnitude_head(pooled),
                self.guidance_head(pooled),
            )

        def predict(self, x: torch.Tensor):
            self.eval()
            with torch.no_grad():
                d, m, g = self.forward(x)
                return (
                    torch.argmax(d, dim=-1),
                    torch.argmax(m, dim=-1),
                    torch.argmax(g, dim=-1),
                )

        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Training ────────────────────────────────────────────────────────────────

def train_model(X, y_dir, y_mag, y_guid, n_epochs=200, lr=3e-3, verbose=True):
    """Train the multi-leader model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"\n[3] Training on {device}  (samples={len(X)}, features={X.shape[1]})")

    model = MultiLeaderDecisionTransformer(n_features=X.shape[1])
    if verbose:
        print(f"  Parameters: {model.count_parameters():,}")

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    yd = torch.tensor(y_dir, dtype=torch.long).to(device)
    ym = torch.tensor(y_mag, dtype=torch.long).to(device)
    yg = torch.tensor(y_guid, dtype=torch.long).to(device)

    # Class weights for direction (handle imbalance)
    dir_counts = np.bincount(y_dir, minlength=3).astype(float)
    dir_counts = np.where(dir_counts == 0, 1, dir_counts)
    dir_weights = torch.tensor(1.0 / dir_counts, dtype=torch.float32).to(device)
    dir_weights /= dir_weights.sum()

    criterion_dir  = nn.CrossEntropyLoss(weight=dir_weights)
    criterion_mag  = nn.CrossEntropyLoss()
    criterion_guid = nn.CrossEntropyLoss()

    history = []
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        d_logits, m_logits, g_logits = model(Xt)
        loss_d = criterion_dir(d_logits, yd)
        loss_m = criterion_mag(m_logits, ym)
        loss_g = criterion_guid(g_logits, yg)
        loss = loss_d + 0.5 * loss_m + 0.3 * loss_g
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                d_pred = torch.argmax(d_logits, dim=-1)
                acc = (d_pred == yd).float().mean().item()
                if verbose:
                    print(f"  epoch {epoch:3d}/{n_epochs}  loss={loss.item():.4f}  "
                          f"dir_acc={acc:.3f}  lr={scheduler.get_last_lr()[0]:.5f}")
            history.append({"epoch": epoch, "loss": float(loss), "dir_acc": acc})

    elapsed = time.time() - t0
    if verbose:
        print(f"  Training completed in {elapsed:.1f}s")

    return model, history


# ─── Leave-One-Out (LOO) Evaluation ──────────────────────────────────────────

def loo_eval_by_leader(samples, leader_embeddings, norm, verbose=True):
    """
    Leave-one-leader-out evaluation.
    For each leader: train on all others, evaluate on this leader.
    Also does standard within-leader LOO.
    """
    print("\n[4] Leave-One-Leader-Out (Cross-Leader Generalization)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Group samples by leader
    by_leader = defaultdict(list)
    for s in samples:
        by_leader[s["leader_id"]].append(s)

    # Only eval leaders with >=5 samples
    eval_leaders = [lid for lid, slist in by_leader.items() if len(slist) >= 5]

    lolo_results = {}

    for test_leader in eval_leaders:
        test_samps = by_leader[test_leader]
        train_samps = [s for lid, slist in by_leader.items()
                       if lid != test_leader for s in slist]

        if len(train_samps) < 10:
            continue

        # Build feature matrices
        X_train = np.array([assemble_feature_vector(s, leader_embeddings)
                            for s in train_samps])
        y_dir_tr = np.array([s["direction"] for s in train_samps])
        y_mag_tr = np.array([s["magnitude"] for s in train_samps])
        y_guid_tr = np.array([s["guidance"] for s in train_samps])

        X_test = np.array([assemble_feature_vector(s, leader_embeddings)
                           for s in test_samps])
        y_dir_te = np.array([s["direction"] for s in test_samps])
        y_mag_te = np.array([s["magnitude"] for s in test_samps])

        # Train a model on all other leaders
        model, _ = train_model(X_train, y_dir_tr, y_mag_tr, y_guid_tr,
                               n_epochs=150, verbose=False)

        # Evaluate on test leader
        model.eval()
        Xt = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            d_pred, m_pred, _ = model.predict(Xt)
        d_pred = d_pred.cpu().numpy()
        m_pred = m_pred.cpu().numpy()

        dir_acc = float(np.mean(d_pred == y_dir_te))
        mag_acc = float(np.mean(m_pred == y_mag_te))
        combined_acc = float(np.mean((d_pred == y_dir_te) & (m_pred == y_mag_te)))

        # Turning point accuracy
        turning_pts = []
        for i in range(1, len(test_samps)):
            prev_dir = test_samps[i-1]["direction"]
            curr_dir = test_samps[i]["direction"]
            if prev_dir != curr_dir:
                turning_pts.append(i)

        tp_acc = None
        if turning_pts:
            tp_correct = sum(1 for i in turning_pts if d_pred[i] == y_dir_te[i])
            tp_acc = tp_correct / len(turning_pts)

        lolo_results[test_leader] = {
            "dir_acc": dir_acc,
            "mag_acc": mag_acc,
            "combined_acc": combined_acc,
            "tp_acc": tp_acc,
            "n_test": len(test_samps),
            "n_turning_pts": len(turning_pts),
            "country": by_leader[test_leader][0]["country"],
        }

        if verbose:
            tp_str = f"TP={tp_acc:.3f}({len(turning_pts)})" if tp_acc is not None else "TP=N/A"
            print(f"  LOLO {test_leader:20s}: dir={dir_acc:.3f}  mag={mag_acc:.3f}  "
                  f"comb={combined_acc:.3f}  {tp_str}  n={len(test_samps)}")

    return lolo_results


def loo_eval_within_leader(leader_id, samples, leader_embeddings, verbose=True):
    """
    Standard Leave-One-Out evaluation within a single leader.
    For fair comparison with v1 Powell LOO (75.8%).
    """
    leader_samps = [s for s in samples if s["leader_id"] == leader_id]
    if len(leader_samps) < 5:
        return None

    if verbose:
        print(f"\n[5] Within-leader LOO for {leader_id} "
              f"(n={len(leader_samps)})...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_pred_dir = []
    all_true_dir = []
    all_pred_mag = []
    all_true_mag = []

    for test_idx in range(len(leader_samps)):
        train_samps = [s for i, s in enumerate(leader_samps) if i != test_idx]
        test_samp = leader_samps[test_idx]

        X_train = np.array([assemble_feature_vector(s, leader_embeddings)
                            for s in train_samps])
        y_dir_tr = np.array([s["direction"] for s in train_samps])
        y_mag_tr = np.array([s["magnitude"] for s in train_samps])
        y_guid_tr = np.array([s["guidance"] for s in train_samps])

        X_test = np.array([assemble_feature_vector(test_samp, leader_embeddings)])
        y_dir_te = test_samp["direction"]
        y_mag_te = test_samp["magnitude"]

        model, _ = train_model(X_train, y_dir_tr, y_mag_tr, y_guid_tr,
                               n_epochs=150, verbose=False)

        model.eval()
        Xt = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            d_pred, m_pred, _ = model.predict(Xt)

        all_pred_dir.append(int(d_pred[0]))
        all_true_dir.append(y_dir_te)
        all_pred_mag.append(int(m_pred[0]))
        all_true_mag.append(y_mag_te)

    dir_acc = float(np.mean(np.array(all_pred_dir) == np.array(all_true_dir)))
    mag_acc = float(np.mean(np.array(all_pred_mag) == np.array(all_true_mag)))

    # Turning point accuracy
    turning_pts = []
    for i in range(1, len(leader_samps)):
        if leader_samps[i-1]["direction"] != leader_samps[i]["direction"]:
            turning_pts.append(i)

    tp_acc = None
    if turning_pts:
        tp_correct = sum(1 for i in turning_pts
                         if all_pred_dir[i] == all_true_dir[i])
        tp_acc = tp_correct / len(turning_pts)

    if verbose:
        tp_str = f"{tp_acc:.3f} ({len(turning_pts)} pts)" if tp_acc is not None else "N/A"
        print(f"  {leader_id} LOO: dir_acc={dir_acc:.3f}  mag_acc={mag_acc:.3f}  "
              f"turning_pts={tp_str}")

    return {
        "dir_acc": dir_acc,
        "mag_acc": mag_acc,
        "tp_acc": tp_acc,
        "n_samples": len(leader_samps),
        "n_turning_pts": len(turning_pts),
        "all_pred_dir": all_pred_dir,
        "all_true_dir": all_true_dir,
    }


# ─── Full Training (All Samples) ─────────────────────────────────────────────

def train_final_model(samples, leader_embeddings, norm):
    """Train the final model on all samples."""
    print("\n[6] Training final model on ALL samples...")

    X = np.array([assemble_feature_vector(s, leader_embeddings) for s in samples])
    y_dir  = np.array([s["direction"] for s in samples])
    y_mag  = np.array([s["magnitude"] for s in samples])
    y_guid = np.array([s["guidance"]  for s in samples])

    model, history = train_model(X, y_dir, y_mag, y_guid, n_epochs=300)

    # Full dataset accuracy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        d_pred, m_pred, _ = model.predict(Xt)
    d_pred = d_pred.cpu().numpy()
    m_pred = m_pred.cpu().numpy()

    train_dir_acc = float(np.mean(d_pred == y_dir))
    train_mag_acc = float(np.mean(m_pred == y_mag))
    print(f"  Training dir_acc={train_dir_acc:.3f}  mag_acc={train_mag_acc:.3f}")

    return model, history, X, y_dir, y_mag, y_guid


# ─── Per-Leader Accuracy on Full Model ────────────────────────────────────────

def per_leader_accuracy(model, samples, leader_embeddings):
    """Compute per-leader accuracy using the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    by_leader = defaultdict(list)
    for s in samples:
        by_leader[s["leader_id"]].append(s)

    results = {}
    for lid, slist in sorted(by_leader.items()):
        X = np.array([assemble_feature_vector(s, leader_embeddings) for s in slist])
        y_dir = np.array([s["direction"] for s in slist])
        y_mag = np.array([s["magnitude"] for s in slist])

        Xt = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            d_pred, m_pred, _ = model.predict(Xt)
        d_pred = d_pred.cpu().numpy()
        m_pred = m_pred.cpu().numpy()

        dir_acc = float(np.mean(d_pred == y_dir))
        mag_acc = float(np.mean(m_pred == y_mag))
        results[lid] = {"dir_acc": dir_acc, "mag_acc": mag_acc,
                        "n_samples": len(slist),
                        "country": slist[0]["country"]}
    return results


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    if not TORCH_OK:
        print("[ERROR] PyTorch required for model training")
        return

    # Step 1: Load economic data
    print("\n[0] Loading economic data...")
    econ = EconomicDataStore()
    print(f"  Loaded {len(econ._data)} indicator series")
    print(f"  BIS policy rates for: {list(econ._bis_rates.keys())}")

    # Step 2: Build multi-leader dataset
    samples, leader_embeddings, stats = build_multi_leader_dataset(econ)

    if len(samples) < 20:
        print("[ERROR] Insufficient samples, cannot train")
        return

    # Step 3: Normalize
    norm = fit_and_normalize(samples)

    # Step 4: Train final model on all data
    model, history, X, y_dir, y_mag, y_guid = train_final_model(
        samples, leader_embeddings, norm
    )

    # Step 5: Per-leader accuracy (in-sample)
    print("\n[7] Per-leader in-sample accuracy...")
    per_leader = per_leader_accuracy(model, samples, leader_embeddings)

    print(f"\n{'Leader':22s} {'Country':7s} {'N':6s} {'Dir%':8s} {'Mag%':8s}")
    print("-" * 55)
    for lid, res in sorted(per_leader.items(), key=lambda x: x[0]):
        print(f"  {lid:20s} {res['country']:7s} {res['n_samples']:6d} "
              f"{res['dir_acc']*100:.1f}%    {res['mag_acc']*100:.1f}%")

    total_dir_acc = float(np.mean([r["dir_acc"] for r in per_leader.values()]))
    total_mag_acc = float(np.mean([r["mag_acc"] for r in per_leader.values()]))
    print(f"\n  Overall (macro-avg): dir={total_dir_acc:.3f}  mag={total_mag_acc:.3f}")
    print(f"  Total samples: {len(samples)}  (v1 Powell-only had 33)")

    # Step 6: Powell within-leader LOO (compare with v1's 75.8%)
    powell_loo = loo_eval_within_leader("powell", samples, leader_embeddings)

    # Step 7: Leave-One-Leader-Out generalization
    lolo_results = loo_eval_by_leader(samples, leader_embeddings, norm)

    # Step 8: Save model
    model_path = MODELS_DIR / "multi_leader_v1.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_features": TOTAL_DIM,
        "n_econ": N_ECON_FEATURES,
        "n_leader_emb": LEADER_EMB_DIM,
        "n_context": N_CONTEXT_FEATURES,
        "leader_embeddings": {k: v.tolist() for k, v in leader_embeddings.items()},
        "normalizer": norm.to_dict(),
        "history": history,
    }, model_path)
    print(f"\n[8] Model saved: {model_path}")

    # Step 9: Save evaluation
    eval_result = {
        "model": "multi_leader_v1",
        "generated_at": datetime.now().isoformat(),
        "training_samples": len(samples),
        "v1_comparison": {
            "v1_samples": 33,
            "v1_powell_loo_dir_acc": 0.758,
            "v1_turning_point_acc": 0.375,
        },
        "dataset_stats": {
            lid: {
                "n_samples": s["n_samples"],
                "n_cut": s["n_cut"],
                "n_hold": s["n_hold"],
                "n_hike": s["n_hike"],
                "country": s["country"],
                "has_profile": s["has_profile"],
            }
            for lid, s in stats.items()
        },
        "per_leader_in_sample_accuracy": per_leader,
        "overall_in_sample_accuracy": {
            "direction": total_dir_acc,
            "magnitude": total_mag_acc,
        },
        "powell_loo": powell_loo,
        "lolo_results": lolo_results,
        "elapsed_seconds": time.time() - t_start,
    }

    eval_path = OUTPUT_DIR / "multi_leader_evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(eval_result, f, indent=2, default=str)
    print(f"[9] Evaluation saved: {eval_path}")

    # Step 10: Print comparison table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  Training samples:  {len(samples)}  (v1: 33, +{len(samples)-33}x more)")
    print(f"  Leaders covered:   {len(stats)} leaders across 6 countries")
    print(f"  Feature dims:      {TOTAL_DIM}D "
          f"({N_ECON_FEATURES} econ + {LEADER_EMB_DIM} leader_emb + {N_CONTEXT_FEATURES} ctx)")

    print("\n  POWELL LOO (compare with v1):")
    if powell_loo:
        v1_dir = 0.758
        v1_tp  = 0.375
        ml_dir = powell_loo["dir_acc"]
        ml_tp  = powell_loo.get("tp_acc")
        delta_dir = (ml_dir - v1_dir) * 100
        print(f"    Direction accuracy:  {ml_dir:.3f} ({delta_dir:+.1f}pp vs v1's {v1_dir:.3f})")
        if ml_tp is not None:
            delta_tp = (ml_tp - v1_tp) * 100
            print(f"    Turning point acc:  {ml_tp:.3f} ({delta_tp:+.1f}pp vs v1's {v1_tp:.3f})")
        else:
            print(f"    Turning point acc:  N/A (insufficient data)")
    else:
        print("    Powell LOO failed (insufficient samples)")

    print("\n  CROSS-LEADER GENERALIZATION (Leave-One-Leader-Out):")
    if lolo_results:
        avg_lolo_dir = np.mean([r["dir_acc"] for r in lolo_results.values()])
        avg_lolo_tp  = np.mean([r["tp_acc"] for r in lolo_results.values()
                                if r["tp_acc"] is not None])
        print(f"    Average LOLO dir acc: {avg_lolo_dir:.3f}")
        print(f"    Average LOLO TP  acc: {avg_lolo_tp:.3f}")
        for lid, r in sorted(lolo_results.items()):
            tp = f"{r['tp_acc']:.3f}" if r["tp_acc"] is not None else "N/A"
            print(f"    {lid:22s}: dir={r['dir_acc']:.3f}  tp={tp}  "
                  f"(trained on everyone else)")

    print("\n  PER-LEADER IN-SAMPLE ACCURACY:")
    for lid, r in sorted(per_leader.items()):
        print(f"    {lid:22s} ({r['country']}): "
              f"dir={r['dir_acc']:.3f}  n={r['n_samples']}")

    print(f"\n  Total elapsed: {time.time() - t_start:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    import os
    os.chdir(BASE)
    main()
