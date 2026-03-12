#!/usr/bin/env python3
"""
Build Transformer Decision Function — Phase 2 Component
=======================================================
Multi-task small Transformer for Powell FOMC decision prediction.

Architecture (based on v1 pilot that achieved 66.7%):
  - Each input feature is its own token: (batch, N_feats, 1) -> projected to d_model
  - 2x TransformerEncoder layers (d_model=32, nhead=4, ff=64)
  - Mean pool -> 3 independent heads:
      direction_head: Linear(32, 3)  [cut | hold | hike]
      magnitude_head: Linear(32, 5)  [0 | 25 | 50 | 75 | 100] bp
      guidance_head:  Linear(32, 3)  [dovish | neutral | hawkish]
  - ~5k-20k parameters total
  - Training: NLLLoss on each head, summed
  - LOO evaluation: final "action" = direction + magnitude combo

Full feature set (~45 dims) vs v1 pilot (10 dims):
  - US: FFR, CPI, core CPI, PCE, unemployment, GDP growth, payrolls,
        yield curve (10y-2y), VIX, credit spread, SP500 yoy, USD index,
        M2 yoy, fed_total_assets yoy, treasury 10y, FFR mom
  - Policy context: hike_streak, cut_streak, hold_streak, partisan
  - Global: CN GDP, CN CPI, EU rate, EU HICP, Nikkei yoy
  - CPI trend (3m), unemployment trend (3m)
  - Leader factors: 15D Powell behavior vector
"""

import json
import math
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np

print("=" * 70)
print("TRANSFORMER DECISION FUNCTION BUILDER  v3 (multi-task)")
print("=" * 70)

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE = Path("/home/user/global-financial-sim")
DATA = BASE / "data"
ECON = DATA / "economic"
PROFILES = DATA / "leaders" / "profiles"
DECISIONS = DATA / "leaders" / "decisions"
OUT_MODELS = BASE / "models" / "decision_functions"
OUT_VIZ = BASE / "output" / "decision_function"

OUT_MODELS.mkdir(parents=True, exist_ok=True)
OUT_VIZ.mkdir(parents=True, exist_ok=True)

# ─── PyTorch ─────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─── Decision Labels ──────────────────────────────────────────────────────────
# Decomposed multi-task targets
DIRECTION = ["cut", "hold", "hike"]       # 3 classes
MAGNITUDE = [0, 25, 50, 75, 100]          # 5 classes (bp)
GUIDANCE  = ["dovish", "neutral", "hawkish"]  # 3 classes

DIR_IDX = {d: i for i, d in enumerate(DIRECTION)}
MAG_IDX = {m: i for i, m in enumerate(MAGNITUDE)}
GUD_IDX = {g: i for i, g in enumerate(GUIDANCE)}

# Combined action names (for display)
def make_action_name(direction, magnitude, guidance):
    if direction == "hold":
        if guidance == "neutral":
            return "hold"
        return f"hold_{guidance}"
    unit = f"{magnitude}bp" if magnitude > 0 else ""
    return f"{direction}_{unit}" if unit else direction


def decode_targets(dir_idx: int, mag_idx: int) -> str:
    """Decode direction + magnitude to action string."""
    d = DIRECTION[dir_idx]
    m = MAGNITUDE[mag_idx]
    if d == "hold":
        return "hold"
    return f"{d}_{m}bp"


def decision_to_targets(decision: dict):
    """Map raw decision dict to (direction, magnitude, guidance) indices."""
    action = decision.get("action", "hold")
    mag = abs(decision.get("magnitude_bp", 0))
    fg = decision.get("forward_guidance", "neutral")

    if action == "hold":
        dir_i = DIR_IDX["hold"]
        mag_i = MAG_IDX[0]
    elif action == "rate_hike":
        dir_i = DIR_IDX["hike"]
        # round to nearest in [25,50,75,100]
        best = min(MAGNITUDE[1:], key=lambda x: abs(x - mag))
        mag_i = MAG_IDX[best]
    elif action == "rate_cut":
        dir_i = DIR_IDX["cut"]
        best = min(MAGNITUDE[1:], key=lambda x: abs(x - mag))
        mag_i = MAG_IDX[best]
    else:
        dir_i = DIR_IDX["hold"]
        mag_i = MAG_IDX[0]

    # Forward guidance
    if fg in GUD_IDX:
        guid_i = GUD_IDX[fg]
    else:
        guid_i = GUD_IDX["neutral"]

    return dir_i, mag_i, guid_i


LEADER_FACTOR_DIM = 15

# ─── Economic Data Loader ─────────────────────────────────────────────────────
class EconomicDataLoader:
    def __init__(self):
        self.cache = {}
        self._load_all()

    def _load_json_indicators(self, entity: str, fp: Path):
        try:
            d = json.load(open(fp))
            for name, ind in d.get("indicators", {}).items():
                series = ind.get("series", [])
                if series:
                    self.cache[(entity, name)] = {
                        e["date"][:7]: e["value"]
                        for e in series if e.get("value") is not None
                    }
        except Exception:
            pass

    def _load_global_indices(self):
        try:
            d = json.load(open(ECON / "indices" / "global_indices.json"))
            for name, ind in d.items():
                data = ind.get("data", [])
                if data:
                    monthly = {}
                    for e in data:
                        if e.get("close") is not None:
                            monthly[e["date"][:7]] = e["close"]
                    self.cache[("global", name)] = monthly
        except Exception:
            pass

    def _load_all(self):
        print("\n[DATA] Loading economic data...")
        for ent in ["us", "cn", "eu", "uk", "jp", "ru"]:
            for fn in ["macro.json", "financial.json"]:
                fp = ECON / ent / fn
                if fp.exists():
                    self._load_json_indicators(ent, fp)
        self._load_global_indices()
        print(f"  Loaded {len(self.cache)} time-series")

    def get(self, entity: str, indicator: str, month: str) -> float:
        k = (entity, indicator)
        if k not in self.cache:
            return float("nan")
        s = self.cache[k]
        if month in s:
            return s[month]
        dates = sorted(s.keys())
        prior = [d for d in dates if d <= month]
        if prior:
            return s[prior[-1]]
        future = [d for d in dates if d > month]
        if future:
            return s[future[0]]
        return float("nan")

    def get_yoy(self, entity: str, indicator: str, month: str) -> float:
        v = self.get(entity, indicator, month)
        dt = datetime.strptime(month, "%Y-%m") - timedelta(days=365)
        vp = self.get(entity, indicator, dt.strftime("%Y-%m"))
        if math.isnan(v) or math.isnan(vp) or vp == 0:
            return float("nan")
        return (v - vp) / abs(vp) * 100.0

    def get_mom(self, entity: str, indicator: str, month: str) -> float:
        v = self.get(entity, indicator, month)
        dt = datetime.strptime(month, "%Y-%m") - timedelta(days=32)
        vp = self.get(entity, indicator, dt.strftime("%Y-%m"))
        if math.isnan(v) or math.isnan(vp):
            return float("nan")
        return v - vp

    def all_months(self, entity: str, indicator: str) -> list:
        k = (entity, indicator)
        return sorted(self.cache.get(k, {}).keys())


# ─── Feature Definitions ─────────────────────────────────────────────────────
# Each feature: (entity, indicator, transform)
# transform: "level" | "yoy" | "mom"
FEATURES = [
    # 0-15: US core
    ("us", "fed_funds_rate",        "level"),
    ("us", "fed_funds_rate",        "mom"),      # rate change momentum
    ("us", "cpi_yoy",               "level"),
    ("us", "core_cpi_yoy",          "level"),
    ("us", "pce_yoy",               "level"),
    ("us", "unemployment",          "level"),
    ("us", "unemployment",          "mom"),      # labor market momentum
    ("us", "gdp_growth",            "level"),
    ("us", "nonfarm_payrolls_change", "level"),
    ("us", "yield_curve_10y2y",     "level"),
    ("us", "treasury_10y",          "level"),
    ("us", "vix",                   "level"),
    ("us", "credit_spread",         "level"),
    ("us", "sp500",                 "yoy"),      # equity returns yoy%
    ("us", "usd_index",             "level"),
    ("us", "m2_money_supply",       "yoy"),
    ("us", "fed_total_assets",      "yoy"),
    # 17-21: Global
    ("cn", "gdp_growth",            "level"),
    ("cn", "cpi_yoy",               "level"),
    ("eu", "interest_rate",         "level"),
    ("eu", "hicp_all_items",        "level"),
    ("global", "nikkei225",         "yoy"),
]
N_ECON = len(FEATURES)  # 22

# Extra context features appended after state (not normalized by Normalizer)
# 0: hike_streak (0-1)
# 1: cut_streak (0-1)
# 2: hold_streak (0-1)
# 3: partisan (0=Dem, 1=Rep)
# 4: cpi_3m_change (clipped)
# 5: unemp_3m_change (clipped)
# 6: time_normalized (year-2020)/10
N_CONTEXT = 7

# Leader factors
# N_LEADER = 15

TOTAL_DIM = N_ECON + N_CONTEXT + LEADER_FACTOR_DIM  # 22+7+15 = 44


def compute_raw_state(econ: EconomicDataLoader, month: str) -> np.ndarray:
    vec = np.full(N_ECON, float("nan"))
    for i, (ent, ind, tr) in enumerate(FEATURES):
        if tr == "level":
            vec[i] = econ.get(ent, ind, month)
        elif tr == "yoy":
            vec[i] = econ.get_yoy(ent, ind, month)
        elif tr == "mom":
            vec[i] = econ.get_mom(ent, ind, month)
    return vec


class ZScoreNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        self.mean = np.zeros(X.shape[1])
        self.std = np.ones(X.shape[1])
        for j in range(X.shape[1]):
            v = X[:, j][~np.isnan(X[:, j])]
            if len(v) > 1:
                self.mean[j] = float(np.mean(v))
                self.std[j] = max(float(np.std(v)), 1e-8)
            elif len(v) == 1:
                self.mean[j] = float(v[0])

    def transform_vec(self, vec: np.ndarray) -> np.ndarray:
        out = vec.astype(np.float32).copy()
        for j in range(min(len(out), len(self.mean))):
            if math.isnan(out[j]):
                out[j] = 0.0
            else:
                out[j] = (out[j] - self.mean[j]) / self.std[j]
        return out

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


# ─── Leader factors ───────────────────────────────────────────────────────────
CB_KEYS = [
    "inflation_sensitivity_headline_cpi",
    "inflation_2pct_target_commitment",
    "unemployment_sensitivity",
    "data_dependency_weight",
    "gradualism_preference_hiking_phase",
    "gradualism_preference_easing_phase",
    "front_loading_preference_inflation_regime",
    "policy_independence_from_executive",
    "financial_stability_concern",
    "market_crash_response_speed",
    "political_pressure_resistance",
    "forward_guidance_reliability",
    "recession_risk_sensitivity",
    "qe_willingness",
    "communication_clarity",
]

HOS_KEYS = [
    "tariff_as_negotiation_weapon",
    "china_confrontation_intensity",
    "fiscal_stimulus_preference",
    "risk_tolerance",
    "nationalism_score",
    "populism_score",
    "institutional_respect",
    "information_fidelity",
    "political_security_over_economic_growth",
    "state_vs_market_balance_preference",
    "trade_openness_preference",
    "military_action_willingness",
    "alliance_commitment",
    "domestic_political_survival_priority",
    "innovation_technology_priority",
]


def extract_factors(path: Path, ltype: str = "cb") -> np.ndarray:
    try:
        p = json.load(open(path))
    except Exception:
        return np.full(LEADER_FACTOR_DIM, 0.5, dtype=np.float32)
    lu = {v["label"]: float(v["value"])
          for v in p.get("behavior_matrix", {}).get("vectors", [])
          if "label" in v and "value" in v}
    keys = CB_KEYS if ltype == "cb" else HOS_KEYS
    out = []
    for k in keys[:LEADER_FACTOR_DIM]:
        if k in lu:
            out.append(lu[k])
        else:
            partial = [v for lbl, v in lu.items() if k.split("_")[0] in lbl]
            out.append(float(np.mean(partial)) if partial else 0.5)
    arr = np.array(out, dtype=np.float32)
    arr = arr[:LEADER_FACTOR_DIM]
    if len(arr) < LEADER_FACTOR_DIM:
        arr = np.pad(arr, (0, LEADER_FACTOR_DIM - len(arr)), constant_values=0.5)
    return arr


# ─── Dataset Builder ──────────────────────────────────────────────────────────
def build_powell_dataset(econ: EconomicDataLoader):
    print("\n[DATASET] Building Powell dataset (full feature set)...")
    d = json.load(open(DECISIONS / "powell_decisions.json"))
    decisions = d["decisions"]

    powell_fac = extract_factors(PROFILES / "powell.json", "cb")

    # Normalizer: fit on wide date range
    months = [f"{y}-{m:02d}" for y in range(2015, 2027) for m in range(1, 13)]
    raw_arr = np.array([compute_raw_state(econ, mo) for mo in months])
    norm = ZScoreNormalizer()
    norm.fit(raw_arr)
    print(f"  Fitted normalizer on {len(months)} months (2015–2026)")

    X, y_dir, y_mag, y_guid = [], [], [], []
    meta = []
    prev_actions = []  # track action history for inertia features

    for dec in decisions:
        month = dec["date"][:7]
        ec = dec.get("context", {}).get("economic_state", {})
        pc = dec.get("context", {}).get("political_state", {})

        raw = compute_raw_state(econ, month)

        # Override with point-in-time values from the decision record
        # (more accurate than FRED monthly averages)
        overrides = {
            0: ec.get("fed_funds_rate_before"),   # fed_funds_rate level
            2: ec.get("cpi_yoy"),                  # cpi_yoy
            5: ec.get("unemployment"),             # unemployment
            7: ec.get("gdp_growth"),               # gdp_growth
        }
        for feat_idx, val in overrides.items():
            if val is not None:
                raw[feat_idx] = float(val)

        state = norm.transform_vec(raw)

        # Policy inertia
        n = len(prev_actions)
        hike_s = sum(1 for a in prev_actions[-8:] if a == "hike") / max(min(n, 8), 1)
        cut_s  = sum(1 for a in prev_actions[-8:] if a == "cut")  / max(min(n, 8), 1)
        hold_s = sum(1 for a in prev_actions[-4:] if a == "hold") / max(min(n, 4), 1)

        # Political
        year = int(month[:4])
        partisan = 1.0 if year >= 2025 else 0.0

        # Inflation / unemployment trend (3-month)
        m3_ago = (datetime.strptime(month, "%Y-%m") - timedelta(days=92)).strftime("%Y-%m")
        raw3 = compute_raw_state(econ, m3_ago)
        cpi_trend  = float(raw[2] - raw3[2]) if not math.isnan(raw[2]) and not math.isnan(raw3[2]) else 0.0
        unemp_trend = float(raw[5] - raw3[5]) if not math.isnan(raw[5]) and not math.isnan(raw3[5]) else 0.0
        year_norm = (year - 2022) / 4.0

        ctx = np.array([
            hike_s, cut_s, hold_s,
            partisan,
            np.clip(cpi_trend / 2.0, -1.5, 1.5),
            np.clip(unemp_trend / 0.5, -1.5, 1.5),
            year_norm,
        ], dtype=np.float32)

        x = np.concatenate([state, ctx, powell_fac]).astype(np.float32)
        assert len(x) == TOTAL_DIM

        dir_i, mag_i, guid_i = decision_to_targets(dec["decision"])
        X.append(x)
        y_dir.append(dir_i)
        y_mag.append(mag_i)
        y_guid.append(guid_i)

        action_str = DIRECTION[dir_i]
        prev_actions.append(action_str)
        meta.append({
            "date": month,
            "direction": DIRECTION[dir_i],
            "magnitude": MAGNITUDE[mag_i],
            "guidance": GUIDANCE[guid_i],
            "raw_action": dec["decision"].get("action"),
            "magnitude_bp": dec["decision"].get("magnitude_bp", 0),
        })

    X = np.array(X, dtype=np.float32)
    yd = np.array(y_dir, dtype=np.int64)
    ym = np.array(y_mag, dtype=np.int64)
    yg = np.array(y_guid, dtype=np.int64)

    print(f"  {len(X)} samples × {TOTAL_DIM} dims")
    dc = defaultdict(int)
    for di, mi in zip(yd, ym):
        dc[f"{DIRECTION[di]}_{MAGNITUDE[mi]}bp" if DIRECTION[di] != "hold" else "hold"] += 1
    print("  Action distribution:")
    for a, c in sorted(dc.items(), key=lambda x: -x[1]):
        print(f"    {a:18s}: {c}")

    return X, yd, ym, yg, meta, norm


# ─── Multi-task Transformer ───────────────────────────────────────────────────
class MultiTaskDecisionTransformer(nn.Module):
    """
    Each input feature is treated as one token.
    Input: (B, N_feats) -> reshaped to (B, N_feats, 1) -> projected to d_model
    2x TransformerEncoder -> mean pool -> 3 heads
    """

    def __init__(self, n_features: int, d_model: int = 32, nhead: int = 4,
                 n_layers: int = 2, dim_ff: int = 64, dropout: float = 0.0):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Project each scalar feature to d_model
        self.input_proj = nn.Linear(1, d_model)

        # Learned position embedding per feature
        self.pos_embed = nn.Embedding(n_features, d_model)

        # Transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Output heads (multi-task)
        self.direction_head = nn.Linear(d_model, len(DIRECTION))  # 3
        self.magnitude_head = nn.Linear(d_model, len(MAGNITUDE))  # 5
        self.guidance_head  = nn.Linear(d_model, len(GUIDANCE))   # 3

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        x: (B, n_features)
        Returns: (dir_logprob, mag_logprob, guid_logprob), each (B, N_classes)
        """
        B = x.shape[0]
        # (B, n_features, 1) -> (B, n_features, d_model)
        tokens = x.unsqueeze(-1)
        tokens = self.input_proj(tokens)
        # Add position embeddings
        pos = torch.arange(self.n_features, device=x.device)
        tokens = tokens + self.pos_embed(pos).unsqueeze(0)
        # Transformer
        enc = self.transformer(tokens)  # (B, n_features, d_model)
        pooled = enc.mean(dim=1)        # (B, d_model)
        # Heads
        dir_lp  = torch.log_softmax(self.direction_head(pooled),  dim=-1)
        mag_lp  = torch.log_softmax(self.magnitude_head(pooled), dim=-1)
        guid_lp = torch.log_softmax(self.guidance_head(pooled),  dim=-1)
        return dir_lp, mag_lp, guid_lp

    def predict(self, x: torch.Tensor):
        """Return (direction_probs, magnitude_probs, guidance_probs)."""
        d, m, g = self(x)
        return torch.exp(d), torch.exp(m), torch.exp(g)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Training ─────────────────────────────────────────────────────────────────
def train_multitask(
    model: MultiTaskDecisionTransformer,
    X: np.ndarray,
    y_dir: np.ndarray, y_mag: np.ndarray, y_guid: np.ndarray,
    n_epochs: int = 500, lr: float = 5e-3,
    batch_size: int = 8, weight_decay: float = 5e-4,
    verbose: bool = True,
) -> dict:
    model.to(DEVICE)
    X_t  = torch.FloatTensor(X).to(DEVICE)
    yd_t = torch.LongTensor(y_dir).to(DEVICE)
    ym_t = torch.LongTensor(y_mag).to(DEVICE)
    yg_t = torch.LongTensor(y_guid).to(DEVICE)

    ds = TensorDataset(X_t, yd_t, ym_t, yg_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # Class weights for direction (most impactful head)
    dir_counts = np.bincount(y_dir, minlength=len(DIRECTION)).astype(np.float32)
    dir_counts = np.maximum(dir_counts, 1)
    dir_weights = torch.FloatTensor(1.0 / dir_counts * dir_counts.sum() / len(DIRECTION)).to(DEVICE)

    crit_dir  = nn.NLLLoss(weight=dir_weights)
    crit_mag  = nn.NLLLoss()
    crit_guid = nn.NLLLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    history = {"loss": [], "dir_acc": [], "joint_acc": []}
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        total_loss, dir_correct, joint_correct, total = 0.0, 0, 0, 0
        for xb, ydb, ymb, ygb in loader:
            optimizer.zero_grad()
            dlp, mlp, glp = model(xb)
            # Direction loss weighted 2x (most important)
            loss = 2.0 * crit_dir(dlp, ydb) + crit_mag(mlp, ymb) + 0.5 * crit_guid(glp, ygb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(ydb)
            dir_correct += (dlp.argmax(1) == ydb).sum().item()
            joint_correct += ((dlp.argmax(1) == ydb) & (mlp.argmax(1) == ymb)).sum().item()
            total += len(ydb)
        sched.step()
        history["loss"].append(total_loss / total)
        history["dir_acc"].append(dir_correct / total)
        history["joint_acc"].append(joint_correct / total)

        if verbose and (epoch == 0 or (epoch + 1) % 100 == 0):
            print(f"  Epoch {epoch+1:4d}/{n_epochs}: "
                  f"loss={history['loss'][-1]:.4f}, "
                  f"dir={history['dir_acc'][-1]:.3f}, "
                  f"joint={history['joint_acc'][-1]:.3f}, "
                  f"lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.1f}s)")

    return history


# ─── Leave-one-out ────────────────────────────────────────────────────────────
def loo_eval(X, y_dir, y_mag, y_guid, meta,
             n_epochs: int = 600,
             model_kwargs: dict = None) -> dict:
    print(f"\n[LOO] Leave-one-out CV ({len(X)} samples, {n_epochs} epochs/fold)...")
    if model_kwargs is None:
        model_kwargs = {}

    dir_correct, joint_correct = 0, 0
    preds_dir, preds_mag, preds_guid = [], [], []
    confs_dir = []

    for i in range(len(X)):
        mask = [j for j in range(len(X)) if j != i]
        Xtr, ydr, ymr, ygr = X[mask], y_dir[mask], y_mag[mask], y_guid[mask]

        m = MultiTaskDecisionTransformer(n_features=TOTAL_DIM, **model_kwargs).to(DEVICE)
        train_multitask(m, Xtr, ydr, ymr, ygr,
                        n_epochs=n_epochs, lr=5e-3,
                        batch_size=min(8, len(Xtr)),
                        weight_decay=5e-4, verbose=False)

        m.eval()
        with torch.no_grad():
            dp, mp, gp = m.predict(torch.FloatTensor(X[i:i+1]).to(DEVICE))
            dp, mp, gp = dp.cpu().numpy()[0], mp.cpu().numpy()[0], gp.cpu().numpy()[0]

        pd_ = int(np.argmax(dp))
        pm_ = int(np.argmax(mp))
        pg_ = int(np.argmax(gp))

        preds_dir.append(pd_)
        preds_mag.append(pm_)
        preds_guid.append(pg_)
        confs_dir.append(float(dp[pd_]))

        dir_ok   = (pd_ == y_dir[i])
        joint_ok = dir_ok and (pm_ == y_mag[i])
        dir_correct   += int(dir_ok)
        joint_correct += int(joint_ok)

        true_action  = decode_targets(y_dir[i], y_mag[i])
        pred_action  = decode_targets(pd_, pm_)
        status = "OK" if joint_ok else ("dir OK" if dir_ok else "--")
        print(f"  [{i+1:2d}/{len(X)}] {meta[i]['date']}  "
              f"true={true_action:14s}  pred={pred_action:14s}  "
              f"conf_dir={dp[pd_]:.2f}  {status}")

    dir_acc   = dir_correct   / len(X)
    joint_acc = joint_correct / len(X)
    print(f"\n  Direction accuracy: {dir_correct}/{len(X)} = {dir_acc:.1%}")
    print(f"  Joint (dir+mag):    {joint_correct}/{len(X)} = {joint_acc:.1%}")

    return {
        "dir_accuracy": dir_acc,
        "joint_accuracy": joint_acc,
        "dir_correct": dir_correct,
        "joint_correct": joint_correct,
        "total": len(X),
        "pred_dir": preds_dir,
        "pred_mag": preds_mag,
        "pred_guid": preds_guid,
        "conf_dir": confs_dir,
        "true_dir": list(y_dir),
        "true_mag": list(y_mag),
        "true_guid": list(y_guid),
        "true_actions": [decode_targets(d, m) for d, m in zip(y_dir, y_mag)],
        "pred_actions": [decode_targets(d, m) for d, m in zip(preds_dir, preds_mag)],
    }


# ─── Synthetic Data ───────────────────────────────────────────────────────────
def generate_synthetic_cb(econ: EconomicDataLoader, norm: ZScoreNormalizer,
                           n_per: int = 200):
    print(f"\n[SYNTHETIC] Generating synthetic CB data ({n_per}/leader)...")

    months = [m for m in econ.all_months("us", "fed_funds_rate") if m >= "1990-01"]
    if not months:
        return np.array([]), [], [], []

    profiles = {
        "powell":       (PROFILES / "powell.json"),
        "bernanke":     (PROFILES / "bernanke.json"),
        "yellen":       (PROFILES / "yellen.json"),
        "greenspan":    (PROFILES / "greenspan.json"),
        "draghi":       (PROFILES / "draghi.json"),
        "lagarde":      (PROFILES / "lagarde.json"),
        "nabiullina":   (PROFILES / "nabiullina.json"),
        "zhou_xiaochuan": (PROFILES / "zhou_xiaochuan.json"),
    }

    all_X, all_yd, all_ym, all_yg = [], [], [], []

    for name, ppath in profiles.items():
        if not ppath.exists():
            continue
        fac = extract_factors(ppath, "cb")
        p = json.load(open(ppath))
        lu = {v["label"]: v["value"]
              for v in p.get("behavior_matrix", {}).get("vectors", [])
              if "label" in v}

        hawk = float(lu.get("inflation_sensitivity_headline_cpi", 0.7))
        dove = float(lu.get("unemployment_sensitivity", 0.6))
        grad = float(lu.get("gradualism_preference_hiking_phase", 0.65))
        qe_w = float(lu.get("qe_willingness", 0.5))

        for mo in random.choices(months, k=n_per):
            raw = compute_raw_state(econ, mo)
            state = norm.transform_vec(raw)

            ffr   = raw[0] if not math.isnan(raw[0]) else 2.0
            cpi   = raw[2] if not math.isnan(raw[2]) else 2.0
            unemp = raw[5] if not math.isnan(raw[5]) else 5.0
            yc    = raw[9] if not math.isnan(raw[9]) else 0.5
            vix_v = raw[11] if not math.isnan(raw[11]) else 20.0

            infl_press  = max(0.0, cpi - 2.0) * hawk
            unemp_press = max(0.0, unemp - 4.5) * dove
            crisis      = vix_v > 30
            inv         = yc < -0.3
            net = infl_press - unemp_press + random.gauss(0, 0.35)
            if inv: net -= 0.25

            # Direction
            if crisis:
                dir_i = DIR_IDX["cut"]
                mag_i = MAG_IDX[50 if not grad > 0.7 else 25]
                guid_i = GUD_IDX["dovish"]
            elif net > 1.5:
                dir_i = DIR_IDX["hike"]
                if net > 2.5 and not grad > 0.7:
                    mag_i = MAG_IDX[75]
                elif net > 1.8 and not grad > 0.75:
                    mag_i = MAG_IDX[50]
                else:
                    mag_i = MAG_IDX[25]
                guid_i = GUD_IDX["hawkish"]
            elif net > 0.35:
                dir_i = DIR_IDX["hike"]
                mag_i = MAG_IDX[25]
                guid_i = GUD_IDX["hawkish"]
            elif net < -1.5:
                dir_i = DIR_IDX["cut"]
                mag_i = MAG_IDX[50 if not grad > 0.7 else 25]
                guid_i = GUD_IDX["dovish"]
            elif net < -0.35:
                dir_i = DIR_IDX["cut"]
                mag_i = MAG_IDX[25]
                guid_i = GUD_IDX["dovish"]
            else:
                dir_i = DIR_IDX["hold"]
                mag_i = MAG_IDX[0]
                if cpi > 2.5:
                    guid_i = GUD_IDX["hawkish"]
                elif unemp > 5.5:
                    guid_i = GUD_IDX["dovish"]
                else:
                    guid_i = GUD_IDX["neutral"]

            hike_s = random.uniform(0, 0.7)
            cut_s  = random.uniform(0, 0.7)
            hold_s = random.uniform(0, 0.6)
            year   = int(mo[:4])
            ctx = np.array([
                hike_s, cut_s, hold_s,
                random.choice([0.0, 1.0]),
                random.gauss(0, 0.3),
                random.gauss(0, 0.3),
                (year - 2022) / 4.0,
            ], dtype=np.float32)

            x = np.concatenate([state, ctx, fac]).astype(np.float32)
            x = x[:TOTAL_DIM]
            if len(x) < TOTAL_DIM:
                x = np.pad(x, (0, TOTAL_DIM - len(x)))

            all_X.append(x)
            all_yd.append(dir_i)
            all_ym.append(mag_i)
            all_yg.append(guid_i)

        print(f"  {name}: {n_per} samples")

    X  = np.array(all_X,  dtype=np.float32)
    yd = np.array(all_yd, dtype=np.int64)
    ym = np.array(all_ym, dtype=np.int64)
    yg = np.array(all_yg, dtype=np.int64)
    print(f"  Total synthetic: {len(X)}")
    return X, yd, ym, yg


# ─── Visualizations ───────────────────────────────────────────────────────────
def save_visualizations(history, loo, meta, model, X, y_dir, y_mag, y_guid):
    print("\n[VIZ] Generating visualizations...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep = list(range(1, len(history["loss"]) + 1))
    axes[0].plot(ep, history["loss"], "b-", lw=1.5)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss (multi-task)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, [a*100 for a in history["dir_acc"]], "g-", lw=1.5, label="Direction")
    axes[1].plot(ep, [a*100 for a in history["joint_acc"]], "b-", lw=1.5, label="Dir+Mag")
    axes[1].axhline(loo["dir_accuracy"]*100,   c="green", ls="--",
                    label=f"LOO Dir: {loo['dir_accuracy']:.1%}")
    axes[1].axhline(loo["joint_accuracy"]*100, c="blue",  ls="--",
                    label=f"LOO Joint: {loo['joint_accuracy']:.1%}")
    axes[1].axhline(66.7, c="orange", ls=":", label="v1 pilot: 66.7%")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_VIZ / "training_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: training_curve.png")

    # 2. Direction confusion matrix
    dirs = DIRECTION
    nd = len(dirs)
    cm = np.zeros((nd, nd), dtype=int)
    for t, p in zip(loo["true_dir"], loo["pred_dir"]):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(nd)); ax.set_yticks(range(nd))
    ax.set_xticklabels(dirs); ax.set_yticklabels(dirs)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Direction Confusion Matrix\n(LOO Dir Acc: {loo['dir_accuracy']:.1%})")
    for i in range(nd):
        for j in range(nd):
            if cm[i, j] > 0:
                clr = "white" if cm[i, j] > cm.max() * 0.5 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color=clr, fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUT_VIZ / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: confusion_matrix.png")

    # 3. Feature importance
    model.eval(); model.to(DEVICE)
    Xt = torch.FloatTensor(X).to(DEVICE)
    Xt.requires_grad_(True)
    dl, ml, gl = model(Xt)
    (dl.max(1).values.sum() + ml.max(1).values.sum()).backward()
    imp = Xt.grad.abs().mean(0).cpu().detach().numpy()

    feat_names = [f"{e}_{ind}_{tr}" for e, ind, tr in FEATURES]
    feat_names += ["hike_streak", "cut_streak", "hold_streak", "partisan",
                   "cpi_trend", "unemp_trend", "year"]
    feat_names += [f"ldr_{k[:22]}" for k in CB_KEYS[:LEADER_FACTOR_DIM]]
    feat_names = (feat_names + [f"f{i}" for i in range(200)])[:len(imp)]

    top_k = min(20, len(imp))
    ti = np.argsort(imp)[-top_k:][::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, top_k))
    ax.barh(range(top_k), imp[ti][::-1], color=colors[::-1])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([feat_names[i] for i in ti][::-1], fontsize=8)
    ax.set_xlabel("Mean |Input Gradient|")
    ax.set_title("Feature Importance — Powell Decision Transformer v3")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(OUT_VIZ / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: feature_importance.png")

    # 4. Decision heatmap: CPI (x) vs Unemployment (y)
    cpi_vals   = np.linspace(-2.5, 3.5, 35)
    unemp_vals = np.linspace(-2.5, 2.5, 30)
    xm = X.mean(0)
    cpi_i   = 2   # us_cpi_yoy_level
    unemp_i = 5   # us_unemployment_level

    grid_X = []
    for uv in unemp_vals:
        for cv in cpi_vals:
            xg = xm.copy(); xg[cpi_i] = cv; xg[unemp_i] = uv
            grid_X.append(xg)

    with torch.no_grad():
        gt = torch.FloatTensor(np.array(grid_X)).to(DEVICE)
        dp, mp, _ = model.predict(gt)
        dp, mp = dp.cpu().numpy(), mp.cpu().numpy()

    dom_dir = dp.argmax(1)
    dom_mag = mp.argmax(1)
    # Create combined label
    dom_label = np.array([
        0 if dom_dir[k] == DIR_IDX["hold"] else
        (3 + dom_mag[k] - 1) if dom_dir[k] == DIR_IDX["hike"] else
        (7 + dom_mag[k] - 1)
        for k in range(len(dom_dir))
    ])

    label_names = ["hold", "hike_25", "hike_50", "hike_75", "hike_100",
                   "cut_25",  "cut_50",  "cut_75",  "cut_100"]

    dom_grid = dom_label.reshape(len(unemp_vals), len(cpi_vals))
    cmap = plt.colormaps["tab10"].resampled(len(label_names))

    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(dom_grid, cmap=cmap, aspect="auto",
                   extent=[cpi_vals[0], cpi_vals[-1], unemp_vals[0], unemp_vals[-1]],
                   origin="lower", vmin=0, vmax=len(label_names)-1)
    ax.set_xlabel("CPI YoY (normalized σ)")
    ax.set_ylabel("Unemployment (normalized σ)")
    ax.set_title("Powell Decision Map: CPI vs Unemployment\n"
                 "(other features held at mean)")
    from matplotlib.patches import Patch
    used = sorted(np.unique(dom_grid))
    leg = [Patch(facecolor=cmap(i), label=label_names[i]) for i in used if i < len(label_names)]
    ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(1.01, 1),
              fontsize=8, title="Action")
    plt.tight_layout()
    plt.savefig(OUT_VIZ / "decision_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: decision_heatmap.png")

    # 5. Timeline
    dates = [m["date"] for m in meta]

    def a2bp(direction, magnitude):
        bp = MAGNITUDE[magnitude] if MAGNITUDE[magnitude] > 0 else 0
        if DIRECTION[direction] == "cut": return -bp
        if DIRECTION[direction] == "hike": return bp
        return 0

    true_bp = [a2bp(d, m) for d, m in zip(loo["true_dir"], loo["true_mag"])]
    pred_bp = [a2bp(d, m) for d, m in zip(loo["pred_dir"], loo["pred_mag"])]
    xi = np.arange(len(dates))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(xi-0.2, true_bp, 0.38, alpha=0.7, label="Actual", color="steelblue")
    ax.bar(xi+0.2, pred_bp, 0.38, alpha=0.7, label="LOO Predicted", color="coral")
    ax.set_xticks(xi)
    ax.set_xticklabels([d[2:7] for d in dates], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Rate Change (bp)")
    ax.set_title(f"Powell FOMC Decisions: Actual vs LOO Predictions\n"
                 f"Direction LOO: {loo['dir_accuracy']:.1%}  "
                 f"Joint (dir+mag) LOO: {loo['joint_accuracy']:.1%}  "
                 f"| Params: {model.count_parameters():,}")
    ax.axhline(0, c="k", lw=0.5)
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT_VIZ / "decision_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: decision_timeline.png")

    # 6. Confidence distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ok_c  = [c for c, ok in zip(loo["conf_dir"],
             [td==pd for td,pd in zip(loo["true_dir"],loo["pred_dir"])]) if ok]
    ng_c  = [c for c, ok in zip(loo["conf_dir"],
             [td==pd for td,pd in zip(loo["true_dir"],loo["pred_dir"])]) if not ok]
    ax.hist(ok_c, bins=12, alpha=0.7, color="green", label=f"Correct dir ({len(ok_c)})")
    ax.hist(ng_c, bins=12, alpha=0.7, color="red",   label=f"Wrong dir ({len(ng_c)})")
    ax.set_xlabel("Direction Confidence"); ax.set_ylabel("Count")
    ax.set_title("LOO Prediction Confidence Distribution")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_VIZ / "confidence_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: confidence_distribution.png")

    # 7. Plotly interactive
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Training Loss", "Accuracy",
                           "Rate Decisions (bp): Actual vs Predicted",
                           "LOO Direction Confidence"]
        )
        ep = list(range(1, len(history["loss"]) + 1))
        fig.add_trace(go.Scatter(x=ep, y=history["loss"], name="Loss", line=dict(color="blue")), 1, 1)
        fig.add_trace(go.Scatter(x=ep, y=[a*100 for a in history["dir_acc"]],
                                 name="Dir Acc%", line=dict(color="green")), 1, 2)
        fig.add_trace(go.Scatter(x=ep, y=[a*100 for a in history["joint_acc"]],
                                 name="Joint Acc%", line=dict(color="blue")), 1, 2)
        fig.add_hline(y=loo["dir_accuracy"]*100,   line_dash="dash", line_color="green",
                      annotation_text=f"LOO Dir {loo['dir_accuracy']:.1%}", row=1, col=2)
        fig.add_hline(y=loo["joint_accuracy"]*100, line_dash="dash", line_color="blue",
                      annotation_text=f"LOO Joint {loo['joint_accuracy']:.1%}", row=1, col=2)
        fig.add_hline(y=66.7, line_dash="dot", line_color="orange",
                      annotation_text="v1 66.7%", row=1, col=2)

        fig.add_trace(go.Bar(x=dates, y=true_bp, name="Actual",
                             marker_color="steelblue", opacity=0.7), 2, 1)
        fig.add_trace(go.Bar(x=dates, y=pred_bp, name="Predicted",
                             marker_color="coral", opacity=0.7), 2, 1)

        dir_ok_list = [td==pd for td,pd in zip(loo["true_dir"],loo["pred_dir"])]
        bar_clr = ["green" if ok else "red" for ok in dir_ok_list]
        fig.add_trace(go.Bar(
            x=dates, y=loo["conf_dir"], name="Confidence",
            marker_color=bar_clr, opacity=0.8,
            text=[f"true={DIRECTION[t]}<br>pred={DIRECTION[p]}"
                  for t, p in zip(loo["true_dir"], loo["pred_dir"])],
            hovertemplate="%{text}<br>conf=%{y:.2f}",
        ), 2, 2)

        fig.update_layout(
            title=(f"Powell Decision Transformer v3 — "
                   f"Dir LOO: {loo['dir_accuracy']:.1%}  "
                   f"Joint LOO: {loo['joint_accuracy']:.1%}  "
                   f"| {model.count_parameters():,} params | {TOTAL_DIM}D input"),
            height=750, barmode="group"
        )
        fig.write_html(str(OUT_VIZ / "interactive_dashboard.html"))
        print("  Saved: interactive_dashboard.html")
    except Exception as e:
        print(f"  Plotly skipped: {e}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    t0 = time.time()

    # ── Load data ─────────────────────────────────────────────────────────
    econ = EconomicDataLoader()

    # ── Build Powell dataset ──────────────────────────────────────────────
    X, y_dir, y_mag, y_guid, meta, norm = build_powell_dataset(econ)

    print(f"\n  Input dim: {TOTAL_DIM}")
    print(f"  Direction classes: {len(DIRECTION)}")
    print(f"  Magnitude classes: {len(MAGNITUDE)}")

    # ── Model kwargs ──────────────────────────────────────────────────────
    model_kwargs = dict(d_model=32, nhead=4, n_layers=2, dim_ff=64, dropout=0.0)

    # ── LOO Evaluation ────────────────────────────────────────────────────
    loo = loo_eval(X, y_dir, y_mag, y_guid, meta,
                   n_epochs=700, model_kwargs=model_kwargs)

    # ── Train full model ──────────────────────────────────────────────────
    print("\n[TRAIN] Training Powell model (full 33 samples)...")
    powell_model = MultiTaskDecisionTransformer(n_features=TOTAL_DIM, **model_kwargs)
    n_params = powell_model.count_parameters()
    print(f"  Parameters: {n_params:,}")

    history = train_multitask(powell_model, X, y_dir, y_mag, y_guid,
                               n_epochs=800, lr=5e-3, batch_size=8, weight_decay=5e-4)
    print(f"  Final dir acc: {history['dir_acc'][-1]:.1%}, joint acc: {history['joint_acc'][-1]:.1%}")

    # ── Synthetic + generalized model ─────────────────────────────────────
    Xs, yds, yms, ygs = generate_synthetic_cb(econ, norm, n_per=200)

    if len(Xs) > 0:
        if Xs.shape[1] < TOTAL_DIM:
            Xs = np.hstack([Xs, np.zeros((len(Xs), TOTAL_DIM - Xs.shape[1]), dtype=np.float32)])
        elif Xs.shape[1] > TOTAL_DIM:
            Xs = Xs[:, :TOTAL_DIM]

        Xc = np.vstack([X, Xs])
        ydc = np.concatenate([y_dir,  yds])
        ymc = np.concatenate([y_mag,  yms])
        ygc = np.concatenate([y_guid, ygs])

        print(f"\n[TRAIN] Generalized CB model: {len(Xc)} samples")
        gen_model = MultiTaskDecisionTransformer(n_features=TOTAL_DIM, **model_kwargs)
        print(f"  Parameters: {gen_model.count_parameters():,}")
        gen_hist = train_multitask(gen_model, Xc, ydc, ymc, ygc,
                                   n_epochs=300, lr=3e-3, batch_size=32, weight_decay=1e-3)
        print(f"  Generalized: dir={gen_hist['dir_acc'][-1]:.1%}, joint={gen_hist['joint_acc'][-1]:.1%}")

        torch.save({
            "model_state_dict": gen_model.state_dict(),
            "model_class": "MultiTaskDecisionTransformer",
            "model_config": {**model_kwargs, "n_features": TOTAL_DIM},
            "n_params": gen_model.count_parameters(),
            "action_heads": {"direction": DIRECTION, "magnitude": MAGNITUDE, "guidance": GUIDANCE},
            "n_real": int(len(X)), "n_synthetic": int(len(Xs)),
        }, OUT_MODELS / "central_banker_generalized.pt")
        print("  Saved: central_banker_generalized.pt")

    # ── Save Powell model ─────────────────────────────────────────────────
    powell_path = OUT_MODELS / "powell_v2.pt"
    torch.save({
        "model_state_dict": powell_model.state_dict(),
        "model_class": "MultiTaskDecisionTransformer",
        "model_config": {**model_kwargs, "n_features": TOTAL_DIM},
        "n_params": n_params,
        "action_heads": {"direction": DIRECTION, "magnitude": MAGNITUDE, "guidance": GUIDANCE},
        "loo_dir_accuracy": float(loo["dir_accuracy"]),
        "loo_joint_accuracy": float(loo["joint_accuracy"]),
        "train_dir_accuracy": float(history["dir_acc"][-1]),
        "train_joint_accuracy": float(history["joint_acc"][-1]),
        "normalization": norm.to_dict(),
        "state_features": FEATURES,
        "total_input_dim": TOTAL_DIM,
        "metadata": meta,
    }, powell_path)
    print(f"\n[SAVE] {powell_path}")

    # ── Save eval JSON ────────────────────────────────────────────────────
    eval_out = {
        "model_version": "v3_multitask",
        "n_params": n_params,
        "total_input_dim": TOTAL_DIM,
        "n_econ_features": N_ECON,
        "n_context_features": N_CONTEXT,
        "n_leader_features": LEADER_FACTOR_DIM,
        "previous_v1": {"n_params": 17770, "input_dim": 10, "loo_accuracy": 0.667},
        "loo_direction_accuracy": float(loo["dir_accuracy"]),
        "loo_joint_accuracy": float(loo["joint_accuracy"]),
        "loo_dir_correct": int(loo["dir_correct"]),
        "loo_joint_correct": int(loo["joint_correct"]),
        "loo_total": int(loo["total"]),
        "loo_predictions": [
            {
                "date": meta[i]["date"],
                "true_action": loo["true_actions"][i],
                "pred_action": loo["pred_actions"][i],
                "true_dir": DIRECTION[loo["true_dir"][i]],
                "pred_dir": DIRECTION[loo["pred_dir"][i]],
                "dir_confidence": float(loo["conf_dir"][i]),
                "dir_correct": bool(loo["true_dir"][i] == loo["pred_dir"][i]),
                "joint_correct": bool(loo["true_dir"][i] == loo["pred_dir"][i] and
                                      loo["true_mag"][i] == loo["pred_mag"][i]),
            }
            for i in range(len(meta))
        ],
        "timestamp": datetime.now().isoformat(),
    }
    json.dump(eval_out, open(OUT_VIZ / "evaluation_results.json", "w"), indent=2)

    # ── Visualizations ────────────────────────────────────────────────────
    save_visualizations(history, loo, meta, powell_model, X, y_dir, y_mag, y_guid)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"  Time:                  {elapsed:.0f}s")
    print(f"  Parameters:            {n_params:,}")
    print(f"  Input dims:            {TOTAL_DIM} ({N_ECON} econ + {N_CONTEXT} ctx + {LEADER_FACTOR_DIM} leader)")
    print(f"  LOO direction acc:     {loo['dir_accuracy']:.1%} ({loo['dir_correct']}/{loo['total']})")
    print(f"  LOO joint (dir+mag):   {loo['joint_accuracy']:.1%} ({loo['joint_correct']}/{loo['total']})")
    print(f"  v1 pilot LOO:          66.7% (10D, single-task, direction only)")
    print(f"  Direction improvement: {(loo['dir_accuracy']-0.667)*100:+.1f} pp")
    print(f"\nSaved models:")
    for f in sorted(OUT_MODELS.iterdir()):
        if f.suffix == ".pt":
            sz = f.stat().st_size // 1024
            print(f"  {f}  ({sz}KB)")
    print(f"\nVisualization:")
    for f in sorted(OUT_VIZ.iterdir()):
        print(f"  {f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
