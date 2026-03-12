#!/usr/bin/env python3
"""
Build Transformer Decision Function v2 — Opponent-Aware Model
==============================================================
Extends v1 (75.8% direction accuracy, 44D input) by adding opponent/environment
leader parameters to the input vector.

Key hypothesis: Turning points correlate with opponent changes.
  - Trump Term 1 pressure → Powell dovish pivot (2019 cuts)
  - Biden no-pressure → Powell independent hawkish (2022-23 hikes)
  - Trump Term 2 pressure → Powell holds despite cut expectations (2025-26)

New input vector structure (~70D total):
  [economic indicators ~22D]     same as v1 FEATURES
  [context features   ~7D]       same as v1: streaks, partisan, trends, year
  [powell params     ~15D]       same as v1 CB_KEYS
  [president params  ~10D]       NEW: Trump/Biden params, switched by date
  [ECB/Lagarde       ~10D]       NEW: ECB stance parameters
  [recent actions    ~5D]        NEW: derived from economic indicator changes

Architecture: Same MultiTaskDecisionTransformer, just wider input.
Training: Same LOO evaluation.
Compare v2 accuracy vs v1 (75.8%) with focus on turning points.

Save: models/decision_functions/powell_v3_opponent.pt
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
print("TRANSFORMER DECISION FUNCTION BUILDER  v2 (opponent-aware)")
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
DIRECTION = ["cut", "hold", "hike"]       # 3 classes
MAGNITUDE = [0, 25, 50, 75, 100]          # 5 classes (bp)
GUIDANCE  = ["dovish", "neutral", "hawkish"]  # 3 classes

DIR_IDX = {d: i for i, d in enumerate(DIRECTION)}
MAG_IDX = {m: i for i, m in enumerate(MAGNITUDE)}
GUD_IDX = {g: i for i, g in enumerate(GUIDANCE)}


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
        best = min(MAGNITUDE[1:], key=lambda x: abs(x - mag))
        mag_i = MAG_IDX[best]
    elif action == "rate_cut":
        dir_i = DIR_IDX["cut"]
        best = min(MAGNITUDE[1:], key=lambda x: abs(x - mag))
        mag_i = MAG_IDX[best]
    else:
        dir_i = DIR_IDX["hold"]
        mag_i = MAG_IDX[0]

    if fg in GUD_IDX:
        guid_i = GUD_IDX[fg]
    else:
        guid_i = GUD_IDX["neutral"]

    return dir_i, mag_i, guid_i


# ─── 维度定义 ─────────────────────────────────────────────────────────────────
LEADER_FACTOR_DIM = 15    # 鲍威尔行为参数（与v1相同）
PRESIDENT_DIM     = 10    # 总统参数（新增）
ECB_DIM           = 10    # ECB/拉加德参数（新增）
RECENT_ACTION_DIM = 5     # 近期行动编码（新增）

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


# ─── Feature Definitions (same as v1) ────────────────────────────────────────
FEATURES = [
    # 0-16: US core
    ("us", "fed_funds_rate",          "level"),
    ("us", "fed_funds_rate",          "mom"),
    ("us", "cpi_yoy",                 "level"),
    ("us", "core_cpi_yoy",            "level"),
    ("us", "pce_yoy",                 "level"),
    ("us", "unemployment",            "level"),
    ("us", "unemployment",            "mom"),
    ("us", "gdp_growth",              "level"),
    ("us", "nonfarm_payrolls_change", "level"),
    ("us", "yield_curve_10y2y",       "level"),
    ("us", "treasury_10y",            "level"),
    ("us", "vix",                     "level"),
    ("us", "credit_spread",           "level"),
    ("us", "sp500",                   "yoy"),
    ("us", "usd_index",               "level"),
    ("us", "m2_money_supply",         "yoy"),
    ("us", "fed_total_assets",        "yoy"),
    # 17-21: Global
    ("cn", "gdp_growth",              "level"),
    ("cn", "cpi_yoy",                 "level"),
    ("eu", "interest_rate",           "level"),
    ("eu", "hicp_all_items",          "level"),
    ("global", "nikkei225",           "yoy"),
]
N_ECON = len(FEATURES)  # 22

# Context features (same as v1)
N_CONTEXT = 7

# ─── 总统任期映射 ─────────────────────────────────────────────────────────────
# 只考虑当前在任者，忽略选举/过渡期
PRESIDENT_TERMS = [
    # (start_date, end_date, president_name, term_label)
    ("2017-01-20", "2021-01-20", "trump", "trump_t1"),
    ("2021-01-20", "2025-01-20", "biden", "biden"),
    ("2025-01-20", "2099-12-31", "trump", "trump_t2"),
]


def get_president(month: str) -> tuple:
    """
    根据月份返回 (president_name, term_label)。
    只考虑正在执政的总统——忽略选举、民调、政权过渡。
    """
    date_str = month + "-15"
    for start, end, name, label in PRESIDENT_TERMS:
        if start <= date_str < end:
            return name, label
    # 默认
    return "trump", "trump_t1"


# ─── 领导人因子键定义 ─────────────────────────────────────────────────────────

# 鲍威尔行为参数（与v1相同）
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

# 总统参数键（对鲍威尔而言：经济压力、贸易政策、财政立场）
# 从 trump.json 取最相关10个维度
TRUMP_KEYS_FOR_POWELL = [
    "fed_political_pressure_intensity",   # 对美联储的政治压力强度
    "fed_independence_respect",           # 对美联储独立性的尊重（低=压力大）
    "tariff_as_negotiation_weapon",       # 关税作为谈判武器（影响通胀/增长）
    "fiscal_deficit_tolerance",           # 财政赤字容忍度（影响供给侧压力）
    "stock_market_as_approval_metric",    # 以股市为政绩指标（偏好低利率）
    "monetary_policy_preference_easing",  # 偏好宽松货币政策
    "trade_deficit_obsession",            # 贸易赤字执念（影响关税政策）
    "china_confrontation_intensity",      # 对华对抗强度（贸易战→通胀）
    "zero_sum_global_economic_worldview", # 零和经济世界观
    "term1_vs_term2_escalation_differential",  # 第一任与第二任的激进程度差
]

# Biden 不存在独立 profile，使用合理默认值（反向于特朗普压力参数）
# 拜登对美联储独立性尊重度高，无直接压力
BIDEN_DEFAULTS_FOR_POWELL = {
    "fed_political_pressure_intensity":   0.05,  # 几乎不施压
    "fed_independence_respect":           0.92,  # 高度尊重独立性
    "tariff_as_negotiation_weapon":       0.25,  # 部分延续特朗普关税但非武器化
    "fiscal_deficit_tolerance":           0.78,  # 大规模财政刺激（ARPA等）
    "stock_market_as_approval_metric":    0.40,  # 股市不是主要政绩指标
    "monetary_policy_preference_easing":  0.30,  # 不强求宽松
    "trade_deficit_obsession":            0.30,  # 关注但非执念
    "china_confrontation_intensity":      0.72,  # 芯片禁令+联盟策略对抗中国
    "zero_sum_global_economic_worldview": 0.35,  # 更重视多边框架
    "term1_vs_term2_escalation_differential": 0.0,  # 只有一任
}

# ECB/拉加德参数（对鲍威尔：利率差距、政策方向、协调程度）
ECB_KEYS_FOR_POWELL = [
    "fed_coordination_vs_independence",       # ECB与美联储的政策协调程度
    "tightening_pace_once_started",           # 加息节奏（影响全球政策空间）
    "inflation_threshold_for_rate_cut_initiation",  # ECB开始降息的通胀门槛
    "late_inflation_response_2021_2022",      # ECB落后于曲线的程度
    "easing_cycle_2024_independent_from_fed", # 2024年ECB独立降息的意愿
    "eu_us_tariff_response_framework",        # 欧美贸易摩擦应对（对美联储影响）
    "geopolitical_shock_response_2022_energy",# 地缘政治冲击应对（欧洲能源危机）
    "forward_guidance_abandonment_2022",      # 放弃前瞻指引的速度（全球政策转变）
    "2025_2026_rate_cutting_pace",            # 2025-26降息节奏（全球政策协调信号）
    "council_coalition_management",           # 委员会管理（ECB内部分歧水平）
]


def get_leader_lookup(path: Path) -> dict:
    """从 profile JSON 提取所有 behavior_matrix.vectors 为 {label: value} 字典。"""
    try:
        p = json.load(open(path))
    except Exception:
        return {}
    return {
        v["label"]: float(v["value"])
        for v in p.get("behavior_matrix", {}).get("vectors", [])
        if "label" in v and "value" in v
    }


def extract_keys(lookup: dict, keys: list, fallback: float = 0.5,
                 fallback_dict: dict = None) -> np.ndarray:
    """
    从 lookup 字典中按 keys 列表提取值。
    找不到时先用 fallback_dict，再用 fallback 默认值。
    """
    out = []
    for k in keys:
        if k in lookup:
            out.append(float(lookup[k]))
        elif fallback_dict and k in fallback_dict:
            out.append(float(fallback_dict[k]))
        else:
            # 尝试前缀匹配（部分词匹配）
            partial = [v for lbl, v in lookup.items() if k.split("_")[0] in lbl]
            out.append(float(np.mean(partial)) if partial else fallback)
    return np.array(out, dtype=np.float32)


def extract_powell_factors(path: Path) -> np.ndarray:
    """提取鲍威尔行为向量（15维，与v1相同）。"""
    lu = get_leader_lookup(path)
    arr = extract_keys(lu, CB_KEYS[:LEADER_FACTOR_DIM])
    if len(arr) < LEADER_FACTOR_DIM:
        arr = np.pad(arr, (0, LEADER_FACTOR_DIM - len(arr)), constant_values=0.5)
    return arr[:LEADER_FACTOR_DIM]


def extract_president_factors(month: str,
                               trump_lookup: dict,
                               biden_defaults: dict) -> np.ndarray:
    """
    根据月份决定使用哪位总统的参数（10维）。
    Trump Term 1/2: 从 trump.json 提取，Term2 调整部分参数
    Biden: 使用合理默认值（无独立 profile）
    """
    pres_name, term_label = get_president(month)

    if pres_name == "trump":
        arr = extract_keys(trump_lookup, TRUMP_KEYS_FOR_POWELL)
        # 特朗普第二任期：对美联储压力更大，激进程度更高
        if term_label == "trump_t2":
            # fed_political_pressure_intensity 在第二任期更高（无连任约束）
            arr[0] = min(1.0, arr[0] * 1.05)
            # liberation_day 关税（2025.4）— 贸易冲击更大
            arr[2] = min(1.0, arr[2] * 1.08)  # tariff_as_negotiation_weapon
        return arr
    else:
        # Biden — 使用硬编码默认值
        arr = extract_keys({}, TRUMP_KEYS_FOR_POWELL,
                           fallback_dict=biden_defaults)
        return arr


def extract_ecb_factors(ecb_lookup: dict) -> np.ndarray:
    """提取 ECB/拉加德参数（10维）。"""
    return extract_keys(ecb_lookup, ECB_KEYS_FOR_POWELL)


def compute_recent_actions(econ: "EconomicDataLoader", month: str) -> np.ndarray:
    """
    编码近期行动（5维），从经济指标变化中推导，无需单独行动数据。
    用FOMC会议前3个月的指标变化来代理总统政策动向：
      0: trade_balance_change  — 贸易差额变化（反映关税/贸易政策效果）
      1: fiscal_impulse        — 财政冲击（政府支出变化代理）
      2: tariff_inflation_pass — 核心CPI-PCE差值变化（关税传导到通胀）
      3: financial_cond_change — 金融条件收紧程度变化（VIX + 利差）
      4: dollar_strength_mom   — 美元指数3个月变化（反映贸易政策/避险）
    """
    m3_ago = (datetime.strptime(month, "%Y-%m") - timedelta(days=92)).strftime("%Y-%m")

    # 贸易差额变化（用 M2 yoy 变化代理财政宽松/收紧）
    m2_now  = econ.get_yoy("us", "m2_money_supply", month)
    m2_ago  = econ.get_yoy("us", "m2_money_supply", m3_ago)
    fiscal_impulse = (m2_now - m2_ago) if not (math.isnan(m2_now) or math.isnan(m2_ago)) else 0.0

    # 关税通胀传导：core_cpi - cpi 差值变化（关税会先打进 core）
    core_now = econ.get("us", "core_cpi_yoy", month)
    cpi_now  = econ.get("us", "cpi_yoy",      month)
    core_ago = econ.get("us", "core_cpi_yoy", m3_ago)
    cpi_ago  = econ.get("us", "cpi_yoy",      m3_ago)
    if not any(math.isnan(v) for v in [core_now, cpi_now, core_ago, cpi_ago]):
        tariff_pass = (core_now - cpi_now) - (core_ago - cpi_ago)
    else:
        tariff_pass = 0.0

    # 金融条件变化：VIX 变化
    vix_now = econ.get("us", "vix", month)
    vix_ago = econ.get("us", "vix", m3_ago)
    fin_cond_change = (vix_now - vix_ago) if not (math.isnan(vix_now) or math.isnan(vix_ago)) else 0.0

    # 美元指数3个月变化
    dxy_now = econ.get("us", "usd_index", month)
    dxy_ago = econ.get("us", "usd_index", m3_ago)
    dollar_mom = (dxy_now - dxy_ago) if not (math.isnan(dxy_now) or math.isnan(dxy_ago)) else 0.0

    # 信用利差变化
    cs_now = econ.get("us", "credit_spread", month)
    cs_ago = econ.get("us", "credit_spread", m3_ago)
    credit_change = (cs_now - cs_ago) if not (math.isnan(cs_now) or math.isnan(cs_ago)) else 0.0

    # 归一化到 [-1, 1] 范围
    vec = np.array([
        np.clip(fiscal_impulse  / 5.0,  -1.5, 1.5),   # M2 变化 ±5%
        np.clip(tariff_pass     / 1.0,  -1.5, 1.5),   # CPI差 ±1%
        np.clip(fin_cond_change / 10.0, -1.5, 1.5),   # VIX ±10点
        np.clip(dollar_mom      / 5.0,  -1.5, 1.5),   # DXY ±5点
        np.clip(credit_change   / 0.5,  -1.5, 1.5),   # 利差 ±0.5%
    ], dtype=np.float32)

    return vec


# ─── Total dimensions ─────────────────────────────────────────────────────────
N_ECON_FEATS = N_ECON
N_CTX_FEATS  = N_CONTEXT

# v2 新增维度
TOTAL_DIM_V2 = (N_ECON + N_CONTEXT + LEADER_FACTOR_DIM +
                PRESIDENT_DIM + ECB_DIM + RECENT_ACTION_DIM)
# 22 + 7 + 15 + 10 + 10 + 5 = 69

print(f"\n[DIM] v1: {N_ECON + N_CONTEXT + LEADER_FACTOR_DIM}D  →  "
      f"v2: {TOTAL_DIM_V2}D  "
      f"(+{PRESIDENT_DIM}D president, +{ECB_DIM}D ECB, +{RECENT_ACTION_DIM}D actions)")


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


# ─── Dataset Builder ──────────────────────────────────────────────────────────
def build_powell_dataset_v2(econ: EconomicDataLoader):
    print("\n[DATASET] Building Powell v2 dataset (opponent-aware)...")
    d = json.load(open(DECISIONS / "powell_decisions.json"))
    decisions = d["decisions"]

    # 加载领导人 profiles
    powell_fac = extract_powell_factors(PROFILES / "powell.json")
    trump_lookup = get_leader_lookup(PROFILES / "trump.json")
    ecb_lookup   = get_leader_lookup(PROFILES / "lagarde.json")

    print(f"  Powell: {LEADER_FACTOR_DIM}D factor loaded")
    print(f"  Trump: {len(trump_lookup)} behavior vectors loaded")
    print(f"  Lagarde (ECB): {len(ecb_lookup)} behavior vectors loaded")
    print(f"  Biden: using hardcoded defaults (no profile file)")

    # Normalizer: fit on wide date range (same as v1)
    months = [f"{y}-{m:02d}" for y in range(2015, 2027) for m in range(1, 13)]
    raw_arr = np.array([compute_raw_state(econ, mo) for mo in months])
    norm = ZScoreNormalizer()
    norm.fit(raw_arr)
    print(f"  Fitted normalizer on {len(months)} months (2015-2026)")

    # ECB factors (constant throughout — Lagarde has been ECB president since Nov 2019)
    # For 2022 decisions Lagarde is in office for all Powell decisions in our dataset
    ecb_fac = extract_ecb_factors(ecb_lookup)

    X, y_dir, y_mag, y_guid = [], [], [], []
    meta = []
    prev_actions = []

    # 跟踪每个总统任期下的预测（用于分析）
    president_decisions = defaultdict(list)

    for dec in decisions:
        month = dec["date"][:7]
        ec = dec.get("context", {}).get("economic_state", {})
        pc = dec.get("context", {}).get("political_state", {})

        raw = compute_raw_state(econ, month)

        # Override with point-in-time values from the decision record
        overrides = {
            0: ec.get("fed_funds_rate_before"),
            2: ec.get("cpi_yoy"),
            5: ec.get("unemployment"),
            7: ec.get("gdp_growth"),
        }
        for feat_idx, val in overrides.items():
            if val is not None:
                raw[feat_idx] = float(val)

        state = norm.transform_vec(raw)

        # ── Context features (v1 部分) ────────────────────────────────────
        n = len(prev_actions)
        hike_s = sum(1 for a in prev_actions[-8:] if a == "hike") / max(min(n, 8), 1)
        cut_s  = sum(1 for a in prev_actions[-8:] if a == "cut")  / max(min(n, 8), 1)
        hold_s = sum(1 for a in prev_actions[-4:] if a == "hold") / max(min(n, 4), 1)

        year = int(month[:4])
        # partisan: 0=Democrat, 1=Republican（v1 用了简化版本，v2 改为精确总统映射）
        pres_name, term_label = get_president(month)
        partisan = 1.0 if pres_name == "trump" else 0.0

        m3_ago = (datetime.strptime(month, "%Y-%m") - timedelta(days=92)).strftime("%Y-%m")
        raw3 = compute_raw_state(econ, m3_ago)
        cpi_trend  = float(raw[2] - raw3[2]) if not math.isnan(raw[2]) and not math.isnan(raw3[2]) else 0.0
        unemp_trend = float(raw[5] - raw3[5]) if not math.isnan(raw[5]) and not math.isnan(raw3[5]) else 0.0
        year_norm = (year - 2022) / 4.0

        ctx = np.array([
            hike_s, cut_s, hold_s,
            partisan,
            np.clip(cpi_trend  / 2.0, -1.5, 1.5),
            np.clip(unemp_trend / 0.5, -1.5, 1.5),
            year_norm,
        ], dtype=np.float32)

        # ── 总统参数（新增，10维） ────────────────────────────────────────
        pres_fac = extract_president_factors(month, trump_lookup, BIDEN_DEFAULTS_FOR_POWELL)

        # ── 近期行动编码（新增，5维） ──────────────────────────────────────
        recent_fac = compute_recent_actions(econ, month)

        # ── 组合完整输入向量 ──────────────────────────────────────────────
        x = np.concatenate([
            state,       # 22D: 经济指标（归一化）
            ctx,         # 7D:  政策惯性 + 政治 + 趋势
            powell_fac,  # 15D: 鲍威尔行为参数
            pres_fac,    # 10D: 总统参数（按日期切换）
            ecb_fac,     # 10D: ECB/拉加德参数
            recent_fac,  # 5D:  近期行动编码
        ]).astype(np.float32)

        assert len(x) == TOTAL_DIM_V2, f"Dim mismatch: {len(x)} != {TOTAL_DIM_V2}"

        dir_i, mag_i, guid_i = decision_to_targets(dec["decision"])
        X.append(x)
        y_dir.append(dir_i)
        y_mag.append(mag_i)
        y_guid.append(guid_i)

        action_str = DIRECTION[dir_i]
        prev_actions.append(action_str)

        meta.append({
            "date":        month,
            "direction":   DIRECTION[dir_i],
            "magnitude":   MAGNITUDE[mag_i],
            "guidance":    GUIDANCE[guid_i],
            "raw_action":  dec["decision"].get("action"),
            "magnitude_bp": dec["decision"].get("magnitude_bp", 0),
            "president":   pres_name,
            "term":        term_label,
        })
        president_decisions[term_label].append(action_str)

    X  = np.array(X,     dtype=np.float32)
    yd = np.array(y_dir, dtype=np.int64)
    ym = np.array(y_mag, dtype=np.int64)
    yg = np.array(y_guid, dtype=np.int64)

    print(f"  {len(X)} samples x {TOTAL_DIM_V2} dims")
    print(f"  Input breakdown: {N_ECON}D econ + {N_CONTEXT}D ctx + "
          f"{LEADER_FACTOR_DIM}D powell + {PRESIDENT_DIM}D president + "
          f"{ECB_DIM}D ECB + {RECENT_ACTION_DIM}D actions")

    dc = defaultdict(int)
    for di, mi in zip(yd, ym):
        dc[f"{DIRECTION[di]}_{MAGNITUDE[mi]}bp" if DIRECTION[di] != "hold" else "hold"] += 1
    print("  Action distribution:")
    for a, c in sorted(dc.items(), key=lambda x: -x[1]):
        print(f"    {a:18s}: {c}")

    print("  Decisions by president:")
    for term, acts in president_decisions.items():
        print(f"    {term}: {len(acts)} decisions  ({dict(defaultdict(int, {a: acts.count(a) for a in set(acts)}))})")

    return X, yd, ym, yg, meta, norm


# ─── Multi-task Transformer (same architecture, wider input) ──────────────────
class MultiTaskDecisionTransformer(nn.Module):
    """
    Each input feature is treated as one token.
    Input: (B, n_features) -> (B, n_features, 1) -> projected to d_model
    2x TransformerEncoder -> mean pool -> 3 heads
    v2: wider input (69D vs 44D), same architecture
    """

    def __init__(self, n_features: int, d_model: int = 32, nhead: int = 4,
                 n_layers: int = 2, dim_ff: int = 64, dropout: float = 0.0):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        self.input_proj = nn.Linear(1, d_model)
        self.pos_embed  = nn.Embedding(n_features, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.direction_head = nn.Linear(d_model, len(DIRECTION))
        self.magnitude_head = nn.Linear(d_model, len(MAGNITUDE))
        self.guidance_head  = nn.Linear(d_model, len(GUIDANCE))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        tokens = x.unsqueeze(-1)
        tokens = self.input_proj(tokens)
        pos    = torch.arange(self.n_features, device=x.device)
        tokens = tokens + self.pos_embed(pos).unsqueeze(0)
        enc    = self.transformer(tokens)
        pooled = enc.mean(dim=1)
        dir_lp  = torch.log_softmax(self.direction_head(pooled),  dim=-1)
        mag_lp  = torch.log_softmax(self.magnitude_head(pooled), dim=-1)
        guid_lp = torch.log_softmax(self.guidance_head(pooled),  dim=-1)
        return dir_lp, mag_lp, guid_lp

    def predict(self, x: torch.Tensor):
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

    ds     = TensorDataset(X_t, yd_t, ym_t, yg_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    dir_counts  = np.bincount(y_dir, minlength=len(DIRECTION)).astype(np.float32)
    dir_counts  = np.maximum(dir_counts, 1)
    dir_weights = torch.FloatTensor(1.0 / dir_counts * dir_counts.sum() / len(DIRECTION)).to(DEVICE)

    crit_dir  = nn.NLLLoss(weight=dir_weights)
    crit_mag  = nn.NLLLoss()
    crit_guid = nn.NLLLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched     = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    history = {"loss": [], "dir_acc": [], "joint_acc": []}
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        total_loss, dir_correct, joint_correct, total = 0.0, 0, 0, 0
        for xb, ydb, ymb, ygb in loader:
            optimizer.zero_grad()
            dlp, mlp, glp = model(xb)
            loss = 2.0 * crit_dir(dlp, ydb) + crit_mag(mlp, ymb) + 0.5 * crit_guid(glp, ygb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss   += loss.item() * len(ydb)
            dir_correct  += (dlp.argmax(1) == ydb).sum().item()
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


# ─── Leave-one-out Evaluation ─────────────────────────────────────────────────
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

        m = MultiTaskDecisionTransformer(n_features=TOTAL_DIM_V2, **model_kwargs).to(DEVICE)
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

        true_action = decode_targets(y_dir[i], y_mag[i])
        pred_action = decode_targets(pd_, pm_)
        pres_info   = meta[i].get("term", "?")
        status = "OK" if joint_ok else ("dir OK" if dir_ok else "--")
        print(f"  [{i+1:2d}/{len(X)}] {meta[i]['date']}  [{pres_info:10s}]  "
              f"true={true_action:14s}  pred={pred_action:14s}  "
              f"conf_dir={dp[pd_]:.2f}  {status}")

    dir_acc   = dir_correct   / len(X)
    joint_acc = joint_correct / len(X)
    print(f"\n  Direction accuracy: {dir_correct}/{len(X)} = {dir_acc:.1%}")
    print(f"  Joint (dir+mag):    {joint_correct}/{len(X)} = {joint_acc:.1%}")

    return {
        "dir_accuracy":   dir_acc,
        "joint_accuracy": joint_acc,
        "dir_correct":    dir_correct,
        "joint_correct":  joint_correct,
        "total":          len(X),
        "pred_dir":       preds_dir,
        "pred_mag":       preds_mag,
        "pred_guid":      preds_guid,
        "conf_dir":       confs_dir,
        "true_dir":       list(y_dir),
        "true_mag":       list(y_mag),
        "true_guid":      list(y_guid),
        "true_actions":   [decode_targets(d, m) for d, m in zip(y_dir, y_mag)],
        "pred_actions":   [decode_targets(d, m) for d, m in zip(preds_dir, preds_mag)],
    }


# ─── Turning Point Analysis ───────────────────────────────────────────────────
def identify_turning_points(meta: list, y_dir: np.ndarray) -> list:
    """
    识别决策序列中的转折点：方向改变的时刻。
    返回转折点索引列表。
    """
    turning = []
    for i in range(1, len(y_dir)):
        prev_d = y_dir[i - 1]
        curr_d = y_dir[i]
        # 方向从 hold/hike 转为 cut，或从 hold/cut 转为 hike
        if prev_d != curr_d:
            if not (prev_d == DIR_IDX["hold"] and curr_d == DIR_IDX["hold"]):
                turning.append(i)
    return turning


def analyze_turning_points(loo: dict, meta: list, y_dir: np.ndarray,
                            v1_preds: list = None) -> dict:
    """
    专门分析转折点准确率，对比 v1 vs v2。
    """
    turning_indices = identify_turning_points(meta, y_dir)
    print(f"\n[TURNING POINTS] Identified {len(turning_indices)} turning points:")

    tp_results = {"v2_correct": 0, "v1_correct": 0, "total": len(turning_indices),
                  "details": []}

    for idx in turning_indices:
        true_dir  = y_dir[idx]
        pred_dir  = loo["pred_dir"][idx]
        v2_ok     = (pred_dir == true_dir)

        v1_ok = None
        if v1_preds:
            v1_ok = (v1_preds[idx] == true_dir)

        tp_results["v2_correct"] += int(bool(v2_ok))
        if v1_ok is not None:
            tp_results["v1_correct"] += int(bool(v1_ok))

        detail = {
            "date":     meta[idx]["date"],
            "president": meta[idx].get("term", "?"),
            "true":     DIRECTION[true_dir],
            "v2_pred":  DIRECTION[pred_dir],
            "v2_ok":    bool(v2_ok),
        }
        if v1_preds:
            detail["v1_pred"] = DIRECTION[v1_preds[idx]]
            detail["v1_ok"]   = bool(v1_ok)

        tp_results["details"].append(detail)

        v1_str = f"  v1={DIRECTION[v1_preds[idx]]}({'OK' if v1_ok else '--'})" if v1_preds else ""
        print(f"  {meta[idx]['date']} [{meta[idx].get('term','?'):10s}]  "
              f"true={DIRECTION[true_dir]:6s}  v2={'OK' if v2_ok else '--':4s}{v1_str}")

    if len(turning_indices) > 0:
        v2_tp_acc = tp_results["v2_correct"] / len(turning_indices)
        print(f"\n  Turning point accuracy — v2: {tp_results['v2_correct']}/{len(turning_indices)} = {v2_tp_acc:.1%}", end="")
        if v1_preds:
            v1_tp_acc = tp_results["v1_correct"] / len(turning_indices)
            print(f"  vs v1: {tp_results['v1_correct']}/{len(turning_indices)} = {v1_tp_acc:.1%}", end="")
            tp_results["v1_tp_accuracy"] = v1_tp_acc
        print()
        tp_results["v2_tp_accuracy"] = v2_tp_acc

    return tp_results


# ─── Visualizations ───────────────────────────────────────────────────────────
def save_visualizations_v2(history, loo, meta, model,
                            X, y_dir, y_mag, y_guid,
                            tp_results, v1_loo_dir_acc=0.7576):
    print("\n[VIZ] Generating v2 visualizations...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep = list(range(1, len(history["loss"]) + 1))
    axes[0].plot(ep, history["loss"], "b-", lw=1.5)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss (v2 opponent-aware)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, [a*100 for a in history["dir_acc"]],  "g-", lw=1.5, label="Direction")
    axes[1].plot(ep, [a*100 for a in history["joint_acc"]], "b-", lw=1.5, label="Dir+Mag")
    axes[1].axhline(loo["dir_accuracy"]*100,   c="green",  ls="--",
                    label=f"LOO Dir v2: {loo['dir_accuracy']:.1%}")
    axes[1].axhline(loo["joint_accuracy"]*100, c="blue",   ls="--",
                    label=f"LOO Joint v2: {loo['joint_accuracy']:.1%}")
    axes[1].axhline(v1_loo_dir_acc*100,        c="orange", ls=":",
                    label=f"v1 LOO Dir: {v1_loo_dir_acc:.1%}")
    axes[1].axhline(66.7, c="red", ls=":", lw=0.8, label="pilot: 66.7%")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy")
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_VIZ / "v2_training_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: v2_training_curve.png")

    # 2. Confusion matrix
    dirs = DIRECTION
    nd   = len(dirs)
    cm   = np.zeros((nd, nd), dtype=int)
    for t, p in zip(loo["true_dir"], loo["pred_dir"]):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(nd)); ax.set_yticks(range(nd))
    ax.set_xticklabels(dirs); ax.set_yticklabels(dirs)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Direction Confusion Matrix — v2\n(LOO Dir Acc: {loo['dir_accuracy']:.1%})")
    for i in range(nd):
        for j in range(nd):
            if cm[i, j] > 0:
                clr = "white" if cm[i, j] > cm.max() * 0.5 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color=clr, fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUT_VIZ / "v2_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: v2_confusion_matrix.png")

    # 3. Feature importance (gradient-based)
    model.eval(); model.to(DEVICE)
    Xt = torch.FloatTensor(X).to(DEVICE)
    Xt.requires_grad_(True)
    dl, ml, gl = model(Xt)
    (dl.max(1).values.sum() + ml.max(1).values.sum()).backward()
    imp = Xt.grad.abs().mean(0).cpu().detach().numpy()

    # 构建特征名称
    feat_names  = [f"{e}_{ind}_{tr}" for e, ind, tr in FEATURES]
    feat_names += ["hike_streak", "cut_streak", "hold_streak", "partisan",
                   "cpi_trend", "unemp_trend", "year"]
    feat_names += [f"powell_{k[:18]}" for k in CB_KEYS[:LEADER_FACTOR_DIM]]
    feat_names += [f"pres_{k[:20]}"   for k in TRUMP_KEYS_FOR_POWELL[:PRESIDENT_DIM]]
    feat_names += [f"ecb_{k[:20]}"    for k in ECB_KEYS_FOR_POWELL[:ECB_DIM]]
    feat_names += ["recent_fiscal", "recent_tariff_cpi", "recent_vix",
                   "recent_dollar", "recent_credit"]
    feat_names = (feat_names + [f"f{i}" for i in range(200)])[:len(imp)]

    top_k = min(25, len(imp))
    ti = np.argsort(imp)[-top_k:][::-1]
    fig, ax = plt.subplots(figsize=(11, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, top_k))
    ax.barh(range(top_k), imp[ti][::-1], color=colors[::-1])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([feat_names[i] for i in ti][::-1], fontsize=7)
    ax.set_xlabel("Mean |Input Gradient|")
    ax.set_title("Feature Importance — Powell v2 Opponent-Aware Transformer")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(OUT_VIZ / "v2_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: v2_feature_importance.png")

    # 4. Decision timeline with president background coloring
    dates   = [m["date"] for m in meta]
    pres_bg = [m.get("term", "?") for m in meta]

    def a2bp(direction, magnitude):
        bp = MAGNITUDE[magnitude] if MAGNITUDE[magnitude] > 0 else 0
        if DIRECTION[direction] == "cut":  return -bp
        if DIRECTION[direction] == "hike": return  bp
        return 0

    true_bp = [a2bp(d, m) for d, m in zip(loo["true_dir"],  loo["true_mag"])]
    pred_bp = [a2bp(d, m) for d, m in zip(loo["pred_dir"],  loo["pred_mag"])]
    xi = np.arange(len(dates))

    fig, ax = plt.subplots(figsize=(16, 5))

    # 总统任期背景色
    pres_colors = {"trump_t1": "#fff0f0", "biden": "#f0f0ff", "trump_t2": "#fff0f0"}
    pres_edge   = {"trump_t1": "#cc0000", "biden": "#0000cc", "trump_t2": "#cc0000"}
    prev_pres = None; band_start = 0
    for k, p in enumerate(pres_bg + ["END"]):
        if p != prev_pres and prev_pres is not None:
            c = pres_colors.get(prev_pres, "#ffffff")
            ax.axvspan(band_start - 0.5, k - 0.5, alpha=0.25, color=c, zorder=0)
        if p != prev_pres:
            band_start = k
            prev_pres = p

    ax.bar(xi - 0.2, true_bp, 0.38, alpha=0.8, label="Actual",        color="steelblue")
    ax.bar(xi + 0.2, pred_bp, 0.38, alpha=0.8, label="LOO Predicted", color="coral")

    # 标注转折点
    for idx in identify_turning_points(meta, np.array(y_dir)):
        ax.axvline(idx - 0.5, color="purple", lw=1.2, ls="--", alpha=0.6)

    ax.set_xticks(xi)
    ax.set_xticklabels([d[2:7] for d in dates], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Rate Change (bp)")
    ax.set_title(f"Powell FOMC Decisions v2: Actual vs LOO Predictions\n"
                 f"Direction LOO: {loo['dir_accuracy']:.1%}  (v1: {v1_loo_dir_acc:.1%})  "
                 f"| Turning Points: {tp_results.get('v2_correct',0)}/{tp_results.get('total',0)} correct  "
                 f"| Params: {model.count_parameters():,}")
    ax.axhline(0, c="k", lw=0.5)

    legend_handles = [
        mpatches.Patch(color="steelblue", alpha=0.8, label="Actual"),
        mpatches.Patch(color="coral",     alpha=0.8, label="LOO Predicted"),
        mpatches.Patch(color="#ffcccc",   alpha=0.6, label="Trump"),
        mpatches.Patch(color="#ccccff",   alpha=0.6, label="Biden"),
        plt.Line2D([0], [0], color="purple", ls="--", lw=1.2, label="Turning Point"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT_VIZ / "v2_decision_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: v2_decision_timeline.png")

    # 5. Opponent parameter effect visualization
    # 展示相同经济条件下，特朗普 vs 拜登参数如何改变决策概率
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 使用数据集均值作为基础
    x_base   = X.mean(0).copy()
    cpi_vals = np.linspace(-2.5, 3.5, 40)

    # 特朗普参数所在维度的 offset
    pres_offset = N_ECON + N_CONTEXT + LEADER_FACTOR_DIM

    # 创建两套：特朗普T2 vs 拜登
    def make_grid(x_base, pres_fac_vec, cpi_vals):
        grid = []
        for cv in cpi_vals:
            x = x_base.copy()
            x[2] = cv   # cpi_yoy normalized
            x[pres_offset:pres_offset + PRESIDENT_DIM] = pres_fac_vec
            grid.append(x)
        return np.array(grid, dtype=np.float32)

    # 提取保存的 trump T2 和 biden 参数
    trump_lu = get_leader_lookup(PROFILES / "trump.json")
    trump_fac = extract_president_factors("2025-06", trump_lu, BIDEN_DEFAULTS_FOR_POWELL)
    biden_fac = extract_president_factors("2023-06", trump_lu, BIDEN_DEFAULTS_FOR_POWELL)

    model.eval()
    with torch.no_grad():
        g_trump = torch.FloatTensor(make_grid(x_base, trump_fac, cpi_vals)).to(DEVICE)
        g_biden = torch.FloatTensor(make_grid(x_base, biden_fac, cpi_vals)).to(DEVICE)
        dp_trump, _, _ = model.predict(g_trump)
        dp_biden, _, _ = model.predict(g_biden)
        dp_trump = dp_trump.cpu().numpy()
        dp_biden = dp_biden.cpu().numpy()

    for ax, dp, title in [(axes[0], dp_trump, "Trump T2 President"),
                          (axes[1], dp_biden, "Biden President")]:
        ax.fill_between(cpi_vals, 0, dp[:, DIR_IDX["cut"]],  alpha=0.6, color="blue",   label="cut")
        ax.fill_between(cpi_vals, dp[:, DIR_IDX["cut"]],
                        dp[:, DIR_IDX["cut"]] + dp[:, DIR_IDX["hold"]], alpha=0.6, color="gray", label="hold")
        ax.fill_between(cpi_vals,
                        dp[:, DIR_IDX["cut"]] + dp[:, DIR_IDX["hold"]], 1.0, alpha=0.6, color="red", label="hike")
        ax.set_xlabel("CPI YoY (normalized σ)")
        ax.set_ylabel("Probability")
        ax.set_title(f"Decision Probability vs CPI\n({title})")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Opponent Effect: Same CPI, Different President → Different Rate Decision",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_VIZ / "v2_opponent_effect.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: v2_opponent_effect.png")

    # 6. Plotly interactive
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Training Loss", "Accuracy",
                           "Rate Decisions (bp): Actual vs Predicted",
                           "LOO Direction Confidence by President"]
        )
        ep_ = list(range(1, len(history["loss"]) + 1))
        fig.add_trace(go.Scatter(x=ep_, y=history["loss"],
                                 name="Loss", line=dict(color="blue")), 1, 1)
        fig.add_trace(go.Scatter(x=ep_, y=[a*100 for a in history["dir_acc"]],
                                 name="Dir Acc%",   line=dict(color="green")), 1, 2)
        fig.add_trace(go.Scatter(x=ep_, y=[a*100 for a in history["joint_acc"]],
                                 name="Joint Acc%", line=dict(color="blue")),  1, 2)
        fig.add_hline(y=loo["dir_accuracy"]*100, line_dash="dash", line_color="green",
                      annotation_text=f"v2 LOO Dir {loo['dir_accuracy']:.1%}", row=1, col=2)
        fig.add_hline(y=v1_loo_dir_acc*100, line_dash="dot", line_color="orange",
                      annotation_text=f"v1 {v1_loo_dir_acc:.1%}", row=1, col=2)

        fig.add_trace(go.Bar(x=dates, y=true_bp, name="Actual",
                             marker_color="steelblue", opacity=0.7), 2, 1)
        fig.add_trace(go.Bar(x=dates, y=pred_bp, name="Predicted",
                             marker_color="coral", opacity=0.7), 2, 1)

        dir_ok_list = [td == pd for td, pd in zip(loo["true_dir"], loo["pred_dir"])]
        pres_clrs   = {"trump_t1": "red", "biden": "blue", "trump_t2": "darkred"}
        bar_clrs    = [pres_clrs.get(meta[i].get("term", "?"), "gray") for i in range(len(meta))]
        bar_opacities = [1.0 if ok else 0.3 for ok in dir_ok_list]

        fig.add_trace(go.Bar(
            x=dates, y=loo["conf_dir"], name="Conf",
            marker_color=bar_clrs,
            opacity=0.8,
            text=[f"pres={meta[i].get('term','?')}<br>true={DIRECTION[loo['true_dir'][i]]}<br>pred={DIRECTION[loo['pred_dir'][i]]}"
                  for i in range(len(meta))],
            hovertemplate="%{text}<br>conf=%{y:.2f}",
        ), 2, 2)

        fig.update_layout(
            title=(f"Powell Decision Transformer v2 (Opponent-Aware) — "
                   f"Dir LOO: {loo['dir_accuracy']:.1%}  "
                   f"vs v1: {v1_loo_dir_acc:.1%}  "
                   f"| {model.count_parameters():,} params | {TOTAL_DIM_V2}D input"),
            height=750, barmode="group"
        )
        fig.write_html(str(OUT_VIZ / "v2_interactive_dashboard.html"))
        print("  Saved: v2_interactive_dashboard.html")
    except Exception as e:
        print(f"  Plotly skipped: {e}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    t0 = time.time()

    # ── v1 baseline（从已有 eval 结果加载）────────────────────────────────
    v1_dir_acc  = 0.7576
    v1_loo_preds = None
    try:
        ev1 = json.load(open(OUT_VIZ / "evaluation_results.json"))
        v1_dir_acc   = ev1["loo_direction_accuracy"]
        v1_loo_preds = [p["true_dir"] == p["pred_dir"]
                        for p in ev1["loo_predictions"]]
        # 提取 v1 预测的方向索引
        v1_pred_dir_idx = [DIR_IDX.get(p["pred_dir"], 1) for p in ev1["loo_predictions"]]
        print(f"\n[V1 BASELINE] Loaded: LOO dir = {v1_dir_acc:.1%}")
    except Exception as e:
        print(f"\n[V1 BASELINE] Could not load: {e}  (using default {v1_dir_acc:.1%})")
        v1_pred_dir_idx = None

    # ── Load data ─────────────────────────────────────────────────────────
    econ = EconomicDataLoader()

    # ── Build v2 dataset ──────────────────────────────────────────────────
    X, y_dir, y_mag, y_guid, meta, norm = build_powell_dataset_v2(econ)

    print(f"\n  Input dim v2: {TOTAL_DIM_V2}  (v1: {44})")
    print(f"  Breakdown: {N_ECON}D econ + {N_CONTEXT}D ctx + {LEADER_FACTOR_DIM}D powell"
          f" + {PRESIDENT_DIM}D president + {ECB_DIM}D ECB + {RECENT_ACTION_DIM}D actions")

    # ── Model kwargs ──────────────────────────────────────────────────────
    # 与v1相同的架构，只是输入更宽
    model_kwargs = dict(d_model=32, nhead=4, n_layers=2, dim_ff=64, dropout=0.0)

    # ── LOO Evaluation ────────────────────────────────────────────────────
    loo = loo_eval(X, y_dir, y_mag, y_guid, meta,
                   n_epochs=700, model_kwargs=model_kwargs)

    # ── Turning point analysis ────────────────────────────────────────────
    tp_results = analyze_turning_points(
        loo, meta, y_dir,
        v1_preds=v1_pred_dir_idx,
    )

    # ── Train full model ──────────────────────────────────────────────────
    print(f"\n[TRAIN] Training Powell v2 model (full {len(X)} samples)...")
    powell_model_v2 = MultiTaskDecisionTransformer(n_features=TOTAL_DIM_V2, **model_kwargs)
    n_params = powell_model_v2.count_parameters()
    print(f"  Parameters: {n_params:,}")

    history = train_multitask(powell_model_v2, X, y_dir, y_mag, y_guid,
                               n_epochs=800, lr=5e-3, batch_size=8, weight_decay=5e-4)
    print(f"  Final dir acc: {history['dir_acc'][-1]:.1%}, joint acc: {history['joint_acc'][-1]:.1%}")

    # ── Save model ────────────────────────────────────────────────────────
    model_path = OUT_MODELS / "powell_v3_opponent.pt"
    torch.save({
        "model_state_dict":       powell_model_v2.state_dict(),
        "model_class":            "MultiTaskDecisionTransformer",
        "model_config":           {**model_kwargs, "n_features": TOTAL_DIM_V2},
        "n_params":               n_params,
        "action_heads":           {"direction": DIRECTION, "magnitude": MAGNITUDE, "guidance": GUIDANCE},
        "loo_dir_accuracy":       float(loo["dir_accuracy"]),
        "loo_joint_accuracy":     float(loo["joint_accuracy"]),
        "train_dir_accuracy":     float(history["dir_acc"][-1]),
        "train_joint_accuracy":   float(history["joint_acc"][-1]),
        "normalization":          norm.to_dict(),
        "state_features":         FEATURES,
        "total_input_dim":        TOTAL_DIM_V2,
        "input_breakdown": {
            "econ":      N_ECON,
            "context":   N_CONTEXT,
            "powell":    LEADER_FACTOR_DIM,
            "president": PRESIDENT_DIM,
            "ecb":       ECB_DIM,
            "actions":   RECENT_ACTION_DIM,
        },
        "president_dim_keys":   TRUMP_KEYS_FOR_POWELL,
        "ecb_dim_keys":         ECB_KEYS_FOR_POWELL,
        "powell_dim_keys":      CB_KEYS,
        "metadata":             meta,
        "v1_comparison": {
            "v1_dir_accuracy":   v1_dir_acc,
            "v2_dir_accuracy":   float(loo["dir_accuracy"]),
            "delta_pp":          float(loo["dir_accuracy"] - v1_dir_acc) * 100,
        },
        "turning_point_results": tp_results,
    }, model_path)
    print(f"\n[SAVE] {model_path}")

    # ── Save eval JSON ────────────────────────────────────────────────────
    # ── JSON-serialize turning point results (convert numpy types) ───────
    def jsonify(obj):
        """Recursively convert numpy scalars to Python native types."""
        if isinstance(obj, dict):
            return {k: jsonify(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [jsonify(v) for v in obj]
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    tp_results_clean = jsonify(tp_results)

    eval_out = {
        "model_version":         "v2_opponent_aware",
        "n_params":              n_params,
        "total_input_dim":       TOTAL_DIM_V2,
        "input_breakdown": {
            "econ_features":     N_ECON,
            "context_features":  N_CONTEXT,
            "powell_features":   LEADER_FACTOR_DIM,
            "president_features": PRESIDENT_DIM,
            "ecb_features":      ECB_DIM,
            "recent_action_features": RECENT_ACTION_DIM,
        },
        "president_mapping": {
            "trump_t1": "2017-01-20 to 2021-01-20",
            "biden":    "2021-01-20 to 2025-01-20",
            "trump_t2": "2025-01-20 onwards",
        },
        "v1_comparison": {
            "v1_pilot":    {"input_dim": 10,  "n_params": 17770, "loo_dir_accuracy": 0.667},
            "v1_full":     {"input_dim": 44,  "loo_dir_accuracy": v1_dir_acc},
            "v2_opponent": {"input_dim": TOTAL_DIM_V2, "loo_dir_accuracy": float(loo["dir_accuracy"])},
        },
        "loo_direction_accuracy": float(loo["dir_accuracy"]),
        "loo_joint_accuracy":     float(loo["joint_accuracy"]),
        "loo_dir_correct":        int(loo["dir_correct"]),
        "loo_joint_correct":      int(loo["joint_correct"]),
        "loo_total":              int(loo["total"]),
        "turning_point_results":  tp_results_clean,
        "loo_predictions": [
            {
                "date":           meta[i]["date"],
                "president":      meta[i].get("term", "?"),
                "true_action":    loo["true_actions"][i],
                "pred_action":    loo["pred_actions"][i],
                "true_dir":       DIRECTION[loo["true_dir"][i]],
                "pred_dir":       DIRECTION[loo["pred_dir"][i]],
                "dir_confidence": float(loo["conf_dir"][i]),
                "dir_correct":    bool(loo["true_dir"][i] == loo["pred_dir"][i]),
                "joint_correct":  bool(loo["true_dir"][i] == loo["pred_dir"][i] and
                                       loo["true_mag"][i]  == loo["pred_mag"][i]),
            }
            for i in range(len(meta))
        ],
        "timestamp": datetime.now().isoformat(),
    }
    eval_path = OUT_VIZ / "v2_evaluation_results.json"
    json.dump(eval_out, open(eval_path, "w"), indent=2)
    print(f"[SAVE] {eval_path}")

    # ── Visualizations ────────────────────────────────────────────────────
    save_visualizations_v2(history, loo, meta, powell_model_v2,
                           X, y_dir, y_mag, y_guid,
                           tp_results, v1_loo_dir_acc=v1_dir_acc)

    # ── Final comparison summary ──────────────────────────────────────────
    elapsed = time.time() - t0
    delta_dir   = (loo["dir_accuracy"]   - v1_dir_acc) * 100
    v1_tp_acc   = tp_results.get("v1_tp_accuracy", None)
    v2_tp_acc   = tp_results.get("v2_tp_accuracy", 0.0)
    tp_delta    = (v2_tp_acc - v1_tp_acc) * 100 if v1_tp_acc is not None else None

    print("\n" + "=" * 70)
    print("v2 OPPONENT-AWARE COMPLETE")
    print("=" * 70)
    print(f"  Time:                   {elapsed:.0f}s")
    print(f"  Parameters:             {n_params:,}  (v1: ~18,923)")
    print(f"  Input dims:             {TOTAL_DIM_V2}D  (v1: 44D)")
    print()
    print("  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │  Accuracy Comparison                                    │")
    print(f"  │  v1 pilot  (10D, single-task):  66.7%                  │")
    print(f"  │  v1 full   (44D, multi-task):   {v1_dir_acc:.1%}                 │")
    print(f"  │  v2 opp.   ({TOTAL_DIM_V2}D, opponent):   {loo['dir_accuracy']:.1%}  ({delta_dir:+.1f} pp) │")
    print(f"  └─────────────────────────────────────────────────────────┘")
    print()
    print(f"  Turning point analysis ({tp_results.get('total',0)} turning points):")
    if v1_tp_acc is not None:
        print(f"    v1 turning point acc: {v1_tp_acc:.1%}")
    print(f"    v2 turning point acc: {v2_tp_acc:.1%}", end="")
    if tp_delta is not None:
        print(f"  ({tp_delta:+.1f} pp)", end="")
    print()
    print()
    print(f"  Direction breakdown:")
    print(f"    LOO dir correct:     {loo['dir_correct']}/{loo['total']} = {loo['dir_accuracy']:.1%}")
    print(f"    LOO joint correct:   {loo['joint_correct']}/{loo['total']} = {loo['joint_accuracy']:.1%}")
    print()
    print(f"  Hypothesis test — opponent params affect turning points:")
    hyp_supported = v2_tp_acc > (v1_tp_acc or 0)
    print(f"    {'SUPPORTED' if hyp_supported else 'NOT SUPPORTED — further tuning needed'}")
    print()
    print(f"  Model saved: {model_path}")
    print()
    print(f"  Visualizations:")
    for f in sorted(OUT_VIZ.glob("v2_*.png")):
        print(f"    {f}")
    for f in sorted(OUT_VIZ.glob("v2_*.html")):
        print(f"    {f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
