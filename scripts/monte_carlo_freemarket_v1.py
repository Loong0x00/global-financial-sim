#!/usr/bin/env python3
"""
Monte Carlo Free Market Analysis — Pure Economic Attribution
=============================================================
排除所有政治因素和中国，只看自由市场经济（US/EU/UK/JP）会因为什么崩溃。

核心问题：如果没有任何领导人干预，纯经济动力学会如何演化？
哪些经济指标在推动路径走向危机？

方法：
1. 仅加载 US/EU/UK/JP 经济数据（排除 CN/RU）
2. 不使用任何领导人决策函数
3. 将PCA分量映射回原始经济指标
4. 追踪哪些经济因子类别驱动路径偏离

经济因子分类（12类）：
  equity      — 股市估值/泡沫
  yield_curve — 收益率曲线（利率+期限结构）
  credit      — 信贷扩张/收缩（BIS credit gap等）
  inflation   — 通胀/物价
  labor       — 就业/失业
  housing     — 房地产
  commodity   — 大宗商品（非地缘因素的供需）
  forex       — 汇率/货币
  volatility  — 市场波动率（VIX等）
  trade_flow  — 贸易流（非政策干预的自然变化）
  debt        — 政府/企业债务
  sentiment   — 消费者/企业信心
"""

import json
import time
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name()}")

# ============================================================
# CONFIG
# ============================================================

DATA_BASE = Path('/home/user/global-financial-sim/data')
ECON_BASE = DATA_BASE / 'economic'
OUT_DIR = Path('/home/user/global-financial-sim/output/monte_carlo')
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SIMULATIONS = 500_000
MAX_HORIZON_MONTHS = 48
N_PCA_COMPONENTS = 14
K_DENSITY = 15
NOISE_SCALE = 0.15

DENSITY_CRISIS_PCT = 5
ACCEL_CRISIS_PCT = 95
DEVIATION_CRISIS_PCT = 95

CRISIS_PERIODS = {
    "Oil73": ("1973-10", "1974-06"),
    "Volcker": ("1980-01", "1982-06"),
    "BlackMon": ("1987-10", "1987-12"),
    "Japan90": ("1990-01", "1992-12"),
    "Dotcom": ("2000-03", "2002-10"),
    "GFC": ("2007-08", "2009-06"),
    "EuroDebt": ("2010-04", "2012-07"),
    "CommodityCrash": ("2014-06", "2016-02"),
    "COVID": ("2020-02", "2020-06"),
    "Inflation22": ("2022-01", "2022-12"),
}

# ============================================================
# ECONOMIC FACTOR CATEGORIES
# ============================================================

# 按关键词分类经济序列到因子类别
FACTOR_RULES = [
    ('equity',      ['sp500', 'nasdaq', 'djia', 'nikkei', 'dax', 'ftse', 'stoxx',
                     'kospi', 'sensex', 'bovespa', 'hang_seng', 'IDX_',
                     'stock', 'equity', 'share_price', 'nikkei225']),
    ('yield_curve', ['treasury', 'yield', 'bond', 'gilt', 'bund', 'jgb', 'interest_rate',
                     'fed_funds', 'libor', 'euribor', 'saron', 'T10Y',
                     'T3M', 'T1Y', 'T2Y', 'T5Y', 'T30Y', 'term_spread',
                     'DGS', 'GS10', 'GS2', 'TB3MS', 'FEDFUNDS',
                     'govt_bond', '10y_bond', '10y2y', 'yield_curve',
                     'fed_funds_rate', 'germany_10y', 'uk_10y']),
    ('credit',      ['credit', 'loan', 'lending', 'BIS_credit', 'debt_service',
                     'bank_credit', 'private_credit', 'credit_gap', 'spread',
                     'BAA', 'AAA', 'high_yield', 'investment_grade',
                     'credit_spread', 'fed_total_assets']),
    ('inflation',   ['cpi', 'pce', 'inflation', 'deflator', 'price_index',
                     'CPIAUCSL', 'CPILFESL', 'consumer_price', 'producer_price', 'ppi',
                     'hicp', 'core_cpi', 'core_pce']),
    ('labor',       ['unemployment', 'employ', 'payroll', 'labor', 'UNRATE',
                     'nonfarm', 'jobless', 'claim', 'wage', 'earnings',
                     'PAYEMS', 'labor_force', 'manufacturing_employment']),
    ('housing',     ['house', 'housing', 'home', 'property', 'real_estate', 'mortgage',
                     'BIS_property', 'CSUSHPINSA', 'MSPUS', 'HOUST',
                     'rent', 'residential', 'home_price']),
    ('commodity',   ['oil', 'gold', 'silver', 'copper', 'iron', 'coal', 'gas',
                     'commodity', 'CMD_', 'WTI', 'brent', 'wheat', 'corn',
                     'DCOILWTICO', 'DCOILBRENTEU', 'natural_gas',
                     'metal', 'energy', 'agricultural']),
    ('forex',       ['exchange', 'forex', 'currency', 'dollar', 'euro', 'yen', 'pound',
                     'DEXJPUS', 'DEXUSEU', 'DEXUSUK', 'DXY', 'FX_',
                     'trade_weighted', 'TWEXB', 'real_effective',
                     'usd_index', 'eur_usd', 'gbp_usd', 'usd_jpy',
                     'exchange_rate']),
    ('volatility',  ['vix', 'volatility', 'vol_', 'VIX', 'MOVE', 'OVX', 'GVZ',
                     'implied_vol', 'fear', 'stress_index']),
    ('liquidity',   ['m2', 'm3', 'money_supply', 'monetary_base', 'liquidity',
                     'M2', 'M3', 'fed_total', 'central_bank_assets']),
    ('trade_flow',  ['trade', 'export', 'import', 'TRADE_', 'current_account',
                     'balance_of_payment', 'BOPG']),
    ('debt',        ['debt', 'deficit', 'fiscal', 'government_debt', 'sovereign',
                     'BIS_total_credit', 'public_debt', 'GFDEBTN']),
    ('sentiment',   ['confidence', 'sentiment', 'consumer_conf', 'business_conf',
                     'PMI', 'ISM', 'michigan', 'UMCSENT', 'expectation',
                     'leading_indicator', 'CLI', 'consumer_sentiment']),
    ('output',      ['gdp', 'industrial_production', 'retail_sales', 'GDP',
                     'output', 'production']),
]

def classify_series(name):
    """Classify an economic series name into a factor category."""
    name_lower = name.lower()
    for category, keywords in FACTOR_RULES:
        for kw in keywords:
            if kw.lower() in name_lower:
                return category
    return 'other'

# ============================================================
# 1. LOAD DATA — FREE MARKET ONLY (no CN, no RU)
# ============================================================

print("=" * 70)
print("1. LOADING DATA — FREE MARKET ECONOMIES ONLY")
print("   US / EU / UK / JP (excluding CN, RU, leaders)")
print("=" * 70)

all_series = {}
FREE_COUNTRIES = ['us', 'jp', 'eu', 'uk']  # 排除 cn, ru

def extract_series_from_data(data, prefix):
    """Recursively extract time series from nested JSON structures."""
    results = {}
    if isinstance(data, dict):
        # Check for indicators structure: {indicators: {name: {series: [{date, value}]}}}
        if 'indicators' in data:
            for ind_name, ind_data in data['indicators'].items():
                if isinstance(ind_data, dict) and 'series' in ind_data:
                    sd = {}
                    for item in ind_data['series']:
                        if isinstance(item, dict):
                            d = item.get('date', '')
                            val = item.get('value', item.get('close'))
                            if d and val is not None:
                                try:
                                    dk = d[:10]
                                    if len(dk) == 7: dk += '-01'
                                    sd[dk] = float(val)
                                except: pass
                    if sd: results[f"{prefix}_{ind_name}"] = sd
        else:
            # Direct dict with date:value pairs or nested
            for k, v in data.items():
                if k in ('entity', 'source', 'last_updated', 'metadata'): continue
                if isinstance(v, dict) and any(isinstance(vv, (int,float)) for vv in v.values()):
                    results[f"{prefix}_{k}"] = {kk: float(vv) for kk, vv in v.items() if isinstance(vv, (int,float))}
                elif isinstance(v, list):
                    sd = {}
                    for item in v:
                        if isinstance(item, dict):
                            d = item.get('date', '')
                            val = item.get('value', item.get('close'))
                            if d and val is not None:
                                try:
                                    dk = d[:10]
                                    if len(dk) == 7: dk += '-01'
                                    sd[dk] = float(val)
                                except: pass
                    if sd: results[f"{prefix}_{k}"] = sd
                elif isinstance(v, dict):
                    results.update(extract_series_from_data(v, f"{prefix}_{k}"))
    return results

for country_dir in FREE_COUNTRIES:
    base = ECON_BASE / country_dir
    for fn in ['macro.json', 'financial.json']:
        fp = base / fn
        if not fp.exists(): continue
        with open(fp) as f: data = json.load(f)
        all_series.update(extract_series_from_data(data, country_dir))

# Supplementary data (filter out CN/RU)
for fn in sorted((ECON_BASE / 'supplementary').glob('*.json')) if (ECON_BASE / 'supplementary').exists() else []:
    with open(fn) as f: data = json.load(f)
    extracted = extract_series_from_data(data, f"supp_{fn.stem}")
    for k, v in extracted.items():
        name_lower = k.lower()
        if any(x in name_lower for x in ['china', 'cn_', 'russia', 'ru_', 'ruble', 'yuan', 'renminbi']):
            continue
        all_series[k] = v

# BIS data (only free market countries)
BIS_COUNTRIES = {'US', 'JP', 'DE', 'FR', 'GB', 'IT', 'ES', 'NL', 'CA', 'AU',
                 'Euro area', 'United States', 'Japan', 'United Kingdom', 'Germany',
                 'France', 'Italy', 'Spain', 'Netherlands', 'Canada', 'Australia'}
for fn in sorted((ECON_BASE / 'bis').glob('*.json')) if (ECON_BASE / 'bis').exists() else []:
    with open(fn) as f: data = json.load(f)
    if isinstance(data, dict):
        for country, cdata in data.items():
            # Only include free market countries
            if country not in BIS_COUNTRIES and not any(x in country for x in ['US', 'JP', 'GB', 'DE', 'FR', 'Euro']):
                continue
            if isinstance(cdata, dict):
                for indicator, idata in cdata.items():
                    if isinstance(idata, dict):
                        vals = {k: float(v) for k, v in idata.items() if isinstance(v, (int, float))}
                        if vals: all_series[f"BIS_{fn.stem}_{country}_{indicator}"] = vals

# Global indices (keep all — they reflect free market dynamics)
fp = ECON_BASE / 'indices' / 'global_indices.json'
if fp.exists():
    with open(fp) as f: data = json.load(f)
    for idx_name, idx_data in data.items():
        # Skip Shanghai Composite and other CN indices
        if any(x in idx_name.lower() for x in ['shanghai', 'shenzhen', 'csi', 'hang_seng']):
            continue
        if isinstance(idx_data, list):
            sd = {}
            for item in idx_data:
                d = item.get('date', '')
                c = item.get('close')
                if d and c is not None:
                    try: sd[d[:10]] = float(c)
                    except: pass
            if sd: all_series[f"IDX_{idx_name}"] = sd

# Commodities (keep — they're market-driven)
for fn in sorted((ECON_BASE / 'commodities').glob('*.json')) if (ECON_BASE / 'commodities').exists() else []:
    if fn.name == 'manifest.json': continue
    with open(fn) as f: data = json.load(f)
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                vals = {kk: float(vv) for kk, vv in v.items() if isinstance(vv, (int, float))}
                if vals: all_series[f"CMD_{fn.stem}_{k}"] = vals
            elif isinstance(v, list):
                sd = {}
                for item in v:
                    if isinstance(item, dict):
                        d = item.get('date', '')
                        c = item.get('close', item.get('value'))
                        if d and c is not None:
                            try: sd[d[:10]] = float(c)
                            except: pass
                if sd: all_series[f"CMD_{fn.stem}_{k}"] = sd

# Crypto (keep — free market asset)
for fn in sorted((ECON_BASE / 'crypto').glob('*.json')) if (ECON_BASE / 'crypto').exists() else []:
    with open(fn) as f: data = json.load(f)
    if isinstance(data, list):
        sd = {}
        for item in data:
            d = item.get('date', '')
            c = item.get('close')
            if d and c is not None:
                try: sd[d[:10]] = float(c)
                except: pass
        if sd: all_series[f"CRYPTO_{fn.stem}"] = sd

# CFTC (keep — market positioning)
fp = ECON_BASE / 'cftc' / 'cot_data.json'
if fp.exists():
    with open(fp) as f: data = json.load(f)
    for cn, recs in data.items():
        if not isinstance(recs, list): continue
        mn, mo = {}, {}
        for r in recs:
            d = r.get('date', '')[:7]
            if not d: continue
            k = d+'-01'
            net, oi = r.get('net_speculative'), r.get('open_interest')
            if net is not None: mn.setdefault(k, []).append(float(net))
            if oi is not None: mo.setdefault(k, []).append(float(oi))
        if mn: all_series[f"CFTC_NET_{cn}"] = {k: np.mean(v) for k, v in mn.items()}
        if mo: all_series[f"CFTC_OI_{cn}"] = {k: np.mean(v) for k, v in mo.items()}

# Trade (exclude CN/RU pairs)
fp = ECON_BASE / 'trade' / 'bilateral_trade.json'
if fp.exists():
    with open(fp) as f: trade = json.load(f)
    for pn, pd in trade.items():
        # Skip pairs involving China or Russia
        if any(x in pn.lower() for x in ['china', 'russia', 'cn', 'ru']):
            continue
        if not isinstance(pd, dict) or 'data' not in pd: continue
        for r in pd['data']:
            yr, total, yoy = r.get('year'), r.get('total'), r.get('yoy_change_pct')
            if yr and total:
                for m in range(1, 13): all_series.setdefault(f"TRADE_{pn}_total", {})[f"{yr}-{m:02d}-01"] = float(total)

# Classify all series
series_categories = {}
category_series = {cat: [] for cat, _ in FACTOR_RULES}
category_series['other'] = []

for name in sorted(all_series.keys()):
    cat = classify_series(name)
    series_categories[name] = cat
    category_series[cat].append(name)

print(f"\nTotal free-market series: {len(all_series)}")
print(f"\nSeries by category:")
for cat in ['equity', 'yield_curve', 'credit', 'inflation', 'labor', 'housing',
            'commodity', 'forex', 'volatility', 'trade_flow', 'debt', 'sentiment', 'other']:
    n = len(category_series.get(cat, []))
    if n > 0:
        print(f"  {cat:15s}: {n:3d} series")

# ============================================================
# 2. BUILD STATE MATRIX (no leaders)
# ============================================================

print("\n" + "=" * 70)
print("2. BUILD STATE MATRIX — PURE ECONOMIC")
print("=" * 70)

def gen_months(sy, sm, ey, em):
    months = []
    y, m = sy, sm
    while (y,m) <= (ey,em):
        months.append(f"{y:04d}-{m:02d}-01")
        m += 1
        if m > 12: m, y = 1, y+1
    return months

all_months = gen_months(1971, 1, 2026, 3)
econ_cols = sorted(all_series.keys())

nm = len(all_months)
nc = len(econ_cols)
matrix = np.full((nm, nc), np.nan)

for j, col in enumerate(econ_cols):
    s = all_series[col]
    for i, mo in enumerate(all_months):
        if mo in s: matrix[i, j] = s[mo]

cov = np.sum(~np.isnan(matrix), axis=0) / nm * 100
gc = cov >= 8
mat = matrix[:, gc]
col_mask = gc
good_cols = [econ_cols[j] for j in range(nc) if gc[j]]
rc = np.sum(~np.isnan(mat), axis=1) / mat.shape[1] * 100
gr = rc >= 25
mat = mat[gr]
months_g = [all_months[i] for i in range(nm) if gr[i]]

# Impute
for j in range(mat.shape[1]):
    last = np.nan
    for i in range(mat.shape[0]):
        if np.isnan(mat[i,j]): mat[i,j] = last
        else: last = mat[i,j]
for j in range(mat.shape[1]):
    first = np.nan
    for i in range(mat.shape[0]):
        if not np.isnan(mat[i,j]): first = mat[i,j]; break
    if not np.isnan(first):
        for i in range(mat.shape[0]):
            if np.isnan(mat[i,j]): mat[i,j] = first
            else: break
for j in range(mat.shape[1]):
    col_data = mat[:,j]; m = np.isnan(col_data)
    if m.any():
        med = np.nanmedian(col_data)
        mat[m,j] = med if not np.isnan(med) else 0.5

print(f"State matrix: {mat.shape} ({len(months_g)} months × {mat.shape[1]} series)")

# Classify good columns
col_categories = [classify_series(c) for c in good_cols]

qt = QuantileTransformer(n_quantiles=min(1000, len(months_g)), output_distribution='uniform', random_state=42)
X_norm = qt.fit_transform(mat)
X_full = np.hstack([X_norm, np.linspace(0,1,len(months_g)).reshape(-1,1)])

max_components = min(X_full.shape[1]-1, X_full.shape[0]-1)
D = min(N_PCA_COMPONENTS, max_components)
n_comp = min(D+5, max_components)
pca = PCA(n_components=n_comp)
X_pca = pca.fit_transform(X_full)[:, :D]
explained = np.cumsum(pca.explained_variance_ratio_)

N = len(months_g)
print(f"PCA: {N}×{D} ({explained[D-1]*100:.1f}% variance)")

# ============================================================
# 3. PCA → ECONOMIC FACTOR MAPPING
# ============================================================

print("\n" + "=" * 70)
print("3. PCA COMPONENT → ECONOMIC FACTOR MAPPING")
print("=" * 70)

# Map PCA components back to economic categories using loadings
components = pca.components_[:D, :-1]  # Exclude time column

# For each PCA component, compute the loading weight per factor category
n_original = len(good_cols)
category_list = ['equity', 'yield_curve', 'credit', 'inflation', 'labor', 'housing',
                 'commodity', 'forex', 'volatility', 'liquidity', 'trade_flow', 'debt',
                 'sentiment', 'output', 'other']

# Loading matrix: [D x n_categories]
pca_to_category = np.zeros((D, len(category_list)))
for pc in range(D):
    loadings = components[pc] if pc < components.shape[0] else np.zeros(n_original)
    # Sum absolute loadings by category
    for j, cat in enumerate(col_categories):
        if j < len(loadings):
            cat_idx = category_list.index(cat) if cat in category_list else category_list.index('other')
            pca_to_category[pc, cat_idx] += loadings[j] ** 2  # Variance contribution

# Normalize each PC's category weights
for pc in range(D):
    total = pca_to_category[pc].sum()
    if total > 0: pca_to_category[pc] /= total

# Print mapping
print(f"\nPCA component → factor category mapping (top contributors):")
for pc in range(min(D, 8)):
    sorted_cats = sorted(range(len(category_list)), key=lambda i: -pca_to_category[pc, i])
    top3 = [(category_list[i], pca_to_category[pc, i]*100) for i in sorted_cats[:3] if pca_to_category[pc, i] > 0.05]
    var_pct = pca.explained_variance_ratio_[pc] * 100
    top_str = ", ".join(f"{c}={p:.0f}%" for c, p in top3)
    print(f"  PC{pc:2d} ({var_pct:5.1f}% var): {top_str}")

# ============================================================
# 4. CALIBRATE CRISIS THRESHOLDS
# ============================================================

print("\n" + "=" * 70)
print("4. CALIBRATE CRISIS THRESHOLDS")
print("=" * 70)

deltas = np.diff(X_pca, axis=0)
delta_norms = np.linalg.norm(deltas, axis=1)

nn_density = NearestNeighbors(n_neighbors=K_DENSITY)
nn_density.fit(X_pca)
knn_dists, _ = nn_density.kneighbors(X_pca)
local_density = knn_dists.mean(axis=1)

accel_full = np.zeros(N)
dn = np.linalg.norm(deltas, axis=1)
accel_full[2:] = np.abs(np.diff(dn))

WINDOW = 24
deviation_full = np.zeros(N)
for i in range(N):
    start = max(0, i-WINDOW)
    if i-start >= 3:
        deviation_full[i] = np.linalg.norm(X_pca[i] - X_pca[start:i].mean(axis=0))

density_thr = np.percentile(local_density, 100-DENSITY_CRISIS_PCT)
accel_thr = np.percentile(np.abs(accel_full), ACCEL_CRISIS_PCT)
deviation_thr = np.percentile(deviation_full, DEVIATION_CRISIS_PCT)

print(f"Density threshold: {density_thr:.4f}")
print(f"Accel threshold:   {accel_thr:.4f}")
print(f"Deviation threshold: {deviation_thr:.4f}")

hist_median = np.median(X_pca, axis=0)

# ============================================================
# 5. PREPARE GPU TENSORS
# ============================================================

print("\n" + "=" * 70)
print("5. PREPARE GPU TENSORS")
print("=" * 70)

X_gpu = torch.tensor(X_pca, dtype=torch.float32, device=DEVICE)
deltas_gpu = torch.tensor(deltas, dtype=torch.float32, device=DEVICE)

N_CLUSTERS = min(80, N // 8)
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_pca[:-1])
cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=DEVICE)

crisis_month_set = set()
for cname, (cs, ce) in CRISIS_PERIODS.items():
    for i, m in enumerate(months_g):
        if cs <= m[:7] <= ce: crisis_month_set.add(i)

cluster_deltas_list = []
cluster_sizes = np.zeros(N_CLUSTERS, dtype=np.int32)
for c in range(N_CLUSTERS):
    members = np.where(cluster_labels == c)[0]
    cluster_sizes[c] = len(members)
    cluster_deltas_list.append(deltas[members] if len(members) > 0 else deltas[:1])

max_pool = max(cluster_sizes)
cluster_delta_pool = np.zeros((N_CLUSTERS, max_pool, D), dtype=np.float32)
for c in range(N_CLUSTERS):
    n = cluster_sizes[c]
    if n > 0: cluster_delta_pool[c, :n] = cluster_deltas_list[c]

cluster_delta_pool_gpu = torch.tensor(cluster_delta_pool, device=DEVICE)
cluster_sizes_gpu = torch.tensor(cluster_sizes, dtype=torch.long, device=DEVICE)

# PCA-to-category mapping on GPU
pca_to_cat_gpu = torch.tensor(pca_to_category, dtype=torch.float32, device=DEVICE)

print(f"Clusters: {N_CLUSTERS}")
print(f"Category mapping: {D} PCs → {len(category_list)} economic factors")

# ============================================================
# 6. MONTE CARLO — PURE FREE MARKET
# ============================================================

print("\n" + "=" * 70)
print(f"6. MONTE CARLO — PURE FREE MARKET")
print(f"   {N_SIMULATIONS:,} paths × {MAX_HORIZON_MONTHS} months")
print(f"   No leaders, no China, no Russia")
print("=" * 70)

N_CATS = len(category_list)
base_state_gpu = torch.tensor(X_pca[-1], dtype=torch.float32, device=DEVICE)
hist_median_gpu = torch.tensor(hist_median, dtype=torch.float32, device=DEVICE)

# Batched simulation
BATCH = N_SIMULATIONS
states = base_state_gpu.unsqueeze(0).expand(BATCH, -1).clone()
noise_std = NOISE_SCALE * torch.tensor(np.std(deltas, axis=0), dtype=torch.float32, device=DEVICE)

# Tracking
crisis_first = torch.full((BATCH,), MAX_HORIZON_MONTHS+1, dtype=torch.int32, device=DEVICE)
crisis_23 = torch.zeros(BATCH, MAX_HORIZON_MONTHS, dtype=torch.bool, device=DEVICE)
crisis_33 = torch.zeros(BATCH, MAX_HORIZON_MONTHS, dtype=torch.bool, device=DEVICE)

# Attribution: cumulative displacement per economic factor category [BATCH x N_CATS]
attr_cumulative = torch.zeros(BATCH, N_CATS, dtype=torch.float32, device=DEVICE)

# Track per-month category contributions for temporal analysis
monthly_cat_contribution = torch.zeros(MAX_HORIZON_MONTHS, N_CATS, dtype=torch.float64, device=DEVICE)

# Track which category was dominant when crisis triggered
crisis_trigger_category = torch.zeros(N_CATS, dtype=torch.float64, device=DEVICE)

t0 = time.time()

for month in range(MAX_HORIZON_MONTHS):
    # 1. Find nearest cluster for each path
    dists = torch.cdist(states, cluster_centers)
    nearest = dists.argmin(dim=1)

    # 2. Sample from cluster-specific delta pools (non-parametric)
    pool_sizes = cluster_sizes_gpu[nearest]
    rand_idx = (torch.rand(BATCH, device=DEVICE) * pool_sizes.float()).long()
    rand_idx = rand_idx.clamp(max=max_pool-1)
    sampled_deltas = cluster_delta_pool_gpu[nearest, rand_idx]

    # 3. Add noise
    noise = torch.randn(BATCH, D, device=DEVICE) * noise_std * 0.5
    total_delta = sampled_deltas + noise

    # 4. Decompose delta into economic factor categories using PCA loadings
    # total_delta: [BATCH, D] — PCA space
    # pca_to_cat_gpu: [D, N_CATS] — mapping from PCA to categories
    # delta contribution per category = sum over PCs of (delta_pc^2 * category_weight_of_pc)
    delta_sq = total_delta ** 2  # [BATCH, D]
    cat_contribution = torch.matmul(delta_sq, pca_to_cat_gpu)  # [BATCH, N_CATS]

    # Normalize to get fraction per category
    cat_total = cat_contribution.sum(dim=1, keepdim=True).clamp(min=1e-10)
    cat_frac = cat_contribution / cat_total  # [BATCH, N_CATS]

    # Weight by delta magnitude
    delta_mag = torch.norm(total_delta, dim=1, keepdim=True)  # [BATCH, 1]
    attr_cumulative += cat_frac * delta_mag

    # Track monthly average
    monthly_cat_contribution[month] = (cat_frac * delta_mag).mean(dim=0).double()

    # 5. Update state
    states = states + total_delta

    # 6. Crisis detection
    knn_d = torch.cdist(states, X_gpu).topk(K_DENSITY, largest=False).values.mean(dim=1)
    prev_delta_norm = torch.norm(total_delta, dim=1)

    dev_from_recent = torch.norm(states - hist_median_gpu, dim=1)

    density_crisis = knn_d > density_thr
    accel_crisis = prev_delta_norm > accel_thr
    deviation_crisis = dev_from_recent > deviation_thr

    crisis_count = density_crisis.int() + accel_crisis.int() + deviation_crisis.int()
    is_crisis_23 = crisis_count >= 2
    is_crisis_33 = crisis_count >= 3

    crisis_23[:, month] = is_crisis_23
    crisis_33[:, month] = is_crisis_33

    # Track which category dominates at crisis trigger
    newly_triggered = is_crisis_23 & (crisis_first > month)
    if newly_triggered.any():
        crisis_first[newly_triggered] = month
        # Get the category attribution for newly triggered paths
        triggered_attr = attr_cumulative[newly_triggered]  # [n_triggered, N_CATS]
        triggered_total = triggered_attr.sum(dim=1, keepdim=True).clamp(min=1e-10)
        triggered_frac = triggered_attr / triggered_total
        crisis_trigger_category += triggered_frac.sum(dim=0).double()

    if (month + 1) % 6 == 0:
        sparse_pct = density_crisis.float().mean() * 100
        crisis_pct = is_crisis_23.float().mean() * 100
        elapsed = time.time() - t0
        print(f"  Month {month+1:2d}: sparse={sparse_pct:.0f}% crisis_23={crisis_pct:.1f}% [{elapsed:.0f}s]")

elapsed = time.time() - t0
print(f"\n  Total time: {elapsed:.1f}s")

# ============================================================
# 7. RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FREE MARKET CRISIS ANALYSIS — RESULTS")
print("=" * 70)

# Crisis probabilities at horizons
horizons = [3, 6, 12, 24, 36, 48]
print(f"\n{'Horizon':>10s}  {'2/3':>8s}  {'3/3':>8s}")
for h in horizons:
    if h <= MAX_HORIZON_MONTHS:
        p23 = crisis_23[:, :h].any(dim=1).float().mean() * 100
        p33 = crisis_33[:, :h].any(dim=1).float().mean() * 100
        print(f"  {h:2d} months  {p23:7.2f}%  {p33:7.2f}%")

# Overall attribution (what's driving paths in general)
print(f"\n--- Overall Economic Factor Attribution ---")
print(f"{'Factor':>15s}  {'%':>6s}  {'Interpretation'}")
print(f"{'-'*60}")
attr_mean = attr_cumulative.mean(dim=0).cpu().numpy()
attr_total = attr_mean.sum()
sorted_cats = sorted(range(N_CATS), key=lambda i: -attr_mean[i])

INTERPRETATIONS = {
    'equity': 'Stock market bubble/crash dynamics',
    'yield_curve': 'Interest rate regime / yield curve inversion',
    'credit': 'Credit cycle — expansion → contraction → crisis',
    'inflation': 'Price stability breakdown',
    'labor': 'Employment deterioration / wage spiral',
    'housing': 'Real estate bubble / mortgage crisis',
    'commodity': 'Energy/commodity supply-demand imbalance',
    'forex': 'Currency instability / capital flows',
    'volatility': 'Market fear / systemic stress',
    'liquidity': 'Money supply / central bank liquidity',
    'trade_flow': 'Trade imbalance correction',
    'debt': 'Sovereign/corporate debt sustainability',
    'sentiment': 'Confidence collapse / animal spirits',
    'output': 'GDP / industrial production slowdown',
    'other': 'Unclassified factors',
}

for idx in sorted_cats:
    cat = category_list[idx]
    pct = attr_mean[idx] / attr_total * 100 if attr_total > 0 else 0
    if pct > 0.5:
        interp = INTERPRETATIONS.get(cat, '')
        print(f"  {cat:>15s}  {pct:5.1f}%  {interp}")

# Attribution at crisis trigger moment (what pushed it over the edge)
print(f"\n--- Attribution AT Crisis Trigger Moment ---")
print(f"(What was dominant when the 2/3 threshold was first crossed)")
trigger_total = crisis_trigger_category.sum().item()
if trigger_total > 0:
    trigger_np = crisis_trigger_category.cpu().numpy()
    sorted_trigger = sorted(range(N_CATS), key=lambda i: -trigger_np[i])
    for idx in sorted_trigger:
        cat = category_list[idx]
        pct = trigger_np[idx] / trigger_total * 100
        if pct > 1.0:
            print(f"  {cat:>15s}  {pct:5.1f}%")

# Temporal analysis: when does each factor dominate?
print(f"\n--- Temporal Dynamics (which factor peaks when) ---")
monthly_np = monthly_cat_contribution.cpu().numpy()
monthly_totals = monthly_np.sum(axis=1, keepdims=True)
monthly_totals[monthly_totals == 0] = 1
monthly_frac = monthly_np / monthly_totals

# Find peak month for each category
for idx in sorted_cats:
    cat = category_list[idx]
    if attr_mean[idx] / attr_total * 100 > 2.0:
        peak_month = np.argmax(monthly_frac[:, idx]) + 1
        early = monthly_frac[:12, idx].mean() * 100
        late = monthly_frac[24:, idx].mean() * 100
        trend = "↗ late riser" if late > early * 1.3 else ("↘ early mover" if early > late * 1.3 else "→ steady")
        print(f"  {cat:>15s}: peak month {peak_month:2d}, early avg {early:.1f}%, late avg {late:.1f}% {trend}")

# ============================================================
# 8. HISTORICAL VALIDATION — do crisis periods match our factors?
# ============================================================

print(f"\n--- Historical Crisis Factor Profiles ---")
print(f"(Which factors were elevated during known crises)")

for cname, (cs, ce) in sorted(CRISIS_PERIODS.items()):
    crisis_months_idx = [i for i, m in enumerate(months_g[:-1]) if cs <= m[:7] <= ce]
    if not crisis_months_idx: continue

    crisis_deltas_local = deltas[crisis_months_idx]
    crisis_delta_sq = crisis_deltas_local ** 2  # [n_months, D]
    # Map to categories
    cat_contrib = crisis_delta_sq @ pca_to_category  # [n_months, N_CATS]
    cat_mean = cat_contrib.mean(axis=0)
    cat_total_local = cat_mean.sum()
    if cat_total_local == 0: continue

    top3 = sorted(range(N_CATS), key=lambda i: -cat_mean[i])[:3]
    top_str = ", ".join(f"{category_list[i]}={cat_mean[i]/cat_total_local*100:.0f}%" for i in top3)
    print(f"  {cname:20s} ({cs}~{ce}): {top_str}")

# ============================================================
# 9. SAVE
# ============================================================

results = {
    'config': {
        'n_simulations': N_SIMULATIONS,
        'max_horizon': MAX_HORIZON_MONTHS,
        'economies': FREE_COUNTRIES,
        'n_series': len(all_series),
        'excluded': ['CN', 'RU', 'all_leaders'],
    },
    'crisis_probability': {
        f'{h}mo': {
            '2of3': round(crisis_23[:, :h].any(dim=1).float().mean().item() * 100, 2),
            '3of3': round(crisis_33[:, :h].any(dim=1).float().mean().item() * 100, 2),
        } for h in horizons if h <= MAX_HORIZON_MONTHS
    },
    'attribution_overall': {
        category_list[i]: round(float(attr_mean[i] / attr_total * 100), 2) if attr_total > 0 else 0
        for i in range(N_CATS)
    },
    'attribution_at_trigger': {
        category_list[i]: round(float(crisis_trigger_category[i].item() / trigger_total * 100), 2) if trigger_total > 0 else 0
        for i in range(N_CATS)
    },
}

outpath = OUT_DIR / 'monte_carlo_results_freemarket.json'
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {outpath}")
print("Done.")
