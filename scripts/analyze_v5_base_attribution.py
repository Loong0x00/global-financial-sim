#!/usr/bin/env python3
"""
分析v5中"base"经济因素34.3%的内部构成
=======================================
v5的base displacement来自集群采样的历史deltas。
本脚本：
1. 复用v5的完整数据管道（含CN/RU/leaders）
2. 将PCA分量映射回原始经济指标类别
3. 分析当前状态所在集群的历史delta特征
4. 回答：经济自然演化中，哪些因子在推动偏离？
"""

import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

DATA_BASE = Path('/home/user/global-financial-sim/data')
ECON_BASE = DATA_BASE / 'economic'
LEADER_BASE = DATA_BASE / 'leaders'

N_PCA_COMPONENTS = 14

# ============================================================
# 1. LOAD DATA (exact same as v5)
# ============================================================

import unicodedata

all_series = {}

for country_dir in ['us', 'cn', 'jp', 'eu', 'uk', 'ru']:
    base = ECON_BASE / country_dir
    for fn in ['macro.json', 'financial.json']:
        fp = base / fn
        if not fp.exists(): continue
        with open(fp) as f: data = json.load(f)
        if isinstance(data, dict):
            # Handle nested indicators structure
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
                        if sd: all_series[f"{country_dir}_{ind_name}"] = sd
            # Also handle flat dict structures
            for k, v in data.items():
                if k in ('entity', 'source', 'last_updated', 'metadata', 'indicators'): continue
                if isinstance(v, dict) and any(isinstance(vv, (int,float)) for vv in v.values()):
                    all_series[f"{country_dir}_{k}"] = {kk: float(vv) for kk, vv in v.items() if isinstance(vv, (int,float))}
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
                    if sd: all_series[f"{country_dir}_{k}"] = sd

# Supplementary
for fn in sorted((ECON_BASE / 'supplementary').glob('*.json')) if (ECON_BASE / 'supplementary').exists() else []:
    with open(fn) as f: data = json.load(f)
    if isinstance(data, dict):
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
                    if sd: all_series[f"supp_{fn.stem}_{ind_name}"] = sd
        for k, v in data.items():
            if k in ('entity', 'source', 'last_updated', 'metadata', 'indicators'): continue
            if isinstance(v, dict) and any(isinstance(vv, (int,float)) for vv in v.values()):
                all_series[f"supp_{fn.stem}_{k}"] = {kk: float(vv) for kk, vv in v.items() if isinstance(vv, (int,float))}

# BIS
for fn in sorted((ECON_BASE / 'bis').glob('*.json')) if (ECON_BASE / 'bis').exists() else []:
    with open(fn) as f: data = json.load(f)
    if isinstance(data, dict):
        for country, cdata in data.items():
            if isinstance(cdata, dict):
                for indicator, idata in cdata.items():
                    if isinstance(idata, dict):
                        vals = {k: float(v) for k, v in idata.items() if isinstance(v, (int, float))}
                        if vals: all_series[f"BIS_{fn.stem}_{country}_{indicator}"] = vals

# Global indices
fp = ECON_BASE / 'indices' / 'global_indices.json'
if fp.exists():
    with open(fp) as f: data = json.load(f)
    for idx_name, idx_data in data.items():
        if isinstance(idx_data, list):
            sd = {}
            for item in idx_data:
                d = item.get('date', '')
                c = item.get('close')
                if d and c is not None:
                    try: sd[d[:10]] = float(c)
                    except: pass
            if sd: all_series[f"IDX_{idx_name}"] = sd

# Commodities
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

# Crypto
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

# CFTC
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

# Trade
fp = ECON_BASE / 'trade' / 'bilateral_trade.json'
if fp.exists():
    with open(fp) as f: trade = json.load(f)
    for pn, pd in trade.items():
        if not isinstance(pd, dict) or 'data' not in pd: continue
        for r in pd['data']:
            yr, total, yoy = r.get('year'), r.get('total'), r.get('yoy_change_pct')
            if yr and total:
                for m in range(1, 13): all_series.setdefault(f"TRADE_{pn}_total", {})[f"{yr}-{m:02d}-01"] = float(total)

print(f"Total economic series: {len(all_series)}")

# ============================================================
# 2. CLASSIFY SERIES
# ============================================================

FACTOR_RULES = [
    ('equity',      ['sp500', 'nasdaq', 'djia', 'nikkei', 'dax', 'ftse', 'stoxx',
                     'kospi', 'sensex', 'bovespa', 'hang_seng', 'IDX_', 'shanghai',
                     'stock', 'equity', 'nikkei225', 'shenzhen']),
    ('yield_curve', ['treasury', 'yield', 'bond', 'gilt', 'bund', 'jgb', 'interest_rate',
                     'fed_funds', 'libor', 'T10Y', 'T3M', 'DGS', 'GS10', 'GS2', 'TB3MS',
                     'FEDFUNDS', 'govt_bond', '10y_bond', '10y2y', 'yield_curve',
                     'fed_funds_rate', 'germany_10y', 'uk_10y', 'discount_rate']),
    ('credit',      ['credit', 'loan', 'lending', 'BIS_credit', 'debt_service',
                     'bank_credit', 'private_credit', 'credit_gap', 'spread',
                     'BAA', 'AAA', 'credit_spread', 'fed_total_assets']),
    ('inflation',   ['cpi', 'pce', 'inflation', 'deflator', 'price_index',
                     'CPIAUCSL', 'CPILFESL', 'consumer_price', 'ppi',
                     'hicp', 'core_cpi', 'core_pce']),
    ('labor',       ['unemployment', 'employ', 'payroll', 'labor', 'UNRATE',
                     'nonfarm', 'jobless', 'claim', 'wage', 'earnings',
                     'PAYEMS', 'labor_force', 'manufacturing_employment']),
    ('housing',     ['house', 'housing', 'home', 'property', 'real_estate', 'mortgage',
                     'BIS_property', 'rent', 'residential', 'home_price']),
    ('commodity',   ['oil', 'gold', 'silver', 'copper', 'iron', 'coal',
                     'commodity', 'CMD_', 'WTI', 'brent', 'wheat', 'corn',
                     'DCOILWTICO', 'natural_gas', 'metal', 'agricultural',
                     'CRYPTO_']),
    ('forex',       ['exchange', 'forex', 'currency', 'dollar', 'euro', 'yen', 'pound',
                     'DEXJPUS', 'DEXUSEU', 'DXY', 'usd_index', 'eur_usd', 'gbp_usd',
                     'usd_jpy', 'exchange_rate', 'ruble', 'yuan', 'real_effective']),
    ('volatility',  ['vix', 'volatility', 'VIX', 'MOVE', 'OVX', 'GVZ',
                     'implied_vol', 'fear', 'stress']),
    ('liquidity',   ['m2', 'm3', 'money_supply', 'monetary_base', 'liquidity',
                     'M2', 'M3', 'fed_total', 'central_bank']),
    ('trade_flow',  ['trade', 'export', 'import', 'TRADE_', 'current_account',
                     'BOPG', 'CFTC_']),
    ('debt',        ['debt', 'deficit', 'fiscal', 'government_debt', 'sovereign',
                     'BIS_total_credit', 'public_debt', 'GFDEBTN']),
    ('sentiment',   ['confidence', 'sentiment', 'consumer_conf', 'business_conf',
                     'PMI', 'ISM', 'michigan', 'UMCSENT', 'consumer_sentiment']),
    ('output',      ['gdp', 'industrial_production', 'retail_sales', 'GDP',
                     'production', 'manufacturing']),
]

CATEGORY_LIST = ['equity', 'yield_curve', 'credit', 'inflation', 'labor', 'housing',
                 'commodity', 'forex', 'volatility', 'liquidity', 'trade_flow', 'debt',
                 'sentiment', 'output', 'other']

def classify_series(name):
    name_lower = name.lower()
    for category, keywords in FACTOR_RULES:
        for kw in keywords:
            if kw.lower() in name_lower:
                return category
    return 'other'

# Also tag by country
def classify_country(name):
    name_lower = name.lower()
    if name_lower.startswith('us_') or 'sp500' in name_lower or 'nasdaq' in name_lower: return 'US'
    if name_lower.startswith('cn_') or 'shanghai' in name_lower or 'yuan' in name_lower: return 'CN'
    if name_lower.startswith('jp_') or 'nikkei' in name_lower or 'jpy' in name_lower: return 'JP'
    if name_lower.startswith('eu_') or 'dax' in name_lower or 'stoxx' in name_lower or 'germany' in name_lower: return 'EU'
    if name_lower.startswith('uk_') or 'ftse' in name_lower or 'gbp' in name_lower: return 'UK'
    if name_lower.startswith('ru_') or 'ruble' in name_lower or 'russia' in name_lower: return 'RU'
    return 'Global'

series_cat = {}
series_country = {}
for name in all_series:
    series_cat[name] = classify_series(name)
    series_country[name] = classify_country(name)

# Print classification stats
cat_counts = {}
for c in series_cat.values(): cat_counts[c] = cat_counts.get(c, 0) + 1
print("\nSeries by category:")
for cat in CATEGORY_LIST:
    if cat in cat_counts:
        print(f"  {cat:15s}: {cat_counts[cat]:3d}")

country_counts = {}
for c in series_country.values(): country_counts[c] = country_counts.get(c, 0) + 1
print("\nSeries by country:")
for co in sorted(country_counts, key=lambda x: -country_counts[x]):
    print(f"  {co:10s}: {country_counts[co]:3d}")

# ============================================================
# 3. BUILD STATE MATRIX + PCA (same as v5 but no leaders)
# ============================================================

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

col_categories = [classify_series(c) for c in good_cols]
col_countries = [classify_country(c) for c in good_cols]

print(f"\nState matrix: {mat.shape}")
print(f"Good columns: {len(good_cols)}")

qt = QuantileTransformer(n_quantiles=min(1000, len(months_g)), output_distribution='uniform', random_state=42)
X_norm = qt.fit_transform(mat)
X_full = np.hstack([X_norm, np.linspace(0,1,len(months_g)).reshape(-1,1)])

max_comp = min(X_full.shape[1]-1, X_full.shape[0]-1)
D = min(N_PCA_COMPONENTS, max_comp)
pca = PCA(n_components=min(D+5, max_comp))
X_pca = pca.fit_transform(X_full)[:, :D]
explained = np.cumsum(pca.explained_variance_ratio_)

N = len(months_g)
print(f"PCA: {N}×{D} ({explained[D-1]*100:.1f}% variance)")

# ============================================================
# 4. PCA LOADINGS → ECONOMIC FACTOR DECOMPOSITION
# ============================================================

print("\n" + "=" * 70)
print("PCA COMPONENT → ECONOMIC FACTOR MAPPING")
print("=" * 70)

components = pca.components_[:D, :-1]  # Exclude time column
N_CATS = len(CATEGORY_LIST)

# Loading matrix: [D x n_categories]
pca_to_category = np.zeros((D, N_CATS))
pca_to_country = np.zeros((D, 7))  # US CN JP EU UK RU Global
COUNTRY_LIST = ['US', 'CN', 'JP', 'EU', 'UK', 'RU', 'Global']

for pc in range(D):
    loadings = components[pc] if pc < components.shape[0] else np.zeros(len(good_cols))
    for j in range(min(len(loadings), len(good_cols))):
        cat = col_categories[j]
        cat_idx = CATEGORY_LIST.index(cat) if cat in CATEGORY_LIST else CATEGORY_LIST.index('other')
        pca_to_category[pc, cat_idx] += loadings[j] ** 2

        co = col_countries[j]
        co_idx = COUNTRY_LIST.index(co) if co in COUNTRY_LIST else COUNTRY_LIST.index('Global')
        pca_to_country[pc, co_idx] += loadings[j] ** 2

# Normalize
for pc in range(D):
    t = pca_to_category[pc].sum()
    if t > 0: pca_to_category[pc] /= t
    t = pca_to_country[pc].sum()
    if t > 0: pca_to_country[pc] /= t

print(f"\nPCA loading breakdown (top factors):")
for pc in range(min(D, 6)):
    var_pct = pca.explained_variance_ratio_[pc] * 100
    sorted_cats = sorted(range(N_CATS), key=lambda i: -pca_to_category[pc, i])
    top = [(CATEGORY_LIST[i], pca_to_category[pc, i]*100) for i in sorted_cats[:3] if pca_to_category[pc, i] > 0.05]
    sorted_cos = sorted(range(7), key=lambda i: -pca_to_country[pc, i])
    top_co = [(COUNTRY_LIST[i], pca_to_country[pc, i]*100) for i in sorted_cos[:2] if pca_to_country[pc, i] > 0.1]
    cat_str = ", ".join(f"{c}={p:.0f}%" for c, p in top)
    co_str = ", ".join(f"{c}={p:.0f}%" for c, p in top_co)
    print(f"  PC{pc} ({var_pct:5.1f}%): {cat_str}  |  {co_str}")

# ============================================================
# 5. ANALYZE CURRENT STATE + CLUSTER DYNAMICS
# ============================================================

print("\n" + "=" * 70)
print("CURRENT STATE ANALYSIS")
print("=" * 70)

deltas = np.diff(X_pca, axis=0)

# Where is the current state?
curr_state = X_pca[-1]
hist_median = np.median(X_pca, axis=0)
curr_displacement = curr_state - hist_median

print(f"\nCurrent state displacement from historical median:")
# Decompose current displacement by category
disp_sq = curr_displacement ** 2
cat_disp = disp_sq @ pca_to_category
cat_total = cat_disp.sum()
sorted_disp = sorted(range(N_CATS), key=lambda i: -cat_disp[i])
for idx in sorted_disp:
    cat = CATEGORY_LIST[idx]
    pct = cat_disp[idx] / cat_total * 100 if cat_total > 0 else 0
    if pct > 2:
        print(f"  {cat:15s}: {pct:5.1f}%")

# Country decomposition
co_disp = disp_sq @ pca_to_country
co_total = co_disp.sum()
print(f"\nCurrent displacement by country:")
sorted_co = sorted(range(7), key=lambda i: -co_disp[i])
for idx in sorted_co:
    co = COUNTRY_LIST[idx]
    pct = co_disp[idx] / co_total * 100 if co_total > 0 else 0
    if pct > 2:
        print(f"  {co:10s}: {pct:5.1f}%")

# ============================================================
# 6. CLUSTER-SPECIFIC DELTA ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CLUSTER DELTA DECOMPOSITION")
print(f"(What drives monthly changes in states similar to current)")
print("=" * 70)

N_CLUSTERS = min(80, N // 8)
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_pca[:-1])

# Find current cluster
curr_cluster = kmeans.predict(curr_state.reshape(1, -1))[0]
print(f"\nCurrent state → cluster {curr_cluster}")

# Analyze deltas from current cluster
curr_cluster_members = np.where(cluster_labels == curr_cluster)[0]
if len(curr_cluster_members) < 3:
    # Use 3 nearest clusters
    dists = np.linalg.norm(kmeans.cluster_centers_ - curr_state, axis=1)
    nearest_clusters = np.argsort(dists)[:5]
    curr_cluster_members = np.concatenate([np.where(cluster_labels == c)[0] for c in nearest_clusters])
    print(f"Sparse cluster — using {len(nearest_clusters)} nearest clusters ({len(curr_cluster_members)} months)")

cluster_deltas = deltas[curr_cluster_members]
print(f"Cluster has {len(curr_cluster_members)} historical months")

# What months are these?
cluster_months = [months_g[i] for i in curr_cluster_members]
print(f"Historical analogue periods: {cluster_months[0][:7]}..{cluster_months[-1][:7]}")

# Show actual months in cluster
from collections import Counter
decade_counts = Counter()
for m in cluster_months:
    yr = int(m[:4])
    decade = f"{yr//10*10}s"
    decade_counts[decade] += 1
print(f"Decade distribution: {dict(sorted(decade_counts.items()))}")

# Decompose cluster deltas by category
delta_sq = cluster_deltas ** 2  # [n_members, D]
cat_contrib = delta_sq @ pca_to_category  # [n_members, N_CATS]
cat_mean = cat_contrib.mean(axis=0)
cat_total = cat_mean.sum()

print(f"\nDelta decomposition for current-state-similar periods:")
sorted_cats = sorted(range(N_CATS), key=lambda i: -cat_mean[i])
for idx in sorted_cats:
    cat = CATEGORY_LIST[idx]
    pct = cat_mean[idx] / cat_total * 100 if cat_total > 0 else 0
    if pct > 2:
        print(f"  {cat:15s}: {pct:5.1f}%")

# Country decomposition of deltas
co_contrib = delta_sq @ pca_to_country
co_mean = co_contrib.mean(axis=0)
co_total = co_mean.sum()
print(f"\nDelta decomposition by country:")
sorted_cos = sorted(range(7), key=lambda i: -co_mean[i])
for idx in sorted_cos:
    co = COUNTRY_LIST[idx]
    pct = co_mean[idx] / co_total * 100 if co_total > 0 else 0
    if pct > 2:
        print(f"  {co:10s}: {pct:5.1f}%")

# ============================================================
# 7. DIRECTION ANALYSIS — are we drifting toward crisis?
# ============================================================

print("\n" + "=" * 70)
print("DRIFT DIRECTION ANALYSIS")
print("(Mean delta vector in current cluster — where are things heading?)")
print("=" * 70)

mean_delta = cluster_deltas.mean(axis=0)
mean_delta_sq = mean_delta ** 2

# Category breakdown of the MEAN drift direction
cat_drift = mean_delta_sq @ pca_to_category
drift_total = cat_drift.sum()
print(f"\nMean drift direction decomposition:")
sorted_drift = sorted(range(N_CATS), key=lambda i: -cat_drift[i])
for idx in sorted_drift:
    cat = CATEGORY_LIST[idx]
    pct = cat_drift[idx] / drift_total * 100 if drift_total > 0 else 0
    if pct > 2:
        print(f"  {cat:15s}: {pct:5.1f}%")

# ============================================================
# 8. COMPARE: WHAT DID CLUSTER DELTAS LOOK LIKE BEFORE PAST CRISES?
# ============================================================

print("\n" + "=" * 70)
print("PRE-CRISIS CLUSTER COMPARISON")
print("(How do current-cluster deltas compare to pre-crisis periods?)")
print("=" * 70)

CRISIS_PERIODS = {
    "Dotcom": ("2000-03", "2002-10"),
    "GFC": ("2007-08", "2009-06"),
    "EuroDebt": ("2010-04", "2012-07"),
    "COVID": ("2020-02", "2020-06"),
    "Inflation22": ("2022-01", "2022-12"),
}

for cname, (cs, ce) in CRISIS_PERIODS.items():
    # Find pre-crisis period (12 months before crisis start)
    pre_months = []
    for i, m in enumerate(months_g[:-1]):
        # 12 months before crisis start
        cs_date = cs + '-01'
        import datetime
        try:
            crisis_start = datetime.datetime.strptime(cs_date, '%Y-%m-%d')
            month_date = datetime.datetime.strptime(m[:10], '%Y-%m-%d')
            diff = (crisis_start - month_date).days
            if 30 < diff < 400:  # 1-13 months before
                pre_months.append(i)
        except: pass

    if not pre_months: continue

    pre_deltas = deltas[pre_months]
    pre_delta_sq = pre_deltas ** 2
    pre_cat = (pre_delta_sq @ pca_to_category).mean(axis=0)
    pre_total = pre_cat.sum()

    sorted_pre = sorted(range(N_CATS), key=lambda i: -pre_cat[i])
    top3 = [(CATEGORY_LIST[i], pre_cat[i]/pre_total*100) for i in sorted_pre[:3]]
    top_str = ", ".join(f"{c}={p:.0f}%" for c, p in top3)
    print(f"  Pre-{cname:12s}: {top_str}")

# Current cluster for comparison
top3_curr = [(CATEGORY_LIST[i], cat_mean[i]/cat_total*100) for i in sorted_cats[:3]]
top_str_curr = ", ".join(f"{c}={p:.0f}%" for c, p in top3_curr)
print(f"  {'CURRENT':14s}: {top_str_curr}")

# ============================================================
# 9. SPECIFIC SERIES ANALYSIS — which exact indicators are extreme?
# ============================================================

print("\n" + "=" * 70)
print("EXTREME INDICATORS IN CURRENT STATE")
print("(Which specific series are at historical extremes?)")
print("=" * 70)

# Current state in normalized space
curr_normalized = X_norm[-1]  # [0,1] quantile-transformed
extreme_high = []
extreme_low = []

for j, col in enumerate(good_cols):
    val = curr_normalized[j]
    cat = col_categories[j]
    co = col_countries[j]
    if val > 0.90:
        extreme_high.append((col, val, cat, co))
    elif val < 0.10:
        extreme_low.append((col, val, cat, co))

extreme_high.sort(key=lambda x: -x[1])
extreme_low.sort(key=lambda x: x[1])

print(f"\n--- At Historical HIGHS (>90th percentile) ---")
for col, val, cat, co in extreme_high[:20]:
    raw = mat[-1, good_cols.index(col)]
    print(f"  [{co:6s}] {col:45s}: p={val:.2f} ({cat:12s}) raw={raw:.2f}")

print(f"\n--- At Historical LOWS (<10th percentile) ---")
for col, val, cat, co in extreme_low[:20]:
    raw = mat[-1, good_cols.index(col)]
    print(f"  [{co:6s}] {col:45s}: p={val:.2f} ({cat:12s}) raw={raw:.2f}")

# Count extremes by category
print(f"\n--- Extreme count by category ---")
all_extremes = extreme_high + extreme_low
cat_extreme = {}
for _, _, cat, _ in all_extremes:
    cat_extreme[cat] = cat_extreme.get(cat, 0) + 1
for cat in sorted(cat_extreme, key=lambda x: -cat_extreme[x]):
    print(f"  {cat:15s}: {cat_extreme[cat]} extreme indicators")

print("\nDone.")
