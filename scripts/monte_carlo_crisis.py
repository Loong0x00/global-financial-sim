#!/usr/bin/env python3
"""
Monte Carlo Crisis Probability Engine
======================================
从当前经济状态出发，基于历史转移概率蒙特卡罗采样，
输出 P(崩溃 | 未来 t 个月) 的概率曲线。

核心思路：
1. 复用 TDA 的状态空间构建（经济指标 + 领导人参数 → 分位数归一化 → PCA）
2. 定义历史危机区域（KNN 从已知危机月份确定危机状态空间边界）
3. 计算历史月度转移分布（状态 t → 状态 t+1 的增量分布）
4. 从当前状态(2026-03)出发，蒙特卡罗采样 N 条路径
5. 输出：P(崩溃 | 未来 t 个月) + 与历史危机的相似度 + 交互式可视化

同时支持决策概率估计：
- 给定状态 → 历史相似状态下的决策分布 → P(hike/hold/cut)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, KernelDensity
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================

DATA_BASE = Path('/home/user/global-financial-sim/data')
ECON_BASE = DATA_BASE / 'economic'
LEADER_BASE = DATA_BASE / 'leaders'
OUT_DIR = Path('/home/user/global-financial-sim/output/monte_carlo')
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SIMULATIONS = 50000       # Monte Carlo 路径数
MAX_HORIZON_MONTHS = 36     # 预测最远 36 个月
N_PCA_COMPONENTS = 14       # 与 TDA full 一致
CRISIS_K_NEIGHBORS = 8      # 定义危机区域的 KNN 半径
NOISE_SCALE = 0.15          # 转移噪声缩放因子

# 已知历史危机时期
CRISIS_PERIODS = {
    "Oil Crisis 1973-74":       ("1973-10", "1974-12"),
    "Volcker Shock 1980-82":    ("1980-01", "1982-12"),
    "Black Monday 1987":        ("1987-08", "1988-03"),
    "Japan Bubble 1990-92":     ("1990-01", "1992-12"),
    "Asian Crisis 1997-98":     ("1997-07", "1998-12"),
    "Dot-com Crash 2000-02":    ("2000-03", "2002-10"),
    "GFC 2007-09":              ("2007-07", "2009-06"),
    "Euro Debt Crisis 2010-12": ("2010-05", "2012-12"),
    "Commodity Crash 2014-16":  ("2014-07", "2016-02"),
    "Trade War 2018-19":        ("2018-06", "2019-12"),
    "COVID 2020":               ("2020-02", "2020-12"),
    "Inflation Crisis 2022":    ("2022-03", "2022-12"),
}

np.random.seed(42)

# ============================================================
# 1. LOAD ALL ECONOMIC DATA (reuse tda_full.py's loading logic)
# ============================================================

print("=" * 70)
print("1. LOADING ECONOMIC DATA")
print("=" * 70)

all_series = {}

def daily_to_monthly(data_points, value_key='close'):
    monthly = {}
    for pt in data_points:
        ym = pt['date'][:7] + "-01"
        val = pt.get(value_key, pt.get('value'))
        if val is None:
            continue
        if ym not in monthly:
            monthly[ym] = []
        monthly[ym].append(float(val))
    return {k: np.mean(v) for k, v in monthly.items()}

# FRED macro/financial data
COUNTRIES = ['us', 'cn', 'eu', 'uk', 'jp', 'ru']
for country in COUNTRIES:
    prefix = country.upper()
    for ftype in ['macro', 'financial']:
        fpath = ECON_BASE / country / f'{ftype}.json'
        if not fpath.exists():
            continue
        with open(fpath) as f:
            data = json.load(f)
        indicators = data.get('indicators', data)
        for name, ind in indicators.items():
            series = ind.get('series', [])
            date_val = {}
            for pt in series:
                if pt['value'] is not None:
                    date_val[pt['date']] = float(pt['value'])
            if date_val:
                all_series[f"ECON_{prefix}_{name}"] = date_val

# Stock indices
indices_path = ECON_BASE / 'indices' / 'global_indices.json'
if indices_path.exists():
    with open(indices_path) as f:
        for name, info in json.load(f).items():
            monthly = daily_to_monthly(info.get('data', []))
            if monthly:
                all_series[f"IDX_{name}"] = monthly

# Gold/Silver
for metal in ['gold', 'silver']:
    fpath = ECON_BASE / 'commodities' / f'{metal}_daily.json'
    if fpath.exists():
        with open(fpath) as f:
            monthly = daily_to_monthly(json.load(f).get('data', []), value_key='close')
        if monthly:
            all_series[f"METAL_{metal}"] = monthly

# FRED Commodities
fred_comm_path = ECON_BASE / 'commodities' / 'fred_commodities.json'
if fred_comm_path.exists():
    with open(fred_comm_path) as f:
        fred_comm = json.load(f)
    for name, info in fred_comm.items():
        if not isinstance(info, dict):
            continue
        series = info.get('data', info.get('series', []))
        date_val = {}
        for pt in series:
            val = pt.get('value', pt.get('close'))
            if val is not None:
                date_val[pt['date']] = float(val)
        if date_val:
            all_series[f"COMM_FRED_{name}"] = date_val

# Yahoo Futures
yahoo_fut_path = ECON_BASE / 'commodities' / 'yahoo_futures.json'
if yahoo_fut_path.exists():
    with open(yahoo_fut_path) as f:
        yahoo_fut = json.load(f)
    for name, info in yahoo_fut.items():
        if not isinstance(info, dict):
            continue
        data = info.get('data', [])
        if data:
            monthly = daily_to_monthly(data)
            if monthly:
                all_series[f"COMM_YF_{name}"] = monthly

# Crypto
crypto_path = ECON_BASE / 'crypto' / 'crypto.json'
if crypto_path.exists():
    with open(crypto_path) as f:
        for name, info in json.load(f).items():
            monthly = daily_to_monthly(info.get('data', []))
            if monthly:
                all_series[f"CRYPTO_{name}"] = monthly

# BIS data
bis_dir = ECON_BASE / 'bis'
if bis_dir.exists():
    for fname in ['credit_gap.json', 'property_prices.json', 'debt_service_ratios.json',
                   'total_credit.json', 'global_liquidity.json']:
        fpath = bis_dir / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            bis_data = json.load(f)
        countries_data = bis_data.get('countries', {})
        ds_name = fname.replace('.json', '')
        for country_code, cdata in countries_data.items():
            series_dict = cdata.get('series', {})
            if not isinstance(series_dict, dict):
                continue
            for idx, (sname, sinfo) in enumerate(series_dict.items()):
                if not isinstance(sinfo, dict) or 'data' not in sinfo:
                    continue
                date_val = {}
                raw_data = sinfo['data']
                items_iter = raw_data.items() if isinstance(raw_data, dict) else raw_data
                for item in items_iter:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        d_raw, val = item
                    elif isinstance(item, dict):
                        d_raw = item.get('date', '')
                        val = item.get('value')
                    else:
                        continue
                    if val is None:
                        continue
                    d_str = str(d_raw)
                    if 'Q' in d_str:
                        parts = d_str.split('-Q')
                        if len(parts) == 2:
                            yr, q = parts[0], int(parts[1])
                            month = (q - 1) * 3 + 1
                            for m_off in range(3):
                                key = f"{yr}-{month+m_off:02d}-01"
                                date_val[key] = float(val)
                    elif len(d_str) >= 7:
                        key = d_str[:7] + '-01'
                        try:
                            date_val[key] = float(val)
                        except:
                            pass
                if date_val:
                    all_series[f"BIS_{ds_name}_{country_code}_{idx}"] = date_val

# Supplementary data
supp_dir = ECON_BASE / 'supplementary'
if supp_dir.exists():
    for fname in ['uk_bank_rate.json', 'russia_supplementary.json',
                   'china_supplementary.json', 'us_additional.json']:
        fpath = supp_dir / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            supp = json.load(f)
        for name, info in supp.items():
            if not isinstance(info, dict) or 'data' not in info:
                continue
            date_val = {}
            for pt in info['data']:
                d = pt.get('date', '')
                v = pt.get('value')
                if d and v is not None:
                    key = d[:7] + '-01' if len(d) >= 7 else d
                    try:
                        date_val[key] = float(v)
                    except:
                        pass
            if date_val:
                all_series[f"SUPP_{name}"] = date_val

# CFTC
cftc_path = ECON_BASE / 'cftc' / 'cot_data.json'
if cftc_path.exists():
    with open(cftc_path) as f:
        cftc = json.load(f)
    for contract_name, records in cftc.items():
        if not isinstance(records, list):
            continue
        monthly_net = {}
        monthly_oi = {}
        for rec in records:
            d = rec.get('date', '')[:7]
            if not d:
                continue
            key = d + '-01'
            net = rec.get('net_speculative')
            oi = rec.get('open_interest')
            if net is not None:
                monthly_net.setdefault(key, []).append(float(net))
            if oi is not None:
                monthly_oi.setdefault(key, []).append(float(oi))
        if monthly_net:
            all_series[f"CFTC_NET_{contract_name}"] = {k: np.mean(v) for k, v in monthly_net.items()}
        if monthly_oi:
            all_series[f"CFTC_OI_{contract_name}"] = {k: np.mean(v) for k, v in monthly_oi.items()}

# Trade
trade_path = ECON_BASE / 'trade' / 'bilateral_trade.json'
if trade_path.exists():
    with open(trade_path) as f:
        trade = json.load(f)
    for pair_name, pair_data in trade.items():
        if not isinstance(pair_data, dict) or 'data' not in pair_data:
            continue
        for rec in pair_data['data']:
            yr = rec.get('year')
            total = rec.get('total')
            if yr and total:
                for m in range(1, 13):
                    key = f"{yr}-{m:02d}-01"
                    all_series.setdefault(f"TRADE_{pair_name}_total", {})[key] = float(total)
            yoy = rec.get('yoy_change_pct')
            if yr and yoy is not None:
                for m in range(1, 13):
                    key = f"{yr}-{m:02d}-01"
                    all_series.setdefault(f"TRADE_{pair_name}_yoy", {})[key] = float(yoy)

print(f"Total economic series loaded: {len(all_series)}")

# ============================================================
# 2. LOAD LEADER PROFILES + TIMELINES (same as tda_full.py)
# ============================================================

print("\n" + "=" * 70)
print("2. LOADING LEADER PROFILES + TIMELINES")
print("=" * 70)

profiles = {}
profile_dir = LEADER_BASE / 'profiles'
for f in sorted(profile_dir.glob('*.json')):
    with open(f) as fp:
        try:
            d = json.load(fp)
        except:
            continue
    vecs = d.get('behavior_matrix', {}).get('vectors', [])
    if not vecs:
        continue
    params = {}
    for v in vecs:
        label = v.get('label', v.get('name', ''))
        val = v.get('value')
        if label and val is not None:
            params[label] = float(val)
    if params:
        profiles[f.stem] = {'params': params}

print(f"Profiles loaded: {len(profiles)}")

# Simplified name mapping (same as tda_full.py)
import unicodedata
name_to_profile = {
    'xi_jinping': 'xi_jinping', 'trump': 'trump', 'donald_trump': 'trump',
    'powell': 'powell', 'jerome_powell': 'powell',
    'putin': 'putin', 'vladimir_putin': 'putin',
    'bernanke': 'bernanke', 'ben_bernanke': 'bernanke',
    'yellen': 'yellen', 'janet_yellen': 'yellen',
    'obama': 'obama', 'barack_obama': 'obama',
    'george_w_bush': 'george_w_bush', 'greenspan': 'greenspan',
    'draghi': 'draghi', 'lagarde': 'lagarde',
    'merkel': 'merkel', 'sarkozy': 'sarkozy',
    'thatcher': 'thatcher', 'blair': 'blair', 'gordon_brown': 'gordon_brown',
    'nabiullina': 'nabiullina', 'yeltsin': 'yeltsin',
    'leonid_brezhnev': 'brezhnev', 'brezhnev': 'brezhnev',
    'nikita_khrushchev': 'khrushchev', 'khrushchev': 'khrushchev',
    'joseph_stalin': 'stalin', 'stalin': 'stalin',
    'vladimir_lenin': 'lenin', 'lenin': 'lenin',
    'zhou_xiaochuan': 'zhou_xiaochuan',
    'hu_jintao': 'hu_wen', 'wen_jiabao': 'hu_wen',
    'abe_shinzo': 'abe_shinzo', 'shinzo_abe': 'abe_shinzo',
    'koizumi_junichiro': 'koizumi_junichiro',
    'kishida_fumio': 'kishida_fumio', 'takaichi_sanae': 'takaichi_sanae',
    'nakasone_yasuhiro': 'nakasone_yasuhiro',
    'hashimoto_ryutaro': 'hashimoto_ryutaro',
    'tanaka_kakuei': 'tanaka_kakuei',
    'ali_khamenei': 'ali_khamenei', 'mojtaba_khamenei': 'mojtaba_khamenei',
    'elon_musk': 'elon_musk', 'netanyahu': 'netanyahu',
}
_last_name_index = {}
for _pk in profiles.keys():
    for p in _pk.split('_'):
        if len(p) > 2 and p not in _last_name_index:
            _last_name_index[p] = _pk

def _strip_accents(s):
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

def normalize_name(name):
    n = _strip_accents(name.lower()).replace('.', '').replace(' ', '_').replace('-', '_').replace("'", '')
    while '__' in n:
        n = n.replace('__', '_')
    n = n.strip('_')
    if n in name_to_profile:
        return name_to_profile[n]
    parts = n.split('_')
    for part in parts:
        if part in name_to_profile:
            return name_to_profile[part]
    if len(parts) == 2:
        rev = f"{parts[1]}_{parts[0]}"
        if rev in name_to_profile:
            return name_to_profile[rev]
    for pk in profiles.keys():
        if pk in n or n in pk:
            return pk
    for part in reversed(parts):
        if part in _last_name_index and len(part) > 3:
            return _last_name_index[part]
    return None

# Load timelines
timeline_dir = LEADER_BASE / 'timelines'
leader_periods = []

for tf in sorted(timeline_dir.glob('*.json')):
    country = tf.stem.upper()
    with open(tf) as fp:
        tl_data = json.load(fp)
    entries = []
    if isinstance(tl_data, list):
        entries = tl_data
    elif isinstance(tl_data, dict):
        roles_dict = tl_data.get('roles', {})
        if roles_dict and isinstance(roles_dict, dict):
            for role_name, role_data in roles_dict.items():
                if isinstance(role_data, dict):
                    for subkey, subval in role_data.items():
                        if isinstance(subval, list) and subval and isinstance(subval[0], dict):
                            entries.extend(subval)
        for key, val in tl_data.items():
            if key == 'roles':
                continue
            if isinstance(val, list) and val and isinstance(val[0], dict):
                entries.extend(val)
            elif isinstance(val, dict):
                if 'entries' in val:
                    entries.extend(val['entries'])
                for subkey, subval in val.items():
                    if isinstance(subval, list) and subval and isinstance(subval[0], dict):
                        entries.extend(subval)

    for entry in entries:
        name = entry.get('name', entry.get('leader', ''))
        start = entry.get('term_start', entry.get('start', ''))
        end = entry.get('term_end', entry.get('end', ''))
        role = entry.get('role', entry.get('position', ''))
        if not name or not start:
            continue
        if len(start) == 7:
            start = start + '-01'
        if end and len(end) == 7:
            end = end + '-01'
        profile_key = normalize_name(name)
        if profile_key and profile_key in profiles:
            if not end or end.lower() in ('present', 'incumbent', ''):
                end = '2026-03-12'
            leader_periods.append((profile_key, start, end, role, country))

print(f"Leader-time mappings: {len(leader_periods)}")

# ============================================================
# 3. BUILD STATE MATRIX
# ============================================================

print("\n" + "=" * 70)
print("3. BUILD STATE MATRIX")
print("=" * 70)

def generate_months(start_ym, end_ym):
    months = []
    y, m = start_ym
    while (y, m) <= end_ym:
        months.append(f"{y:04d}-{m:02d}-01")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months

all_months = generate_months((1971, 1), (2026, 3))
econ_cols = sorted(all_series.keys())

# Role slots (same as tda_full.py)
ROLE_SLOTS = {
    'US_PRES': ['president'],
    'US_FED': ['fed_chair', 'federal_reserve_chair', 'chairman_of_the_federal_reserve'],
    'CN_LEADER': ['general_secretary', 'paramount_leader', 'chairman_of_cpc', 'president_of_prc'],
    'CN_PREMIER': ['premier_of_state_council', 'premier'],
    'CN_PBOC': ['pboc_governor', 'governor_of_pboc', "governor_of_people's_bank", 'governor,_people'],
    'EU_ECB': ['ecb_president', 'president_of_ecb', 'bundesbank_president'],
    'EU_LEADER': ['german_chancellor', 'french_president'],
    'UK_PM': ['prime_minister'],
    'UK_BOE': ['boe_governor', 'governor_of_bank_of_england', 'bank_of_england_governor'],
    'JP_PM': ['prime_minister'],
    'RU_LEADER': ['president', 'head_of_state', 'general_secretary', 'general_secretary_of_the_cpsu'],
    'RU_CB': ['chairman,_gosbank', 'governor,_central_bank', 'chairman_of_the_central_bank',
              'governor_of_the_central_bank', 'chairman,_bank_of_russia'],
    'IR_LEADER': ['supreme_leader'],
}

def match_role_slot(role_str, country):
    role_lower = role_str.lower().replace('-', '_').replace(' ', '_')
    for slot, keywords in ROLE_SLOTS.items():
        slot_country = slot.split('_')[0]
        if country != slot_country:
            continue
        for kw in keywords:
            if kw in role_lower:
                return slot
    return None

# Behavioral dimensions
dim_counts = {}
for pk, pdata in profiles.items():
    for dim_name in pdata['params'].keys():
        dim_counts[dim_name] = dim_counts.get(dim_name, 0) + 1
top_dims = sorted(dim_counts.keys(), key=lambda d: -dim_counts[d])[:40]
N_LEADER_DIMS = 15

# Role assignments per month
role_assignments = {}
for month in all_months:
    role_assignments[month] = {}
    for profile_key, start, end, role, country in leader_periods:
        if start[:10] <= month <= end[:10]:
            slot = match_role_slot(role, country)
            if slot:
                role_assignments[month][slot] = profile_key

slots_used = sorted(set(s for m in all_months for s in role_assignments[m]))
leader_col_names = [f"L_{slot}_{dim}" for slot in slots_used for dim in top_dims[:N_LEADER_DIMS]]

all_col_names = econ_cols + leader_col_names
n_months = len(all_months)
n_total_cols = len(all_col_names)

matrix = np.full((n_months, n_total_cols), np.nan)

# Fill economic data
for j, col in enumerate(econ_cols):
    series = all_series[col]
    for i, month in enumerate(all_months):
        if month in series:
            matrix[i, j] = series[month]

# Fill leader data
leader_col_start = len(econ_cols)
for i, month in enumerate(all_months):
    assignments = role_assignments[month]
    for s, slot in enumerate(slots_used):
        if slot in assignments:
            pk = assignments[slot]
            params = profiles[pk]['params']
            for d, dim in enumerate(top_dims[:N_LEADER_DIMS]):
                col_idx = leader_col_start + s * N_LEADER_DIMS + d
                if dim in params:
                    matrix[i, col_idx] = params[dim]

# Filter columns by coverage
coverage = np.sum(~np.isnan(matrix), axis=0) / n_months * 100
good_cols = coverage >= 8
matrix_f = matrix[:, good_cols]
col_names_f = [all_col_names[j] for j in range(n_total_cols) if good_cols[j]]

# Filter rows
row_coverage = np.sum(~np.isnan(matrix_f), axis=1) / matrix_f.shape[1] * 100
good_rows = row_coverage >= 25
matrix_g = matrix_f[good_rows]
months_g = [all_months[i] for i in range(n_months) if good_rows[i]]

print(f"State matrix: {matrix_g.shape} ({len(months_g)} months × {matrix_g.shape[1]} features)")

# Impute: forward-fill then backward-fill then median
for j in range(matrix_g.shape[1]):
    last = np.nan
    for i in range(matrix_g.shape[0]):
        if np.isnan(matrix_g[i, j]):
            matrix_g[i, j] = last
        else:
            last = matrix_g[i, j]
for j in range(matrix_g.shape[1]):
    first = np.nan
    for i in range(matrix_g.shape[0]):
        if not np.isnan(matrix_g[i, j]):
            first = matrix_g[i, j]
            break
    for i in range(matrix_g.shape[0]):
        if np.isnan(matrix_g[i, j]):
            matrix_g[i, j] = first
        else:
            break
for j in range(matrix_g.shape[1]):
    col = matrix_g[:, j]
    mask = np.isnan(col)
    if mask.any():
        med = np.nanmedian(col)
        matrix_g[mask, j] = med if not np.isnan(med) else 0.5

# ============================================================
# 4. NORMALIZE + PCA
# ============================================================

print("\n" + "=" * 70)
print("4. QUANTILE NORMALIZATION + PCA")
print("=" * 70)

qt = QuantileTransformer(n_quantiles=min(1000, len(months_g)),
                         output_distribution='uniform', random_state=42)
X_norm = qt.fit_transform(matrix_g)

# Add time dimension
time_vals = np.linspace(0, 1, len(months_g)).reshape(-1, 1)
X_full = np.hstack([X_norm, time_vals])

# PCA
n_comp = min(N_PCA_COMPONENTS + 5, X_full.shape[1] - 1, X_full.shape[0] - 1)
pca = PCA(n_components=n_comp)
X_pca_all = pca.fit_transform(X_full)

# Keep target components
X_pca = X_pca_all[:, :N_PCA_COMPONENTS]
explained = np.cumsum(pca.explained_variance_ratio_)
print(f"PCA: {X_full.shape[1]}D → {N_PCA_COMPONENTS}D ({explained[N_PCA_COMPONENTS-1]*100:.1f}% variance)")
print(f"State space: {X_pca.shape}")

# Month-to-index lookup
month_to_idx = {m: i for i, m in enumerate(months_g)}

# ============================================================
# 5. DEFINE CRISIS REGIONS
# ============================================================

print("\n" + "=" * 70)
print("5. DEFINE CRISIS REGIONS IN STATE SPACE")
print("=" * 70)

crisis_indices = {}  # crisis_name → list of month indices
crisis_all_idx = []  # all crisis month indices combined

for cname, (cstart, cend) in CRISIS_PERIODS.items():
    indices = []
    for i, m in enumerate(months_g):
        ym = m[:7]
        if cstart <= ym <= cend:
            indices.append(i)
    crisis_indices[cname] = indices
    crisis_all_idx.extend(indices)
    if indices:
        print(f"  {cname:35s}: {len(indices)} months")

crisis_all_idx = sorted(set(crisis_all_idx))
print(f"\nTotal crisis months: {len(crisis_all_idx)} / {len(months_g)}")

# Crisis states in PCA space
X_crisis = X_pca[crisis_all_idx]

# Build KNN model for crisis region detection
# A state is "in crisis zone" if it's within the crisis KNN radius
nn_crisis = NearestNeighbors(n_neighbors=min(CRISIS_K_NEIGHBORS, len(X_crisis)))
nn_crisis.fit(X_crisis)

# Compute the radius threshold: 90th percentile of intra-crisis distances
crisis_dists, _ = nn_crisis.kneighbors(X_crisis)
crisis_radius = np.percentile(crisis_dists[:, -1], 90)
print(f"Crisis region radius (90th pct): {crisis_radius:.4f}")

# Also compute per-crisis centroids and radii for similarity analysis
crisis_centroids = {}
crisis_radii = {}
for cname, indices in crisis_indices.items():
    if len(indices) >= 3:
        pts = X_pca[indices]
        centroid = pts.mean(axis=0)
        radius = np.max(np.linalg.norm(pts - centroid, axis=1))
        crisis_centroids[cname] = centroid
        crisis_radii[cname] = radius

def is_in_crisis_zone(state_vec):
    """Check if a state vector is within the crisis region."""
    dist, _ = nn_crisis.kneighbors(state_vec.reshape(1, -1))
    return dist[0, -1] <= crisis_radius

def crisis_proximity(state_vec):
    """Return distance to nearest crisis state (lower = closer to crisis)."""
    dist, _ = nn_crisis.kneighbors(state_vec.reshape(1, -1), n_neighbors=1)
    return dist[0, 0]

# ============================================================
# 6. COMPUTE HISTORICAL TRANSITION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("6. COMPUTE HISTORICAL TRANSITION DISTRIBUTION")
print("=" * 70)

# Monthly deltas in PCA space: delta[t] = X[t+1] - X[t]
deltas = X_pca[1:] - X_pca[:-1]
print(f"Historical transitions: {len(deltas)}")

# Statistics of deltas
delta_mean = deltas.mean(axis=0)
delta_cov = np.cov(deltas.T)
print(f"Delta mean norm: {np.linalg.norm(delta_mean):.6f}")
print(f"Delta std per dim: {np.sqrt(np.diag(delta_cov)).mean():.6f}")

# State-dependent transitions: for each state, find K nearest historical states
# and use their transitions (weighted by distance)
K_NEIGHBORS_TRANSITION = 20

nn_states = NearestNeighbors(n_neighbors=K_NEIGHBORS_TRANSITION)
# Only use states that have a next state (i.e., exclude last month)
nn_states.fit(X_pca[:-1])

def sample_transition(current_state, n_samples=1):
    """Sample transitions from current state using state-dependent historical transitions.

    For each state:
    1. Find K nearest historical states
    2. Get their actual next-month deltas
    3. Sample from these deltas with Gaussian noise

    This preserves state-dependent dynamics: transitions near crisis states
    will use crisis-era transitions, transitions in normal states use normal transitions.
    """
    dists, indices = nn_states.kneighbors(current_state.reshape(1, -1))
    dists = dists[0]
    indices = indices[0]

    # Inverse distance weighting
    weights = 1.0 / (dists + 1e-10)
    weights /= weights.sum()

    # Get the deltas for these neighbors
    neighbor_deltas = deltas[indices]

    # Sample from weighted deltas + noise
    results = np.zeros((n_samples, X_pca.shape[1]))
    chosen = np.random.choice(len(indices), size=n_samples, p=weights)
    for i, c in enumerate(chosen):
        # Base delta from historical transition
        base_delta = neighbor_deltas[c]
        # Add scaled noise based on local variance
        local_std = np.std(neighbor_deltas, axis=0)
        noise = np.random.randn(X_pca.shape[1]) * local_std * NOISE_SCALE
        results[i] = base_delta + noise

    return results

# ============================================================
# 7. CURRENT STATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("7. CURRENT STATE ANALYSIS (2026-03)")
print("=" * 70)

# Current state = last available month
current_idx = len(months_g) - 1
current_month = months_g[current_idx]
current_state = X_pca[current_idx]

print(f"Current month: {current_month}")
print(f"Current state (PCA 14D): [{', '.join(f'{v:.3f}' for v in current_state[:5])}...]")

# Distance to each historical crisis
print(f"\nDistance to historical crises:")
for cname, centroid in sorted(crisis_centroids.items(),
                               key=lambda x: np.linalg.norm(current_state - x[1])):
    dist = np.linalg.norm(current_state - centroid)
    radius = crisis_radii[cname]
    ratio = dist / radius if radius > 0 else float('inf')
    print(f"  {cname:35s}: dist={dist:.4f}  (radius={radius:.4f}, ratio={ratio:.2f}x)")

# Is current state already in crisis zone?
curr_crisis_dist = crisis_proximity(current_state)
print(f"\nNearest crisis state distance: {curr_crisis_dist:.4f} (threshold: {crisis_radius:.4f})")
print(f"Currently in crisis zone: {'YES' if curr_crisis_dist <= crisis_radius else 'NO'}")

# ============================================================
# 8. MONTE CARLO SIMULATION
# ============================================================

print("\n" + "=" * 70)
print(f"8. MONTE CARLO SIMULATION ({N_SIMULATIONS:,} paths, {MAX_HORIZON_MONTHS} months)")
print("=" * 70)

# Track: for each path, when (if ever) does it enter crisis zone
first_crisis_month = np.full(N_SIMULATIONS, MAX_HORIZON_MONTHS + 1, dtype=int)
# Track crisis proximity over time
proximity_over_time = np.zeros((N_SIMULATIONS, MAX_HORIZON_MONTHS))
# Track which crisis each path is closest to
closest_crisis_at_end = []

# Run simulation in batches for efficiency
BATCH_SIZE = 5000
n_batches = (N_SIMULATIONS + BATCH_SIZE - 1) // BATCH_SIZE

for batch in range(n_batches):
    batch_start = batch * BATCH_SIZE
    batch_end = min(batch_start + BATCH_SIZE, N_SIMULATIONS)
    batch_n = batch_end - batch_start

    # Initialize all paths at current state
    paths = np.tile(current_state, (batch_n, 1))  # (batch_n, 14)

    for t in range(MAX_HORIZON_MONTHS):
        # Sample transitions for all paths
        for p in range(batch_n):
            sim_idx = batch_start + p
            if first_crisis_month[sim_idx] <= t:
                # Already hit crisis, continue but mark
                pass

            # Sample one transition
            trans = sample_transition(paths[p], n_samples=1)[0]
            paths[p] = paths[p] + trans

            # Check crisis proximity
            prox = crisis_proximity(paths[p])
            proximity_over_time[sim_idx, t] = prox

            # Check if entered crisis zone
            if prox <= crisis_radius and first_crisis_month[sim_idx] > t:
                first_crisis_month[sim_idx] = t + 1  # 1-indexed month

    # Find closest crisis for each path at end
    for p in range(batch_n):
        min_dist = float('inf')
        min_crisis = "none"
        for cname, centroid in crisis_centroids.items():
            d = np.linalg.norm(paths[p] - centroid)
            if d < min_dist:
                min_dist = d
                min_crisis = cname
        closest_crisis_at_end.append(min_crisis)

    if (batch + 1) % 2 == 0 or batch == n_batches - 1:
        done = batch_end
        crisis_so_far = np.sum(first_crisis_month[:done] <= MAX_HORIZON_MONTHS)
        print(f"  Batch {batch+1}/{n_batches}: {done:,}/{N_SIMULATIONS:,} paths, "
              f"{crisis_so_far} ({crisis_so_far/done*100:.1f}%) hit crisis zone")

# ============================================================
# 9. COMPUTE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("9. RESULTS — Crisis Probability Curve")
print("=" * 70)

# Cumulative probability: P(crisis by month t)
cum_prob = np.zeros(MAX_HORIZON_MONTHS)
for t in range(MAX_HORIZON_MONTHS):
    cum_prob[t] = np.mean(first_crisis_month <= (t + 1))

print(f"\nP(crisis | next t months):")
for t in [1, 3, 6, 12, 18, 24, 36]:
    if t <= MAX_HORIZON_MONTHS:
        print(f"  {t:2d} months: {cum_prob[t-1]*100:6.2f}%")

# Most likely crisis type
from collections import Counter
crisis_type_counts = Counter(closest_crisis_at_end)
print(f"\nMost similar crisis type (at path endpoints):")
for ctype, cnt in crisis_type_counts.most_common(5):
    print(f"  {ctype:35s}: {cnt/N_SIMULATIONS*100:.1f}%")

# Mean proximity over time
mean_prox = np.mean(proximity_over_time, axis=0)
std_prox = np.std(proximity_over_time, axis=0)
p10_prox = np.percentile(proximity_over_time, 10, axis=0)
p90_prox = np.percentile(proximity_over_time, 90, axis=0)

print(f"\nMean crisis proximity over time:")
for t in [0, 5, 11, 17, 23, 35]:
    if t < MAX_HORIZON_MONTHS:
        print(f"  Month {t+1:2d}: mean={mean_prox[t]:.4f}  "
              f"(10th pct={p10_prox[t]:.4f}, 90th pct={p90_prox[t]:.4f})")

# ============================================================
# 10. GENERATE VISUALIZATIONS
# ============================================================

print("\n" + "=" * 70)
print("10. GENERATING VISUALIZATIONS")
print("=" * 70)

# --- Plot 1: Crisis Probability Curve ---
fig, ax = plt.subplots(figsize=(14, 8))
months_range = np.arange(1, MAX_HORIZON_MONTHS + 1)

ax.fill_between(months_range, 0, cum_prob * 100, alpha=0.3, color='red')
ax.plot(months_range, cum_prob * 100, 'r-', linewidth=2.5, label='P(crisis)')

# Add confidence band via bootstrap
n_bootstrap = 1000
bootstrap_probs = np.zeros((n_bootstrap, MAX_HORIZON_MONTHS))
for b in range(n_bootstrap):
    boot_idx = np.random.choice(N_SIMULATIONS, N_SIMULATIONS, replace=True)
    boot_fcm = first_crisis_month[boot_idx]
    for t in range(MAX_HORIZON_MONTHS):
        bootstrap_probs[b, t] = np.mean(boot_fcm <= (t + 1))

p5 = np.percentile(bootstrap_probs, 5, axis=0) * 100
p95 = np.percentile(bootstrap_probs, 95, axis=0) * 100
ax.fill_between(months_range, p5, p95, alpha=0.15, color='red', label='90% CI')

ax.set_xlabel('Months from Now (2026-03)', fontsize=13)
ax.set_ylabel('Cumulative Probability (%)', fontsize=13)
ax.set_title('Monte Carlo Crisis Probability Forecast\n'
             f'From Current State → P(entering crisis zone within t months)\n'
             f'{N_SIMULATIONS:,} simulations, {len(months_g)} months history',
             fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, MAX_HORIZON_MONTHS)
ax.set_ylim(0, max(cum_prob[-1] * 100 * 1.2, 10))

# Add key time markers
for t_mark, label in [(6, '6mo'), (12, '1yr'), (24, '2yr'), (36, '3yr')]:
    if t_mark <= MAX_HORIZON_MONTHS:
        ax.axvline(t_mark, color='gray', linestyle='--', alpha=0.5)
        ax.text(t_mark, ax.get_ylim()[1] * 0.95, label,
                ha='center', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig(OUT_DIR / 'crisis_probability_curve.png', dpi=150)
print(f"  Saved: crisis_probability_curve.png")

# --- Plot 2: Crisis Proximity Over Time ---
fig, ax = plt.subplots(figsize=(14, 8))

ax.fill_between(months_range, p10_prox, p90_prox, alpha=0.2, color='blue', label='10-90th pct')
ax.plot(months_range, mean_prox, 'b-', linewidth=2, label='Mean proximity')
ax.axhline(crisis_radius, color='red', linestyle='--', linewidth=2, label=f'Crisis threshold ({crisis_radius:.3f})')

ax.set_xlabel('Months from Now', fontsize=13)
ax.set_ylabel('Distance to Nearest Crisis State', fontsize=13)
ax.set_title('Crisis Proximity Over Monte Carlo Paths\n'
             'Lower = closer to historical crisis states', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, MAX_HORIZON_MONTHS)

plt.tight_layout()
plt.savefig(OUT_DIR / 'crisis_proximity_over_time.png', dpi=150)
print(f"  Saved: crisis_proximity_over_time.png")

# --- Plot 3: Historical State Space + Current Position + Crisis Zones ---
try:
    import umap
    has_umap = True
except ImportError:
    has_umap = False

if has_umap:
    print("  Computing UMAP projection...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_2d = reducer.fit_transform(X_pca)

    fig, ax = plt.subplots(figsize=(16, 12))

    # Color by year
    years = np.array([int(m[:4]) for m in months_g])
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=years, cmap='viridis',
                         s=15, alpha=0.5, zorder=1)
    plt.colorbar(scatter, label='Year', ax=ax)

    # Highlight crisis periods
    crisis_colors = plt.cm.Set1(np.linspace(0, 1, len(crisis_indices)))
    for idx, (cname, c_indices) in enumerate(crisis_indices.items()):
        if c_indices:
            ax.scatter(X_2d[c_indices, 0], X_2d[c_indices, 1],
                      c=[crisis_colors[idx]], s=60, marker='x',
                      label=cname, zorder=3, linewidths=2)

    # Mark current position
    ax.scatter(X_2d[current_idx, 0], X_2d[current_idx, 1],
              c='red', s=300, marker='*', zorder=5, edgecolors='black',
              linewidths=2, label=f'NOW ({current_month[:7]})')

    # Mark recent trajectory (last 24 months)
    recent_start = max(0, current_idx - 24)
    ax.plot(X_2d[recent_start:current_idx+1, 0],
            X_2d[recent_start:current_idx+1, 1],
            'r-', linewidth=2, alpha=0.7, zorder=4)

    ax.set_title('Economic State Space (UMAP 2D)\n'
                 'Historical trajectory + Crisis zones + Current position', fontsize=14)
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'state_space_crisis_map.png', dpi=150)
    print(f"  Saved: state_space_crisis_map.png")

# --- Plot 4: Crisis Type Distribution ---
fig, ax = plt.subplots(figsize=(12, 6))
types = [t for t, _ in crisis_type_counts.most_common()]
counts = [c/N_SIMULATIONS*100 for _, c in crisis_type_counts.most_common()]
bars = ax.barh(types, counts, color='coral')
ax.set_xlabel('% of Paths Most Similar To', fontsize=13)
ax.set_title('Crisis Type Distribution\n'
             'Which historical crisis does each simulated path most resemble?', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')

for bar, pct in zip(bars, counts):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{pct:.1f}%', va='center', fontsize=11)

plt.tight_layout()
plt.savefig(OUT_DIR / 'crisis_type_distribution.png', dpi=150)
print(f"  Saved: crisis_type_distribution.png")

# ============================================================
# 11. SAVE RESULTS
# ============================================================

results = {
    "timestamp": "2026-03-13",
    "config": {
        "n_simulations": N_SIMULATIONS,
        "max_horizon_months": MAX_HORIZON_MONTHS,
        "n_pca_components": N_PCA_COMPONENTS,
        "crisis_k_neighbors": CRISIS_K_NEIGHBORS,
        "noise_scale": NOISE_SCALE,
        "n_economic_series": len(all_series),
        "n_months_history": len(months_g),
        "state_matrix_shape": list(matrix_g.shape),
    },
    "current_state": {
        "month": current_month,
        "pca_vector": current_state.tolist(),
        "nearest_crisis_distance": float(curr_crisis_dist),
        "crisis_threshold": float(crisis_radius),
        "in_crisis_zone": bool(curr_crisis_dist <= crisis_radius),
    },
    "crisis_probability_curve": {
        f"{t+1}_months": float(cum_prob[t])
        for t in range(MAX_HORIZON_MONTHS)
    },
    "key_probabilities": {
        "3_months": float(cum_prob[2]),
        "6_months": float(cum_prob[5]),
        "12_months": float(cum_prob[11]),
        "18_months": float(cum_prob[17]) if MAX_HORIZON_MONTHS >= 18 else None,
        "24_months": float(cum_prob[23]) if MAX_HORIZON_MONTHS >= 24 else None,
        "36_months": float(cum_prob[35]) if MAX_HORIZON_MONTHS >= 36 else None,
    },
    "crisis_similarity": {
        cname: {
            "distance": float(np.linalg.norm(current_state - centroid)),
            "radius": float(crisis_radii[cname]),
            "ratio": float(np.linalg.norm(current_state - centroid) / crisis_radii[cname]) if crisis_radii[cname] > 0 else None,
        }
        for cname, centroid in crisis_centroids.items()
    },
    "crisis_type_distribution": {
        ctype: cnt / N_SIMULATIONS
        for ctype, cnt in crisis_type_counts.most_common()
    },
    "bootstrap_confidence_intervals": {
        "5th_percentile": {f"{t+1}_months": float(p5[t]/100) for t in range(MAX_HORIZON_MONTHS)},
        "95th_percentile": {f"{t+1}_months": float(p95[t]/100) for t in range(MAX_HORIZON_MONTHS)},
    },
}

with open(OUT_DIR / 'monte_carlo_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved: {OUT_DIR / 'monte_carlo_results.json'}")

print("\n" + "=" * 70)
print("MONTE CARLO CRISIS SIMULATION COMPLETE")
print("=" * 70)
print(f"\nKey findings:")
print(f"  P(crisis within  6 months): {cum_prob[5]*100:.1f}%")
print(f"  P(crisis within 12 months): {cum_prob[11]*100:.1f}%")
print(f"  P(crisis within 24 months): {cum_prob[23]*100:.1f}%" if MAX_HORIZON_MONTHS >= 24 else "")
print(f"  P(crisis within 36 months): {cum_prob[35]*100:.1f}%" if MAX_HORIZON_MONTHS >= 36 else "")
print(f"  Currently in crisis zone: {'YES' if curr_crisis_dist <= crisis_radius else 'NO'}")
print(f"\nOutput directory: {OUT_DIR}")
