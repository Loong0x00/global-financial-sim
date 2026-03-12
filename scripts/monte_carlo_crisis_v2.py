#!/usr/bin/env python3
"""
Monte Carlo Crisis Probability Engine v2 — GPU Accelerated
============================================================
v1 的问题：用"进入历史危机区域"定义崩溃 → 0% 概率，因为 TDA 已证明 2024+ 是拓扑孤岛。

v2 改进：
1. GPU 并行（PyTorch）— 50K 路径全部在 GPU 上跑
2. 三个新的危机指标（不依赖"与历史危机相似"）：
   A. 拓扑密度骤降 — 进入稀疏区域 = 未知领域 = 风险
   B. 状态变化加速度 — 月度 delta 突然增大 = 不稳定
   C. 历史轨迹偏离度 — 偏离所有已知模式的程度
3. 综合崩溃概率 = 三个指标的联合阈值
"""

import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ============================================================
# CONFIG
# ============================================================

DATA_BASE = Path('/home/user/global-financial-sim/data')
ECON_BASE = DATA_BASE / 'economic'
LEADER_BASE = DATA_BASE / 'leaders'
OUT_DIR = Path('/home/user/global-financial-sim/output/monte_carlo')
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SIMULATIONS = 100000       # 10万条路径（GPU 轻松搞定）
MAX_HORIZON_MONTHS = 36      # 预测 36 个月
N_PCA_COMPONENTS = 14        # 与 TDA full 一致
K_DENSITY = 15               # 计算局部密度的 KNN
K_TRANSITION = 20            # 转移采样的 KNN
NOISE_SCALE = 0.15           # 转移噪声缩放

# 危机指标阈值（基于历史分布自动校准）
DENSITY_CRISIS_PERCENTILE = 5     # 密度低于历史 5% 分位 = 危机
ACCEL_CRISIS_PERCENTILE = 95      # 加速度高于历史 95% 分位 = 危机
DEVIATION_CRISIS_PERCENTILE = 95  # 偏离度高于历史 95% 分位 = 危机

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
torch.manual_seed(42)

# ============================================================
# 1. LOAD ALL DATA (same loading pipeline as v1)
# ============================================================

print("=" * 70)
print("1. LOADING DATA")
print("=" * 70)

all_series = {}

def daily_to_monthly(data_points, value_key='close'):
    monthly = {}
    for pt in data_points:
        ym = pt['date'][:7] + "-01"
        val = pt.get(value_key, pt.get('value'))
        if val is None:
            continue
        monthly.setdefault(ym, []).append(float(val))
    return {k: np.mean(v) for k, v in monthly.items()}

# --- FRED macro/financial ---
for country in ['us', 'cn', 'eu', 'uk', 'jp', 'ru']:
    prefix = country.upper()
    for ftype in ['macro', 'financial']:
        fpath = ECON_BASE / country / f'{ftype}.json'
        if not fpath.exists():
            continue
        with open(fpath) as f:
            data = json.load(f)
        for name, ind in data.get('indicators', data).items():
            dv = {}
            for pt in ind.get('series', []):
                if pt['value'] is not None:
                    dv[pt['date']] = float(pt['value'])
            if dv:
                all_series[f"ECON_{prefix}_{name}"] = dv

# --- Indices ---
p = ECON_BASE / 'indices' / 'global_indices.json'
if p.exists():
    with open(p) as f:
        for name, info in json.load(f).items():
            m = daily_to_monthly(info.get('data', []))
            if m:
                all_series[f"IDX_{name}"] = m

# --- Metals ---
for metal in ['gold', 'silver']:
    p = ECON_BASE / 'commodities' / f'{metal}_daily.json'
    if p.exists():
        with open(p) as f:
            m = daily_to_monthly(json.load(f).get('data', []), 'close')
        if m:
            all_series[f"METAL_{metal}"] = m

# --- FRED Commodities ---
p = ECON_BASE / 'commodities' / 'fred_commodities.json'
if p.exists():
    with open(p) as f:
        fc = json.load(f)
    for name, info in fc.items():
        if not isinstance(info, dict):
            continue
        dv = {}
        for pt in info.get('data', info.get('series', [])):
            val = pt.get('value', pt.get('close'))
            if val is not None:
                dv[pt['date']] = float(val)
        if dv:
            all_series[f"COMM_FRED_{name}"] = dv

# --- Yahoo Futures ---
p = ECON_BASE / 'commodities' / 'yahoo_futures.json'
if p.exists():
    with open(p) as f:
        yf = json.load(f)
    for name, info in yf.items():
        if isinstance(info, dict) and info.get('data'):
            m = daily_to_monthly(info['data'])
            if m:
                all_series[f"COMM_YF_{name}"] = m

# --- Crypto ---
p = ECON_BASE / 'crypto' / 'crypto.json'
if p.exists():
    with open(p) as f:
        for name, info in json.load(f).items():
            m = daily_to_monthly(info.get('data', []))
            if m:
                all_series[f"CRYPTO_{name}"] = m

# --- BIS ---
bis_dir = ECON_BASE / 'bis'
if bis_dir.exists():
    for fname in ['credit_gap.json', 'property_prices.json', 'debt_service_ratios.json',
                   'total_credit.json', 'global_liquidity.json']:
        fpath = bis_dir / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            bis_data = json.load(f)
        ds_name = fname.replace('.json', '')
        for cc, cdata in bis_data.get('countries', {}).items():
            sd = cdata.get('series', {})
            if not isinstance(sd, dict):
                continue
            for idx, (sn, si) in enumerate(sd.items()):
                if not isinstance(si, dict) or 'data' not in si:
                    continue
                dv = {}
                raw = si['data']
                items_iter = raw.items() if isinstance(raw, dict) else raw
                for item in items_iter:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        d_raw, val = item
                    elif isinstance(item, dict):
                        d_raw, val = item.get('date', ''), item.get('value')
                    else:
                        continue
                    if val is None:
                        continue
                    d_str = str(d_raw)
                    if 'Q' in d_str:
                        parts = d_str.split('-Q')
                        if len(parts) == 2:
                            yr, q = parts[0], int(parts[1])
                            mo = (q - 1) * 3 + 1
                            for off in range(3):
                                dv[f"{yr}-{mo+off:02d}-01"] = float(val)
                    elif len(d_str) >= 7:
                        try:
                            dv[d_str[:7] + '-01'] = float(val)
                        except:
                            pass
                if dv:
                    all_series[f"BIS_{ds_name}_{cc}_{idx}"] = dv

# --- Supplementary ---
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
            dv = {}
            for pt in info['data']:
                d, v = pt.get('date', ''), pt.get('value')
                if d and v is not None:
                    try:
                        dv[d[:7] + '-01'] = float(v)
                    except:
                        pass
            if dv:
                all_series[f"SUPP_{name}"] = dv

# --- CFTC ---
p = ECON_BASE / 'cftc' / 'cot_data.json'
if p.exists():
    with open(p) as f:
        cftc = json.load(f)
    for cn, recs in cftc.items():
        if not isinstance(recs, list):
            continue
        mn, mo = {}, {}
        for r in recs:
            d = r.get('date', '')[:7]
            if not d:
                continue
            k = d + '-01'
            net, oi = r.get('net_speculative'), r.get('open_interest')
            if net is not None:
                mn.setdefault(k, []).append(float(net))
            if oi is not None:
                mo.setdefault(k, []).append(float(oi))
        if mn:
            all_series[f"CFTC_NET_{cn}"] = {k: np.mean(v) for k, v in mn.items()}
        if mo:
            all_series[f"CFTC_OI_{cn}"] = {k: np.mean(v) for k, v in mo.items()}

# --- Trade ---
p = ECON_BASE / 'trade' / 'bilateral_trade.json'
if p.exists():
    with open(p) as f:
        trade = json.load(f)
    for pn, pd in trade.items():
        if not isinstance(pd, dict) or 'data' not in pd:
            continue
        for r in pd['data']:
            yr, total, yoy = r.get('year'), r.get('total'), r.get('yoy_change_pct')
            if yr and total:
                for m in range(1, 13):
                    all_series.setdefault(f"TRADE_{pn}_total", {})[f"{yr}-{m:02d}-01"] = float(total)
            if yr and yoy is not None:
                for m in range(1, 13):
                    all_series.setdefault(f"TRADE_{pn}_yoy", {})[f"{yr}-{m:02d}-01"] = float(yoy)

print(f"Economic series: {len(all_series)}")

# --- Leader profiles + timelines (simplified) ---
import unicodedata
profiles = {}
for f in sorted((LEADER_BASE / 'profiles').glob('*.json')):
    with open(f) as fp:
        try:
            d = json.load(fp)
        except:
            continue
    vecs = d.get('behavior_matrix', {}).get('vectors', [])
    if vecs:
        params = {}
        for v in vecs:
            lbl = v.get('label', v.get('name', ''))
            val = v.get('value')
            if lbl and val is not None:
                params[lbl] = float(val)
        if params:
            profiles[f.stem] = {'params': params}

name_to_profile = {
    'xi_jinping': 'xi_jinping', 'trump': 'trump', 'donald_trump': 'trump',
    'powell': 'powell', 'jerome_powell': 'powell',
    'putin': 'putin', 'vladimir_putin': 'putin',
    'bernanke': 'bernanke', 'yellen': 'yellen',
    'obama': 'obama', 'george_w_bush': 'george_w_bush', 'greenspan': 'greenspan',
    'draghi': 'draghi', 'lagarde': 'lagarde', 'merkel': 'merkel', 'sarkozy': 'sarkozy',
    'thatcher': 'thatcher', 'blair': 'blair', 'gordon_brown': 'gordon_brown',
    'nabiullina': 'nabiullina', 'yeltsin': 'yeltsin',
    'brezhnev': 'brezhnev', 'khrushchev': 'khrushchev', 'stalin': 'stalin', 'lenin': 'lenin',
    'zhou_xiaochuan': 'zhou_xiaochuan', 'hu_jintao': 'hu_wen', 'wen_jiabao': 'hu_wen',
    'abe_shinzo': 'abe_shinzo', 'koizumi_junichiro': 'koizumi_junichiro',
    'kishida_fumio': 'kishida_fumio', 'takaichi_sanae': 'takaichi_sanae',
    'nakasone_yasuhiro': 'nakasone_yasuhiro', 'hashimoto_ryutaro': 'hashimoto_ryutaro',
    'tanaka_kakuei': 'tanaka_kakuei',
    'ali_khamenei': 'ali_khamenei', 'mojtaba_khamenei': 'mojtaba_khamenei',
    'elon_musk': 'elon_musk', 'netanyahu': 'netanyahu',
}
_lni = {}
for _pk in profiles:
    for p in _pk.split('_'):
        if len(p) > 2 and p not in _lni:
            _lni[p] = _pk

def normalize_name(name):
    n = unicodedata.normalize('NFKD', name.lower())
    n = ''.join(c for c in n if not unicodedata.combining(c))
    n = n.replace('.', '').replace(' ', '_').replace('-', '_').replace("'", '')
    while '__' in n:
        n = n.replace('__', '_')
    n = n.strip('_')
    if n in name_to_profile:
        return name_to_profile[n]
    for part in n.split('_'):
        if part in name_to_profile:
            return name_to_profile[part]
    parts = n.split('_')
    if len(parts) == 2:
        rev = f"{parts[1]}_{parts[0]}"
        if rev in name_to_profile:
            return name_to_profile[rev]
    for pk in profiles:
        if pk in n or n in pk:
            return pk
    for part in reversed(parts):
        if part in _lni and len(part) > 3:
            return _lni[part]
    return None

leader_periods = []
for tf in sorted((LEADER_BASE / 'timelines').glob('*.json')):
    country = tf.stem.upper()
    with open(tf) as fp:
        tl = json.load(fp)
    entries = []
    if isinstance(tl, list):
        entries = tl
    elif isinstance(tl, dict):
        rd = tl.get('roles', {})
        if isinstance(rd, dict):
            for rn, rdata in rd.items():
                if isinstance(rdata, dict):
                    for sk, sv in rdata.items():
                        if isinstance(sv, list) and sv and isinstance(sv[0], dict):
                            entries.extend(sv)
        for key, val in tl.items():
            if key == 'roles':
                continue
            if isinstance(val, list) and val and isinstance(val[0], dict):
                entries.extend(val)
            elif isinstance(val, dict):
                if 'entries' in val:
                    entries.extend(val['entries'])
                for sk, sv in val.items():
                    if isinstance(sv, list) and sv and isinstance(sv[0], dict):
                        entries.extend(sv)
    for e in entries:
        nm = e.get('name', e.get('leader', ''))
        st = e.get('term_start', e.get('start', ''))
        en = e.get('term_end', e.get('end', ''))
        rl = e.get('role', e.get('position', ''))
        if not nm or not st:
            continue
        if len(st) == 7: st += '-01'
        if en and len(en) == 7: en += '-01'
        pk = normalize_name(nm)
        if pk and pk in profiles:
            if not en or en.lower() in ('present', 'incumbent', ''):
                en = '2026-03-12'
            leader_periods.append((pk, st, en, rl, country))

print(f"Leader profiles: {len(profiles)}, time mappings: {len(leader_periods)}")

# ============================================================
# 2. BUILD STATE MATRIX
# ============================================================

print("\n" + "=" * 70)
print("2. BUILD STATE MATRIX")
print("=" * 70)

def gen_months(sy, sm, ey, em):
    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}-01")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months

all_months = gen_months(1971, 1, 2026, 3)
econ_cols = sorted(all_series.keys())

ROLE_SLOTS = {
    'US_PRES': ['president'], 'US_FED': ['fed_chair', 'federal_reserve_chair', 'chairman_of_the_federal_reserve'],
    'CN_LEADER': ['general_secretary', 'paramount_leader', 'chairman_of_cpc', 'president_of_prc'],
    'CN_PREMIER': ['premier_of_state_council', 'premier'],
    'CN_PBOC': ['pboc_governor', 'governor_of_pboc', "governor_of_people's_bank", 'governor,_people'],
    'EU_ECB': ['ecb_president', 'president_of_ecb', 'bundesbank_president'],
    'EU_LEADER': ['german_chancellor', 'french_president'],
    'UK_PM': ['prime_minister'], 'UK_BOE': ['boe_governor', 'governor_of_bank_of_england'],
    'JP_PM': ['prime_minister'],
    'RU_LEADER': ['president', 'head_of_state', 'general_secretary', 'general_secretary_of_the_cpsu'],
    'RU_CB': ['chairman,_gosbank', 'governor,_central_bank', 'chairman_of_the_central_bank',
              'governor_of_the_central_bank', 'chairman,_bank_of_russia'],
    'IR_LEADER': ['supreme_leader'],
}

def match_role_slot(role_str, country):
    rl = role_str.lower().replace('-', '_').replace(' ', '_')
    for slot, kws in ROLE_SLOTS.items():
        if country != slot.split('_')[0]:
            continue
        for kw in kws:
            if kw in rl:
                return slot
    return None

dim_counts = {}
for pdata in profiles.values():
    for dn in pdata['params']:
        dim_counts[dn] = dim_counts.get(dn, 0) + 1
top_dims = sorted(dim_counts, key=lambda d: -dim_counts[d])[:40]
N_LEADER_DIMS = 15

role_assignments = {}
for month in all_months:
    role_assignments[month] = {}
    for pk, st, en, rl, co in leader_periods:
        if st[:10] <= month <= en[:10]:
            slot = match_role_slot(rl, co)
            if slot:
                role_assignments[month][slot] = pk

slots_used = sorted(set(s for m in all_months for s in role_assignments[m]))
leader_col_names = [f"L_{sl}_{dm}" for sl in slots_used for dm in top_dims[:N_LEADER_DIMS]]
all_col_names = econ_cols + leader_col_names

n_months = len(all_months)
n_cols = len(all_col_names)
matrix = np.full((n_months, n_cols), np.nan)

for j, col in enumerate(econ_cols):
    s = all_series[col]
    for i, mo in enumerate(all_months):
        if mo in s:
            matrix[i, j] = s[mo]

lcs = len(econ_cols)
for i, mo in enumerate(all_months):
    asgn = role_assignments[mo]
    for s_idx, slot in enumerate(slots_used):
        if slot in asgn:
            pk = asgn[slot]
            par = profiles[pk]['params']
            for d, dm in enumerate(top_dims[:N_LEADER_DIMS]):
                ci = lcs + s_idx * N_LEADER_DIMS + d
                if dm in par:
                    matrix[i, ci] = par[dm]

# Filter
cov = np.sum(~np.isnan(matrix), axis=0) / n_months * 100
gc = cov >= 8
mat = matrix[:, gc]
cn_f = [all_col_names[j] for j in range(n_cols) if gc[j]]

rc = np.sum(~np.isnan(mat), axis=1) / mat.shape[1] * 100
gr = rc >= 25
mat = mat[gr]
months_g = [all_months[i] for i in range(n_months) if gr[i]]

# Impute
for j in range(mat.shape[1]):
    last = np.nan
    for i in range(mat.shape[0]):
        if np.isnan(mat[i, j]):
            mat[i, j] = last
        else:
            last = mat[i, j]
for j in range(mat.shape[1]):
    first = np.nan
    for i in range(mat.shape[0]):
        if not np.isnan(mat[i, j]):
            first = mat[i, j]
            break
    if not np.isnan(first):
        for i in range(mat.shape[0]):
            if np.isnan(mat[i, j]):
                mat[i, j] = first
            else:
                break
for j in range(mat.shape[1]):
    col = mat[:, j]
    m = np.isnan(col)
    if m.any():
        med = np.nanmedian(col)
        mat[m, j] = med if not np.isnan(med) else 0.5

print(f"State matrix: {mat.shape}")

# ============================================================
# 3. NORMALIZE + PCA
# ============================================================

print("\n" + "=" * 70)
print("3. NORMALIZE + PCA")
print("=" * 70)

qt = QuantileTransformer(n_quantiles=min(1000, len(months_g)),
                         output_distribution='uniform', random_state=42)
X_norm = qt.fit_transform(mat)

time_vals = np.linspace(0, 1, len(months_g)).reshape(-1, 1)
X_full = np.hstack([X_norm, time_vals])

n_comp = min(N_PCA_COMPONENTS + 5, X_full.shape[1] - 1, X_full.shape[0] - 1)
pca = PCA(n_components=n_comp)
X_pca_all = pca.fit_transform(X_full)
X_pca = X_pca_all[:, :N_PCA_COMPONENTS]

explained = np.cumsum(pca.explained_variance_ratio_)
print(f"PCA: {X_full.shape[1]}D → {N_PCA_COMPONENTS}D ({explained[N_PCA_COMPONENTS-1]*100:.1f}% variance)")

N = len(months_g)
D = N_PCA_COMPONENTS

# ============================================================
# 4. COMPUTE HISTORICAL CRISIS INDICATORS (for calibration)
# ============================================================

print("\n" + "=" * 70)
print("4. CALIBRATE CRISIS INDICATORS FROM HISTORY")
print("=" * 70)

# --- Indicator A: Local density (KNN average distance, lower = denser) ---
nn_density = NearestNeighbors(n_neighbors=K_DENSITY)
nn_density.fit(X_pca)
knn_dists, _ = nn_density.kneighbors(X_pca)
local_density = knn_dists.mean(axis=1)  # higher = sparser = more dangerous

# --- Indicator B: State change acceleration ---
# delta[t] = X[t] - X[t-1], accel[t] = |delta[t]| - |delta[t-1]|
deltas = np.diff(X_pca, axis=0)  # (N-1, D)
delta_norms = np.linalg.norm(deltas, axis=1)  # (N-1,)
# acceleration = change in speed (positive = speeding up)
accel = np.diff(delta_norms)  # (N-2,)
# Pad to align with months (first 2 months have no accel)
accel_full = np.zeros(N)
accel_full[2:] = accel

# --- Indicator C: Deviation from historical mean trajectory ---
# Rolling window deviation: how far is current state from the mean of recent K months?
WINDOW = 24  # 2-year rolling window
deviation_full = np.zeros(N)
for i in range(N):
    start = max(0, i - WINDOW)
    if i - start < 3:
        deviation_full[i] = 0
    else:
        local_mean = X_pca[start:i].mean(axis=0)
        deviation_full[i] = np.linalg.norm(X_pca[i] - local_mean)

# Calibrate thresholds from historical data
density_threshold = np.percentile(local_density, DENSITY_CRISIS_PERCENTILE)
# For density: LOWER density = denser = safer; we want the SPARSE threshold
density_crisis_threshold = np.percentile(local_density, 100 - DENSITY_CRISIS_PERCENTILE)
accel_crisis_threshold = np.percentile(np.abs(accel_full), ACCEL_CRISIS_PERCENTILE)
deviation_crisis_threshold = np.percentile(deviation_full, DEVIATION_CRISIS_PERCENTILE)

print(f"Historical density:    mean={local_density.mean():.4f}, crisis threshold (>{100-DENSITY_CRISIS_PERCENTILE}th pct)={density_crisis_threshold:.4f}")
print(f"Historical |accel|:    mean={np.abs(accel_full).mean():.4f}, crisis threshold (>{ACCEL_CRISIS_PERCENTILE}th pct)={accel_crisis_threshold:.4f}")
print(f"Historical deviation:  mean={deviation_full.mean():.4f}, crisis threshold (>{DEVIATION_CRISIS_PERCENTILE}th pct)={deviation_crisis_threshold:.4f}")

# Validate: check which historical months exceed these thresholds
print(f"\nHistorical crisis indicator validation:")
crisis_month_set = set()
for cname, (cs, ce) in CRISIS_PERIODS.items():
    for i, m in enumerate(months_g):
        if cs <= m[:7] <= ce:
            crisis_month_set.add(i)

# For each indicator, what % of flagged months are actual crisis months?
for name, vals, thr, direction in [
    ("Density (sparse)", local_density, density_crisis_threshold, "above"),
    ("|Acceleration|", np.abs(accel_full), accel_crisis_threshold, "above"),
    ("Deviation", deviation_full, deviation_crisis_threshold, "above"),
]:
    if direction == "above":
        flagged = set(np.where(vals > thr)[0])
    else:
        flagged = set(np.where(vals < thr)[0])
    overlap = flagged & crisis_month_set
    precision = len(overlap) / len(flagged) * 100 if flagged else 0
    recall = len(overlap) / len(crisis_month_set) * 100 if crisis_month_set else 0
    print(f"  {name:25s}: flagged={len(flagged):3d}, crisis overlap={len(overlap):3d}, "
          f"precision={precision:.1f}%, recall={recall:.1f}%")

# Current state indicators
curr_idx = N - 1
curr_density = local_density[curr_idx]
curr_accel = np.abs(accel_full[curr_idx])
curr_deviation = deviation_full[curr_idx]

print(f"\nCurrent state (month {months_g[curr_idx][:7]}):")
print(f"  Density:   {curr_density:.4f} (threshold: {density_crisis_threshold:.4f}) {'⚠ SPARSE' if curr_density > density_crisis_threshold else '✓ OK'}")
print(f"  |Accel|:   {curr_accel:.4f} (threshold: {accel_crisis_threshold:.4f}) {'⚠ HIGH' if curr_accel > accel_crisis_threshold else '✓ OK'}")
print(f"  Deviation: {curr_deviation:.4f} (threshold: {deviation_crisis_threshold:.4f}) {'⚠ HIGH' if curr_deviation > deviation_crisis_threshold else '✓ OK'}")

# ============================================================
# 5. PREPARE GPU TENSORS FOR MONTE CARLO
# ============================================================

print("\n" + "=" * 70)
print("5. PREPARE GPU TENSORS")
print("=" * 70)

# Historical states and deltas on GPU
X_gpu = torch.tensor(X_pca, dtype=torch.float32, device=DEVICE)       # (N, D)
deltas_gpu = torch.tensor(deltas, dtype=torch.float32, device=DEVICE)  # (N-1, D)
delta_norms_gpu = torch.tensor(delta_norms, dtype=torch.float32, device=DEVICE)

# Precompute KNN for transition sampling (on CPU, results to GPU)
nn_trans = NearestNeighbors(n_neighbors=K_TRANSITION)
nn_trans.fit(X_pca[:-1])  # exclude last month (no next delta)

# For GPU-accelerated KNN approximation during simulation,
# precompute the full historical state matrix norm for fast distance computation
X_trans_gpu = X_gpu[:-1]  # (N-1, D) — states that have transitions

# Precompute delta statistics per local region
# For efficiency, cluster the state space into regions
from sklearn.cluster import KMeans
N_CLUSTERS = min(50, N // 10)
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_pca[:-1])
cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=DEVICE)

# For each cluster, precompute delta mean and covariance
cluster_delta_mean = torch.zeros(N_CLUSTERS, D, device=DEVICE)
cluster_delta_std = torch.zeros(N_CLUSTERS, D, device=DEVICE)
cluster_delta_all = [[] for _ in range(N_CLUSTERS)]

for i in range(len(cluster_labels)):
    cluster_delta_all[cluster_labels[i]].append(deltas[i])

for c in range(N_CLUSTERS):
    if cluster_delta_all[c]:
        cd = np.array(cluster_delta_all[c])
        cluster_delta_mean[c] = torch.tensor(cd.mean(axis=0), dtype=torch.float32, device=DEVICE)
        cluster_delta_std[c] = torch.tensor(cd.std(axis=0) + 1e-8, dtype=torch.float32, device=DEVICE)

print(f"GPU tensors ready: {DEVICE}")
print(f"  States: {X_gpu.shape}")
print(f"  Deltas: {deltas_gpu.shape}")
print(f"  Clusters: {N_CLUSTERS}")

# Also put density KNN data on GPU for fast proximity computation
# Precompute all historical states for batch distance computation
X_all_gpu = X_gpu  # (N, D)

# ============================================================
# 6. GPU MONTE CARLO SIMULATION
# ============================================================

print("\n" + "=" * 70)
print(f"6. GPU MONTE CARLO ({N_SIMULATIONS:,} paths × {MAX_HORIZON_MONTHS} months)")
print("=" * 70)

t_start = time.time()

# Initialize all paths at current state
current_state = X_gpu[curr_idx]  # (D,)
paths = current_state.unsqueeze(0).expand(N_SIMULATIONS, -1).clone()  # (N_SIM, D)

# Track indicators over time
# Shape: (N_SIM, MAX_HORIZON)
density_over_time = torch.zeros(N_SIMULATIONS, MAX_HORIZON_MONTHS, device=DEVICE)
accel_over_time = torch.zeros(N_SIMULATIONS, MAX_HORIZON_MONTHS, device=DEVICE)
deviation_over_time = torch.zeros(N_SIMULATIONS, MAX_HORIZON_MONTHS, device=DEVICE)

# Track previous delta norm for acceleration
prev_delta_norm = torch.zeros(N_SIMULATIONS, device=DEVICE)

# Rolling window buffer for deviation (last WINDOW states per path)
# To save memory, only track mean and count
path_sum = current_state.unsqueeze(0).expand(N_SIMULATIONS, -1).clone()  # running sum
path_count = torch.ones(N_SIMULATIONS, device=DEVICE)

for t in range(MAX_HORIZON_MONTHS):
    # 1. Find nearest cluster for each path
    # distances: (N_SIM, N_CLUSTERS)
    diffs = paths.unsqueeze(1) - cluster_centers.unsqueeze(0)  # (N_SIM, N_CL, D)
    dist_sq = (diffs ** 2).sum(dim=2)  # (N_SIM, N_CL)
    nearest_cluster = dist_sq.argmin(dim=1)  # (N_SIM,)

    # 2. Sample transition from cluster statistics
    means = cluster_delta_mean[nearest_cluster]  # (N_SIM, D)
    stds = cluster_delta_std[nearest_cluster]    # (N_SIM, D)
    noise = torch.randn_like(means)
    delta = means + stds * noise * NOISE_SCALE   # (N_SIM, D)

    # 3. Apply transition
    paths = paths + delta

    # 4. Compute indicators

    # A. Density: average distance to K nearest historical states
    # Batch computation: (N_SIM, N_HISTORY)
    # For memory efficiency, process in chunks
    CHUNK = 10000
    density_vals = torch.zeros(N_SIMULATIONS, device=DEVICE)
    for ch_start in range(0, N_SIMULATIONS, CHUNK):
        ch_end = min(ch_start + CHUNK, N_SIMULATIONS)
        chunk_paths = paths[ch_start:ch_end]  # (chunk, D)
        # Distance to all historical states
        d = torch.cdist(chunk_paths, X_all_gpu)  # (chunk, N)
        # Top-K nearest
        topk_dists, _ = d.topk(K_DENSITY, dim=1, largest=False)  # (chunk, K)
        density_vals[ch_start:ch_end] = topk_dists.mean(dim=1)

    density_over_time[:, t] = density_vals

    # B. Acceleration: |current delta| - |previous delta|
    curr_delta_norm = torch.norm(delta, dim=1)  # (N_SIM,)
    if t > 0:
        accel_over_time[:, t] = torch.abs(curr_delta_norm - prev_delta_norm)
    prev_delta_norm = curr_delta_norm

    # C. Deviation: distance from rolling mean
    path_sum = path_sum + paths
    path_count = path_count + 1
    effective_count = torch.clamp(path_count, max=WINDOW)
    # Approximate rolling mean using exponential decay
    alpha = 1.0 / effective_count.unsqueeze(1)
    running_mean = path_sum / path_count.unsqueeze(1)
    deviation_over_time[:, t] = torch.norm(paths - running_mean, dim=1)

    if (t + 1) % 6 == 0:
        elapsed = time.time() - t_start
        mean_dens = density_vals.mean().item()
        mean_dev = deviation_over_time[:, t].mean().item()
        print(f"  Month {t+1:2d}/{MAX_HORIZON_MONTHS}: "
              f"density={mean_dens:.4f}, deviation={mean_dev:.4f}, "
              f"elapsed={elapsed:.1f}s")

total_time = time.time() - t_start
print(f"\nSimulation complete: {total_time:.1f}s ({N_SIMULATIONS:,} paths × {MAX_HORIZON_MONTHS} months)")
print(f"  Throughput: {N_SIMULATIONS * MAX_HORIZON_MONTHS / total_time:.0f} state-transitions/sec")

# ============================================================
# 7. COMPUTE CRISIS PROBABILITIES
# ============================================================

print("\n" + "=" * 70)
print("7. CRISIS PROBABILITY ANALYSIS")
print("=" * 70)

# Move to CPU for analysis
density_cpu = density_over_time.cpu().numpy()
accel_cpu = accel_over_time.cpu().numpy()
deviation_cpu = deviation_over_time.cpu().numpy()

# Crisis condition: ANY of the three indicators exceeds its threshold
density_crisis = density_cpu > density_crisis_threshold
accel_crisis = accel_cpu > accel_crisis_threshold
deviation_crisis = deviation_cpu > deviation_crisis_threshold

# Three crisis definitions:
# 1. Any single indicator (most sensitive)
any_crisis = density_crisis | accel_crisis | deviation_crisis
# 2. At least 2 of 3 indicators (balanced)
two_of_three = ((density_crisis.astype(int) + accel_crisis.astype(int) + deviation_crisis.astype(int)) >= 2)
# 3. All three (most conservative)
all_three = density_crisis & accel_crisis & deviation_crisis

def compute_first_crisis(crisis_matrix):
    """For each path, find the first month where crisis condition is met."""
    first = np.full(N_SIMULATIONS, MAX_HORIZON_MONTHS + 1, dtype=int)
    for i in range(N_SIMULATIONS):
        hits = np.where(crisis_matrix[i])[0]
        if len(hits) > 0:
            first[i] = hits[0] + 1  # 1-indexed
    return first

first_any = compute_first_crisis(any_crisis)
first_two = compute_first_crisis(two_of_three)
first_all = compute_first_crisis(all_three)

# Cumulative probability curves
def cum_prob_curve(first_crisis_array):
    return np.array([np.mean(first_crisis_array <= (t + 1)) for t in range(MAX_HORIZON_MONTHS)])

cum_any = cum_prob_curve(first_any)
cum_two = cum_prob_curve(first_two)
cum_all = cum_prob_curve(first_all)

print(f"\n{'Horizon':>10s}  {'Any 1/3':>10s}  {'2/3':>10s}  {'All 3':>10s}")
print("-" * 45)
for t in [1, 3, 6, 12, 18, 24, 36]:
    if t <= MAX_HORIZON_MONTHS:
        print(f"  {t:2d} months  {cum_any[t-1]*100:9.2f}%  {cum_two[t-1]*100:9.2f}%  {cum_all[t-1]*100:9.2f}%")

# Per-indicator breakdown
cum_density = cum_prob_curve(compute_first_crisis(density_crisis))
cum_accel = cum_prob_curve(compute_first_crisis(accel_crisis))
cum_deviation = cum_prob_curve(compute_first_crisis(deviation_crisis))

print(f"\nPer-indicator breakdown:")
print(f"{'Horizon':>10s}  {'Density':>10s}  {'Accel':>10s}  {'Deviation':>10s}")
print("-" * 45)
for t in [1, 3, 6, 12, 18, 24, 36]:
    if t <= MAX_HORIZON_MONTHS:
        print(f"  {t:2d} months  {cum_density[t-1]*100:9.2f}%  {cum_accel[t-1]*100:9.2f}%  {cum_deviation[t-1]*100:9.2f}%")

# ============================================================
# 8. VISUALIZATIONS
# ============================================================

print("\n" + "=" * 70)
print("8. GENERATING VISUALIZATIONS")
print("=" * 70)

months_range = np.arange(1, MAX_HORIZON_MONTHS + 1)

# --- Plot 1: Triple Crisis Probability Curve ---
fig, ax = plt.subplots(figsize=(14, 8))

ax.fill_between(months_range, 0, cum_any * 100, alpha=0.15, color='red')
ax.plot(months_range, cum_any * 100, 'r-', linewidth=2.5, label='Any 1 of 3 indicators (sensitive)')
ax.plot(months_range, cum_two * 100, 'orange', linewidth=2.5, linestyle='--', label='2 of 3 indicators (balanced)')
ax.plot(months_range, cum_all * 100, 'darkred', linewidth=2.5, linestyle=':', label='All 3 indicators (conservative)')

ax.set_xlabel('Months from Now (2026-03)', fontsize=13)
ax.set_ylabel('Cumulative Crisis Probability (%)', fontsize=13)
ax.set_title(f'Monte Carlo Crisis Probability — New State-Based Indicators\n'
             f'{N_SIMULATIONS:,} simulations, {len(months_g)} months history, GPU accelerated\n'
             f'Indicators: Density drop | Acceleration spike | Trajectory deviation',
             fontsize=13)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(1, MAX_HORIZON_MONTHS)

for t_mark, label in [(6, '6mo'), (12, '1yr'), (24, '2yr'), (36, '3yr')]:
    if t_mark <= MAX_HORIZON_MONTHS:
        ax.axvline(t_mark, color='gray', linestyle='--', alpha=0.4)
        ax.text(t_mark + 0.3, ax.get_ylim()[1] * 0.95, label, fontsize=10, color='gray')

plt.tight_layout()
plt.savefig(OUT_DIR / 'crisis_probability_v2.png', dpi=150)
print(f"  Saved: crisis_probability_v2.png")

# --- Plot 2: Per-Indicator Probability ---
fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(months_range, cum_density * 100, 'b-', linewidth=2, label=f'Density (sparse zone, >{100-DENSITY_CRISIS_PERCENTILE}th pct)')
ax.plot(months_range, cum_accel * 100, 'r-', linewidth=2, label=f'Acceleration (speed spike, >{ACCEL_CRISIS_PERCENTILE}th pct)')
ax.plot(months_range, cum_deviation * 100, 'g-', linewidth=2, label=f'Deviation (trajectory divergence, >{DEVIATION_CRISIS_PERCENTILE}th pct)')

ax.set_xlabel('Months from Now', fontsize=13)
ax.set_ylabel('Cumulative Probability (%)', fontsize=13)
ax.set_title('Individual Crisis Indicator Probabilities', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, MAX_HORIZON_MONTHS)

plt.tight_layout()
plt.savefig(OUT_DIR / 'crisis_indicators_breakdown.png', dpi=150)
print(f"  Saved: crisis_indicators_breakdown.png")

# --- Plot 3: Indicator Evolution Over Time (mean + percentile bands) ---
fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

for ax, data, name, thr, color in [
    (axes[0], density_cpu, 'Density (sparseness)', density_crisis_threshold, 'blue'),
    (axes[1], accel_cpu, '|Acceleration| (instability)', accel_crisis_threshold, 'red'),
    (axes[2], deviation_cpu, 'Deviation (trajectory divergence)', deviation_crisis_threshold, 'green'),
]:
    mean_val = data.mean(axis=0)
    p10 = np.percentile(data, 10, axis=0)
    p25 = np.percentile(data, 25, axis=0)
    p75 = np.percentile(data, 75, axis=0)
    p90 = np.percentile(data, 90, axis=0)

    ax.fill_between(months_range, p10, p90, alpha=0.15, color=color)
    ax.fill_between(months_range, p25, p75, alpha=0.25, color=color)
    ax.plot(months_range, mean_val, f'{color[0]}-', linewidth=2, label='Mean')
    ax.axhline(thr, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Crisis threshold')

    ax.set_ylabel(name, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Months from Now', fontsize=13)
axes[0].set_title('Crisis Indicator Evolution Over Monte Carlo Paths\n'
                  '(bands: 10-90th and 25-75th percentiles)', fontsize=14)

plt.tight_layout()
plt.savefig(OUT_DIR / 'crisis_indicator_evolution.png', dpi=150)
print(f"  Saved: crisis_indicator_evolution.png")

# --- Plot 4: State Space Map with UMAP ---
try:
    import umap
    print("  Computing UMAP projection...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_2d = reducer.fit_transform(X_pca)

    fig, ax = plt.subplots(figsize=(16, 12))

    years = np.array([int(m[:4]) for m in months_g])
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=years, cmap='viridis', s=15, alpha=0.5, zorder=1)
    plt.colorbar(scatter, label='Year', ax=ax)

    # Crisis periods
    crisis_colors = plt.cm.Set1(np.linspace(0, 1, min(len(CRISIS_PERIODS), 9)))
    for idx, (cname, (cs, ce)) in enumerate(CRISIS_PERIODS.items()):
        c_idx = [i for i, m in enumerate(months_g) if cs <= m[:7] <= ce]
        if c_idx:
            ax.scatter(X_2d[c_idx, 0], X_2d[c_idx, 1],
                      c=[crisis_colors[idx % 9]], s=60, marker='x',
                      label=cname, zorder=3, linewidths=2)

    # Current position
    ax.scatter(X_2d[curr_idx, 0], X_2d[curr_idx, 1],
              c='red', s=300, marker='*', zorder=5, edgecolors='black',
              linewidths=2, label=f'NOW ({months_g[curr_idx][:7]})')

    # Recent trajectory
    recent = max(0, curr_idx - 24)
    ax.plot(X_2d[recent:curr_idx+1, 0], X_2d[recent:curr_idx+1, 1],
            'r-', linewidth=2, alpha=0.7, zorder=4)

    ax.set_title('State Space (UMAP) — Current Position vs Historical Crises', fontsize=14)
    ax.legend(fontsize=7, loc='upper left', ncol=2)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'state_space_v2.png', dpi=150)
    print(f"  Saved: state_space_v2.png")
except ImportError:
    print("  UMAP not available, skipping state space plot")

# --- Plot 5: Historical Indicators Timeline with Crisis Overlay ---
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

hist_months_num = np.arange(N)
# Use actual year labels
year_labels = [int(m[:4]) for m in months_g]

for ax, data, name, thr, color in [
    (axes[0], local_density, 'Density (sparseness)', density_crisis_threshold, 'blue'),
    (axes[1], np.abs(accel_full), '|Acceleration|', accel_crisis_threshold, 'red'),
    (axes[2], deviation_full, 'Deviation', deviation_crisis_threshold, 'green'),
]:
    ax.plot(hist_months_num, data, color=color, linewidth=0.8, alpha=0.7)
    ax.axhline(thr, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Crisis threshold')

    # Shade crisis periods
    for cname, (cs, ce) in CRISIS_PERIODS.items():
        c_idx = [i for i, m in enumerate(months_g) if cs <= m[:7] <= ce]
        if c_idx:
            ax.axvspan(min(c_idx), max(c_idx), alpha=0.1, color='red')

    ax.set_ylabel(name, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# X-axis labels every 5 years
tick_positions = [i for i, m in enumerate(months_g) if m[5:7] == '01' and int(m[:4]) % 5 == 0]
tick_labels = [months_g[i][:4] for i in tick_positions]
axes[-1].set_xticks(tick_positions)
axes[-1].set_xticklabels(tick_labels, rotation=45)
axes[0].set_title('Historical Crisis Indicators (1971-2026)\n'
                  'Red shading = known crisis periods', fontsize=14)

plt.tight_layout()
plt.savefig(OUT_DIR / 'historical_indicators.png', dpi=150)
print(f"  Saved: historical_indicators.png")

# ============================================================
# 9. SAVE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("9. SAVE RESULTS")
print("=" * 70)

results = {
    "version": "v2_gpu",
    "timestamp": "2026-03-13",
    "config": {
        "n_simulations": N_SIMULATIONS,
        "max_horizon_months": MAX_HORIZON_MONTHS,
        "n_pca_components": N_PCA_COMPONENTS,
        "device": str(DEVICE),
        "noise_scale": NOISE_SCALE,
        "n_clusters": N_CLUSTERS,
        "n_economic_series": len(all_series),
        "n_months_history": len(months_g),
        "simulation_time_seconds": round(total_time, 1),
    },
    "crisis_indicator_thresholds": {
        "density_sparse": {
            "threshold": float(density_crisis_threshold),
            "percentile": 100 - DENSITY_CRISIS_PERCENTILE,
            "current_value": float(curr_density),
            "breached": bool(curr_density > density_crisis_threshold),
        },
        "acceleration": {
            "threshold": float(accel_crisis_threshold),
            "percentile": ACCEL_CRISIS_PERCENTILE,
            "current_value": float(curr_accel),
            "breached": bool(curr_accel > accel_crisis_threshold),
        },
        "deviation": {
            "threshold": float(deviation_crisis_threshold),
            "percentile": DEVIATION_CRISIS_PERCENTILE,
            "current_value": float(curr_deviation),
            "breached": bool(curr_deviation > deviation_crisis_threshold),
        },
    },
    "crisis_probability_curves": {
        "any_1_of_3": {f"{t+1}_months": float(cum_any[t]) for t in range(MAX_HORIZON_MONTHS)},
        "2_of_3": {f"{t+1}_months": float(cum_two[t]) for t in range(MAX_HORIZON_MONTHS)},
        "all_3": {f"{t+1}_months": float(cum_all[t]) for t in range(MAX_HORIZON_MONTHS)},
    },
    "per_indicator_curves": {
        "density": {f"{t+1}_months": float(cum_density[t]) for t in range(MAX_HORIZON_MONTHS)},
        "acceleration": {f"{t+1}_months": float(cum_accel[t]) for t in range(MAX_HORIZON_MONTHS)},
        "deviation": {f"{t+1}_months": float(cum_deviation[t]) for t in range(MAX_HORIZON_MONTHS)},
    },
    "key_probabilities": {
        "6_months": {"any": float(cum_any[5]), "two_of_three": float(cum_two[5]), "all_three": float(cum_all[5])},
        "12_months": {"any": float(cum_any[11]), "two_of_three": float(cum_two[11]), "all_three": float(cum_all[11])},
        "24_months": {"any": float(cum_any[23]), "two_of_three": float(cum_two[23]), "all_three": float(cum_all[23])},
        "36_months": {"any": float(cum_any[35]), "two_of_three": float(cum_two[35]), "all_three": float(cum_all[35])},
    },
}

with open(OUT_DIR / 'monte_carlo_results_v2.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved: {OUT_DIR / 'monte_carlo_results_v2.json'}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MONTE CARLO v2 — SUMMARY")
print("=" * 70)

print(f"\nSimulation: {N_SIMULATIONS:,} paths × {MAX_HORIZON_MONTHS} months on {DEVICE}")
print(f"Time: {total_time:.1f}s ({N_SIMULATIONS * MAX_HORIZON_MONTHS / total_time:,.0f} transitions/sec)")

print(f"\nCurrent state indicators:")
print(f"  Density:   {curr_density:.4f} {'⚠ SPARSE' if curr_density > density_crisis_threshold else '✓ Normal'}")
print(f"  |Accel|:   {curr_accel:.4f} {'⚠ HIGH' if curr_accel > accel_crisis_threshold else '✓ Normal'}")
print(f"  Deviation: {curr_deviation:.4f} {'⚠ HIGH' if curr_deviation > deviation_crisis_threshold else '✓ Normal'}")

print(f"\nCrisis probability (balanced — 2 of 3 indicators):")
for t in [3, 6, 12, 24, 36]:
    if t <= MAX_HORIZON_MONTHS:
        print(f"  {t:2d} months: {cum_two[t-1]*100:.1f}%")

print(f"\nOutput: {OUT_DIR}/")
