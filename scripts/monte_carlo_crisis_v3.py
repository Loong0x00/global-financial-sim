#!/usr/bin/env python3
"""
Monte Carlo Crisis Engine v3 — Shock-Adjusted, High Budget
============================================================
v2 的数据截止到 2025-09。但 2026.2-3 月已发生多个重大冲击：
  - 霍尔木兹海峡封锁（油价 +40%）
  - 2 月非农 -92,000
  - 最高法院判关税违宪 → Trump 换法条重征
  - 消费者信心 12 年新低

v3 改进：
1. 用历史类似冲击估算当前状态在 PCA 空间的位移
2. 从位移后的状态开始蒙特卡罗
3. 大幅提高预算：100 万条路径
4. 增加"危机态转移"：当路径进入稀疏区域时，用危机时期的转移分布
5. 多情景对比：基线 / 霍尔木兹 / 全冲击叠加
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
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# CONFIG
# ============================================================

DATA_BASE = Path('/home/user/global-financial-sim/data')
ECON_BASE = DATA_BASE / 'economic'
LEADER_BASE = DATA_BASE / 'leaders'
OUT_DIR = Path('/home/user/global-financial-sim/output/monte_carlo')
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SIMULATIONS = 1_000_000    # 100 万条路径
MAX_HORIZON_MONTHS = 48      # 4 年
N_PCA_COMPONENTS = 14
K_DENSITY = 15
K_TRANSITION = 20
NOISE_SCALE = 0.15

# 危机指标阈值百分位
DENSITY_CRISIS_PCT = 5
ACCEL_CRISIS_PCT = 95
DEVIATION_CRISIS_PCT = 95

# 已知危机时期
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
# 1-3: DATA LOADING + STATE MATRIX + PCA (same as v2, condensed)
# ============================================================

print("=" * 70)
print("1-3. LOADING DATA → STATE MATRIX → PCA")
print("=" * 70)

all_series = {}

def d2m(pts, vk='close'):
    mo = {}
    for p in pts:
        ym = p['date'][:7] + "-01"
        v = p.get(vk, p.get('value'))
        if v is not None:
            mo.setdefault(ym, []).append(float(v))
    return {k: np.mean(v) for k, v in mo.items()}

# Load all economic data
for country in ['us', 'cn', 'eu', 'uk', 'jp', 'ru']:
    pfx = country.upper()
    for ft in ['macro', 'financial']:
        fp = ECON_BASE / country / f'{ft}.json'
        if not fp.exists(): continue
        with open(fp) as f: data = json.load(f)
        for nm, ind in data.get('indicators', data).items():
            dv = {p['date']: float(p['value']) for p in ind.get('series', []) if p['value'] is not None}
            if dv: all_series[f"ECON_{pfx}_{nm}"] = dv

for src, loader in [
    (ECON_BASE / 'indices' / 'global_indices.json', lambda d: {n: d2m(i.get('data', [])) for n, i in d.items()}),
]:
    if src.exists():
        with open(src) as f:
            for n, m in loader(json.load(f)).items():
                if m: all_series[f"IDX_{n}"] = m

for metal in ['gold', 'silver']:
    fp = ECON_BASE / 'commodities' / f'{metal}_daily.json'
    if fp.exists():
        with open(fp) as f: m = d2m(json.load(f).get('data', []), 'close')
        if m: all_series[f"METAL_{metal}"] = m

for fname, prefix in [('fred_commodities.json', 'COMM_FRED'), ('yahoo_futures.json', 'COMM_YF')]:
    fp = ECON_BASE / 'commodities' / fname
    if not fp.exists(): continue
    with open(fp) as f: raw = json.load(f)
    for nm, info in raw.items():
        if not isinstance(info, dict): continue
        if 'COMM_YF' in prefix:
            m = d2m(info.get('data', []))
        else:
            dv = {}
            for p in info.get('data', info.get('series', [])):
                v = p.get('value', p.get('close'))
                if v is not None: dv[p['date']] = float(v)
            m = dv
        if m: all_series[f"{prefix}_{nm}"] = m

fp = ECON_BASE / 'crypto' / 'crypto.json'
if fp.exists():
    with open(fp) as f:
        for nm, info in json.load(f).items():
            m = d2m(info.get('data', []))
            if m: all_series[f"CRYPTO_{nm}"] = m

bis_dir = ECON_BASE / 'bis'
if bis_dir.exists():
    for fname in ['credit_gap.json', 'property_prices.json', 'debt_service_ratios.json',
                   'total_credit.json', 'global_liquidity.json']:
        fp = bis_dir / fname
        if not fp.exists(): continue
        with open(fp) as f: bd = json.load(f)
        dsn = fname.replace('.json', '')
        for cc, cd in bd.get('countries', {}).items():
            sd = cd.get('series', {})
            if not isinstance(sd, dict): continue
            for idx, (sn, si) in enumerate(sd.items()):
                if not isinstance(si, dict) or 'data' not in si: continue
                dv = {}
                raw = si['data']
                it = raw.items() if isinstance(raw, dict) else raw
                for item in it:
                    if isinstance(item, (list, tuple)) and len(item) == 2: dr, val = item
                    elif isinstance(item, dict): dr, val = item.get('date', ''), item.get('value')
                    else: continue
                    if val is None: continue
                    ds = str(dr)
                    if 'Q' in ds:
                        parts = ds.split('-Q')
                        if len(parts) == 2:
                            yr, q = parts[0], int(parts[1])
                            mo = (q-1)*3+1
                            for off in range(3): dv[f"{yr}-{mo+off:02d}-01"] = float(val)
                    elif len(ds) >= 7:
                        try: dv[ds[:7]+'-01'] = float(val)
                        except: pass
                if dv: all_series[f"BIS_{dsn}_{cc}_{idx}"] = dv

supp_dir = ECON_BASE / 'supplementary'
if supp_dir.exists():
    for fname in ['uk_bank_rate.json', 'russia_supplementary.json', 'china_supplementary.json', 'us_additional.json']:
        fp = supp_dir / fname
        if not fp.exists(): continue
        with open(fp) as f: supp = json.load(f)
        for nm, info in supp.items():
            if not isinstance(info, dict) or 'data' not in info: continue
            dv = {}
            for p in info['data']:
                d, v = p.get('date', ''), p.get('value')
                if d and v is not None:
                    try: dv[d[:7]+'-01'] = float(v)
                    except: pass
            if dv: all_series[f"SUPP_{nm}"] = dv

fp = ECON_BASE / 'cftc' / 'cot_data.json'
if fp.exists():
    with open(fp) as f: cftc = json.load(f)
    for cn, recs in cftc.items():
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

fp = ECON_BASE / 'trade' / 'bilateral_trade.json'
if fp.exists():
    with open(fp) as f: trade = json.load(f)
    for pn, pd in trade.items():
        if not isinstance(pd, dict) or 'data' not in pd: continue
        for r in pd['data']:
            yr, total, yoy = r.get('year'), r.get('total'), r.get('yoy_change_pct')
            if yr and total:
                for m in range(1, 13): all_series.setdefault(f"TRADE_{pn}_total", {})[f"{yr}-{m:02d}-01"] = float(total)
            if yr and yoy is not None:
                for m in range(1, 13): all_series.setdefault(f"TRADE_{pn}_yoy", {})[f"{yr}-{m:02d}-01"] = float(yoy)

print(f"Economic series: {len(all_series)}")

# Leader profiles
import unicodedata
profiles = {}
for f in sorted((LEADER_BASE / 'profiles').glob('*.json')):
    with open(f) as fp:
        try: d = json.load(fp)
        except: continue
    vecs = d.get('behavior_matrix', {}).get('vectors', [])
    if vecs:
        params = {v.get('label', v.get('name', '')): float(v['value']) for v in vecs if v.get('label') or v.get('name') and v.get('value') is not None}
        if params: profiles[f.stem] = {'params': params}

n2p = {
    'xi_jinping': 'xi_jinping', 'trump': 'trump', 'powell': 'powell',
    'putin': 'putin', 'bernanke': 'bernanke', 'yellen': 'yellen',
    'obama': 'obama', 'george_w_bush': 'george_w_bush', 'greenspan': 'greenspan',
    'draghi': 'draghi', 'lagarde': 'lagarde', 'merkel': 'merkel',
    'thatcher': 'thatcher', 'blair': 'blair', 'gordon_brown': 'gordon_brown',
    'nabiullina': 'nabiullina', 'yeltsin': 'yeltsin', 'brezhnev': 'brezhnev',
    'khrushchev': 'khrushchev', 'stalin': 'stalin', 'lenin': 'lenin',
    'zhou_xiaochuan': 'zhou_xiaochuan', 'hu_jintao': 'hu_wen', 'wen_jiabao': 'hu_wen',
    'abe_shinzo': 'abe_shinzo', 'kishida_fumio': 'kishida_fumio',
    'takaichi_sanae': 'takaichi_sanae', 'ali_khamenei': 'ali_khamenei',
    'mojtaba_khamenei': 'mojtaba_khamenei', 'elon_musk': 'elon_musk', 'netanyahu': 'netanyahu',
}
_lni = {}
for _pk in profiles:
    for p in _pk.split('_'):
        if len(p) > 2 and p not in _lni: _lni[p] = _pk

def normalize_name(name):
    n = unicodedata.normalize('NFKD', name.lower())
    n = ''.join(c for c in n if not unicodedata.combining(c))
    n = n.replace('.','').replace(' ','_').replace('-','_').replace("'",'')
    while '__' in n: n = n.replace('__','_')
    n = n.strip('_')
    if n in n2p: return n2p[n]
    for part in n.split('_'):
        if part in n2p: return n2p[part]
    parts = n.split('_')
    if len(parts)==2:
        rev = f"{parts[1]}_{parts[0]}"
        if rev in n2p: return n2p[rev]
    for pk in profiles:
        if pk in n or n in pk: return pk
    for part in reversed(parts):
        if part in _lni and len(part)>3: return _lni[part]
    return None

leader_periods = []
for tf in sorted((LEADER_BASE / 'timelines').glob('*.json')):
    co = tf.stem.upper()
    with open(tf) as fp: tl = json.load(fp)
    entries = []
    if isinstance(tl, list): entries = tl
    elif isinstance(tl, dict):
        rd = tl.get('roles', {})
        if isinstance(rd, dict):
            for rn, rdata in rd.items():
                if isinstance(rdata, dict):
                    for sk, sv in rdata.items():
                        if isinstance(sv, list) and sv and isinstance(sv[0], dict): entries.extend(sv)
        for key, val in tl.items():
            if key == 'roles': continue
            if isinstance(val, list) and val and isinstance(val[0], dict): entries.extend(val)
            elif isinstance(val, dict):
                if 'entries' in val: entries.extend(val['entries'])
                for sk, sv in val.items():
                    if isinstance(sv, list) and sv and isinstance(sv[0], dict): entries.extend(sv)
    for e in entries:
        nm = e.get('name', e.get('leader', ''))
        st = e.get('term_start', e.get('start', ''))
        en = e.get('term_end', e.get('end', ''))
        rl = e.get('role', e.get('position', ''))
        if not nm or not st: continue
        if len(st)==7: st += '-01'
        if en and len(en)==7: en += '-01'
        pk = normalize_name(nm)
        if pk and pk in profiles:
            if not en or en.lower() in ('present','incumbent',''): en = '2026-03-12'
            leader_periods.append((pk, st, en, rl, co))

# Build state matrix
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

def match_role(role_str, country):
    rl = role_str.lower().replace('-','_').replace(' ','_')
    for slot, kws in ROLE_SLOTS.items():
        if country != slot.split('_')[0]: continue
        for kw in kws:
            if kw in rl: return slot
    return None

dim_counts = {}
for pdata in profiles.values():
    for dn in pdata['params']: dim_counts[dn] = dim_counts.get(dn,0)+1
top_dims = sorted(dim_counts, key=lambda d: -dim_counts[d])[:40]
NLD = 15

role_assignments = {}
for month in all_months:
    role_assignments[month] = {}
    for pk, st, en, rl, co in leader_periods:
        if st[:10] <= month <= en[:10]:
            slot = match_role(rl, co)
            if slot: role_assignments[month][slot] = pk

slots_used = sorted(set(s for m in all_months for s in role_assignments[m]))
all_col_names = econ_cols + [f"L_{sl}_{dm}" for sl in slots_used for dm in top_dims[:NLD]]

nm = len(all_months)
nc = len(all_col_names)
matrix = np.full((nm, nc), np.nan)

for j, col in enumerate(econ_cols):
    s = all_series[col]
    for i, mo in enumerate(all_months):
        if mo in s: matrix[i, j] = s[mo]

lcs = len(econ_cols)
for i, mo in enumerate(all_months):
    asgn = role_assignments[mo]
    for si, slot in enumerate(slots_used):
        if slot in asgn:
            par = profiles[asgn[slot]]['params']
            for d, dm in enumerate(top_dims[:NLD]):
                ci = lcs + si*NLD + d
                if dm in par: matrix[i, ci] = par[dm]

cov = np.sum(~np.isnan(matrix), axis=0) / nm * 100
gc = cov >= 8
mat = matrix[:, gc]
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
    col = mat[:,j]; m = np.isnan(col)
    if m.any():
        med = np.nanmedian(col)
        mat[m,j] = med if not np.isnan(med) else 0.5

qt = QuantileTransformer(n_quantiles=min(1000, len(months_g)), output_distribution='uniform', random_state=42)
X_norm = qt.fit_transform(mat)
X_full = np.hstack([X_norm, np.linspace(0,1,len(months_g)).reshape(-1,1)])

n_comp = min(N_PCA_COMPONENTS+5, X_full.shape[1]-1, X_full.shape[0]-1)
pca = PCA(n_components=n_comp)
X_pca = pca.fit_transform(X_full)[:, :N_PCA_COMPONENTS]
explained = np.cumsum(pca.explained_variance_ratio_)

N = len(months_g)
D = N_PCA_COMPONENTS
print(f"State: {mat.shape} → PCA {N}×{D} ({explained[D-1]*100:.1f}% var)")

# ============================================================
# 4. ESTIMATE SHOCK DISPLACEMENT IN PCA SPACE
# ============================================================

print("\n" + "=" * 70)
print("4. ESTIMATE CURRENT SHOCK DISPLACEMENT")
print("=" * 70)

# Find historical analogues for each shock and measure their PCA displacement
deltas = np.diff(X_pca, axis=0)  # (N-1, D)
delta_norms = np.linalg.norm(deltas, axis=1)

# --- Shock 1: Oil price spike +40% (Hormuz) ---
# Historical analogue: 1973 oil crisis, 1990 Gulf War oil spike
# Find months where oil-related series spiked most
oil_shock_months = []
for cname, (cs, ce) in [("Oil Crisis 1973-74", ("1973-10", "1974-03")),
                          ("Gulf War 1990", ("1990-08", "1990-10"))]:
    for i, m in enumerate(months_g[:-1]):
        if cs <= m[:7] <= ce:
            oil_shock_months.append(i)

if oil_shock_months:
    oil_shock_delta = deltas[oil_shock_months].mean(axis=0)
    oil_shock_magnitude = np.linalg.norm(oil_shock_delta)
    print(f"Oil shock displacement (from {len(oil_shock_months)} months): magnitude={oil_shock_magnitude:.4f}")
else:
    oil_shock_delta = np.zeros(D)
    print("No oil shock analogues found")

# --- Shock 2: Employment collapse ---
# Historical: 2008-09 GFC job losses, 2020 COVID
employment_shock_months = []
for i, m in enumerate(months_g[:-1]):
    ym = m[:7]
    if ym in ['2008-10', '2008-11', '2008-12', '2009-01', '2009-02',
              '2020-03', '2020-04']:
        employment_shock_months.append(i)

if employment_shock_months:
    emp_shock_delta = deltas[employment_shock_months].mean(axis=0)
    # Scale: -92K is bad but not -20M (COVID) or -800K (GFC peak)
    # NFP -92K is about 10-15% of GFC peak severity
    emp_scale = 0.12
    emp_shock_delta = emp_shock_delta * emp_scale
    print(f"Employment shock displacement (scaled {emp_scale}x): magnitude={np.linalg.norm(emp_shock_delta):.4f}")
else:
    emp_shock_delta = np.zeros(D)

# --- Shock 3: Trade policy upheaval (tariff ruling + re-imposition) ---
# Historical: 2018-19 trade war escalation
trade_shock_months = []
for i, m in enumerate(months_g[:-1]):
    if '2018-06' <= m[:7] <= '2018-09':
        trade_shock_months.append(i)

if trade_shock_months:
    trade_shock_delta = deltas[trade_shock_months].mean(axis=0)
    # Supreme Court ruling + reimposition = roughly same magnitude as 2018 escalation
    print(f"Trade shock displacement: magnitude={np.linalg.norm(trade_shock_delta):.4f}")
else:
    trade_shock_delta = np.zeros(D)

# --- Shock 4: Consumer confidence collapse ---
# Historical: 2008 confidence crash
conf_shock_months = []
for i, m in enumerate(months_g[:-1]):
    if '2008-09' <= m[:7] <= '2008-12':
        conf_shock_months.append(i)

if conf_shock_months:
    conf_shock_delta = deltas[conf_shock_months].mean(axis=0)
    # Scale: current is 57.3 (12yr low) but not 2008 level
    conf_scale = 0.5
    conf_shock_delta = conf_shock_delta * conf_scale
    print(f"Confidence shock displacement (scaled {conf_scale}x): magnitude={np.linalg.norm(conf_shock_delta):.4f}")
else:
    conf_shock_delta = np.zeros(D)

# Combined shock displacement
# Not fully additive — shocks are correlated, use 70% of sum
SHOCK_CORRELATION_DISCOUNT = 0.70
combined_shock = (oil_shock_delta + emp_shock_delta + trade_shock_delta + conf_shock_delta) * SHOCK_CORRELATION_DISCOUNT
combined_magnitude = np.linalg.norm(combined_shock)

# Historical context: how does this compare to typical monthly deltas?
median_delta = np.median(delta_norms)
p95_delta = np.percentile(delta_norms, 95)
p99_delta = np.percentile(delta_norms, 99)

print(f"\nCombined shock magnitude: {combined_magnitude:.4f}")
print(f"  vs median monthly delta: {median_delta:.4f} ({combined_magnitude/median_delta:.1f}x)")
print(f"  vs 95th pct delta:       {p95_delta:.4f} ({combined_magnitude/p95_delta:.1f}x)")
print(f"  vs 99th pct delta:       {p99_delta:.4f} ({combined_magnitude/p99_delta:.1f}x)")

# ============================================================
# 5. DEFINE SCENARIOS
# ============================================================

print("\n" + "=" * 70)
print("5. DEFINE SIMULATION SCENARIOS")
print("=" * 70)

curr_idx = N - 1
base_state = X_pca[curr_idx].copy()

scenarios = {
    "Baseline (2025-09 data)": base_state.copy(),
    "Hormuz Only (+40% oil)": base_state + oil_shock_delta,
    "Full Shock (Hormuz + NFP + Tariff + Confidence)": base_state + combined_shock,
    "Worst Case (Full Shock × 1.5)": base_state + combined_shock * 1.5,
}

for sname, sstate in scenarios.items():
    dist_from_base = np.linalg.norm(sstate - base_state)
    print(f"  {sname:45s}: displacement={dist_from_base:.4f}")

# ============================================================
# 6. CALIBRATE CRISIS THRESHOLDS
# ============================================================

print("\n" + "=" * 70)
print("6. CALIBRATE CRISIS THRESHOLDS")
print("=" * 70)

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

print(f"Density threshold (sparse): {density_thr:.4f}")
print(f"|Accel| threshold:          {accel_thr:.4f}")
print(f"Deviation threshold:        {deviation_thr:.4f}")

# ============================================================
# 7. PREPARE GPU + CLUSTER-BASED TRANSITIONS
# ============================================================

print("\n" + "=" * 70)
print("7. PREPARE GPU TENSORS")
print("=" * 70)

X_gpu = torch.tensor(X_pca, dtype=torch.float32, device=DEVICE)
deltas_gpu = torch.tensor(deltas, dtype=torch.float32, device=DEVICE)

# Cluster for fast transition sampling
N_CLUSTERS = min(80, N // 8)
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_pca[:-1])
cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=DEVICE)

# Build per-cluster delta pools (non-parametric: sample actual historical deltas)
crisis_month_set = set()
for cname, (cs, ce) in CRISIS_PERIODS.items():
    for i, m in enumerate(months_g):
        if cs <= m[:7] <= ce: crisis_month_set.add(i)

# Store actual deltas per cluster as padded tensor for GPU sampling
cluster_deltas_list = []  # list of (n_members, D) arrays
cluster_crisis_deltas_list = []
cluster_sizes = np.zeros(N_CLUSTERS, dtype=np.int32)
cluster_crisis_sizes = np.zeros(N_CLUSTERS, dtype=np.int32)

for c in range(N_CLUSTERS):
    members = np.where(cluster_labels == c)[0]
    cluster_sizes[c] = len(members)
    cluster_deltas_list.append(deltas[members] if len(members) > 0 else deltas[:1])
    crisis_members = [m for m in members if m in crisis_month_set]
    cluster_crisis_sizes[c] = len(crisis_members)
    cluster_crisis_deltas_list.append(deltas[crisis_members] if len(crisis_members) > 0 else deltas[:1])

# Pad to max size for efficient GPU indexing
max_pool = max(cluster_sizes)
max_crisis_pool = max(max(cluster_crisis_sizes), 1)
cluster_delta_pool = np.zeros((N_CLUSTERS, max_pool, D), dtype=np.float32)
cluster_crisis_pool = np.zeros((N_CLUSTERS, max_crisis_pool, D), dtype=np.float32)

for c in range(N_CLUSTERS):
    n = cluster_sizes[c]
    if n > 0:
        cluster_delta_pool[c, :n] = cluster_deltas_list[c]
    nc = cluster_crisis_sizes[c]
    if nc > 0:
        cluster_crisis_pool[c, :nc] = cluster_crisis_deltas_list[c]

cluster_delta_pool_gpu = torch.tensor(cluster_delta_pool, device=DEVICE)
cluster_crisis_pool_gpu = torch.tensor(cluster_crisis_pool, device=DEVICE)
cluster_sizes_gpu = torch.tensor(cluster_sizes, dtype=torch.long, device=DEVICE)
cluster_crisis_sizes_gpu = torch.tensor(cluster_crisis_sizes, dtype=torch.long, device=DEVICE)

# Also keep all crisis deltas as a global pool for sparse-region sampling
all_crisis_deltas = deltas[list(crisis_month_set)] if crisis_month_set else deltas[:10]
all_crisis_pool_gpu = torch.tensor(all_crisis_deltas, dtype=torch.float32, device=DEVICE)

print(f"Clusters: {N_CLUSTERS} ({(cluster_crisis_sizes > 0).sum()} with crisis data)")
print(f"Non-parametric pools: max {max_pool} deltas/cluster, {len(all_crisis_deltas)} global crisis deltas")
print(f"GPU ready: {DEVICE}")

# ============================================================
# 8. RUN MONTE CARLO FOR ALL SCENARIOS
# ============================================================

print("\n" + "=" * 70)
print(f"8. MONTE CARLO: {N_SIMULATIONS:,} paths × {MAX_HORIZON_MONTHS} months × {len(scenarios)} scenarios")
print("=" * 70)

all_results = {}

for sc_name, sc_start in scenarios.items():
    print(f"\n--- Scenario: {sc_name} ---")
    t0 = time.time()

    start_gpu = torch.tensor(sc_start, dtype=torch.float32, device=DEVICE)
    paths = start_gpu.unsqueeze(0).expand(N_SIMULATIONS, -1).clone()

    density_ot = torch.zeros(N_SIMULATIONS, MAX_HORIZON_MONTHS, device=DEVICE)
    accel_ot = torch.zeros(N_SIMULATIONS, MAX_HORIZON_MONTHS, device=DEVICE)
    deviation_ot = torch.zeros(N_SIMULATIONS, MAX_HORIZON_MONTHS, device=DEVICE)

    prev_dnorm = torch.zeros(N_SIMULATIONS, device=DEVICE)
    path_sum = start_gpu.unsqueeze(0).expand(N_SIMULATIONS, -1).clone()
    path_count = torch.ones(N_SIMULATIONS, device=DEVICE)

    for t in range(MAX_HORIZON_MONTHS):
        # Find nearest cluster
        diffs = paths.unsqueeze(1) - cluster_centers.unsqueeze(0)
        dist_sq = (diffs ** 2).sum(dim=2)
        nearest = dist_sq.argmin(dim=1)

        # Check if path is in sparse region → use crisis transitions if available
        # Compute density for all paths
        CHUNK = 50000
        dens_vals = torch.zeros(N_SIMULATIONS, device=DEVICE)
        for cs in range(0, N_SIMULATIONS, CHUNK):
            ce = min(cs+CHUNK, N_SIMULATIONS)
            d = torch.cdist(paths[cs:ce], X_gpu)
            topk, _ = d.topk(K_DENSITY, dim=1, largest=False)
            dens_vals[cs:ce] = topk.mean(dim=1)

        density_ot[:, t] = dens_vals

        # Determine which paths use crisis transitions (in sparse regions)
        in_sparse = dens_vals > density_thr
        crisis_boost = in_sparse.float() * 0.5  # 50% more volatile when sparse

        # Non-parametric: sample actual historical deltas from nearest cluster
        sizes = cluster_sizes_gpu[nearest]  # (N_SIM,)
        rand_idx = (torch.rand(N_SIMULATIONS, device=DEVICE) * sizes.float()).long()
        rand_idx = rand_idx.clamp(max=max_pool - 1)
        delta = cluster_delta_pool_gpu[nearest, rand_idx]  # (N_SIM, D)

        # For paths in sparse regions: 40% chance to sample from global crisis pool instead
        if in_sparse.any():
            crisis_rand = torch.rand(N_SIMULATIONS, device=DEVICE) < 0.4
            use_crisis = in_sparse & crisis_rand
            if use_crisis.any():
                n_crisis = use_crisis.sum()
                crisis_idx = torch.randint(0, len(all_crisis_pool_gpu), (n_crisis.item(),), device=DEVICE)
                delta[use_crisis] = all_crisis_pool_gpu[crisis_idx]

        # Extra volatility in sparse regions (bootstrap noise amplification)
        noise_amp = 1.0 + crisis_boost * 0.3  # 30% more volatile when sparse
        delta = delta * noise_amp.unsqueeze(1)

        paths = paths + delta

        # Acceleration
        curr_dnorm = torch.norm(delta, dim=1)
        if t > 0:
            accel_ot[:, t] = torch.abs(curr_dnorm - prev_dnorm)
        prev_dnorm = curr_dnorm

        # Deviation
        path_sum = path_sum + paths
        path_count = path_count + 1
        running_mean = path_sum / path_count.unsqueeze(1)
        deviation_ot[:, t] = torch.norm(paths - running_mean, dim=1)

        if (t+1) % 12 == 0:
            sparse_pct = in_sparse.float().mean().item() * 100
            print(f"  Month {t+1:2d}: density={dens_vals.mean().item():.3f}, "
                  f"sparse={sparse_pct:.1f}%, elapsed={time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    throughput = N_SIMULATIONS * MAX_HORIZON_MONTHS / elapsed

    # Compute crisis probabilities
    dc = density_ot.cpu().numpy()
    ac = accel_ot.cpu().numpy()
    dv = deviation_ot.cpu().numpy()

    d_crisis = dc > density_thr
    a_crisis = ac > accel_thr
    v_crisis = dv > deviation_thr

    any_c = d_crisis | a_crisis | v_crisis
    two_c = (d_crisis.astype(int) + a_crisis.astype(int) + v_crisis.astype(int)) >= 2
    all_c = d_crisis & a_crisis & v_crisis

    def first_hit(cm):
        first = np.full(N_SIMULATIONS, MAX_HORIZON_MONTHS+1, dtype=int)
        for i in range(N_SIMULATIONS):
            h = np.where(cm[i])[0]
            if len(h) > 0: first[i] = h[0]+1
        return first

    f_any = first_hit(any_c)
    f_two = first_hit(two_c)
    f_all = first_hit(all_c)

    cum_any = np.array([np.mean(f_any <= t+1) for t in range(MAX_HORIZON_MONTHS)])
    cum_two = np.array([np.mean(f_two <= t+1) for t in range(MAX_HORIZON_MONTHS)])
    cum_all = np.array([np.mean(f_all <= t+1) for t in range(MAX_HORIZON_MONTHS)])

    # Per-indicator
    cum_dens = np.array([np.mean(first_hit(d_crisis) <= t+1) for t in range(MAX_HORIZON_MONTHS)])
    cum_acc = np.array([np.mean(first_hit(a_crisis) <= t+1) for t in range(MAX_HORIZON_MONTHS)])
    cum_dev = np.array([np.mean(first_hit(v_crisis) <= t+1) for t in range(MAX_HORIZON_MONTHS)])

    all_results[sc_name] = {
        'cum_any': cum_any, 'cum_two': cum_two, 'cum_all': cum_all,
        'cum_dens': cum_dens, 'cum_acc': cum_acc, 'cum_dev': cum_dev,
        'mean_density': dc.mean(axis=0), 'mean_accel': ac.mean(axis=0),
        'mean_deviation': dv.mean(axis=0),
        'elapsed': elapsed, 'throughput': throughput,
    }

    print(f"  Time: {elapsed:.1f}s ({throughput:,.0f} transitions/sec)")
    print(f"  {'Horizon':>10s}  {'Any 1/3':>8s}  {'2/3':>8s}  {'All 3':>8s}")
    for t in [3, 6, 12, 24, 36, 48]:
        if t <= MAX_HORIZON_MONTHS:
            print(f"  {t:2d} months  {cum_any[t-1]*100:7.2f}%  {cum_two[t-1]*100:7.2f}%  {cum_all[t-1]*100:7.2f}%")

# ============================================================
# 9. SAVE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("9. SAVE RESULTS")
print("=" * 70)

results_json = {
    "version": "v3_shock_adjusted",
    "timestamp": "2026-03-13",
    "config": {
        "n_simulations": N_SIMULATIONS,
        "max_horizon_months": MAX_HORIZON_MONTHS,
        "n_pca_components": D,
        "device": str(DEVICE),
        "n_clusters": N_CLUSTERS,
        "shock_correlation_discount": SHOCK_CORRELATION_DISCOUNT,
    },
    "shock_analysis": {
        "oil_shock_magnitude": float(np.linalg.norm(oil_shock_delta)),
        "employment_shock_magnitude": float(np.linalg.norm(emp_shock_delta)),
        "trade_shock_magnitude": float(np.linalg.norm(trade_shock_delta)),
        "confidence_shock_magnitude": float(np.linalg.norm(conf_shock_delta)),
        "combined_magnitude": float(combined_magnitude),
        "vs_median_monthly_delta": float(combined_magnitude / median_delta),
        "vs_p95_delta": float(combined_magnitude / p95_delta),
    },
    "scenarios": {},
}

for sname, sdata in all_results.items():
    results_json["scenarios"][sname] = {
        "crisis_probability_2of3": {
            f"{t+1}_months": float(sdata['cum_two'][t]) for t in range(MAX_HORIZON_MONTHS)
        },
        "key_probabilities": {
            "6_months": {"any": float(sdata['cum_any'][5]), "two": float(sdata['cum_two'][5]), "all": float(sdata['cum_all'][5])},
            "12_months": {"any": float(sdata['cum_any'][11]), "two": float(sdata['cum_two'][11]), "all": float(sdata['cum_all'][11])},
            "24_months": {"any": float(sdata['cum_any'][23]), "two": float(sdata['cum_two'][23]), "all": float(sdata['cum_all'][23])},
            "36_months": {"any": float(sdata['cum_any'][35]), "two": float(sdata['cum_two'][35]), "all": float(sdata['cum_all'][35])},
            "48_months": {"any": float(sdata['cum_any'][47]), "two": float(sdata['cum_two'][47]), "all": float(sdata['cum_all'][47])},
        },
        "elapsed_seconds": round(sdata['elapsed'], 1),
    }

with open(OUT_DIR / 'monte_carlo_results_v3.json', 'w') as f:
    json.dump(results_json, f, indent=2)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MONTE CARLO v3 — FINAL SUMMARY")
print("=" * 70)

print(f"\n{N_SIMULATIONS:,} paths × {MAX_HORIZON_MONTHS} months × {len(scenarios)} scenarios on {DEVICE}")

print(f"\nShock displacement: {combined_magnitude:.4f} ({combined_magnitude/median_delta:.1f}x median monthly change)")

print(f"\n--- 2/3 indicators breached ---")
print(f"{'':45s}  {'6mo':>7s}  {'12mo':>7s}  {'24mo':>7s}  {'36mo':>7s}  {'48mo':>7s}")
print("-" * 90)
for sname, sdata in all_results.items():
    c = sdata['cum_two']
    vals = [c[5], c[11], c[23], c[35], c[47]]
    print(f"{sname:45s}  " + "  ".join(f"{v*100:6.1f}%" for v in vals))

print(f"\n--- Per-indicator detail (Full Shock scenario) ---")
fs_key = [k for k in all_results if "Full Shock" in k][0]
fs = all_results[fs_key]
print(f"{'Indicator':30s}  {'6mo':>7s}  {'12mo':>7s}  {'24mo':>7s}  {'36mo':>7s}  {'48mo':>7s}")
print("-" * 75)
for iname, ikey in [("Density (unknown territory)", 'cum_dens'), ("Acceleration (instability)", 'cum_acc'), ("Deviation (trajectory break)", 'cum_dev')]:
    c = fs[ikey]
    vals = [c[5], c[11], c[23], c[35], c[47]]
    print(f"{iname:30s}  " + "  ".join(f"{v*100:6.1f}%" for v in vals))

print(f"\n--- All 3 indicators breached ---")
print(f"{'':45s}  {'6mo':>7s}  {'12mo':>7s}  {'24mo':>7s}  {'36mo':>7s}  {'48mo':>7s}")
print("-" * 90)
for sname, sdata in all_results.items():
    c = sdata['cum_all']
    vals = [c[5], c[11], c[23], c[35], c[47]]
    print(f"{sname:45s}  " + "  ".join(f"{v*100:6.1f}%" for v in vals))

print(f"\nOutput: {OUT_DIR}/")
