#!/usr/bin/env python3
"""
Monte Carlo v4b — Crisis Attribution Analysis
===============================================
在 v4 基础上追踪：
1. 每类决策对路径位移的贡献占比（谁推的最多）
2. 台湾触发 vs 未触发路径的分别统计
3. Trump-only vs Xi-only vs Both 分离实验
4. 按崩溃原因分类路径
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
LEADER_BASE = DATA_BASE / 'leaders'
OUT_DIR = Path('/home/user/global-financial-sim/output/monte_carlo')
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SIMULATIONS = 500_000  # 50万够用，跑4个场景
MAX_HORIZON_MONTHS = 48
N_PCA_COMPONENTS = 14
K_DENSITY = 15

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

DENSITY_CRISIS_PCT = 5
ACCEL_CRISIS_PCT = 95
DEVIATION_CRISIS_PCT = 95

np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# DATA LOADING (condensed, same as v4)
# ============================================================

print("=" * 70)
print("LOADING DATA")
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
        if 'COMM_YF' in prefix: m = d2m(info.get('data', []))
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
                raw_data = si['data']
                it = raw_data.items() if isinstance(raw_data, dict) else raw_data
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
        mn_d, mo_d = {}, {}
        for r in recs:
            d = r.get('date', '')[:7]
            if not d: continue
            k = d+'-01'
            net, oi = r.get('net_speculative'), r.get('open_interest')
            if net is not None: mn_d.setdefault(k, []).append(float(net))
            if oi is not None: mo_d.setdefault(k, []).append(float(oi))
        if mn_d: all_series[f"CFTC_NET_{cn}"] = {k: np.mean(v) for k, v in mn_d.items()}
        if mo_d: all_series[f"CFTC_OI_{cn}"] = {k: np.mean(v) for k, v in mo_d.items()}

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
    with open(f) as fp_f:
        try: d = json.load(fp_f)
        except: continue
    vecs = d.get('behavior_matrix', {}).get('vectors', [])
    if vecs:
        params = {v.get('label', v.get('name', '')): float(v['value']) for v in vecs if (v.get('label') or v.get('name')) and v.get('value') is not None}
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
    with open(tf) as fp_f: tl = json.load(fp_f)
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
        enm = e.get('name', e.get('leader', ''))
        st = e.get('term_start', e.get('start', ''))
        en = e.get('term_end', e.get('end', ''))
        rl = e.get('role', e.get('position', ''))
        if not enm or not st: continue
        if len(st)==7: st += '-01'
        if en and len(en)==7: en += '-01'
        pk = normalize_name(enm)
        if pk and pk in profiles:
            if not en or en.lower() in ('present','incumbent',''): en = '2026-03-12'
            leader_periods.append((pk, st, en, rl, co))

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

nmonths = len(all_months)
nc = len(all_col_names)
matrix = np.full((nmonths, nc), np.nan)

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

cov = np.sum(~np.isnan(matrix), axis=0) / nmonths * 100
gc = cov >= 8
mat = matrix[:, gc]
rc = np.sum(~np.isnan(mat), axis=1) / mat.shape[1] * 100
gr = rc >= 25
mat = mat[gr]
months_g = [all_months[i] for i in range(nmonths) if gr[i]]

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
pca_model = PCA(n_components=n_comp)
X_pca = pca_model.fit_transform(X_full)[:, :N_PCA_COMPONENTS]

N = len(months_g)
D = N_PCA_COMPONENTS
print(f"State: {mat.shape} → PCA {N}×{D}")

# ============================================================
# TEMPLATES + THRESHOLDS (same as v4)
# ============================================================

deltas = np.diff(X_pca, axis=0)
delta_norms = np.linalg.norm(deltas, axis=1)
median_delta = np.median(delta_norms)

def month_range_delta(start_ym, end_ym):
    idxs = [i for i, m in enumerate(months_g[:-1]) if start_ym <= m[:7] <= end_ym]
    return (deltas[idxs].mean(axis=0), len(idxs)) if idxs else (np.zeros(D), 0)

tariff_esc_d, _ = month_range_delta("2018-06", "2018-09")
tariff_deesc_d, _ = month_range_delta("2019-06", "2019-12")
mil_d, _ = month_range_delta("1990-08", "1990-10")
oil_d, _ = month_range_delta("1973-10", "1974-03")
mil_d = mil_d * 0.7 + oil_d * 0.3
fed_d, _ = month_range_delta("2018-10", "2018-12")
fed_d *= 0.5
gfc_d, _ = month_range_delta("2008-09", "2008-12")
tw_d, _ = month_range_delta("2018-06", "2019-06")
taiwan_d = gfc_d * 1.5 + tw_d * 1.0 + mil_d * 1.0
stim09_d, n09 = month_range_delta("2009-03", "2009-12")
stim20_d, n20 = month_range_delta("2020-05", "2020-12")
stim_d = (stim09_d + stim20_d) / 2 if n09 > 0 and n20 > 0 else (stim09_d if n09 > 0 else stim20_d)
capctrl_d, _ = month_range_delta("2015-08", "2016-02")
capctrl_d *= 0.6
tech_d, _ = month_range_delta("2019-05", "2019-08")

templates_np = {
    'tariff_escalate': tariff_esc_d, 'tariff_deesc': tariff_deesc_d,
    'military': mil_d, 'fed_pressure': fed_d,
    'taiwan': taiwan_d, 'stimulus': stim_d,
    'capital_control': capctrl_d, 'tech_retaliation': tech_d,
}
templates = {k: torch.tensor(v, dtype=torch.float32, device=DEVICE) for k, v in templates_np.items()}

nn_density = NearestNeighbors(n_neighbors=K_DENSITY)
nn_density.fit(X_pca)
knn_dists, _ = nn_density.kneighbors(X_pca)
local_density = knn_dists.mean(axis=1)

accel_full = np.zeros(N)
dn = np.linalg.norm(deltas, axis=1)
accel_full[2:] = np.abs(np.diff(dn))
deviation_full = np.zeros(N)
for i in range(N):
    start = max(0, i-24)
    if i-start >= 3:
        deviation_full[i] = np.linalg.norm(X_pca[i] - X_pca[start:i].mean(axis=0))

density_thr = np.percentile(local_density, 100-DENSITY_CRISIS_PCT)
accel_thr = np.percentile(np.abs(accel_full), ACCEL_CRISIS_PCT)
deviation_thr = np.percentile(deviation_full, DEVIATION_CRISIS_PCT)

hist_median = np.median(X_pca, axis=0)
market_stress_ref = np.percentile(np.linalg.norm(X_pca - hist_median, axis=1), 90)

# GPU tensors
X_gpu = torch.tensor(X_pca, dtype=torch.float32, device=DEVICE)
hist_median_gpu = torch.tensor(hist_median, dtype=torch.float32, device=DEVICE)

N_CLUSTERS = min(80, N // 8)
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_pca[:-1])
cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=DEVICE)

crisis_month_set = set()
for cname, (cs, ce) in CRISIS_PERIODS.items():
    for i, m in enumerate(months_g):
        if cs <= m[:7] <= ce: crisis_month_set.add(i)

cluster_sizes = np.zeros(N_CLUSTERS, dtype=np.int32)
cluster_deltas_list = []
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

all_crisis_deltas = deltas[list(crisis_month_set)] if crisis_month_set else deltas[:10]
all_crisis_pool_gpu = torch.tensor(all_crisis_deltas, dtype=torch.float32, device=DEVICE)

# Starting state (Full Shock from v3)
curr_idx = N - 1
base_state = X_pca[curr_idx].copy()
oil_sm = [i for i, m in enumerate(months_g[:-1]) if m[:7] in ['1973-10','1973-11','1973-12','1974-01','1974-02','1974-03','1990-08','1990-09','1990-10']]
emp_sm = [i for i, m in enumerate(months_g[:-1]) if m[:7] in ['2008-10','2008-11','2008-12','2009-01','2009-02','2020-03','2020-04']]
trd_sm = [i for i, m in enumerate(months_g[:-1]) if '2018-06' <= m[:7] <= '2018-09']
cnf_sm = [i for i, m in enumerate(months_g[:-1]) if '2008-09' <= m[:7] <= '2008-12']
combined_shock = (
    (deltas[oil_sm].mean(axis=0) if oil_sm else np.zeros(D)) +
    (deltas[emp_sm].mean(axis=0) * 0.12 if emp_sm else np.zeros(D)) +
    (deltas[trd_sm].mean(axis=0) if trd_sm else np.zeros(D)) +
    (deltas[cnf_sm].mean(axis=0) * 0.5 if cnf_sm else np.zeros(D))
) * 0.70
shock_state = base_state + combined_shock

print(f"Templates: {len(templates)}, Clusters: {N_CLUSTERS}")
print(f"Taiwan template magnitude: {np.linalg.norm(templates_np['taiwan']):.3f} "
      f"({np.linalg.norm(templates_np['taiwan'])/median_delta:.1f}x median delta)")

# ============================================================
# LEADER PARAMETERS
# ============================================================

TRUMP = {
    'market_crash_threshold': 0.08, 'tariff_escalation_rate': 0.15,
    'tariff_pause_on_crash': 0.85, 'military_momentum_init': 0.4,
    'military_momentum_increment': 0.08, 'military_action_base_prob': 0.05,
    'military_cooldown_months': 4, 'fed_pressure_probability': 0.30,
    'fed_pressure_escalation': 0.05, 'deal_month_threshold': 18,
}
XI = {
    'taiwan_base_prob': 0.005, 'taiwan_prob_2027_peak': 0.03,
    'taiwan_ramp_start_month': 6, 'taiwan_ramp_peak_month': 18,
    'gdp_red_line_trigger': 0.85, 'stimulus_probability': 0.60,
    'flood_stimulus_aversion': 0.82, 'stimulus_scale': 0.4,
    'capital_control_threshold': 0.80, 'capital_control_probability': 0.40,
    'delayed_response_months': 3, 'tech_retaliation_probability': 0.25,
}

# ============================================================
# SIMULATION FUNCTION WITH ATTRIBUTION TRACKING
# ============================================================

def run_simulation(n_sim, use_trump, use_xi, label):
    """Run Monte Carlo with optional Trump/Xi and track displacement attribution."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  {n_sim:,} paths × {MAX_HORIZON_MONTHS} months | Trump={'ON' if use_trump else 'OFF'} Xi={'ON' if use_xi else 'OFF'}")
    print(f"{'='*70}")

    t0 = time.time()
    start_gpu = torch.tensor(shock_state, dtype=torch.float32, device=DEVICE)
    paths = start_gpu.unsqueeze(0).expand(n_sim, -1).clone()

    density_ot = torch.zeros(n_sim, MAX_HORIZON_MONTHS, device=DEVICE)
    accel_ot = torch.zeros(n_sim, MAX_HORIZON_MONTHS, device=DEVICE)
    deviation_ot = torch.zeros(n_sim, MAX_HORIZON_MONTHS, device=DEVICE)

    prev_dnorm = torch.zeros(n_sim, device=DEVICE)
    path_sum = start_gpu.unsqueeze(0).expand(n_sim, -1).clone()
    path_count = torch.ones(n_sim, device=DEVICE)

    # Attribution accumulators: cumulative |displacement| from each source
    attr_base = torch.zeros(n_sim, device=DEVICE)        # Random walk
    attr_tariff = torch.zeros(n_sim, device=DEVICE)       # Trump tariffs
    attr_military = torch.zeros(n_sim, device=DEVICE)     # Trump military
    attr_fed = torch.zeros(n_sim, device=DEVICE)           # Trump Fed pressure
    attr_taiwan = torch.zeros(n_sim, device=DEVICE)        # Xi Taiwan
    attr_taiwan_ongoing = torch.zeros(n_sim, device=DEVICE)# Xi Taiwan aftermath
    attr_stimulus = torch.zeros(n_sim, device=DEVICE)      # Xi stimulus
    attr_capctrl = torch.zeros(n_sim, device=DEVICE)       # Xi capital controls
    attr_tech = torch.zeros(n_sim, device=DEVICE)          # Xi tech retaliation

    # State variables
    trump_tariff = torch.ones(n_sim, device=DEVICE) * 0.5
    trump_mil_mom = torch.ones(n_sim, device=DEVICE) * TRUMP['military_momentum_init']
    trump_mil_cd = torch.zeros(n_sim, device=DEVICE)
    trump_fed_lvl = torch.zeros(n_sim, device=DEVICE)
    trump_dealing = torch.zeros(n_sim, dtype=torch.bool, device=DEVICE)

    xi_taiwan_triggered = torch.zeros(n_sim, dtype=torch.bool, device=DEVICE)
    xi_taiwan_month = torch.full((n_sim,), 999, dtype=torch.long, device=DEVICE)  # When Taiwan triggered
    xi_stim_active = torch.zeros(n_sim, device=DEVICE)
    xi_delay = torch.zeros(n_sim, device=DEVICE)

    for t in range(MAX_HORIZON_MONTHS):
        # Base transition
        diffs = paths.unsqueeze(1) - cluster_centers.unsqueeze(0)
        dist_sq = (diffs ** 2).sum(dim=2)
        nearest = dist_sq.argmin(dim=1)

        CHUNK = 50000
        dens_vals = torch.zeros(n_sim, device=DEVICE)
        for cs in range(0, n_sim, CHUNK):
            ce = min(cs+CHUNK, n_sim)
            d = torch.cdist(paths[cs:ce], X_gpu)
            topk, _ = d.topk(K_DENSITY, dim=1, largest=False)
            dens_vals[cs:ce] = topk.mean(dim=1)

        density_ot[:, t] = dens_vals
        in_sparse = dens_vals > density_thr
        crisis_boost = in_sparse.float() * 0.5

        sizes = cluster_sizes_gpu[nearest]
        rand_idx = (torch.rand(n_sim, device=DEVICE) * sizes.float()).long().clamp(max=max_pool-1)
        base_delta = cluster_delta_pool_gpu[nearest, rand_idx]

        if in_sparse.any():
            crisis_rand = torch.rand(n_sim, device=DEVICE) < 0.4
            use_crisis = in_sparse & crisis_rand
            if use_crisis.any():
                nc = use_crisis.sum()
                cidx = torch.randint(0, len(all_crisis_pool_gpu), (nc.item(),), device=DEVICE)
                base_delta[use_crisis] = all_crisis_pool_gpu[cidx]

        noise_amp = 1.0 + crisis_boost * 0.3
        base_delta = base_delta * noise_amp.unsqueeze(1)
        attr_base += torch.norm(base_delta, dim=1)

        delta = base_delta.clone()

        stress = torch.norm(paths - hist_median_gpu, dim=1) / market_stress_ref

        # ---- TRUMP ----
        if use_trump:
            rand = torch.rand(n_sim, device=DEVICE)
            market_crashing = stress > 1.3

            # Tariff escalation
            esc_prob = TRUMP['tariff_escalation_rate'] * (1 + trump_tariff * 0.5)
            escalate = ~market_crashing & (rand < esc_prob) & ~trump_dealing
            if escalate.any():
                scale = trump_tariff[escalate].unsqueeze(1) * 0.5 + 0.5
                td = templates['tariff_escalate'] * scale
                delta[escalate] += td
                attr_tariff[escalate] += torch.norm(td, dim=1)
                trump_tariff[escalate] = (trump_tariff[escalate] + 0.05).clamp(max=1.0)

            # Tariff pause
            rand2 = torch.rand(n_sim, device=DEVICE)
            pause = market_crashing & (trump_tariff > 0.3) & (rand2 < TRUMP['tariff_pause_on_crash'])
            if pause.any():
                td = templates['tariff_deesc'] * 0.5
                delta[pause] += td
                attr_tariff[pause] += torch.norm(td.expand(pause.sum().item(), -1), dim=1)
                trump_tariff[pause] = (trump_tariff[pause] - 0.1).clamp(min=0.1)

            if t >= TRUMP['deal_month_threshold']:
                rand3 = torch.rand(n_sim, device=DEVICE)
                trump_dealing = trump_dealing | (~trump_dealing & (rand3 < 0.1))

            # Military action
            trump_mil_cd = (trump_mil_cd - 1).clamp(min=0)
            mil_prob = TRUMP['military_action_base_prob'] + trump_mil_mom * 0.05
            rand4 = torch.rand(n_sim, device=DEVICE)
            mil_act = (trump_mil_cd == 0) & (rand4 < mil_prob)
            if mil_act.any():
                ms = trump_mil_mom[mil_act].unsqueeze(1) * 1.5 + 0.5
                md = templates['military'] * ms
                delta[mil_act] += md
                attr_military[mil_act] += torch.norm(md, dim=1)
                trump_mil_mom[mil_act] = (trump_mil_mom[mil_act] + TRUMP['military_momentum_increment']).clamp(max=1.0)
                trump_mil_cd[mil_act] = TRUMP['military_cooldown_months']

            # Fed pressure
            rand5 = torch.rand(n_sim, device=DEVICE)
            fp = rand5 < (TRUMP['fed_pressure_probability'] + trump_fed_lvl * 0.1)
            if fp.any():
                delta[fp] += templates['fed_pressure']
                attr_fed[fp] += torch.norm(templates['fed_pressure']).item()
                trump_fed_lvl[fp] += TRUMP['fed_pressure_escalation']

        # ---- XI ----
        if use_xi:
            # Taiwan
            if t < XI['taiwan_ramp_start_month']:
                tp = XI['taiwan_base_prob']
            elif t < XI['taiwan_ramp_peak_month']:
                frac = (t - XI['taiwan_ramp_start_month']) / (XI['taiwan_ramp_peak_month'] - XI['taiwan_ramp_start_month'])
                tp = XI['taiwan_base_prob'] + frac * (XI['taiwan_prob_2027_peak'] - XI['taiwan_base_prob'])
            else:
                tp = XI['taiwan_prob_2027_peak'] * (0.98 ** (t - XI['taiwan_ramp_peak_month']))

            stress_boost_tw = (stress - 1.0).clamp(min=0) * 0.02
            eff_tp = tp + stress_boost_tw

            rand6 = torch.rand(n_sim, device=DEVICE)
            tw_act = ~xi_taiwan_triggered & (rand6 < eff_tp)
            if tw_act.any():
                noise = torch.randn(tw_act.sum().item(), D, device=DEVICE) * 0.3
                td = templates['taiwan'] + noise * templates['taiwan'].abs()
                delta[tw_act] += td
                attr_taiwan[tw_act] += torch.norm(td, dim=1)
                xi_taiwan_triggered = xi_taiwan_triggered | tw_act
                xi_taiwan_month[tw_act] = t

            # Taiwan ongoing disruption
            tw_ongoing = xi_taiwan_triggered & (torch.rand(n_sim, device=DEVICE) < 0.6)
            if tw_ongoing.any():
                td = templates['taiwan'] * 0.15
                delta[tw_ongoing] += td
                attr_taiwan_ongoing[tw_ongoing] += torch.norm(td).item()

            # Stimulus
            economy_bad = stress > XI['gdp_red_line_trigger']
            xi_delay[economy_bad] += 1
            xi_delay[~economy_bad] = (xi_delay[~economy_bad] - 0.5).clamp(min=0)
            perceived = xi_delay >= XI['delayed_response_months']
            rand7 = torch.rand(n_sim, device=DEVICE)
            do_stim = perceived & (xi_stim_active <= 0) & (rand7 < XI['stimulus_probability'])
            if do_stim.any():
                ss = XI['stimulus_scale'] * (1 - XI['flood_stimulus_aversion'] * 0.3)
                td = templates['stimulus'] * ss
                delta[do_stim] += td
                attr_stimulus[do_stim] += torch.norm(td).item()
                xi_stim_active[do_stim] = 12

            stim_on = xi_stim_active > 0
            if stim_on.any():
                strength = (xi_stim_active[stim_on] / 12).unsqueeze(1)
                td = templates['stimulus'] * XI['stimulus_scale'] * 0.3 * strength
                delta[stim_on] += torch.norm(td, dim=1).unsqueeze(1).expand_as(td) * 0  # just track
                attr_stimulus[stim_on] += torch.norm(td, dim=1)
                delta[stim_on] += td
                xi_stim_active[stim_on] -= 1

            # Capital controls
            fin_stress = dens_vals > density_thr * XI['capital_control_threshold']
            rand8 = torch.rand(n_sim, device=DEVICE)
            do_cc = fin_stress & (rand8 < XI['capital_control_probability'])
            if do_cc.any():
                delta[do_cc] += templates['capital_control']
                attr_capctrl[do_cc] += torch.norm(templates['capital_control']).item()

            # Tech retaliation (needs Trump tariff state, use default 0.6 threshold if Trump off)
            tariff_high = trump_tariff > 0.6 if use_trump else torch.ones(n_sim, dtype=torch.bool, device=DEVICE) * 0.3
            if use_trump:
                tariff_high = trump_tariff > 0.6
            else:
                tariff_high = torch.rand(n_sim, device=DEVICE) < 0.3  # Assume some baseline tariff environment
            rand9 = torch.rand(n_sim, device=DEVICE)
            do_tech = tariff_high & (rand9 < XI['tech_retaliation_probability'])
            if do_tech.any():
                delta[do_tech] += templates['tech_retaliation']
                attr_tech[do_tech] += torch.norm(templates['tech_retaliation']).item()

        # Noise on leader decisions
        if use_trump or use_xi:
            leader_part = delta - base_delta
            noise = torch.randn_like(leader_part) * 0.2
            delta = delta + noise * leader_part.abs()

        paths = paths + delta

        curr_dnorm = torch.norm(delta, dim=1)
        if t > 0: accel_ot[:, t] = torch.abs(curr_dnorm - prev_dnorm)
        prev_dnorm = curr_dnorm

        path_sum = path_sum + paths
        path_count = path_count + 1
        running_mean = path_sum / path_count.unsqueeze(1)
        deviation_ot[:, t] = torch.norm(paths - running_mean, dim=1)

        if (t+1) % 12 == 0:
            tw_pct = xi_taiwan_triggered.float().mean().item() * 100 if use_xi else 0
            print(f"  Month {t+1:2d}: density={dens_vals.mean().item():.3f}, "
                  f"sparse={in_sparse.float().mean().item()*100:.0f}%, "
                  f"taiwan={tw_pct:.1f}%, "
                  f"elapsed={time.time()-t0:.1f}s")

    elapsed = time.time() - t0

    # Crisis probabilities
    dc = density_ot.cpu().numpy()
    ac = accel_ot.cpu().numpy()
    dv = deviation_ot.cpu().numpy()
    d_c = dc > density_thr
    a_c = ac > accel_thr
    v_c = dv > deviation_thr
    two_c = (d_c.astype(int) + a_c.astype(int) + v_c.astype(int)) >= 2
    all_c = d_c & a_c & v_c

    def first_hit(cm):
        first = np.full(n_sim, MAX_HORIZON_MONTHS+1, dtype=int)
        for i in range(n_sim):
            h = np.where(cm[i])[0]
            if len(h) > 0: first[i] = h[0]+1
        return first

    cum_two = np.array([np.mean(first_hit(two_c) <= t+1) for t in range(MAX_HORIZON_MONTHS)])
    cum_all = np.array([np.mean(first_hit(all_c) <= t+1) for t in range(MAX_HORIZON_MONTHS)])

    # Attribution analysis
    total_disp = (attr_base + attr_tariff + attr_military + attr_fed +
                  attr_taiwan + attr_taiwan_ongoing + attr_stimulus + attr_capctrl + attr_tech)
    total_disp = total_disp.clamp(min=1e-8)

    attr_pcts = {
        'Random walk (economy)': attr_base / total_disp,
        'Trump: Tariffs': attr_tariff / total_disp,
        'Trump: Military': attr_military / total_disp,
        'Trump: Fed pressure': attr_fed / total_disp,
        'Xi: Taiwan (initial)': attr_taiwan / total_disp,
        'Xi: Taiwan (ongoing)': attr_taiwan_ongoing / total_disp,
        'Xi: Stimulus': attr_stimulus / total_disp,
        'Xi: Capital controls': attr_capctrl / total_disp,
        'Xi: Tech retaliation': attr_tech / total_disp,
    }

    print(f"\n  Time: {elapsed:.1f}s")
    print(f"\n  2/3 crisis:  6mo={cum_two[5]*100:.1f}%  12mo={cum_two[11]*100:.1f}%  "
          f"24mo={cum_two[23]*100:.1f}%  36mo={cum_two[35]*100:.1f}%  48mo={cum_two[47]*100:.1f}%")
    print(f"  3/3 crisis:  6mo={cum_all[5]*100:.1f}%  12mo={cum_all[11]*100:.1f}%  "
          f"24mo={cum_all[23]*100:.1f}%  36mo={cum_all[35]*100:.1f}%  48mo={cum_all[47]*100:.1f}%")

    # Attribution report
    if use_trump or use_xi:
        print(f"\n  --- DISPLACEMENT ATTRIBUTION (mean across all paths) ---")
        for name, pct_tensor in attr_pcts.items():
            mean_pct = pct_tensor.mean().item() * 100
            if mean_pct > 0.1:
                print(f"    {name:30s}: {mean_pct:5.1f}%")

        # Separate Taiwan vs non-Taiwan paths
        if use_xi:
            tw_mask = xi_taiwan_triggered.cpu().numpy()
            no_tw_mask = ~xi_taiwan_triggered.cpu().numpy()
            n_tw = tw_mask.sum()
            n_no_tw = no_tw_mask.sum()

            if n_tw > 0 and n_no_tw > 0:
                print(f"\n  --- TAIWAN SPLIT ---")
                print(f"  Taiwan triggered: {n_tw:,} paths ({n_tw/n_sim*100:.1f}%)")
                print(f"  No Taiwan:        {n_no_tw:,} paths ({n_no_tw/n_sim*100:.1f}%)")

                # Crisis rates for each group
                tw_two = two_c[tw_mask]
                no_tw_two = two_c[no_tw_mask]
                tw_all = all_c[tw_mask]
                no_tw_all = all_c[no_tw_mask]

                def cum_from_matrix(cm):
                    first = np.full(cm.shape[0], MAX_HORIZON_MONTHS+1, dtype=int)
                    for i in range(cm.shape[0]):
                        h = np.where(cm[i])[0]
                        if len(h) > 0: first[i] = h[0]+1
                    return np.array([np.mean(first <= t+1) for t in range(MAX_HORIZON_MONTHS)])

                tw_cum2 = cum_from_matrix(tw_two)
                no_tw_cum2 = cum_from_matrix(no_tw_two)
                tw_cum3 = cum_from_matrix(tw_all)
                no_tw_cum3 = cum_from_matrix(no_tw_all)

                print(f"\n  {'':30s}  {'12mo':>7s}  {'24mo':>7s}  {'36mo':>7s}  {'48mo':>7s}")
                print(f"  {'-'*65}")
                print(f"  {'2/3 crisis WITH Taiwan':30s}  {tw_cum2[11]*100:6.1f}%  {tw_cum2[23]*100:6.1f}%  {tw_cum2[35]*100:6.1f}%  {tw_cum2[47]*100:6.1f}%")
                print(f"  {'2/3 crisis NO Taiwan':30s}  {no_tw_cum2[11]*100:6.1f}%  {no_tw_cum2[23]*100:6.1f}%  {no_tw_cum2[35]*100:6.1f}%  {no_tw_cum2[47]*100:6.1f}%")
                print(f"  {'3/3 crisis WITH Taiwan':30s}  {tw_cum3[11]*100:6.1f}%  {tw_cum3[23]*100:6.1f}%  {tw_cum3[35]*100:6.1f}%  {tw_cum3[47]*100:6.1f}%")
                print(f"  {'3/3 crisis NO Taiwan':30s}  {no_tw_cum3[11]*100:6.1f}%  {no_tw_cum3[23]*100:6.1f}%  {no_tw_cum3[35]*100:6.1f}%  {no_tw_cum3[47]*100:6.1f}%")

                # Taiwan timing distribution
                tw_months = xi_taiwan_month[xi_taiwan_triggered].cpu().numpy()
                print(f"\n  --- TAIWAN TIMING DISTRIBUTION ---")
                for bucket_start, bucket_end, label in [(0,6,"0-6mo"), (6,12,"6-12mo"), (12,18,"12-18mo"),
                                                         (18,24,"18-24mo"), (24,36,"24-36mo"), (36,48,"36-48mo")]:
                    count = ((tw_months >= bucket_start) & (tw_months < bucket_end)).sum()
                    print(f"    {label:10s}: {count:>8,} ({count/n_tw*100:5.1f}%)")
                print(f"    Median trigger month: {np.median(tw_months):.0f}")

                # Attribution split
                print(f"\n  --- ATTRIBUTION: Taiwan paths vs Non-Taiwan paths ---")
                for name, pct_tensor in attr_pcts.items():
                    tw_pct = pct_tensor[xi_taiwan_triggered].mean().item() * 100
                    no_tw_pct = pct_tensor[~xi_taiwan_triggered].mean().item() * 100
                    if tw_pct > 0.1 or no_tw_pct > 0.1:
                        print(f"    {name:30s}:  Taiwan={tw_pct:5.1f}%  NoTaiwan={no_tw_pct:5.1f}%")

    return {
        'cum_two': cum_two, 'cum_all': cum_all, 'elapsed': elapsed,
        'taiwan_pct': xi_taiwan_triggered.float().mean().item() if use_xi else 0,
    }


# ============================================================
# RUN 4 SCENARIOS
# ============================================================

results = {}

# 1. No leaders (baseline)
results['No Leaders'] = run_simulation(N_SIMULATIONS, False, False, "SCENARIO 1: No Leaders (pure random walk)")

# 2. Trump only
results['Trump Only'] = run_simulation(N_SIMULATIONS, True, False, "SCENARIO 2: Trump Only")

# 3. Xi only
results['Xi Only'] = run_simulation(N_SIMULATIONS, False, True, "SCENARIO 3: Xi Only")

# 4. Both
results['Trump + Xi'] = run_simulation(N_SIMULATIONS, True, True, "SCENARIO 4: Trump + Xi (Full Model)")


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL COMPARISON — WHO BREAKS THE WORLD?")
print("=" * 70)

print(f"\n--- 2/3 crisis probability ---")
print(f"{'Scenario':25s}  {'6mo':>7s}  {'12mo':>7s}  {'24mo':>7s}  {'36mo':>7s}  {'48mo':>7s}")
print("-" * 70)
for sname, sdata in results.items():
    c = sdata['cum_two']
    vals = [c[5], c[11], c[23], c[35], c[47]]
    print(f"{sname:25s}  " + "  ".join(f"{v*100:6.1f}%" for v in vals))

print(f"\n--- 3/3 crisis probability ---")
print(f"{'Scenario':25s}  {'6mo':>7s}  {'12mo':>7s}  {'24mo':>7s}  {'36mo':>7s}  {'48mo':>7s}")
print("-" * 70)
for sname, sdata in results.items():
    c = sdata['cum_all']
    vals = [c[5], c[11], c[23], c[35], c[47]]
    print(f"{sname:25s}  " + "  ".join(f"{v*100:6.1f}%" for v in vals))

# Marginal contribution
print(f"\n--- MARGINAL CONTRIBUTION (Δ from No Leaders baseline) ---")
bl = results['No Leaders']
print(f"{'Scenario':25s}  {'6mo':>7s}  {'12mo':>7s}  {'24mo':>7s}  {'36mo':>7s}  {'48mo':>7s}")
print("-" * 70)
for sname in ['Trump Only', 'Xi Only', 'Trump + Xi']:
    c = results[sname]['cum_two']
    b = bl['cum_two']
    vals = [c[5]-b[5], c[11]-b[11], c[23]-b[23], c[35]-b[35], c[47]-b[47]]
    print(f"{sname:25s}  " + "  ".join(f"{v*100:+6.1f}%" for v in vals))

# Interaction effect
print(f"\n--- INTERACTION EFFECT (Both > sum of individual) ---")
trump_marginal = results['Trump Only']['cum_two'] - bl['cum_two']
xi_marginal = results['Xi Only']['cum_two'] - bl['cum_two']
both_marginal = results['Trump + Xi']['cum_two'] - bl['cum_two']
interaction = both_marginal - (trump_marginal + xi_marginal)
for t_idx, label in [(5,"6mo"), (11,"12mo"), (23,"24mo"), (35,"36mo"), (47,"48mo")]:
    synergy = interaction[t_idx] * 100
    sign = "amplifying" if synergy > 0 else "dampening"
    print(f"  {label}: Trump+Xi together adds {synergy:+.1f}% beyond sum of parts ({sign})")

print(f"\nOutput: {OUT_DIR}/")

# Save
results_json = {
    "version": "v4b_attribution",
    "scenarios": {sname: {
        "crisis_2of3_48mo": float(sdata['cum_two'][47]),
        "crisis_3of3_48mo": float(sdata['cum_all'][47]),
        "taiwan_triggered_pct": float(sdata['taiwan_pct']),
    } for sname, sdata in results.items()}
}
with open(OUT_DIR / 'monte_carlo_results_v4_attribution.json', 'w') as f:
    json.dump(results_json, f, indent=2)
