#!/usr/bin/env python3
"""
Monte Carlo Crisis Engine v5 — Multi-Leader Chain Reactions
============================================================
v4 + Putin决策函数 + 中东模块 + 欧洲团结度 + 交叉耦合矩阵

核心改进：
1. 5个决策模块（Trump扩展/Xi扩展/Putin/中东/欧洲）
2. 领导人之间的决策不再独立——交叉耦合矩阵调制概率
3. 当前状态编入：霍尔木兹已封锁、油价$100+、真主党已开火
4. 详细归因追踪：每条路径记录崩溃原因

归因维度（15个）：
  base        — 经济基本面漂移
  tariff      — Trump关税
  military_us — Trump军事行动
  fed_press   — Trump施压美联储
  taiwan      — Xi台湾行动
  stimulus_cn — Xi经济刺激
  capctrl     — Xi资本管制
  tech_ret    — Xi科技报复
  energy_ru   — Putin能源武器化
  mil_ru      — Putin军事升级（乌克兰）
  nuclear_ru  — Putin核姿态
  hormuz      — 霍尔木兹海峡封锁（已是当前状态）
  iran_war    — 伊朗战争冲击（持续）
  hezbollah   — 真主党/代理人战争
  eu_fracture — 欧洲分裂效应
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
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# CONFIG
# ============================================================

DATA_BASE = Path('/home/user/global-financial-sim/data')
ECON_BASE = DATA_BASE / 'economic'
LEADER_BASE = DATA_BASE / 'leaders'
OUT_DIR = Path('/home/user/global-financial-sim/output/monte_carlo')
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SIMULATIONS = 500_000  # 500K for speed (attribution needs 4+ scenarios)
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
    "Asia97": ("1997-07", "1998-06"),
    "Dotcom": ("2000-03", "2002-10"),
    "GFC": ("2007-08", "2009-06"),
    "EuroDebt": ("2010-04", "2012-07"),
    "CommodityCrash": ("2014-06", "2016-02"),
    "TradeWar": ("2018-06", "2019-12"),
    "COVID": ("2020-02", "2020-06"),
    "Inflation22": ("2022-01", "2022-12"),
}

N_ATTR = 15  # Attribution dimensions
ATTR_NAMES = [
    'base', 'tariff', 'military_us', 'fed_press',
    'taiwan', 'stimulus_cn', 'capctrl', 'tech_ret',
    'energy_ru', 'mil_ru', 'nuclear_ru',
    'hormuz', 'iran_war', 'hezbollah',
    'eu_fracture'
]

# ============================================================
# 1-3. LOAD DATA + PCA (same pipeline as v4)
# ============================================================

print("=" * 70)
print("1-3. LOADING DATA + PCA PIPELINE")
print("=" * 70)

# --- Load all economic series ---
all_series = {}

for country_dir in ['us', 'cn', 'jp', 'eu', 'uk', 'ru']:
    base = ECON_BASE / country_dir
    for fn in ['macro.json', 'financial.json']:
        fp = base / fn
        if not fp.exists(): continue
        with open(fp) as f: data = json.load(f)
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict) and any(isinstance(vv, (int,float)) for vv in v.values()):
                    all_series[f"{country_dir}_{k}"] = {kk: float(vv) for kk, vv in v.items() if isinstance(vv, (int,float))}
                elif isinstance(v, list):
                    sd = {}
                    for item in v:
                        if isinstance(item, dict):
                            d = item.get('date', '')
                            val = item.get('value', item.get('close'))
                            if d and val is not None:
                                try: sd[d[:10]] = float(val)
                                except: pass
                    if sd: all_series[f"{country_dir}_{k}"] = sd

# Supplementary data
for fn in sorted((ECON_BASE / 'supplementary').glob('*.json')) if (ECON_BASE / 'supplementary').exists() else []:
    with open(fn) as f: data = json.load(f)
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                all_series[f"supp_{fn.stem}_{k}"] = {kk: float(vv) for kk, vv in v.items() if isinstance(vv, (int,float))}

# BIS data
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
    'mojtaba_khamenei': 'mojtaba_khamenei', 'elon_musk': 'elon_musk',
    'netanyahu': 'netanyahu', 'zelenskyy': 'zelenskyy', 'rutte': 'rutte',
    'merz': 'merz', 'macron': 'macron', 'starmer': 'starmer',
    'orban': 'orban', 'tusk': 'tusk', 'meloni': 'meloni',
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
            if not en or en.lower() in ('present','incumbent',''): en = '2026-03-13'
            leader_periods.append((pk, st, en, rl, co))

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
# 4. BUILD DISPLACEMENT TEMPLATES (expanded for v5)
# ============================================================

print("\n" + "=" * 70)
print("4. BUILD DISPLACEMENT TEMPLATES (v5 expanded)")
print("=" * 70)

deltas = np.diff(X_pca, axis=0)
delta_norms = np.linalg.norm(deltas, axis=1)
median_delta = np.median(delta_norms)

def month_range_delta(start_ym, end_ym):
    idxs = [i for i, m in enumerate(months_g[:-1]) if start_ym <= m[:7] <= end_ym]
    if idxs:
        return deltas[idxs].mean(axis=0), len(idxs)
    return np.zeros(D), 0

# --- TRUMP TEMPLATES (same as v4) ---
tariff_escalate_delta, n_te = month_range_delta("2018-06", "2018-09")
tariff_deesc_delta, n_td = month_range_delta("2019-06", "2019-12")
mil_action_delta, n_mil = month_range_delta("1990-08", "1990-10")
oil_crisis_delta_73, n_oc = month_range_delta("1973-10", "1974-03")
if n_mil > 0 and n_oc > 0:
    mil_action_delta = mil_action_delta * 0.7 + oil_crisis_delta_73 * 0.3
fed_pressure_delta, n_fp = month_range_delta("2018-10", "2018-12")
fed_pressure_delta = fed_pressure_delta * 0.5

# --- XI TEMPLATES (same as v4) ---
gfc_delta, n_gfc = month_range_delta("2008-09", "2008-12")
trade_war_delta, n_tw = month_range_delta("2018-06", "2019-06")
taiwan_action_delta = gfc_delta * 1.5 + trade_war_delta * 1.0 + mil_action_delta * 1.0

stim_delta_09, n_s09 = month_range_delta("2009-03", "2009-12")
stim_delta_20, n_s20 = month_range_delta("2020-05", "2020-12")
if n_s09 > 0 and n_s20 > 0:
    stimulus_delta = (stim_delta_09 + stim_delta_20) / 2
else:
    stimulus_delta = stim_delta_09 if n_s09 > 0 else stim_delta_20

capctrl_delta, n_cc = month_range_delta("2015-08", "2016-02")
capctrl_delta = capctrl_delta * 0.6
tech_war_delta, n_twr = month_range_delta("2019-05", "2019-08")

# --- PUTIN TEMPLATES (new for v5) ---
# Energy weapon: 2022 Russia gas cutoff to Europe
energy_weapon_delta, n_ew = month_range_delta("2022-02", "2022-06")
print(f"  Putin energy weapon template: mag={np.linalg.norm(energy_weapon_delta):.4f} ({n_ew} months)")

# Military escalation Ukraine: 2022 invasion shock
mil_ru_delta, n_mru = month_range_delta("2022-02", "2022-04")
print(f"  Putin military escalation template: mag={np.linalg.norm(mil_ru_delta):.4f} ({n_mru} months)")

# Nuclear posture: scaled version of military (tail risk amplifier)
# No direct historical template — use military × 2.0 as proxy for nuclear scare
nuclear_posture_delta = mil_ru_delta * 2.0 if n_mru > 0 else mil_action_delta * 2.5
print(f"  Putin nuclear posture template: mag={np.linalg.norm(nuclear_posture_delta):.4f}")

# Sanctions adaptation / Russia pivot to China: stabilization effect
# 2022 H2 partial recovery
sanctions_adapt_delta, n_sa = month_range_delta("2022-07", "2022-12")
sanctions_adapt_delta = sanctions_adapt_delta * 0.4  # Partial, dampening
print(f"  Putin sanctions adaptation template: mag={np.linalg.norm(sanctions_adapt_delta):.4f} ({n_sa} months)")

# --- MIDDLE EAST TEMPLATES (new for v5) ---
# Hormuz blockade: 1973 oil crisis × 1.5 (bigger impact — 20% of global oil)
hormuz_delta = oil_crisis_delta_73 * 1.5 if n_oc > 0 else np.zeros(D)
print(f"  Hormuz blockade template: mag={np.linalg.norm(hormuz_delta):.4f}")

# Iran war ongoing: Gulf War 1990 + 2022 energy crisis blend
iran_war_delta_90, _ = month_range_delta("1990-08", "1990-12")
iran_war_delta_22, _ = month_range_delta("2022-03", "2022-06")
iran_war_delta = iran_war_delta_90 * 0.6 + iran_war_delta_22 * 0.4
print(f"  Iran war ongoing template: mag={np.linalg.norm(iran_war_delta):.4f}")

# Hezbollah/proxy war: 2006 Lebanon war equivalent + trade disruption
# Use trade war as proxy for supply chain disruption component
hezbollah_delta = mil_action_delta * 0.5 + trade_war_delta * 0.3
print(f"  Hezbollah proxy war template: mag={np.linalg.norm(hezbollah_delta):.4f}")

# Oil price stabilization (if Hormuz reopens): recovery template
oil_recovery_delta, n_or = month_range_delta("1991-01", "1991-06")
print(f"  Oil recovery template: mag={np.linalg.norm(oil_recovery_delta):.4f} ({n_or} months)")

# --- EUROPE FRACTURE TEMPLATE ---
# Euro debt crisis dynamics: when European unity fails
eu_fracture_delta, n_ef = month_range_delta("2010-05", "2011-06")
print(f"  EU fracture template: mag={np.linalg.norm(eu_fracture_delta):.4f} ({n_ef} months)")

# ============================================================
# 5. LEADER DECISION PARAMETERS (expanded)
# ============================================================

print("\n" + "=" * 70)
print("5. LEADER DECISION PARAMETERS (v5)")
print("=" * 70)

# --- TRUMP (expanded with Russia/ME interactions) ---
TRUMP = {
    'market_crash_threshold': 0.08,
    'stock_market_report_card': 0.93,
    'tariff_volatility': 0.90,
    'tariff_escalation_rate': 0.15,
    'tariff_pause_on_crash': 0.85,
    'military_action_willingness': 0.92,
    'military_momentum_init': 0.6,    # Higher: already did Iran, Venezuela, Nigeria
    'military_momentum_increment': 0.06,
    'military_action_base_prob': 0.04,
    'military_cooldown_months': 4,
    'fed_pressure_probability': 0.30,
    'fed_pressure_escalation': 0.05,
    'deal_aspiration': 0.85,
    'deal_month_threshold': 18,
    # v5 new: Russia/ME interactions
    'russia_deal_probability': 0.15,       # Monthly prob of pursuing Russia deal
    'russia_deal_start_month': 6,          # After 6 months, starts seeking deal
    'iran_pressure_continuation': 0.70,    # Probability of maintaining Iran pressure
    'israel_green_light': 0.88,            # Netanyahu action enablement
    'oil_price_domestic_pain': 0.80,       # How much oil price hurts his base
}

# --- XI (expanded with ME reaction) ---
XI = {
    'taiwan_base_prob': 0.005,
    'taiwan_prob_2027_peak': 0.03,
    'taiwan_ramp_start_month': 6,
    'taiwan_ramp_peak_month': 18,
    'taiwan_action_prob_overall': 0.70,
    'gdp_red_line_trigger': 0.85,
    'stimulus_probability': 0.60,
    'flood_stimulus_aversion': 0.82,
    'stimulus_scale': 0.4,
    'financial_crisis_response': 0.94,
    'capital_control_threshold': 0.80,
    'capital_control_probability': 0.40,
    'information_fidelity': 0.41,
    'delayed_response_months': 3,
    'political_security_priority': 0.97,
    'tech_retaliation_probability': 0.25,
    # v5 new: ME reaction
    'hormuz_energy_vulnerability': 0.70,   # 70% oil via Hormuz
    'russia_pipeline_substitution': 0.25,  # How much Russia pipeline can replace
    'mediation_attempt_prob': 0.10,        # Monthly prob of attempting ME mediation
    'us_distracted_taiwan_boost': 0.015,   # Extra taiwan prob when US bogged in ME
}

# --- PUTIN (new for v5) ---
PUTIN = {
    # Energy weapon
    'energy_weapon_propensity': 0.91,      # From profile
    'energy_cutoff_probability': 0.05,     # Monthly prob of new energy cutoff
    'energy_cutoff_escalation': 0.02,      # Increases each month
    'oil_price_benefit_threshold': 80,     # Above $80/bbl → Russia benefits
    # Military (Ukraine)
    'ukraine_fixation': 0.96,              # From profile
    'military_escalation_base_prob': 0.08, # Monthly prob of escalation
    'sunken_cost_commitment': 0.93,        # Won't de-escalate easily
    'war_intensity_init': 0.65,            # Current war intensity
    'war_intensity_increment': 0.05,       # Per escalation
    # Nuclear
    'nuclear_posture_init': 0.20,          # Current nuclear rhetoric level
    'nuclear_escalation_prob': 0.02,       # Monthly prob of nuclear rhetoric increase
    'nuclear_action_threshold': 0.85,      # If posture reaches this → tactical nuke risk
    'nuclear_action_prob': 0.005,          # Monthly prob at threshold
    # Trump interaction
    'trump_deal_receptivity': 0.40,        # How likely to accept Trump deal
    'deal_minimum_territory': 0.80,        # Won't give back occupied territory
    # Sanctions adaptation
    'sanctions_resilience': 0.87,          # From profile
    'china_pivot_depth': 0.75,             # Dependence on China trade
}

# --- MIDDLE EAST (combined Mojtaba + Netanyahu) ---
MIDEAST = {
    # Mojtaba Khamenei / Iran
    'revenge_motivation': 0.95,            # From profile
    'hormuz_closure_current': 0.96,        # Already nearly closed
    'hormuz_reopening_prob': 0.02,         # Very low monthly prob of reopening
    'hormuz_reopening_conditions': 0.15,   # Conditions for reopening (ceasefire etc)
    'nuclear_acceleration': 0.97,          # From profile
    'nuclear_threshold_months': 18,        # Estimated months to weapon capability
    'proxy_network_activation': 0.97,      # Already activated
    'iran_war_ongoing_intensity': 0.85,    # Current war intensity
    'iran_war_deescalation_prob': 0.03,    # Monthly prob of de-escalation
    'gulf_infrastructure_damage': 0.60,    # Current damage to Gulf oil infra
    # Netanyahu / Israel
    'netanyahu_survival_primacy': 0.97,    # From profile
    'iran_obsession': 0.95,               # From profile
    'trump_netanyahu_alignment': 0.91,     # From profile
    'continued_strikes_prob': 0.40,        # Monthly prob of continued Israel strikes on Iran
    # Hezbollah
    'hezbollah_active': True,              # Already resumed (March 2, 2026)
    'hezbollah_intensity': 0.70,           # Current intensity
    'hezbollah_escalation_prob': 0.10,     # Monthly prob of further escalation
    # Houthi
    'houthi_resumption_prob': 0.60,        # Probability of full Red Sea resumption
    'houthi_shipping_disruption': 0.40,    # Current disruption level
}

# --- EUROPE ---
EUROPE = {
    'unity_index_init': 0.45,              # From structural research
    'unity_decay_rate': 0.005,             # Monthly baseline decay
    'russia_threat_boost': 0.03,           # Unity boost per Russian escalation
    'trump_pressure_damage': 0.02,         # Unity damage per Trump pressure event
    'orban_sabotage_prob': 0.30,           # Monthly prob of Orbán veto/sabotage
    'orban_sabotage_damage': 0.03,         # Unity damage per sabotage
    'france_crisis_prob': 0.08,            # Monthly prob of French government crisis (35%/quarter)
    'france_crisis_damage': 0.05,          # Unity damage per French crisis
    'fiscal_stress_threshold': 0.60,       # Unity below this → fiscal stress amplified
    'defense_spending_gap': 0.40,          # Gap between NATO target and actual
    'migration_pressure': 0.35,            # Background migration pressure on unity
}

print(f"  Trump: mil_momentum=0.6, iran_pressure=0.70")
print(f"  Xi: taiwan_base=0.005, hormuz_vulnerability=0.70")
print(f"  Putin: war_intensity=0.65, energy_weapon=0.91, nuclear_init=0.20")
print(f"  MidEast: hormuz_closed=0.96, revenge=0.95, nuclear_accel=0.97")
print(f"  Europe: unity=0.45, france_crisis_prob=0.08")

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

print(f"Density threshold: {density_thr:.4f}")
print(f"Accel threshold:   {accel_thr:.4f}")
print(f"Deviation threshold: {deviation_thr:.4f}")

hist_median = np.median(X_pca, axis=0)
market_stress_ref = np.percentile(np.linalg.norm(X_pca - hist_median, axis=1), 90)

# ============================================================
# 7. PREPARE GPU TENSORS
# ============================================================

print("\n" + "=" * 70)
print("7. PREPARE GPU TENSORS")
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

all_crisis_deltas = deltas[list(crisis_month_set)] if crisis_month_set else deltas[:10]
all_crisis_pool_gpu = torch.tensor(all_crisis_deltas, dtype=torch.float32, device=DEVICE)

# Templates on GPU
templates = {
    'tariff_escalate': torch.tensor(tariff_escalate_delta, dtype=torch.float32, device=DEVICE),
    'tariff_deesc': torch.tensor(tariff_deesc_delta, dtype=torch.float32, device=DEVICE),
    'military_action': torch.tensor(mil_action_delta, dtype=torch.float32, device=DEVICE),
    'fed_pressure': torch.tensor(fed_pressure_delta, dtype=torch.float32, device=DEVICE),
    'taiwan_action': torch.tensor(taiwan_action_delta, dtype=torch.float32, device=DEVICE),
    'stimulus': torch.tensor(stimulus_delta, dtype=torch.float32, device=DEVICE),
    'capital_control': torch.tensor(capctrl_delta, dtype=torch.float32, device=DEVICE),
    'tech_retaliation': torch.tensor(tech_war_delta, dtype=torch.float32, device=DEVICE),
    # v5 new
    'energy_weapon': torch.tensor(energy_weapon_delta, dtype=torch.float32, device=DEVICE),
    'mil_ru': torch.tensor(mil_ru_delta, dtype=torch.float32, device=DEVICE),
    'nuclear_posture': torch.tensor(nuclear_posture_delta, dtype=torch.float32, device=DEVICE),
    'sanctions_adapt': torch.tensor(sanctions_adapt_delta, dtype=torch.float32, device=DEVICE),
    'hormuz': torch.tensor(hormuz_delta, dtype=torch.float32, device=DEVICE),
    'iran_war': torch.tensor(iran_war_delta, dtype=torch.float32, device=DEVICE),
    'hezbollah': torch.tensor(hezbollah_delta, dtype=torch.float32, device=DEVICE),
    'oil_recovery': torch.tensor(oil_recovery_delta, dtype=torch.float32, device=DEVICE),
    'eu_fracture': torch.tensor(eu_fracture_delta, dtype=torch.float32, device=DEVICE),
}

hist_median_gpu = torch.tensor(hist_median, dtype=torch.float32, device=DEVICE)

print(f"Clusters: {N_CLUSTERS}, crisis deltas: {len(all_crisis_deltas)}")
print(f"Templates: {len(templates)} displacement vectors on GPU")

# ============================================================
# 8. MONTE CARLO WITH MULTI-LEADER CHAIN REACTIONS
# ============================================================

print("\n" + "=" * 70)
print("8. MONTE CARLO v5 — MULTI-LEADER CHAIN REACTIONS")
print(f"   {N_SIMULATIONS:,} paths × {MAX_HORIZON_MONTHS} months")
print("=" * 70)

curr_idx = N - 1
base_state = X_pca[curr_idx].copy()

# v5 starting condition: current state + ACTIVE Middle East crisis
# Hormuz is ALREADY closed, oil is ALREADY at $100+, Hezbollah is ALREADY firing
# This is not a probability — it's the initial condition
oil_shock_months = []
for cname, (cs, ce) in [("Oil73", ("1973-10", "1974-03")), ("Gulf90", ("1990-08", "1990-10"))]:
    for i, m in enumerate(months_g[:-1]):
        if cs <= m[:7] <= ce: oil_shock_months.append(i)
oil_shock_delta = deltas[oil_shock_months].mean(axis=0) if oil_shock_months else np.zeros(D)

emp_shock_months = [i for i, m in enumerate(months_g[:-1]) if m[:7] in ['2008-10','2008-11','2008-12','2009-01','2009-02','2020-03','2020-04']]
emp_shock_delta = deltas[emp_shock_months].mean(axis=0) * 0.12 if emp_shock_months else np.zeros(D)

trade_shock_months = [i for i, m in enumerate(months_g[:-1]) if '2018-06' <= m[:7] <= '2018-09']
trade_shock_delta_v = deltas[trade_shock_months].mean(axis=0) if trade_shock_months else np.zeros(D)

conf_shock_months = [i for i, m in enumerate(months_g[:-1]) if '2008-09' <= m[:7] <= '2008-12']
conf_shock_delta = deltas[conf_shock_months].mean(axis=0) * 0.5 if conf_shock_months else np.zeros(D)

combined_shock = (oil_shock_delta + emp_shock_delta + trade_shock_delta_v + conf_shock_delta) * 0.70

# Additional ME crisis initial displacement (Hormuz already closed)
me_initial_shock = (hormuz_delta if isinstance(hormuz_delta, np.ndarray) else hormuz_delta.numpy()) * 0.5
shock_state = base_state + combined_shock + me_initial_shock

scenarios = {
    "Baseline (econ only)": {'leaders': False, 'putin': False, 'mideast': False, 'europe': False},
    "Trump+Xi only (v4)": {'leaders': True, 'putin': False, 'mideast': False, 'europe': False},
    "Full v5 (all actors)": {'leaders': True, 'putin': True, 'mideast': True, 'europe': True},
}

all_results = {}

for sc_name, sc_flags in scenarios.items():
    print(f"\n{'='*60}")
    print(f"--- Scenario: {sc_name} ---")
    print(f"{'='*60}")
    t0 = time.time()

    use_leaders = sc_flags['leaders']
    use_putin = sc_flags['putin']
    use_mideast = sc_flags['mideast']
    use_europe = sc_flags['europe']

    start_gpu = torch.tensor(shock_state, dtype=torch.float32, device=DEVICE)
    paths = start_gpu.unsqueeze(0).expand(N_SIMULATIONS, -1).clone()

    density_ot = torch.zeros(N_SIMULATIONS, MAX_HORIZON_MONTHS, device=DEVICE)
    accel_ot = torch.zeros(N_SIMULATIONS, MAX_HORIZON_MONTHS, device=DEVICE)
    deviation_ot = torch.zeros(N_SIMULATIONS, MAX_HORIZON_MONTHS, device=DEVICE)

    # Attribution tracking: cumulative |displacement| from each source
    attr = torch.zeros(N_SIMULATIONS, N_ATTR, device=DEVICE)

    prev_dnorm = torch.zeros(N_SIMULATIONS, device=DEVICE)
    path_sum = start_gpu.unsqueeze(0).expand(N_SIMULATIONS, -1).clone()
    path_count = torch.ones(N_SIMULATIONS, device=DEVICE)

    # --- State variables ---
    if use_leaders:
        trump_tariff_level = torch.ones(N_SIMULATIONS, device=DEVICE) * 0.5
        trump_mil_momentum = torch.ones(N_SIMULATIONS, device=DEVICE) * TRUMP['military_momentum_init']
        trump_mil_cooldown = torch.zeros(N_SIMULATIONS, device=DEVICE)
        trump_fed_pressure_level = torch.zeros(N_SIMULATIONS, device=DEVICE)
        trump_seeking_deal = torch.zeros(N_SIMULATIONS, dtype=torch.bool, device=DEVICE)
        trump_russia_deal_active = torch.zeros(N_SIMULATIONS, dtype=torch.bool, device=DEVICE)

        xi_taiwan_triggered = torch.zeros(N_SIMULATIONS, dtype=torch.bool, device=DEVICE)
        xi_stimulus_active = torch.zeros(N_SIMULATIONS, device=DEVICE)
        xi_response_delay = torch.zeros(N_SIMULATIONS, device=DEVICE)

    if use_putin:
        putin_war_intensity = torch.ones(N_SIMULATIONS, device=DEVICE) * PUTIN['war_intensity_init']
        putin_nuclear_posture = torch.ones(N_SIMULATIONS, device=DEVICE) * PUTIN['nuclear_posture_init']
        putin_energy_cutoff_level = torch.zeros(N_SIMULATIONS, device=DEVICE)

    if use_mideast:
        hormuz_closed = torch.ones(N_SIMULATIONS, dtype=torch.bool, device=DEVICE)  # STARTS CLOSED
        iran_war_active = torch.ones(N_SIMULATIONS, dtype=torch.bool, device=DEVICE)  # STARTS ACTIVE
        hezbollah_active = torch.ones(N_SIMULATIONS, dtype=torch.bool, device=DEVICE)  # STARTS ACTIVE
        houthi_active = torch.zeros(N_SIMULATIONS, dtype=torch.bool, device=DEVICE)  # Debating
        iran_nuclear_progress = torch.zeros(N_SIMULATIONS, device=DEVICE)
        hormuz_months_closed = torch.zeros(N_SIMULATIONS, device=DEVICE)

    if use_europe:
        eu_unity = torch.ones(N_SIMULATIONS, device=DEVICE) * EUROPE['unity_index_init']

    # Event counters
    evt = {k: 0 for k in ATTR_NAMES}

    for t in range(MAX_HORIZON_MONTHS):
        # --- Standard transition ---
        diffs = paths.unsqueeze(1) - cluster_centers.unsqueeze(0)
        dist_sq = (diffs ** 2).sum(dim=2)
        nearest = dist_sq.argmin(dim=1)

        CHUNK = 50000
        dens_vals = torch.zeros(N_SIMULATIONS, device=DEVICE)
        for cs in range(0, N_SIMULATIONS, CHUNK):
            ce = min(cs+CHUNK, N_SIMULATIONS)
            d = torch.cdist(paths[cs:ce], X_gpu)
            topk, _ = d.topk(K_DENSITY, dim=1, largest=False)
            dens_vals[cs:ce] = topk.mean(dim=1)

        density_ot[:, t] = dens_vals
        in_sparse = dens_vals > density_thr
        crisis_boost = in_sparse.float() * 0.5

        sizes = cluster_sizes_gpu[nearest]
        rand_idx = (torch.rand(N_SIMULATIONS, device=DEVICE) * sizes.float()).long()
        rand_idx = rand_idx.clamp(max=max_pool - 1)
        delta = cluster_delta_pool_gpu[nearest, rand_idx]

        if in_sparse.any():
            crisis_rand = torch.rand(N_SIMULATIONS, device=DEVICE) < 0.4
            use_crisis = in_sparse & crisis_rand
            if use_crisis.any():
                n_crisis = use_crisis.sum()
                crisis_idx = torch.randint(0, len(all_crisis_pool_gpu), (n_crisis.item(),), device=DEVICE)
                delta[use_crisis] = all_crisis_pool_gpu[crisis_idx]

        noise_amp = 1.0 + crisis_boost * 0.3
        base_delta = delta * noise_amp.unsqueeze(1)
        attr[:, 0] += torch.norm(base_delta, dim=1)  # base attribution

        # Economic stress proxy
        stress = torch.norm(paths - hist_median_gpu, dim=1) / market_stress_ref

        leader_delta = torch.zeros_like(delta)

        # =====================================================
        # TRUMP DECISIONS (expanded)
        # =====================================================
        if use_leaders:
            rand = torch.rand(N_SIMULATIONS, device=DEVICE)

            market_crashing = stress > 1.3
            not_crashing = ~market_crashing

            # (A) Tariffs
            esc_prob = TRUMP['tariff_escalation_rate'] * (1 + trump_tariff_level * 0.5)
            # Cross-coupling: if oil price high (ME crisis), less tariff escalation (domestic pain)
            if use_mideast:
                oil_pain = hormuz_closed.float() * TRUMP['oil_price_domestic_pain'] * 0.3
                esc_prob = esc_prob * (1 - oil_pain)

            escalate = not_crashing & (rand < esc_prob) & ~trump_seeking_deal
            if escalate.any():
                d_tariff = templates['tariff_escalate'] * (trump_tariff_level[escalate].unsqueeze(1) * 0.5 + 0.5)
                leader_delta[escalate] += d_tariff
                attr[escalate, 1] += torch.norm(d_tariff, dim=1)
                trump_tariff_level[escalate] = (trump_tariff_level[escalate] + 0.05).clamp(max=1.0)
                evt['tariff'] += escalate.sum().item()

            rand2 = torch.rand(N_SIMULATIONS, device=DEVICE)
            pause = market_crashing & (trump_tariff_level > 0.3) & (rand2 < TRUMP['tariff_pause_on_crash'])
            if pause.any():
                leader_delta[pause] += templates['tariff_deesc'] * 0.5
                trump_tariff_level[pause] = (trump_tariff_level[pause] - 0.1).clamp(min=0.1)

            if t >= TRUMP['deal_month_threshold']:
                rand3 = torch.rand(N_SIMULATIONS, device=DEVICE)
                new_deals = ~trump_seeking_deal & (rand3 < 0.1)
                trump_seeking_deal = trump_seeking_deal | new_deals

            # (B) Military action
            trump_mil_cooldown = (trump_mil_cooldown - 1).clamp(min=0)
            can_act = trump_mil_cooldown == 0
            mil_prob = TRUMP['military_action_base_prob'] + trump_mil_momentum * 0.05
            # Cross-coupling: if already in Iran war, military momentum stays high but new actions less likely
            if use_mideast:
                mil_prob = mil_prob * (1 - iran_war_active.float() * 0.3)  # Already busy

            rand4 = torch.rand(N_SIMULATIONS, device=DEVICE)
            mil_act = can_act & (rand4 < mil_prob)
            if mil_act.any():
                mil_scale = trump_mil_momentum[mil_act].unsqueeze(1) * 1.5 + 0.5
                d_mil = templates['military_action'] * mil_scale
                leader_delta[mil_act] += d_mil
                attr[mil_act, 2] += torch.norm(d_mil, dim=1)
                trump_mil_momentum[mil_act] = (trump_mil_momentum[mil_act] + TRUMP['military_momentum_increment']).clamp(max=1.0)
                trump_mil_cooldown[mil_act] = TRUMP['military_cooldown_months']
                evt['military_us'] += mil_act.sum().item()

            # (C) Fed pressure
            rand5 = torch.rand(N_SIMULATIONS, device=DEVICE)
            fed_pres = rand5 < (TRUMP['fed_pressure_probability'] + trump_fed_pressure_level * 0.1)
            if fed_pres.any():
                leader_delta[fed_pres] += templates['fed_pressure']
                attr[fed_pres, 3] += torch.norm(templates['fed_pressure'].unsqueeze(0).expand(fed_pres.sum().item(), -1), dim=1)
                trump_fed_pressure_level[fed_pres] += TRUMP['fed_pressure_escalation']
                evt['fed_press'] += fed_pres.sum().item()

            # (D) Russia deal seeking (v5 new)
            if use_putin and t >= TRUMP['russia_deal_start_month']:
                rand_rd = torch.rand(N_SIMULATIONS, device=DEVICE)
                seek_deal = ~trump_russia_deal_active & (rand_rd < TRUMP['russia_deal_probability'])
                # Cross-coupling: if oil price high, more motivation to deal with Russia
                if use_mideast:
                    seek_deal = seek_deal | (~trump_russia_deal_active & hormuz_closed & (rand_rd < TRUMP['russia_deal_probability'] * 2))
                trump_russia_deal_active = trump_russia_deal_active | seek_deal

        # =====================================================
        # XI DECISIONS (expanded with ME reaction)
        # =====================================================
        if use_leaders:
            # (E) Taiwan
            if t < XI['taiwan_ramp_start_month']:
                taiwan_prob = XI['taiwan_base_prob']
            elif t < XI['taiwan_ramp_peak_month']:
                frac = (t - XI['taiwan_ramp_start_month']) / (XI['taiwan_ramp_peak_month'] - XI['taiwan_ramp_start_month'])
                taiwan_prob = XI['taiwan_base_prob'] + frac * (XI['taiwan_prob_2027_peak'] - XI['taiwan_base_prob'])
            else:
                decay = 0.98 ** (t - XI['taiwan_ramp_peak_month'])
                taiwan_prob = XI['taiwan_prob_2027_peak'] * decay

            stress_boost = (stress - 1.0).clamp(min=0) * 0.02

            # Cross-coupling: US distracted by Iran → Taiwan window
            us_distracted_boost = torch.zeros(N_SIMULATIONS, device=DEVICE)
            if use_mideast:
                us_distracted_boost = iran_war_active.float() * XI['us_distracted_taiwan_boost']

            effective_taiwan_prob = taiwan_prob + stress_boost + us_distracted_boost

            # Cross-coupling: Hormuz closed → China energy crisis → REDUCES taiwan probability
            if use_mideast:
                energy_crisis_penalty = hormuz_closed.float() * XI['hormuz_energy_vulnerability'] * 0.01
                effective_taiwan_prob = effective_taiwan_prob - energy_crisis_penalty

            rand6 = torch.rand(N_SIMULATIONS, device=DEVICE)
            taiwan_act = ~xi_taiwan_triggered & (rand6 < effective_taiwan_prob)
            if taiwan_act.any():
                noise = torch.randn(taiwan_act.sum().item(), D, device=DEVICE) * 0.3
                d_tw = templates['taiwan_action'] + noise * templates['taiwan_action'].abs()
                leader_delta[taiwan_act] += d_tw
                attr[taiwan_act, 4] += torch.norm(d_tw, dim=1)
                xi_taiwan_triggered = xi_taiwan_triggered | taiwan_act
                evt['taiwan'] += taiwan_act.sum().item()

            if xi_taiwan_triggered.any():
                ongoing_taiwan = xi_taiwan_triggered & (torch.rand(N_SIMULATIONS, device=DEVICE) < 0.6)
                if ongoing_taiwan.any():
                    d_tw_ongoing = templates['taiwan_action'] * 0.15
                    leader_delta[ongoing_taiwan] += d_tw_ongoing
                    attr[ongoing_taiwan, 4] += torch.norm(d_tw_ongoing.unsqueeze(0).expand(ongoing_taiwan.sum().item(), -1), dim=1)

            # (F) Stimulus (with info delay)
            economy_bad = stress > XI['gdp_red_line_trigger']
            xi_response_delay[economy_bad] += 1
            xi_response_delay[~economy_bad] = (xi_response_delay[~economy_bad] - 0.5).clamp(min=0)

            perceived_crisis = xi_response_delay >= XI['delayed_response_months']
            rand7 = torch.rand(N_SIMULATIONS, device=DEVICE)
            do_stimulus = perceived_crisis & (xi_stimulus_active <= 0) & (rand7 < XI['stimulus_probability'])
            if do_stimulus.any():
                stim_scale = XI['stimulus_scale'] * (1 - XI['flood_stimulus_aversion'] * 0.3)
                d_stim = templates['stimulus'] * stim_scale
                leader_delta[do_stimulus] += d_stim
                attr[do_stimulus, 5] += torch.norm(d_stim.unsqueeze(0).expand(do_stimulus.sum().item(), -1), dim=1)
                xi_stimulus_active[do_stimulus] = 12
                evt['stimulus_cn'] += do_stimulus.sum().item()

            if (xi_stimulus_active > 0).any():
                stim_ongoing = xi_stimulus_active > 0
                stim_strength = (xi_stimulus_active[stim_ongoing] / 12).unsqueeze(1)
                d_stim_ong = templates['stimulus'] * XI['stimulus_scale'] * 0.3 * stim_strength
                leader_delta[stim_ongoing] += d_stim_ong
                attr[stim_ongoing, 5] += torch.norm(d_stim_ong, dim=1)
                xi_stimulus_active[stim_ongoing] -= 1

            # (G) Capital controls
            fin_stress = dens_vals > density_thr * XI['capital_control_threshold']
            rand8 = torch.rand(N_SIMULATIONS, device=DEVICE)
            do_capctrl = fin_stress & (rand8 < XI['capital_control_probability'])
            if do_capctrl.any():
                leader_delta[do_capctrl] += templates['capital_control']
                attr[do_capctrl, 6] += torch.norm(templates['capital_control'].unsqueeze(0).expand(do_capctrl.sum().item(), -1), dim=1)
                evt['capctrl'] += do_capctrl.sum().item()

            # (H) Tech retaliation
            rand9 = torch.rand(N_SIMULATIONS, device=DEVICE)
            tech_ret = (trump_tariff_level > 0.6) & (rand9 < XI['tech_retaliation_probability'])
            if tech_ret.any():
                leader_delta[tech_ret] += templates['tech_retaliation']
                attr[tech_ret, 7] += torch.norm(templates['tech_retaliation'].unsqueeze(0).expand(tech_ret.sum().item(), -1), dim=1)
                evt['tech_ret'] += tech_ret.sum().item()

        # =====================================================
        # PUTIN DECISIONS (new for v5)
        # =====================================================
        if use_putin:
            rand_p = torch.rand(N_SIMULATIONS, device=DEVICE)

            # (I) Energy weapon: further cutoffs to Europe
            energy_prob = PUTIN['energy_cutoff_probability'] + putin_energy_cutoff_level * PUTIN['energy_cutoff_escalation']
            # Cross-coupling: oil price high → MORE incentive to weaponize energy
            if use_mideast:
                energy_prob = energy_prob * (1 + hormuz_closed.float() * 0.5)
            # Cross-coupling: Trump deal active → LESS energy weaponization
            if use_leaders and trump_russia_deal_active.any():
                energy_prob = energy_prob * (1 - trump_russia_deal_active.float() * 0.5)

            do_energy = rand_p < energy_prob
            if do_energy.any():
                d_en = templates['energy_weapon'] * (putin_energy_cutoff_level[do_energy].unsqueeze(1) * 0.5 + 0.5)
                leader_delta[do_energy] += d_en
                attr[do_energy, 8] += torch.norm(d_en, dim=1)
                putin_energy_cutoff_level[do_energy] = (putin_energy_cutoff_level[do_energy] + 0.1).clamp(max=1.0)
                evt['energy_ru'] += do_energy.sum().item()

            # (J) Military escalation Ukraine
            rand_pm = torch.rand(N_SIMULATIONS, device=DEVICE)
            mil_esc_prob = PUTIN['military_escalation_base_prob'] * (1 + putin_war_intensity * 0.3)
            # Cross-coupling: EU fractured → MORE escalation
            if use_europe:
                mil_esc_prob = mil_esc_prob * (1 + (1 - eu_unity) * 0.3)
            # Cross-coupling: Trump deal active → LESS escalation
            if use_leaders:
                mil_esc_prob = mil_esc_prob * (1 - trump_russia_deal_active.float() * 0.4)

            do_mil_ru = rand_pm < mil_esc_prob
            if do_mil_ru.any():
                d_mru = templates['mil_ru'] * (putin_war_intensity[do_mil_ru].unsqueeze(1) * 0.8 + 0.2)
                leader_delta[do_mil_ru] += d_mru
                attr[do_mil_ru, 9] += torch.norm(d_mru, dim=1)
                putin_war_intensity[do_mil_ru] = (putin_war_intensity[do_mil_ru] + PUTIN['war_intensity_increment']).clamp(max=1.0)
                evt['mil_ru'] += do_mil_ru.sum().item()

            # (K) Nuclear posture
            rand_pn = torch.rand(N_SIMULATIONS, device=DEVICE)
            nuc_prob = PUTIN['nuclear_escalation_prob']
            # Cross-coupling: high war intensity + stress → higher nuclear probability
            nuc_prob = nuc_prob * (1 + putin_war_intensity * 0.5)
            nuc_escalate = rand_pn < nuc_prob
            if nuc_escalate.any():
                putin_nuclear_posture[nuc_escalate] = (putin_nuclear_posture[nuc_escalate] + 0.05).clamp(max=1.0)

            # Nuclear action: only if posture reaches threshold (very rare)
            at_threshold = putin_nuclear_posture > PUTIN['nuclear_action_threshold']
            rand_na = torch.rand(N_SIMULATIONS, device=DEVICE)
            nuc_action = at_threshold & (rand_na < PUTIN['nuclear_action_prob'])
            if nuc_action.any():
                d_nuc = templates['nuclear_posture']
                leader_delta[nuc_action] += d_nuc
                attr[nuc_action, 10] += torch.norm(d_nuc.unsqueeze(0).expand(nuc_action.sum().item(), -1), dim=1)
                evt['nuclear_ru'] += nuc_action.sum().item()

            # Sanctions adaptation (dampening — partial recovery)
            if t % 6 == 0 and t > 0:  # Every 6 months
                leader_delta += templates['sanctions_adapt'].unsqueeze(0) * 0.1

        # =====================================================
        # MIDDLE EAST (new for v5)
        # =====================================================
        if use_mideast:
            # (L) Hormuz — starts closed, can reopen
            rand_h = torch.rand(N_SIMULATIONS, device=DEVICE)

            # Reopening probability increases over time (economic pressure on Iran too)
            reopen_prob = MIDEAST['hormuz_reopening_prob'] + hormuz_months_closed * 0.002
            # Cross-coupling: Trump deal with Russia → less pressure on Iran → might reopen
            if use_leaders:
                reopen_prob = reopen_prob + trump_russia_deal_active.float() * 0.01
            # Cross-coupling: Iran nuclear progress → LESS likely to reopen (leverage)
            reopen_prob = reopen_prob * (1 - iran_nuclear_progress * 0.5)

            reopen = hormuz_closed & (rand_h < reopen_prob)
            if reopen.any():
                hormuz_closed[reopen] = False
                # Recovery when Hormuz reopens
                d_rec = templates['oil_recovery'] * 0.5
                leader_delta[reopen] += d_rec
                attr[reopen, 11] -= torch.norm(d_rec.unsqueeze(0).expand(reopen.sum().item(), -1), dim=1)

            # Ongoing Hormuz impact: continuous economic drag while closed
            if hormuz_closed.any():
                # Diminishing marginal impact (markets adapt somewhat)
                hormuz_impact = 1.0 / (1.0 + hormuz_months_closed[hormuz_closed] * 0.1)
                d_hormuz = templates['hormuz'] * 0.15 * hormuz_impact.unsqueeze(1)
                leader_delta[hormuz_closed] += d_hormuz
                attr[hormuz_closed, 11] += torch.norm(d_hormuz, dim=1)
                hormuz_months_closed[hormuz_closed] += 1
                evt['hormuz'] += hormuz_closed.sum().item()

            # (M) Iran war ongoing
            rand_iw = torch.rand(N_SIMULATIONS, device=DEVICE)
            deesc = iran_war_active & (rand_iw < MIDEAST['iran_war_deescalation_prob'])
            if deesc.any():
                iran_war_active[deesc] = False

            if iran_war_active.any():
                # Diminishing but persistent impact
                d_iw = templates['iran_war'] * 0.10
                leader_delta[iran_war_active] += d_iw
                attr[iran_war_active, 12] += torch.norm(d_iw.unsqueeze(0).expand(iran_war_active.sum().item(), -1), dim=1)
                evt['iran_war'] += iran_war_active.sum().item()

            # Iran nuclear progress
            iran_nuclear_progress = (iran_nuclear_progress + 1.0 / MIDEAST['nuclear_threshold_months']).clamp(max=1.0)

            # (N) Continued Israel strikes
            rand_is = torch.rand(N_SIMULATIONS, device=DEVICE)
            # Cross-coupling: Trump green lights Netanyahu
            strike_prob = MIDEAST['continued_strikes_prob']
            if use_leaders:
                strike_prob = strike_prob * TRUMP['israel_green_light']
            israel_strikes = iran_war_active & (rand_is < strike_prob)
            if israel_strikes.any():
                d_is = templates['iran_war'] * 0.2
                leader_delta[israel_strikes] += d_is
                attr[israel_strikes, 12] += torch.norm(d_is.unsqueeze(0).expand(israel_strikes.sum().item(), -1), dim=1)

            # (O) Hezbollah
            if hezbollah_active.any():
                d_hez = templates['hezbollah'] * 0.10 * MIDEAST['hezbollah_intensity']
                leader_delta[hezbollah_active] += d_hez
                attr[hezbollah_active, 13] += torch.norm(d_hez.unsqueeze(0).expand(hezbollah_active.sum().item(), -1), dim=1)
                evt['hezbollah'] += hezbollah_active.sum().item()

                # Hezbollah can de-escalate
                rand_hez = torch.rand(N_SIMULATIONS, device=DEVICE)
                hez_deesc = hezbollah_active & (~iran_war_active) & (rand_hez < 0.05)
                hezbollah_active[hez_deesc] = False

            # (P) Houthi Red Sea
            if not houthi_active.all():
                rand_hou = torch.rand(N_SIMULATIONS, device=DEVICE)
                houthi_resume = (~houthi_active) & (rand_hou < MIDEAST['houthi_resumption_prob'] * 0.1)
                houthi_active = houthi_active | houthi_resume

            if houthi_active.any():
                d_hou = templates['hezbollah'] * 0.05  # Smaller than Hezbollah but persistent
                leader_delta[houthi_active] += d_hou
                attr[houthi_active, 13] += torch.norm(d_hou.unsqueeze(0).expand(houthi_active.sum().item(), -1), dim=1)

        # =====================================================
        # EUROPE (new for v5)
        # =====================================================
        if use_europe:
            # Unity decay
            eu_unity = eu_unity - EUROPE['unity_decay_rate']

            # Cross-coupling: Russia escalation boosts unity
            if use_putin:
                russia_threat = (putin_war_intensity > 0.7).float() * EUROPE['russia_threat_boost']
                eu_unity = eu_unity + russia_threat
                # Nuclear posture especially unifying
                nuc_scare = (putin_nuclear_posture > 0.5).float() * 0.02
                eu_unity = eu_unity + nuc_scare

            # Cross-coupling: Trump pressure damages unity
            if use_leaders:
                trump_pressure = (trump_tariff_level > 0.6).float() * EUROPE['trump_pressure_damage']
                eu_unity = eu_unity - trump_pressure

            # Orbán sabotage
            rand_orb = torch.rand(N_SIMULATIONS, device=DEVICE)
            orban_sab = rand_orb < EUROPE['orban_sabotage_prob']
            eu_unity[orban_sab] -= EUROPE['orban_sabotage_damage']

            # France crisis
            rand_fr = torch.rand(N_SIMULATIONS, device=DEVICE)
            france_crisis = rand_fr < EUROPE['france_crisis_prob']
            eu_unity[france_crisis] -= EUROPE['france_crisis_damage']

            # Migration background pressure
            eu_unity = eu_unity - EUROPE['migration_pressure'] * 0.001

            eu_unity = eu_unity.clamp(0.05, 0.95)

            # EU fracture effect: when unity is low, amplify economic stress
            low_unity = eu_unity < EUROPE['fiscal_stress_threshold']
            if low_unity.any():
                fracture_intensity = (EUROPE['fiscal_stress_threshold'] - eu_unity[low_unity]).unsqueeze(1) * 2
                d_frac = templates['eu_fracture'] * 0.05 * fracture_intensity
                leader_delta[low_unity] += d_frac
                attr[low_unity, 14] += torch.norm(d_frac, dim=1)
                evt['eu_fracture'] += low_unity.sum().item()

        # Apply leader decisions with noise
        if use_leaders or use_putin or use_mideast or use_europe:
            leader_noise = torch.randn_like(leader_delta) * 0.2
            delta = base_delta + leader_delta + leader_noise * leader_delta.abs()
        else:
            delta = base_delta

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

        if (t+1) % 6 == 0:
            sparse_pct = in_sparse.float().mean().item() * 100
            extras = ""
            if use_leaders:
                taiwan_pct = xi_taiwan_triggered.float().mean().item() * 100
                extras += f" taiwan={taiwan_pct:.0f}%"
            if use_putin:
                extras += f" ru_war={putin_war_intensity.mean().item():.2f} nuc={putin_nuclear_posture.mean().item():.2f}"
            if use_mideast:
                hormuz_pct = hormuz_closed.float().mean().item() * 100
                extras += f" hormuz_closed={hormuz_pct:.0f}%"
            if use_europe:
                extras += f" eu_unity={eu_unity.mean().item():.2f}"
            print(f"  Month {t+1:2d}: sparse={sparse_pct:.0f}%{extras} [{time.time()-t0:.0f}s]")

    elapsed = time.time() - t0

    # Compute crisis probabilities
    dc = density_ot.cpu().numpy()
    ac = accel_ot.cpu().numpy()
    dv = deviation_ot.cpu().numpy()

    d_crisis = dc > density_thr
    a_crisis = ac > accel_thr
    v_crisis = dv > deviation_thr

    two_c = (d_crisis.astype(int) + a_crisis.astype(int) + v_crisis.astype(int)) >= 2
    all_c = d_crisis & a_crisis & v_crisis

    def cum_prob(cm):
        first = np.full(N_SIMULATIONS, MAX_HORIZON_MONTHS+1, dtype=int)
        for i in range(N_SIMULATIONS):
            h = np.where(cm[i])[0]
            if len(h) > 0: first[i] = h[0]+1
        return np.array([np.mean(first <= t+1) for t in range(MAX_HORIZON_MONTHS)])

    cum_two = cum_prob(two_c)
    cum_all = cum_prob(all_c)

    cum_dens = cum_prob(d_crisis)
    cum_acc = cum_prob(a_crisis)
    cum_dev = cum_prob(v_crisis)

    # Attribution: normalize to percentages
    attr_cpu = attr.cpu().numpy()
    attr_total = attr_cpu.sum(axis=1, keepdims=True)
    attr_total[attr_total == 0] = 1
    attr_pct = (attr_cpu / attr_total * 100).mean(axis=0)

    all_results[sc_name] = {
        'cum_two': cum_two, 'cum_all': cum_all,
        'cum_dens': cum_dens, 'cum_acc': cum_acc, 'cum_dev': cum_dev,
        'elapsed': elapsed, 'attr_pct': attr_pct,
    }

    # Store final state variables for reporting
    if sc_name == "Full v5 (all actors)":
        final_state = {}
        if use_leaders:
            final_state['taiwan_triggered_pct'] = xi_taiwan_triggered.float().mean().item() * 100
            final_state['tariff_level'] = trump_tariff_level.mean().item()
            final_state['mil_momentum'] = trump_mil_momentum.mean().item()
            final_state['russia_deal_pct'] = trump_russia_deal_active.float().mean().item() * 100
        if use_putin:
            final_state['war_intensity'] = putin_war_intensity.mean().item()
            final_state['nuclear_posture'] = putin_nuclear_posture.mean().item()
            final_state['energy_cutoff'] = putin_energy_cutoff_level.mean().item()
        if use_mideast:
            final_state['hormuz_still_closed_pct'] = hormuz_closed.float().mean().item() * 100
            final_state['iran_war_still_active_pct'] = iran_war_active.float().mean().item() * 100
            final_state['nuclear_progress'] = iran_nuclear_progress.mean().item()
            final_state['houthi_active_pct'] = houthi_active.float().mean().item() * 100
        if use_europe:
            final_state['eu_unity_final'] = eu_unity.mean().item()

    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  {'Horizon':>10s}  {'2/3':>8s}  {'All 3':>8s}")
    for t_val in [3, 6, 12, 24, 36, 48]:
        if t_val <= MAX_HORIZON_MONTHS:
            print(f"  {t_val:2d} months  {cum_two[t_val-1]*100:7.2f}%  {cum_all[t_val-1]*100:7.2f}%")

# ============================================================
# 9. RESULTS SUMMARY + ATTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("MONTE CARLO v5 — MULTI-LEADER CHAIN REACTIONS — RESULTS")
print("=" * 70)

print(f"\n{N_SIMULATIONS:,} paths × {MAX_HORIZON_MONTHS} months on {DEVICE}")
print(f"Leaders: Trump + Xi + Putin + Middle East (Iran/Israel/Hezbollah) + Europe unity")
print(f"Starting condition: Hormuz CLOSED, Iran war ACTIVE, Hezbollah ACTIVE, oil $100+")

print(f"\n--- Crisis Probability Comparison (2/3 standard) ---")
print(f"{'Scenario':30s}  {'6mo':>7s}  {'12mo':>7s}  {'24mo':>7s}  {'36mo':>7s}  {'48mo':>7s}")
print("-" * 80)
for sname, sdata in all_results.items():
    c = sdata['cum_two']
    vals = [c[5], c[11], c[23], c[35], c[47]]
    print(f"{sname:30s}  " + "  ".join(f"{v*100:6.1f}%" for v in vals))

print(f"\n--- Crisis Probability Comparison (3/3 standard) ---")
print(f"{'Scenario':30s}  {'6mo':>7s}  {'12mo':>7s}  {'24mo':>7s}  {'36mo':>7s}  {'48mo':>7s}")
print("-" * 80)
for sname, sdata in all_results.items():
    c = sdata['cum_all']
    vals = [c[5], c[11], c[23], c[35], c[47]]
    print(f"{sname:30s}  " + "  ".join(f"{v*100:6.1f}%" for v in vals))

print(f"\n--- Attribution (Full v5, % of total displacement) ---")
full_attr = all_results["Full v5 (all actors)"]['attr_pct']
print(f"{'Source':25s}  {'%':>7s}")
print("-" * 35)
sorted_attr = sorted(zip(ATTR_NAMES, full_attr), key=lambda x: -x[1])
for name, pct in sorted_attr:
    if pct > 0.1:
        print(f"  {name:23s}  {pct:6.1f}%")

print(f"\n--- Final State Variables (Full v5, 48 months) ---")
if 'final_state' in dir():
    for k, v in final_state.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.2f}" + ("%" if 'pct' in k else ""))

# Marginal contribution analysis
print(f"\n--- Marginal Contribution (Δ from baseline at 48mo, 2/3) ---")
baseline_48 = all_results["Baseline (econ only)"]['cum_two'][47]
for sname, sdata in all_results.items():
    if sname == "Baseline (econ only)": continue
    delta_pct = (sdata['cum_two'][47] - baseline_48) * 100
    print(f"  {sname:30s}: +{delta_pct:.1f}pp")

# ============================================================
# 10. SAVE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("10. SAVE RESULTS")
print("=" * 70)

results_json = {
    "version": "v5_multi_leader_chain_reactions",
    "timestamp": "2026-03-13",
    "config": {
        "n_simulations": N_SIMULATIONS,
        "max_horizon_months": MAX_HORIZON_MONTHS,
        "n_pca_components": D,
        "device": str(DEVICE),
        "n_clusters": N_CLUSTERS,
        "leaders": ["trump", "xi_jinping", "putin", "mojtaba_khamenei", "netanyahu",
                     "rutte", "merz", "macron", "starmer", "orban", "tusk", "meloni", "zelenskyy"],
        "starting_condition": "Hormuz CLOSED, Iran war ACTIVE, Hezbollah ACTIVE",
    },
    "leader_parameters": {
        "trump": TRUMP, "xi": XI, "putin": PUTIN,
        "mideast": {k: v for k, v in MIDEAST.items() if not isinstance(v, bool)},
        "europe": EUROPE,
    },
    "scenarios": {},
    "attribution": {name: float(pct) for name, pct in zip(ATTR_NAMES, full_attr)},
}

if 'final_state' in dir():
    results_json["final_state_48mo"] = {k: round(v, 4) if isinstance(v, float) else v for k, v in final_state.items()}

for sname, sdata in all_results.items():
    results_json["scenarios"][sname] = {
        "key_probabilities_2of3": {
            f"{t}_months": float(sdata['cum_two'][t-1])
            for t in [6, 12, 24, 36, 48] if t <= MAX_HORIZON_MONTHS
        },
        "key_probabilities_3of3": {
            f"{t}_months": float(sdata['cum_all'][t-1])
            for t in [6, 12, 24, 36, 48] if t <= MAX_HORIZON_MONTHS
        },
        "elapsed_seconds": round(sdata['elapsed'], 1),
    }

with open(OUT_DIR / 'monte_carlo_results_v5.json', 'w') as f:
    json.dump(results_json, f, indent=2)

print(f"Results saved to {OUT_DIR}/monte_carlo_results_v5.json")
print("\nDone.")
