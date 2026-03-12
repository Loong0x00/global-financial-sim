#!/usr/bin/env python3
"""
Monte Carlo Crisis Engine v4 — Leader Decision Functions
=========================================================
v3 + Trump/Xi 决策函数注入蒙特卡洛循环

核心改进：每一步蒙特卡洛不只是随机漂移，
还要评估两个领导人"此时会做什么"→ 决策产生额外位移

Trump 决策函数：
  - 关税升级/暂停（取决于市场状态）
  - 军事行动动能（state variable，不可逆）
  - 对美联储施压

Xi 决策函数：
  - 独裁者陷阱时间线（台湾行动概率递增到 2027）
  - GDP 红线触发刺激
  - 金融系统维稳
  - 资本管制升级

决策 → PCA 空间位移：用历史类似事件的月度增量作为模板
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

N_SIMULATIONS = 1_000_000
MAX_HORIZON_MONTHS = 48
N_PCA_COMPONENTS = 14
K_DENSITY = 15
NOISE_SCALE = 0.15

DENSITY_CRISIS_PCT = 5
ACCEL_CRISIS_PCT = 95
DEVIATION_CRISIS_PCT = 95

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
# 1-3: DATA LOADING + STATE MATRIX + PCA (identical to v3)
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
# 4. BUILD LEADER DECISION DISPLACEMENT TEMPLATES
# ============================================================

print("\n" + "=" * 70)
print("4. BUILD LEADER DECISION DISPLACEMENT TEMPLATES")
print("=" * 70)

deltas = np.diff(X_pca, axis=0)
delta_norms = np.linalg.norm(deltas, axis=1)
median_delta = np.median(delta_norms)
p95_delta = np.percentile(delta_norms, 95)

# Helper: average PCA delta for a set of historical months
def month_range_delta(start_ym, end_ym):
    """Average PCA-space delta during a historical period."""
    idxs = [i for i, m in enumerate(months_g[:-1]) if start_ym <= m[:7] <= end_ym]
    if idxs:
        return deltas[idxs].mean(axis=0), len(idxs)
    return np.zeros(D), 0

# ---------- TRUMP DECISION TEMPLATES ----------
# Template 1: Tariff escalation (2018 trade war months)
tariff_escalate_delta, n_te = month_range_delta("2018-06", "2018-09")
print(f"  Trump tariff escalation template: mag={np.linalg.norm(tariff_escalate_delta):.4f} ({n_te} months)")

# Template 2: Tariff pause/de-escalation (2019-01 Phase 1 truce area)
tariff_deesc_delta, n_td = month_range_delta("2019-06", "2019-12")
print(f"  Trump tariff de-escalation template: mag={np.linalg.norm(tariff_deesc_delta):.4f} ({n_td} months)")

# Template 3: Military action → oil + volatility shock
# Combine Gulf War 1990 + oil crisis component
mil_action_delta, n_mil = month_range_delta("1990-08", "1990-10")
# Also blend with recent Hormuz-type event
oil_crisis_delta, n_oc = month_range_delta("1973-10", "1974-03")
# Military action = Gulf War dynamics + smaller oil component
if n_mil > 0 and n_oc > 0:
    mil_action_delta = mil_action_delta * 0.7 + oil_crisis_delta * 0.3
print(f"  Trump military action template: mag={np.linalg.norm(mil_action_delta):.4f}")

# Template 4: Fed pressure → uncertainty (no direct historical analogue)
# Use 2018 Q4 when Trump publicly attacked Fed
fed_pressure_delta, n_fp = month_range_delta("2018-10", "2018-12")
fed_pressure_delta = fed_pressure_delta * 0.5  # Pressure effect is subtler
print(f"  Trump Fed pressure template: mag={np.linalg.norm(fed_pressure_delta):.4f} ({n_fp} months)")

# ---------- XI DECISION TEMPLATES ----------
# Template 5: Taiwan action → maximum global shock
# Combine Lehman + Gulf War + trade war × amplification
# This would be unprecedented — estimate as 2x GFC magnitude in trade/military dimensions
gfc_delta, n_gfc = month_range_delta("2008-09", "2008-12")
trade_war_delta, n_tw = month_range_delta("2018-06", "2019-06")
taiwan_action_delta = gfc_delta * 1.5 + trade_war_delta * 1.0 + mil_action_delta * 1.0
print(f"  Xi Taiwan action template: mag={np.linalg.norm(taiwan_action_delta):.4f}")

# Template 6: Economic stimulus (2009 China 4万亿, 2020 recovery)
stim_delta_09, n_s09 = month_range_delta("2009-03", "2009-12")
stim_delta_20, n_s20 = month_range_delta("2020-05", "2020-12")
if n_s09 > 0 and n_s20 > 0:
    stimulus_delta = (stim_delta_09 + stim_delta_20) / 2
else:
    stimulus_delta = stim_delta_09 if n_s09 > 0 else stim_delta_20
# Stimulus partially counteracts crisis drift (reduces magnitude)
print(f"  Xi stimulus template: mag={np.linalg.norm(stimulus_delta):.4f}")

# Template 7: Capital controls tightening (2015-16 China capital flight response)
capctrl_delta, n_cc = month_range_delta("2015-08", "2016-02")
capctrl_delta = capctrl_delta * 0.6  # Capital controls dampen but don't reverse
print(f"  Xi capital control template: mag={np.linalg.norm(capctrl_delta):.4f} ({n_cc} months)")

# Template 8: Tech war retaliation (2019 Huawei ban response period)
tech_war_delta, n_twr = month_range_delta("2019-05", "2019-08")
print(f"  Xi tech retaliation template: mag={np.linalg.norm(tech_war_delta):.4f} ({n_twr} months)")

# ============================================================
# 5. LEADER DECISION FUNCTION PARAMETERS
# ============================================================

print("\n" + "=" * 70)
print("5. LEADER DECISION FUNCTION PARAMETERS")
print("=" * 70)

# --- TRUMP PARAMETERS (from profile) ---
TRUMP = {
    # Market sensitivity
    'market_crash_threshold': 0.08,      # 8% decline triggers policy reversal
    'stock_market_report_card': 0.93,    # How much he cares about markets
    'tariff_volatility': 0.90,           # Probability of tariff action each month
    'tariff_escalation_rate': 0.15,      # Monthly probability of escalation (when not pausing)
    'tariff_pause_on_crash': 0.85,       # Probability of pausing if market crashes

    # Military
    'military_action_willingness': 0.92,
    'military_momentum_init': 0.4,       # Starting momentum (Yemen/Nigeria/Venezuela/Iran already done)
    'military_momentum_increment': 0.08, # Each action increases by this
    'military_action_base_prob': 0.05,   # Base monthly probability
    'military_cooldown_months': 4,       # Minimum months between major actions

    # Fed pressure
    'fed_pressure_probability': 0.30,    # Monthly probability of public Fed pressure
    'fed_pressure_escalation': 0.05,     # Increases if rates don't drop

    # Deal-seeking
    'deal_aspiration': 0.85,             # Eventually wants a deal
    'deal_month_threshold': 18,          # After this many months, seeks de-escalation
}

# --- XI PARAMETERS (from profile) ---
XI = {
    # Taiwan
    'taiwan_base_prob': 0.005,           # Monthly base probability of Taiwan action
    'taiwan_prob_2027_peak': 0.03,       # Monthly probability peaks before 2027 Congress
    'taiwan_ramp_start_month': 6,        # Start ramping at month 6 (Sep 2026)
    'taiwan_ramp_peak_month': 18,        # Peak at month 18 (Sep 2027, before 2027 Congress)
    'taiwan_action_prob_overall': 0.70,  # 70% within timeframe (from profile)

    # GDP red line
    'gdp_red_line_trigger': 0.85,        # Density percentile that triggers "economy bad"
    'stimulus_probability': 0.60,        # When GDP bad, probability of stimulus
    'flood_stimulus_aversion': 0.82,     # Prefers targeted → scale stimulus down
    'stimulus_scale': 0.4,              # Stimulus effectiveness vs historical (Xi is more restrained)

    # Financial system
    'financial_crisis_response': 0.94,   # RL004: unlimited state support
    'capital_control_threshold': 0.80,   # Density percentile triggering cap controls
    'capital_control_probability': 0.40, # Monthly probability when triggered

    # Information fidelity
    'information_fidelity': 0.41,        # Only sees 41% of reality accurately
    'delayed_response_months': 3,        # Average delay before responding to crisis

    # Political security override
    'political_security_priority': 0.97,
    'tech_retaliation_probability': 0.25, # Monthly probability of tech war counter
}

print("  Trump: tariff_vol=0.90, mil_momentum=0.4, market_sensitivity=0.93")
print(f"  Xi: taiwan_base=0.005/mo, info_fidelity=0.41, stimulus_scale=0.4")

# ============================================================
# 6. CALIBRATE CRISIS THRESHOLDS (same as v3)
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

# Economy stress proxy: distance from historical median state
hist_median = np.median(X_pca, axis=0)
market_stress_ref = np.percentile(np.linalg.norm(X_pca - hist_median, axis=1), 90)
print(f"Market stress reference (90th pct distance from median): {market_stress_ref:.4f}")

# ============================================================
# 7. PREPARE GPU + CLUSTER-BASED TRANSITIONS (same as v3)
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

max_pool = max(cluster_sizes)
max_crisis_pool = max(max(cluster_crisis_sizes), 1)
cluster_delta_pool = np.zeros((N_CLUSTERS, max_pool, D), dtype=np.float32)
cluster_crisis_pool = np.zeros((N_CLUSTERS, max_crisis_pool, D), dtype=np.float32)

for c in range(N_CLUSTERS):
    n = cluster_sizes[c]
    if n > 0: cluster_delta_pool[c, :n] = cluster_deltas_list[c]
    nc = cluster_crisis_sizes[c]
    if nc > 0: cluster_crisis_pool[c, :nc] = cluster_crisis_deltas_list[c]

cluster_delta_pool_gpu = torch.tensor(cluster_delta_pool, device=DEVICE)
cluster_sizes_gpu = torch.tensor(cluster_sizes, dtype=torch.long, device=DEVICE)

all_crisis_deltas = deltas[list(crisis_month_set)] if crisis_month_set else deltas[:10]
all_crisis_pool_gpu = torch.tensor(all_crisis_deltas, dtype=torch.float32, device=DEVICE)

# Pre-compute leader displacement templates on GPU
templates = {
    'tariff_escalate': torch.tensor(tariff_escalate_delta, dtype=torch.float32, device=DEVICE),
    'tariff_deesc': torch.tensor(tariff_deesc_delta, dtype=torch.float32, device=DEVICE),
    'military_action': torch.tensor(mil_action_delta, dtype=torch.float32, device=DEVICE),
    'fed_pressure': torch.tensor(fed_pressure_delta, dtype=torch.float32, device=DEVICE),
    'taiwan_action': torch.tensor(taiwan_action_delta, dtype=torch.float32, device=DEVICE),
    'stimulus': torch.tensor(stimulus_delta, dtype=torch.float32, device=DEVICE),
    'capital_control': torch.tensor(capctrl_delta, dtype=torch.float32, device=DEVICE),
    'tech_retaliation': torch.tensor(tech_war_delta, dtype=torch.float32, device=DEVICE),
}

hist_median_gpu = torch.tensor(hist_median, dtype=torch.float32, device=DEVICE)

print(f"Clusters: {N_CLUSTERS} ({(cluster_crisis_sizes > 0).sum()} with crisis data)")
print(f"Non-parametric pools: max {max_pool} deltas/cluster, {len(all_crisis_deltas)} global crisis deltas")
print(f"Leader templates: {len(templates)} displacement vectors on GPU")

# ============================================================
# 8. MONTE CARLO WITH LEADER DECISIONS
# ============================================================

print("\n" + "=" * 70)
print("8. MONTE CARLO WITH LEADER DECISIONS")
print(f"   {N_SIMULATIONS:,} paths × {MAX_HORIZON_MONTHS} months")
print("=" * 70)

# Scenarios: only Full Shock (current state) with and without leader decisions
curr_idx = N - 1
base_state = X_pca[curr_idx].copy()

# Combined shock (same as v3 Full Shock)
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
shock_state = base_state + combined_shock

scenarios = {
    "No Leaders (v3 Full Shock)": (shock_state.copy(), False),
    "With Trump + Xi Decisions": (shock_state.copy(), True),
}

all_results = {}
leader_event_counts = {}

for sc_name, (sc_start, use_leaders) in scenarios.items():
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

    # Leader state variables (per path)
    if use_leaders:
        # Trump state
        trump_tariff_level = torch.ones(N_SIMULATIONS, device=DEVICE) * 0.5  # Current tariff level [0,1]
        trump_mil_momentum = torch.ones(N_SIMULATIONS, device=DEVICE) * TRUMP['military_momentum_init']
        trump_mil_cooldown = torch.zeros(N_SIMULATIONS, device=DEVICE)  # Months since last action
        trump_fed_pressure_level = torch.zeros(N_SIMULATIONS, device=DEVICE)
        trump_seeking_deal = torch.zeros(N_SIMULATIONS, dtype=torch.bool, device=DEVICE)

        # Xi state
        xi_taiwan_triggered = torch.zeros(N_SIMULATIONS, dtype=torch.bool, device=DEVICE)
        xi_stimulus_active = torch.zeros(N_SIMULATIONS, device=DEVICE)  # Remaining months of stimulus
        xi_response_delay = torch.zeros(N_SIMULATIONS, device=DEVICE)  # Delayed perception counter

        # Event counters for analysis
        evt_tariff_esc = 0
        evt_tariff_pause = 0
        evt_military = 0
        evt_fed_pressure = 0
        evt_taiwan = 0
        evt_stimulus = 0
        evt_capctrl = 0
        evt_tech_ret = 0

    for t in range(MAX_HORIZON_MONTHS):
        # --- Standard transition (same as v3) ---
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
        delta = delta * noise_amp.unsqueeze(1)

        # --- LEADER DECISION LAYER ---
        if use_leaders:
            leader_delta = torch.zeros_like(delta)
            rand = torch.rand(N_SIMULATIONS, device=DEVICE)

            # Economic stress proxy: distance from historical median
            stress = torch.norm(paths - hist_median_gpu, dim=1) / market_stress_ref
            # stress > 1.0 means above 90th percentile historical stress

            # =====================
            # TRUMP DECISIONS
            # =====================

            # (A) Tariff dynamics: escalate unless market crashes
            market_crashing = stress > 1.3  # Proxy for >10% market decline equivalent
            not_crashing = ~market_crashing

            # Escalation: monthly probability depends on tariff volatility
            esc_prob = TRUMP['tariff_escalation_rate'] * (1 + trump_tariff_level * 0.5)
            escalate = not_crashing & (rand < esc_prob) & ~trump_seeking_deal
            if escalate.any():
                # Tariff escalation adds trade shock displacement, scaled by current level
                scale = trump_tariff_level[escalate].unsqueeze(1) * 0.5 + 0.5  # 0.5-1.0x
                leader_delta[escalate] += templates['tariff_escalate'] * scale
                trump_tariff_level[escalate] = (trump_tariff_level[escalate] + 0.05).clamp(max=1.0)
                evt_tariff_esc += escalate.sum().item()

            # Pause: if market crashing and tariff level high
            rand2 = torch.rand(N_SIMULATIONS, device=DEVICE)
            pause = market_crashing & (trump_tariff_level > 0.3) & (rand2 < TRUMP['tariff_pause_on_crash'])
            if pause.any():
                leader_delta[pause] += templates['tariff_deesc'] * 0.5  # Partial de-escalation
                trump_tariff_level[pause] = (trump_tariff_level[pause] - 0.1).clamp(min=0.1)
                evt_tariff_pause += pause.sum().item()

            # Deal-seeking after threshold months
            if t >= TRUMP['deal_month_threshold']:
                rand3 = torch.rand(N_SIMULATIONS, device=DEVICE)
                new_deals = ~trump_seeking_deal & (rand3 < 0.1)  # 10% monthly chance of switching to deal mode
                trump_seeking_deal = trump_seeking_deal | new_deals

            # (B) Military action: momentum-driven, each success raises probability
            trump_mil_cooldown = (trump_mil_cooldown - 1).clamp(min=0)
            can_act = trump_mil_cooldown == 0
            # Military probability = base + momentum bonus
            mil_prob = TRUMP['military_action_base_prob'] + trump_mil_momentum * 0.05
            rand4 = torch.rand(N_SIMULATIONS, device=DEVICE)
            mil_act = can_act & (rand4 < mil_prob)
            if mil_act.any():
                # Military action: oil + volatility shock
                # Scale by momentum (higher momentum = bigger actions)
                mil_scale = trump_mil_momentum[mil_act].unsqueeze(1) * 1.5 + 0.5
                leader_delta[mil_act] += templates['military_action'] * mil_scale
                trump_mil_momentum[mil_act] = (trump_mil_momentum[mil_act] + TRUMP['military_momentum_increment']).clamp(max=1.0)
                trump_mil_cooldown[mil_act] = TRUMP['military_cooldown_months']
                evt_military += mil_act.sum().item()

            # (C) Fed pressure: escalates over time if rates perceived too high
            rand5 = torch.rand(N_SIMULATIONS, device=DEVICE)
            fed_pres = rand5 < (TRUMP['fed_pressure_probability'] + trump_fed_pressure_level * 0.1)
            if fed_pres.any():
                leader_delta[fed_pres] += templates['fed_pressure']
                trump_fed_pressure_level[fed_pres] += TRUMP['fed_pressure_escalation']
                evt_fed_pressure += fed_pres.sum().item()

            # =====================
            # XI DECISIONS
            # =====================

            # (D) Taiwan action: probability ramps toward 2027
            if t < XI['taiwan_ramp_start_month']:
                taiwan_prob = XI['taiwan_base_prob']
            elif t < XI['taiwan_ramp_peak_month']:
                # Linear ramp from base to peak
                frac = (t - XI['taiwan_ramp_start_month']) / (XI['taiwan_ramp_peak_month'] - XI['taiwan_ramp_start_month'])
                taiwan_prob = XI['taiwan_base_prob'] + frac * (XI['taiwan_prob_2027_peak'] - XI['taiwan_base_prob'])
            else:
                # After peak, slowly decrease (window passing)
                decay = 0.98 ** (t - XI['taiwan_ramp_peak_month'])
                taiwan_prob = XI['taiwan_prob_2027_peak'] * decay

            # Stress increases Taiwan probability (dictator's trap: worse economy → more likely)
            stress_boost = (stress - 1.0).clamp(min=0) * 0.02  # Extra 2% per unit of stress above threshold
            effective_taiwan_prob = taiwan_prob + stress_boost

            rand6 = torch.rand(N_SIMULATIONS, device=DEVICE)
            taiwan_act = ~xi_taiwan_triggered & (rand6 < effective_taiwan_prob)
            if taiwan_act.any():
                # Taiwan action: MASSIVE shock to all paths that trigger it
                # This is a one-time event per path — irreversible
                # Add noise to the template (not all Taiwan scenarios are identical)
                noise = torch.randn(taiwan_act.sum().item(), D, device=DEVICE) * 0.3
                leader_delta[taiwan_act] += templates['taiwan_action'] + noise * templates['taiwan_action'].abs()
                xi_taiwan_triggered = xi_taiwan_triggered | taiwan_act
                evt_taiwan += taiwan_act.sum().item()

            # Post-Taiwan: ongoing disruption for paths where Taiwan happened
            if xi_taiwan_triggered.any():
                # Continuous disruption: 40% of Taiwan shock per month for 12 months
                ongoing_taiwan = xi_taiwan_triggered & (torch.rand(N_SIMULATIONS, device=DEVICE) < 0.6)
                if ongoing_taiwan.any():
                    leader_delta[ongoing_taiwan] += templates['taiwan_action'] * 0.15

            # (E) GDP red line → stimulus (with information delay)
            economy_bad = stress > XI['gdp_red_line_trigger']
            # Xi doesn't see reality instantly — delayed response
            xi_response_delay[economy_bad] += 1
            xi_response_delay[~economy_bad] = (xi_response_delay[~economy_bad] - 0.5).clamp(min=0)

            perceived_crisis = xi_response_delay >= XI['delayed_response_months']
            rand7 = torch.rand(N_SIMULATIONS, device=DEVICE)
            do_stimulus = perceived_crisis & (xi_stimulus_active <= 0) & (rand7 < XI['stimulus_probability'])
            if do_stimulus.any():
                # Stimulus: partially counteracts drift (negative direction = mean reversion)
                stim_scale = XI['stimulus_scale'] * (1 - XI['flood_stimulus_aversion'] * 0.3)
                leader_delta[do_stimulus] += templates['stimulus'] * stim_scale
                xi_stimulus_active[do_stimulus] = 12  # 12 months of active stimulus
                evt_stimulus += do_stimulus.sum().item()

            # Ongoing stimulus effect
            stim_ongoing = xi_stimulus_active > 0
            if stim_ongoing.any():
                # Diminishing stimulus effect
                stim_strength = (xi_stimulus_active[stim_ongoing] / 12).unsqueeze(1)
                leader_delta[stim_ongoing] += templates['stimulus'] * XI['stimulus_scale'] * 0.3 * stim_strength
                xi_stimulus_active[stim_ongoing] -= 1

            # (F) Capital controls when financial stress high
            fin_stress = dens_vals > density_thr * XI['capital_control_threshold']
            rand8 = torch.rand(N_SIMULATIONS, device=DEVICE)
            do_capctrl = fin_stress & (rand8 < XI['capital_control_probability'])
            if do_capctrl.any():
                leader_delta[do_capctrl] += templates['capital_control']
                evt_capctrl += do_capctrl.sum().item()

            # (G) Tech war retaliation (periodic, driven by Trump tariff escalation)
            rand9 = torch.rand(N_SIMULATIONS, device=DEVICE)
            tech_ret = (trump_tariff_level > 0.6) & (rand9 < XI['tech_retaliation_probability'])
            if tech_ret.any():
                leader_delta[tech_ret] += templates['tech_retaliation']
                evt_tech_ret += tech_ret.sum().item()

            # Apply leader decisions with information fidelity noise
            # Leaders don't perfectly control outcomes — add 20% noise to decision effects
            leader_noise = torch.randn_like(leader_delta) * 0.2
            delta = delta + leader_delta + leader_noise * leader_delta.abs()

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
            extras = ""
            if use_leaders:
                taiwan_pct = xi_taiwan_triggered.float().mean().item() * 100
                extras = f", taiwan_triggered={taiwan_pct:.1f}%, tariff_lvl={trump_tariff_level.mean().item():.2f}"
            print(f"  Month {t+1:2d}: density={dens_vals.mean().item():.3f}, "
                  f"sparse={sparse_pct:.1f}%{extras}, elapsed={time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    throughput = N_SIMULATIONS * MAX_HORIZON_MONTHS / elapsed

    if use_leaders:
        total_events = N_SIMULATIONS * MAX_HORIZON_MONTHS
        leader_event_counts = {
            'tariff_escalation': evt_tariff_esc,
            'tariff_pause': evt_tariff_pause,
            'military_action': evt_military,
            'fed_pressure': evt_fed_pressure,
            'taiwan_action': evt_taiwan,
            'stimulus': evt_stimulus,
            'capital_controls': evt_capctrl,
            'tech_retaliation': evt_tech_ret,
        }
        print(f"\n  Leader events (total across all paths × months):")
        for ename, ecount in leader_event_counts.items():
            print(f"    {ename:25s}: {ecount:>10,} ({ecount/total_events*100:.2f}% of path-months)")
        print(f"  Taiwan triggered: {xi_taiwan_triggered.sum().item():,} / {N_SIMULATIONS:,} paths "
              f"({xi_taiwan_triggered.float().mean().item()*100:.1f}%)")
        print(f"  Mean tariff level at end: {trump_tariff_level.mean().item():.3f}")
        print(f"  Mean military momentum at end: {trump_mil_momentum.mean().item():.3f}")

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

    print(f"\n  Time: {elapsed:.1f}s ({throughput:,.0f} transitions/sec)")
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
    "version": "v4_leader_decisions",
    "timestamp": "2026-03-13",
    "config": {
        "n_simulations": N_SIMULATIONS,
        "max_horizon_months": MAX_HORIZON_MONTHS,
        "n_pca_components": D,
        "device": str(DEVICE),
        "n_clusters": N_CLUSTERS,
        "leaders": ["trump", "xi_jinping"],
    },
    "leader_parameters": {
        "trump": TRUMP,
        "xi": XI,
    },
    "leader_events": leader_event_counts,
    "scenarios": {},
}

for sname, sdata in all_results.items():
    results_json["scenarios"][sname] = {
        "key_probabilities": {
            "6_months": {"any": float(sdata['cum_any'][5]), "two": float(sdata['cum_two'][5]), "all": float(sdata['cum_all'][5])},
            "12_months": {"any": float(sdata['cum_any'][11]), "two": float(sdata['cum_two'][11]), "all": float(sdata['cum_all'][11])},
            "24_months": {"any": float(sdata['cum_any'][23]), "two": float(sdata['cum_two'][23]), "all": float(sdata['cum_all'][23])},
            "36_months": {"any": float(sdata['cum_any'][35]), "two": float(sdata['cum_two'][35]), "all": float(sdata['cum_all'][35])},
            "48_months": {"any": float(sdata['cum_any'][47]), "two": float(sdata['cum_two'][47]), "all": float(sdata['cum_all'][47])},
        },
        "elapsed_seconds": round(sdata['elapsed'], 1),
    }

with open(OUT_DIR / 'monte_carlo_results_v4.json', 'w') as f:
    json.dump(results_json, f, indent=2)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MONTE CARLO v4 — LEADER DECISION FUNCTIONS")
print("=" * 70)

print(f"\n{N_SIMULATIONS:,} paths × {MAX_HORIZON_MONTHS} months on {DEVICE}")
print(f"Leaders: Trump (tariff/military/Fed) + Xi (Taiwan/stimulus/capctrl/tech)")

print(f"\n--- Comparison: Without vs With Leader Decisions ---")
print(f"{'Criterion':15s}  {'':35s}  {'6mo':>7s}  {'12mo':>7s}  {'24mo':>7s}  {'36mo':>7s}  {'48mo':>7s}")
print("-" * 100)

for criterion, key in [("Any 1/3", 'cum_any'), ("2/3 (balanced)", 'cum_two'), ("All 3", 'cum_all')]:
    for sname, sdata in all_results.items():
        c = sdata[key]
        vals = [c[5], c[11], c[23], c[35], c[47]]
        label = f"{criterion}" if sname == list(all_results.keys())[0] else ""
        print(f"{label:15s}  {sname:35s}  " + "  ".join(f"{v*100:6.1f}%" for v in vals))
    print()

# Per-indicator for leader scenario
print(f"\n--- Per-indicator detail (With Leaders) ---")
fs = all_results["With Trump + Xi Decisions"]
print(f"{'Indicator':30s}  {'6mo':>7s}  {'12mo':>7s}  {'24mo':>7s}  {'36mo':>7s}  {'48mo':>7s}")
print("-" * 75)
for iname, ikey in [("Density (unknown territory)", 'cum_dens'), ("Acceleration (instability)", 'cum_acc'), ("Deviation (trajectory break)", 'cum_dev')]:
    c = fs[ikey]
    vals = [c[5], c[11], c[23], c[35], c[47]]
    print(f"{iname:30s}  " + "  ".join(f"{v*100:6.1f}%" for v in vals))

print(f"\nOutput: {OUT_DIR}/")
