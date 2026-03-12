#!/usr/bin/env python3
"""
TDA Full Analysis — Economic + Market + Leader Behavior Matrices
================================================================
The core innovation: combine economic state vectors with active leader
behavioral parameters into a unified state space, then run TDA.

For each month:
  state = [economic_indicators... | active_leader_params...]

Leader mapping: check who was in power that month via timelines,
then inject their behavioral parameters into the state vector.
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
from sklearn.neighbors import NearestNeighbors
from ripser import ripser
from persim import plot_diagrams
import umap
import kmapper as km
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

DATA_BASE = Path('/home/user/global-financial-sim/data')
ECON_BASE = DATA_BASE / 'economic'
LEADER_BASE = DATA_BASE / 'leaders'
OUT_DIR = Path('/home/user/global-financial-sim/output/tda_full')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. LOAD ECONOMIC DATA
# ============================================================

print("=" * 70)
print("1. LOADING ECONOMIC DATA")
print("=" * 70)

all_series = {}

# FRED economic data
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

indices_path = ECON_BASE / 'indices' / 'global_indices.json'
if indices_path.exists():
    with open(indices_path) as f:
        for name, info in json.load(f).items():
            monthly = daily_to_monthly(info.get('data', []))
            if monthly:
                all_series[f"IDX_{name}"] = monthly

# Gold/Silver (standalone daily files — use 'close' key)
for metal in ['gold', 'silver']:
    fpath = ECON_BASE / 'commodities' / f'{metal}_daily.json'
    if fpath.exists():
        with open(fpath) as f:
            monthly = daily_to_monthly(json.load(f).get('data', []), value_key='close')
        if monthly:
            all_series[f"METAL_{metal}"] = monthly

# FRED Commodities (monthly series with 'value' key)
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

# Yahoo Futures (daily data → monthly average)
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

# World Bank Commodities (if exists)
wb_comm_path = ECON_BASE / 'commodities' / 'world_bank_commodities.json'
if wb_comm_path.exists():
    with open(wb_comm_path) as f:
        wb_comm = json.load(f)
    if isinstance(wb_comm, dict):
        items = wb_comm.get('commodities', wb_comm.get('data', wb_comm))
        if isinstance(items, dict):
            for name, info in items.items():
                if not isinstance(info, dict):
                    continue
                series = info.get('data', info.get('series', []))
                date_val = {}
                for pt in series:
                    val = pt.get('value', pt.get('close'))
                    if val is not None:
                        date_val[pt['date']] = float(val)
                if date_val:
                    all_series[f"COMM_WB_{name}"] = date_val

# Crypto
crypto_path = ECON_BASE / 'crypto' / 'crypto.json'
if crypto_path.exists():
    with open(crypto_path) as f:
        for name, info in json.load(f).items():
            monthly = daily_to_monthly(info.get('data', []))
            if monthly:
                all_series[f"CRYPTO_{name}"] = monthly

# BIS data (credit gap, property prices, debt service, exchange rates, etc.)
bis_dir = ECON_BASE / 'bis'
if bis_dir.exists():
    # Only load smaller, most useful files (skip 49MB/74MB ones)
    for fname in ['credit_gap.json', 'property_prices.json', 'debt_service_ratios.json',
                   'total_credit.json', 'global_liquidity.json']:
        fpath = bis_dir / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            bis_data = json.load(f)
        countries = bis_data.get('countries', {})
        ds_name = fname.replace('.json', '')
        for country_code, cdata in countries.items():
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
                    # Convert quarterly dates (2020-Q1) to monthly
                    d_str = str(d_raw)
                    if 'Q' in d_str:
                        parts = d_str.split('-Q')
                        if len(parts) == 2:
                            yr, q = parts[0], int(parts[1])
                            month = (q - 1) * 3 + 1
                            # Spread quarterly to 3 months
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
                    # Use short label: BIS_credit_gap_CN_0
                    all_series[f"BIS_{ds_name}_{country_code}_{idx}"] = date_val

# Supplementary economic data (UK/RU/CN gaps + US stress indicators)
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

# CFTC COT data (weekly → monthly average)
cftc_path = ECON_BASE / 'cftc' / 'cot_data.json'
if cftc_path.exists():
    with open(cftc_path) as f:
        cftc = json.load(f)
    for contract_name, records in cftc.items():
        if not isinstance(records, list):
            continue
        # Net speculative position (monthly avg)
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

# Bilateral trade data (annual → spread to monthly for alignment)
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
                # Spread annual value to all 12 months
                for m in range(1, 13):
                    key = f"{yr}-{m:02d}-01"
                    all_series.setdefault(f"TRADE_{pair_name}_total", {})[key] = float(total)
            yoy = rec.get('yoy_change_pct')
            if yr and yoy is not None:
                for m in range(1, 13):
                    key = f"{yr}-{m:02d}-01"
                    all_series.setdefault(f"TRADE_{pair_name}_yoy", {})[key] = float(yoy)

print(f"Economic/market indicators loaded: {len(all_series)}")

# ============================================================
# 2. LOAD LEADER PROFILES + TIMELINES
# ============================================================

print("\n" + "=" * 70)
print("2. LOADING LEADER PROFILES + TIMELINES")
print("=" * 70)

# Load all profiles
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
    # Extract parameter values
    params = {}
    for v in vecs:
        label = v.get('label', v.get('name', ''))
        val = v.get('value')
        if label and val is not None:
            params[label] = float(val)
    if params:
        profiles[f.stem] = {
            'params': params,
            'metadata': d.get('metadata', {}),
        }

print(f"Profiles loaded: {len(profiles)}")
for name, p in sorted(profiles.items(), key=lambda x: -len(x[1]['params'])):
    print(f"  {name:35s}: {len(p['params']):4d} dims")

# Load timelines to map leaders to time periods
timeline_dir = LEADER_BASE / 'timelines'
leader_periods = []  # [(profile_key, start_date, end_date, role, country)]

# Mapping from normalized timeline names to profile filenames.
# normalize_name() strips dots, accents, spaces→underscores, then looks up here.
# Also supports last-name-only matching and Japanese family_given ↔ profile key matching.
name_to_profile = {
    # === US ===
    'xi_jinping': 'xi_jinping', 'trump': 'trump', 'donald_trump': 'trump',
    'powell': 'powell', 'jerome_powell': 'powell',
    'putin': 'putin', 'vladimir_putin': 'putin',
    'bernanke': 'bernanke', 'ben_bernanke': 'bernanke',
    'yellen': 'yellen', 'janet_yellen': 'yellen',
    'obama': 'obama', 'barack_obama': 'obama',
    'george_w_bush': 'george_w_bush',
    'greenspan': 'greenspan', 'alan_greenspan': 'greenspan',
    # === EU ===
    'draghi': 'draghi', 'mario_draghi': 'draghi',
    'lagarde': 'lagarde', 'christine_lagarde': 'lagarde',
    'merkel': 'merkel', 'angela_merkel': 'merkel',
    'sarkozy': 'sarkozy', 'nicolas_sarkozy': 'sarkozy',
    # === UK ===
    'thatcher': 'thatcher', 'margaret_thatcher': 'thatcher',
    'blair': 'blair', 'tony_blair': 'blair',
    'gordon_brown': 'gordon_brown',
    # === RU ===
    'nabiullina': 'nabiullina', 'elvira_nabiullina': 'nabiullina',
    'yeltsin': 'yeltsin', 'boris_yeltsin': 'yeltsin',
    'leonid_brezhnev': 'brezhnev', 'brezhnev': 'brezhnev',
    'nikita_khrushchev': 'khrushchev', 'khrushchev': 'khrushchev',
    'joseph_stalin': 'stalin', 'stalin': 'stalin',
    'vladimir_lenin': 'lenin', 'lenin': 'lenin',
    # === CN ===
    'zhou_xiaochuan': 'zhou_xiaochuan',
    'hu_jintao': 'hu_wen',      # Hu-Wen era combined profile
    'wen_jiabao': 'hu_wen',     # Hu-Wen era combined profile
    # === JP (family_given format in timelines) ===
    'abe_shinzo': 'abe_shinzo', 'shinzo_abe': 'abe_shinzo',
    'koizumi_junichiro': 'koizumi_junichiro', 'junichiro_koizumi': 'koizumi_junichiro',
    'kishida_fumio': 'kishida_fumio', 'fumio_kishida': 'kishida_fumio',
    'takaichi_sanae': 'takaichi_sanae', 'sanae_takaichi': 'takaichi_sanae',
    'nakasone_yasuhiro': 'nakasone_yasuhiro', 'yasuhiro_nakasone': 'nakasone_yasuhiro',
    'hashimoto_ryutaro': 'hashimoto_ryutaro', 'ryutaro_hashimoto': 'hashimoto_ryutaro',
    'tanaka_kakuei': 'tanaka_kakuei', 'kakuei_tanaka': 'tanaka_kakuei',
    # === IR ===
    'ali_khamenei': 'ali_khamenei',
    'mojtaba_khamenei': 'mojtaba_khamenei',
    # === Non-state actors ===
    'elon_musk': 'elon_musk',
    # === Special / meta profiles ===
    'netanyahu': 'netanyahu', 'benjamin_netanyahu': 'netanyahu',
}

# Build a reverse index: last-name → profile key (for fallback matching)
_last_name_index = {}
for _pk in profiles.keys():
    parts = _pk.split('_')
    # For multi-part names, index the last part AND the first part (Japanese family name first)
    for p in parts:
        if len(p) > 2:  # skip short fragments like 'w'
            if p not in _last_name_index:
                _last_name_index[p] = _pk

import unicodedata
def _strip_accents(s):
    """Remove diacritics: é→e, ö→o, ü→u, etc."""
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

def normalize_name(name):
    """Match a timeline name to a profile key.

    Handles:
    - Full name → lookup ("George W. Bush" → george_w_bush)
    - Dots in initials ("G. William Miller" → g_william_miller)
    - Accented chars ("Schröder" → schroder)
    - Japanese family-first names ("Abe Shinzo" → abe_shinzo)
    - Last-name fallback ("Leonid Brezhnev" → brezhnev via last name)
    - Substring matching against profile keys
    """
    # Normalize: lowercase, strip accents, remove dots, spaces→underscores
    n = _strip_accents(name.lower()).replace('.', '').replace(' ', '_').replace('-', '_').replace("'", '')
    # Remove double underscores from stripping dots
    while '__' in n:
        n = n.replace('__', '_')
    n = n.strip('_')

    # 1. Direct lookup in explicit mapping
    if n in name_to_profile:
        return name_to_profile[n]

    # 2. Try each word/part against the explicit mapping (catches last names like "brezhnev")
    parts = n.split('_')
    for part in parts:
        if part in name_to_profile:
            return name_to_profile[part]

    # 3. Try reversed order (Japanese: "Abe Shinzo" → timeline has family_given,
    #    but also try given_family in case of inconsistency)
    if len(parts) == 2:
        reversed_n = f"{parts[1]}_{parts[0]}"
        if reversed_n in name_to_profile:
            return name_to_profile[reversed_n]

    # 4. Substring matching against profile keys
    for pk in profiles.keys():
        if pk in n or n in pk:
            return pk

    # 5. Last-name index fallback (matches last name part against known profile names)
    for part in reversed(parts):  # try last name first, then first name (Japanese)
        if part in _last_name_index and len(part) > 3:  # require 4+ chars to avoid false matches
            return _last_name_index[part]

    return None

for tf in sorted(timeline_dir.glob('*.json')):
    country = tf.stem.upper()
    with open(tf) as fp:
        tl_data = json.load(fp)

    # Handle different timeline formats:
    # Format A (US/CN/EU/UK): {"heads_of_state": [...], "central_bank": [...]}
    # Format B (JP/RU/IR): {"roles": {"head_of_state": {"leaders": [...]}}}
    # Format C: flat list
    entries = []
    if isinstance(tl_data, list):
        entries = tl_data
    elif isinstance(tl_data, dict):
        # Check for nested "roles" structure first (JP/RU/IR format)
        roles_dict = tl_data.get('roles', {})
        if roles_dict and isinstance(roles_dict, dict):
            for role_name, role_data in roles_dict.items():
                if isinstance(role_data, dict):
                    # Look for "leaders" list inside role
                    for subkey, subval in role_data.items():
                        if isinstance(subval, list) and subval and isinstance(subval[0], dict):
                            entries.extend(subval)

        # Also check top-level keys (US/CN/EU/UK format)
        for key, val in tl_data.items():
            if key == 'roles':
                continue  # already handled above
            if isinstance(val, list):
                if val and isinstance(val[0], dict):
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

        # Normalize dates: ensure "YYYY-MM" → "YYYY-MM-01" for comparison
        if len(start) == 7:  # "1964-11" → "1964-11-01"
            start = start + '-01'
        if end and len(end) == 7:
            end = end + '-01'

        profile_key = normalize_name(name)
        if profile_key and profile_key in profiles:
            if not end or end.lower() in ('present', 'incumbent', ''):
                end = '2026-03-12'
            leader_periods.append((profile_key, start, end, role, country))

print(f"\nLeader-time mappings found: {len(leader_periods)}")

# Show matched leaders
matched = set(lp[0] for lp in leader_periods)
unmatched = set(profiles.keys()) - matched
print(f"Profiles matched to timelines: {len(matched)}/{len(profiles)}")
for pk in sorted(matched):
    roles_for = [(r, c) for (p, s, e, r, c) in leader_periods if p == pk]
    print(f"  MATCHED: {pk:30s} → {roles_for[0][1]}/{roles_for[0][0]}")
if unmatched:
    print(f"Unmatched profiles ({len(unmatched)}, will use as constants):")
    for pk in sorted(unmatched):
        print(f"  UNMATCHED: {pk}")

# ============================================================
# 3. SELECT KEY BEHAVIORAL DIMENSIONS
# ============================================================

print("\n" + "=" * 70)
print("3. SELECT KEY BEHAVIORAL DIMENSIONS")
print("=" * 70)

# Find dimensions that appear across multiple leaders (universal dimensions)
dim_counts = {}
for pk, pdata in profiles.items():
    for dim_name in pdata['params'].keys():
        dim_counts[dim_name] = dim_counts.get(dim_name, 0) + 1

# Keep dims that appear in at least 3 profiles (reasonably universal)
universal_dims = {d for d, c in dim_counts.items() if c >= 3}
print(f"Universal dimensions (in >=3 profiles): {len(universal_dims)}")

# Also keep key dimensions that are highly important even if not universal
key_dims = {
    'risk_tolerance', 'crisis_response_speed', 'inflation_tolerance',
    'market_intervention_willingness', 'fiscal_stimulus_preference',
    'trade_protectionism', 'military_action_willingness',
    'international_cooperation', 'institutional_independence',
    'transparency_preference', 'political_survival_priority',
    'information_environment_control', 'reform_willingness',
    'monetary_policy_hawkishness', 'financial_regulation_stringency',
}

# For each active leader at each time point, we'll use their top N dims
# To keep dimensionality manageable, select top 30 most common dims
top_dims = sorted(dim_counts.keys(), key=lambda d: -dim_counts[d])[:40]
print(f"Top 40 behavioral dimensions selected:")
for i, d in enumerate(top_dims[:10]):
    print(f"  {i+1}. {d} (in {dim_counts[d]} profiles)")
print(f"  ... and {len(top_dims)-10} more")

# ============================================================
# 4. BUILD UNIFIED STATE MATRIX
# ============================================================

print("\n" + "=" * 70)
print("4. BUILD UNIFIED STATE MATRIX (Economic + Leaders)")
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

# Economic columns
econ_cols = sorted(all_series.keys())

# Leader behavior columns: for key roles, create role-specific dim columns
# We'll track: US_President, US_FedChair, CN_Leader, CN_PBOC, EU_ECB, UK_PM, UK_BOE, JP_PM, RU_Leader
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
    'JP_BOJ': ['governor,_bank_of_japan', 'governor_of_bank_of_japan', 'boj_governor'],
    'RU_LEADER': ['president', 'head_of_state', 'general_secretary', 'general_secretary_of_the_cpsu'],
    'RU_CB': ['chairman,_gosbank', 'governor,_central_bank', 'chairman_of_the_central_bank',
              'governor_of_the_central_bank', 'chairman,_bank_of_russia'],
    'IR_LEADER': ['supreme_leader'],
    'IR_PRES': ['president_of_iran'],
}

def match_role_slot(role_str, country):
    """Match a timeline role to a role slot."""
    role_lower = role_str.lower().replace('-', '_').replace(' ', '_')
    for slot, keywords in ROLE_SLOTS.items():
        slot_country = slot.split('_')[0]
        if country != slot_country:
            continue
        for kw in keywords:
            if kw in role_lower:
                return slot
    return None

# Build leader assignment: for each month, who fills each role slot
def month_in_range(month, start, end):
    return start <= month <= end

# Pre-compute role assignments per month
role_assignments = {}  # {month: {slot: profile_key}}
for month in all_months:
    role_assignments[month] = {}
    for profile_key, start, end, role, country in leader_periods:
        if month_in_range(month, start[:10], end[:10]):
            slot = match_role_slot(role, country)
            if slot:
                role_assignments[month][slot] = profile_key

# Create leader columns: slot_dim for each role slot and behavioral dim
leader_col_names = []
slots_used = set()
for month in all_months:
    for slot in role_assignments[month]:
        slots_used.add(slot)

slots_used = sorted(slots_used)
print(f"Active role slots: {slots_used}")

# Use top behavioral dims per slot
N_LEADER_DIMS = 15  # dims per role slot
for slot in slots_used:
    for dim in top_dims[:N_LEADER_DIMS]:
        leader_col_names.append(f"L_{slot}_{dim}")

print(f"Leader behavior columns: {len(leader_col_names)} ({len(slots_used)} slots × {N_LEADER_DIMS} dims)")

# Build full matrix
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

print(f"\nFull matrix: {matrix.shape}")
print(f"Economic columns: {len(econ_cols)}")
print(f"Leader columns: {len(leader_col_names)}")
print(f"Total: {n_total_cols}")

# Filter: need reasonable coverage
coverage = np.sum(~np.isnan(matrix), axis=0) / n_months * 100
good_cols = coverage >= 8  # Lower threshold since leader data is sparse pre-1990
matrix_f = matrix[:, good_cols]
col_names_f = [all_col_names[j] for j in range(n_total_cols) if good_cols[j]]

n_econ_kept = sum(1 for c in col_names_f if not c.startswith('L_'))
n_leader_kept = sum(1 for c in col_names_f if c.startswith('L_'))
print(f"After coverage filter: {matrix_f.shape[1]} cols ({n_econ_kept} econ + {n_leader_kept} leader)")

# Filter rows
row_coverage = np.sum(~np.isnan(matrix_f), axis=1) / matrix_f.shape[1] * 100
good_rows = row_coverage >= 25
matrix_g = matrix_f[good_rows]
months_g = [all_months[i] for i in range(n_months) if good_rows[i]]
print(f"Months with >=25% coverage: {len(months_g)}")

# Impute: forward-fill then backward-fill
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

# Fill remaining NaN with column median
for j in range(matrix_g.shape[1]):
    col = matrix_g[:, j]
    mask = np.isnan(col)
    if mask.any():
        med = np.nanmedian(col)
        matrix_g[mask, j] = med if not np.isnan(med) else 0.5

remaining_nan = np.sum(np.isnan(matrix_g))
print(f"Remaining NaN: {remaining_nan}")
print(f"Final state matrix: {matrix_g.shape}")

# ============================================================
# 5. NORMALIZE + PCA + TDA
# ============================================================

print("\n" + "=" * 70)
print("5. QUANTILE NORMALIZATION + PCA + TDA")
print("=" * 70)

qt = QuantileTransformer(n_quantiles=min(1000, len(months_g)),
                         output_distribution='uniform', random_state=42)
X_norm = qt.fit_transform(matrix_g)

# Add time
time_vals = np.linspace(0, 1, len(months_g)).reshape(-1, 1)
X_full = np.hstack([X_norm, time_vals])

# PCA
n_comp = min(30, X_full.shape[1] - 1, X_full.shape[0] - 1)
pca = PCA(n_components=n_comp)
X_pca = pca.fit_transform(X_full)
explained = np.cumsum(pca.explained_variance_ratio_)
n_keep = np.searchsorted(explained, 0.95) + 1
X_pca = X_pca[:, :n_keep]
print(f"Input: {X_full.shape[1]}D → PCA: {n_keep}D ({explained[n_keep-1]*100:.1f}% variance)")

# Landmarks
n_lm = min(500, len(months_g))
if len(months_g) > n_lm:
    idx = np.linspace(0, len(months_g)-1, n_lm, dtype=int)
    X_tda = X_pca[idx]
    months_tda = [months_g[i] for i in idx]
else:
    X_tda = X_pca
    months_tda = months_g
    idx = np.arange(len(months_g))

print(f"TDA input: {X_tda.shape}")

# Ripser
result = ripser(X_tda, maxdim=2, do_cocycles=True)
dgms = result['dgms']

for dim in range(len(dgms)):
    dgm = dgms[dim]
    finite = dgm[dgm[:, 1] < np.inf]
    if len(finite) > 0:
        p = finite[:, 1] - finite[:, 0]
        thr = np.percentile(p, 75)
        ll = (p > thr).sum()
        print(f"  H{dim}: {len(finite)} features, {ll} long-lived (>{thr:.4f})")
    else:
        print(f"  H{dim}: {len(dgm)} features")

# ============================================================
# 6. CRISIS BOTTLENECK
# ============================================================

print("\n" + "=" * 70)
print("6. CRISIS BOTTLENECK ANALYSIS")
print("=" * 70)

nn = NearestNeighbors(n_neighbors=min(10, len(X_tda)-1))
nn.fit(X_tda)
dists, _ = nn.kneighbors(X_tda)
avg_dists = dists.mean(axis=1)
edge_thr = np.percentile(avg_dists, 80)

crises = {
    "Oil Crisis 1973-74": ("1973-10-01", "1974-12-01"),
    "Volcker 1980-82": ("1980-01-01", "1982-12-01"),
    "Black Monday 1987": ("1987-08-01", "1988-03-01"),
    "Japan Bubble 1990-92": ("1990-01-01", "1992-12-01"),
    "Asian Crisis 1997-98": ("1997-07-01", "1998-12-01"),
    "Dot-com 2000-02": ("2000-03-01", "2002-10-01"),
    "GFC 2007-09": ("2007-07-01", "2009-06-01"),
    "Euro Crisis 2010-12": ("2010-05-01", "2012-12-01"),
    "Commodity 2014-16": ("2014-07-01", "2016-02-01"),
    "Trade War 2018-19": ("2018-03-01", "2019-12-01"),
    "COVID 2020": ("2020-02-01", "2020-12-01"),
    "Inflation 2022": ("2022-01-01", "2022-12-01"),
    "Trump 2.0 2025": ("2025-01-01", "2026-03-01"),
}

for name, (s, e) in crises.items():
    ci = [i for i, m in enumerate(months_tda) if s <= m <= e]
    if not ci:
        print(f"  {name:35s}: no data")
        continue
    cd = avg_dists[ci]
    ep = np.mean(cd > edge_thr) * 100
    md = np.mean(cd)
    print(f"  {name:35s}: {ep:5.1f}% edge, dist={md:.3f} ({len(ci)} pts)")

# ============================================================
# 7. LEADER TRANSITION DETECTION
# ============================================================

print("\n" + "=" * 70)
print("7. LEADER TRANSITION IMPACT ON TOPOLOGY")
print("=" * 70)

# Find months where leader changes happened
transitions = []
for slot in slots_used:
    prev_leader = None
    for i, month in enumerate(months_g):
        current = role_assignments.get(month, {}).get(slot)
        if current != prev_leader and prev_leader is not None and current is not None:
            transitions.append((month, slot, prev_leader, current))
        prev_leader = current

# For each transition, check if it coincides with topological edge
print(f"Leader transitions detected: {len(transitions)}")
for month, slot, old, new in transitions:
    # Find closest TDA point
    mi = [i for i, m in enumerate(months_tda) if m == month]
    if mi:
        dist = avg_dists[mi[0]]
        is_edge = "EDGE" if dist > edge_thr else "    "
        print(f"  {month} {slot:10s}: {old:25s} → {new:25s}  dist={dist:.3f} {is_edge}")

# ============================================================
# 8. VISUALIZATION
# ============================================================

print("\n" + "=" * 70)
print("8. VISUALIZATION")
print("=" * 70)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_tda)

def month_to_decade(m):
    y = int(m[:4])
    if y < 1980: return 0
    elif y < 1990: return 1
    elif y < 2000: return 2
    elif y < 2010: return 3
    elif y < 2020: return 4
    else: return 5

decade_labels = ['1971-79', '1980s', '1990s', '2000s', '2010s', '2020s']
decades = [month_to_decade(m) for m in months_tda]
colors = cm.rainbow(np.linspace(0, 1, 6))

fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# 1: Decades
ax = axes[0, 0]
for d in range(6):
    mask = [i for i, dd in enumerate(decades) if dd == d]
    if mask:
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1], c=[colors[d]], s=15, alpha=0.7, label=decade_labels[d])
ax.set_title('Economic + Leaders State Space — by Decade', fontsize=13)
ax.legend()

# 2: Edge distance
ax = axes[0, 1]
sc = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=avg_dists, cmap='hot_r', s=15, alpha=0.7)
plt.colorbar(sc, ax=ax, label='Topological Edge Distance')
ax.set_title('Bottlenecks (bright = crisis zone)', fontsize=13)
# Annotate crises
for cname, (s, e) in crises.items():
    ci = [i for i, m in enumerate(months_tda) if s <= m <= e]
    if ci:
        cx, cy = np.mean(X_umap[ci, 0]), np.mean(X_umap[ci, 1])
        short = cname.split('(')[0].strip()[:20]
        ax.annotate(short, (cx, cy), fontsize=7, alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.5))

# 3: Persistence diagram
ax = axes[1, 0]
plot_diagrams(dgms, ax=ax, show=False)
ax.set_title(f'Persistence Diagram ({X_full.shape[1]}D → {n_keep}D PCA)', fontsize=13)

# 4: Leader transitions on UMAP
ax = axes[1, 1]
ax.scatter(X_umap[:, 0], X_umap[:, 1], c='lightgray', s=10, alpha=0.3)
for month, slot, old, new in transitions:
    mi = [i for i, m in enumerate(months_tda) if m == month]
    if mi:
        i = mi[0]
        c = 'red' if avg_dists[i] > edge_thr else 'blue'
        ax.scatter(X_umap[i, 0], X_umap[i, 1], c=c, s=40, marker='^', zorder=5)
        label = f"{new[:10]}"
        ax.annotate(label, (X_umap[i, 0], X_umap[i, 1]), fontsize=5, alpha=0.7)
ax.set_title('Leader Transitions (red=at bottleneck, blue=normal)', fontsize=13)

plt.tight_layout()
plt.savefig(OUT_DIR / 'tda_full.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'tda_full.png'}")

# Mapper
print("Building Mapper network...")
mapper = km.KeplerMapper(verbose=0)
lens = mapper.fit_transform(X_tda, projection=umap.UMAP(n_components=2, random_state=42))
graph = mapper.map(lens, X_tda,
                   cover=km.Cover(n_cubes=30, perc_overlap=0.45),
                   clusterer=DBSCAN(eps=0.5, min_samples=3))
print(f"Mapper: {len(graph['nodes'])} nodes, {len(graph['links'])} edges")

mapper.visualize(graph, path_html=str(OUT_DIR / 'mapper_full.html'),
                 title="Global State Space — Economic + Leader Behaviors",
                 color_values=np.array([decades[i] for i in range(len(months_tda))]),
                 color_function_name="Decade",
                 node_color_function=np.array(['mean']))
print(f"Saved: {OUT_DIR / 'mapper_full.html'}")

# Save metadata
meta = {
    "n_econ_indicators": n_econ_kept,
    "n_leader_dims": n_leader_kept,
    "n_total_dims": int(X_full.shape[1]),
    "n_months": len(months_g),
    "n_pca_dims": int(n_keep),
    "pca_variance": float(explained[n_keep-1]),
    "n_profiles_used": len(matched),
    "n_transitions": len(transitions),
    "role_slots": slots_used,
}
with open(OUT_DIR / 'metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\nDone! All output in {OUT_DIR}/")
