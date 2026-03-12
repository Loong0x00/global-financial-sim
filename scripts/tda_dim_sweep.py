#!/usr/bin/env python3
"""
TDA Dimensionality Sweep: find optimal PCA dimensions that preserve H2 voids
while using ALL data sources (including BIS).

Strategy: sweep PCA variance thresholds from 70% to 99%, report H0/H1/H2 counts.
"""
import json, numpy as np, time
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from ripser import ripser
import warnings
warnings.filterwarnings('ignore')

DATA_BASE = Path('data')
ECON_BASE = DATA_BASE / 'economic'

# === LOAD ALL DATA (same as tda_feature_check.py) ===
all_series = {}
COUNTRIES = ['us', 'cn', 'eu', 'uk', 'jp', 'ru']
for country in COUNTRIES:
    for ftype in ['macro', 'financial']:
        fpath = ECON_BASE / country / f'{ftype}.json'
        if not fpath.exists(): continue
        with open(fpath) as f: data = json.load(f)
        indicators = data.get('indicators', data)
        for name, ind in indicators.items():
            dv = {}
            for pt in ind.get('series', []):
                if pt['value'] is not None: dv[pt['date']] = float(pt['value'])
            if dv: all_series[f"ECON_{country.upper()}_{name}"] = dv

def daily_to_monthly(data_points, value_key='close'):
    monthly = {}
    for pt in data_points:
        ym = pt['date'][:7] + "-01"
        val = pt.get(value_key, pt.get('value'))
        if val is None: continue
        monthly.setdefault(ym, []).append(float(val))
    return {k: np.mean(v) for k, v in monthly.items()}

for f in ['indices/global_indices.json']:
    fp = ECON_BASE / f
    if fp.exists():
        with open(fp) as fh:
            for name, info in json.load(fh).items():
                m = daily_to_monthly(info.get('data', []))
                if m: all_series[f"IDX_{name}"] = m

for metal in ['gold', 'silver']:
    fp = ECON_BASE / 'commodities' / f'{metal}_daily.json'
    if fp.exists():
        with open(fp) as fh:
            m = daily_to_monthly(json.load(fh).get('data', []), 'close')
        if m: all_series[f"METAL_{metal}"] = m

for fn in ['fred_commodities.json', 'yahoo_futures.json']:
    fp = ECON_BASE / 'commodities' / fn
    if fp.exists():
        with open(fp) as fh: cd = json.load(fh)
        for name, info in cd.items():
            if not isinstance(info, dict): continue
            if 'data' in info:
                dv = {}
                for pt in info['data']:
                    v = pt.get('value', pt.get('close'))
                    if v is not None: dv[pt['date'][:7]+'-01'] = float(v)
                if dv: all_series[f"COMM_{name}"] = dv

fp = ECON_BASE / 'crypto' / 'crypto.json'
if fp.exists():
    with open(fp) as fh:
        for name, info in json.load(fh).items():
            m = daily_to_monthly(info.get('data', []))
            if m: all_series[f"CRYPTO_{name}"] = m

# BIS
bis_dir = ECON_BASE / 'bis'
if bis_dir.exists():
    for fname in ['credit_gap.json', 'property_prices.json', 'debt_service_ratios.json',
                   'total_credit.json', 'global_liquidity.json']:
        fp = bis_dir / fname
        if not fp.exists(): continue
        with open(fp) as fh: bis_data = json.load(fh)
        ds_name = fname.replace('.json', '')
        for cc, cdata in bis_data.get('countries', {}).items():
            for idx, (sn, si) in enumerate(cdata.get('series', {}).items()):
                if not isinstance(si, dict) or 'data' not in si: continue
                dv = {}
                raw_data = si['data']
                items = raw_data.items() if isinstance(raw_data, dict) else raw_data
                for item in items:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        d_raw, val = item
                    elif isinstance(item, dict):
                        d_raw, val = item.get('date',''), item.get('value')
                    else: continue
                    if val is None: continue
                    d_str = str(d_raw)
                    if 'Q' in d_str:
                        parts = d_str.split('-Q')
                        if len(parts) == 2:
                            yr, q = parts[0], int(parts[1])
                            mo = (q-1)*3+1
                            for o in range(3): dv[f"{yr}-{mo+o:02d}-01"] = float(val)
                    elif len(d_str) >= 7:
                        try: dv[d_str[:7]+'-01'] = float(val)
                        except: pass
                if dv: all_series[f"BIS_{ds_name}_{cc}_{idx}"] = dv

# Supplementary
supp_dir = ECON_BASE / 'supplementary'
if supp_dir.exists():
    for fname in ['uk_bank_rate.json', 'russia_supplementary.json', 'china_supplementary.json', 'us_additional.json']:
        fp = supp_dir / fname
        if not fp.exists(): continue
        with open(fp) as fh: supp = json.load(fh)
        for name, info in supp.items():
            if not isinstance(info, dict) or 'data' not in info: continue
            dv = {}
            for pt in info['data']:
                d, v = pt.get('date',''), pt.get('value')
                if d and v is not None:
                    try: dv[d[:7]+'-01'] = float(v)
                    except: pass
            if dv: all_series[f"SUPP_{name}"] = dv

# CFTC
fp = ECON_BASE / 'cftc' / 'cot_data.json'
if fp.exists():
    with open(fp) as fh: cftc = json.load(fh)
    for cn, recs in cftc.items():
        if not isinstance(recs, list): continue
        mn = {}
        for r in recs:
            d = r.get('date','')[:7]
            net = r.get('net_speculative')
            if d and net is not None: mn.setdefault(d+'-01', []).append(float(net))
        if mn: all_series[f"CFTC_{cn}"] = {k: np.mean(v) for k,v in mn.items()}

# Trade
fp = ECON_BASE / 'trade' / 'bilateral_trade.json'
if fp.exists():
    with open(fp) as fh: trade = json.load(fh)
    for pn, pd in trade.items():
        if not isinstance(pd, dict) or 'data' not in pd: continue
        for r in pd['data']:
            yr, tot = r.get('year'), r.get('total')
            if yr and tot:
                for m in range(1,13):
                    all_series.setdefault(f"TRADE_{pn}", {})[f"{yr}-{m:02d}-01"] = float(tot)

print(f"Total series loaded: {len(all_series)}")

# === BUILD MATRIX ===
all_months_set = set()
for s in all_series.values(): all_months_set.update(s.keys())
all_months = sorted([m for m in all_months_set if '1964-01-01' <= m <= '2026-12-01'])

econ_cols = sorted(all_series.keys())
matrix = np.full((len(all_months), len(econ_cols)), np.nan)
for j, col in enumerate(econ_cols):
    for i, month in enumerate(all_months):
        if month in all_series[col]: matrix[i, j] = all_series[col][month]

# Filter columns: >=8% coverage
coverage = np.sum(~np.isnan(matrix), axis=0) / len(all_months) * 100
good_cols = coverage >= 8
matrix_f = matrix[:, good_cols]
col_names_f = [econ_cols[j] for j in range(len(econ_cols)) if good_cols[j]]

# Filter rows: >=25% coverage
row_cov = np.sum(~np.isnan(matrix_f), axis=1) / matrix_f.shape[1] * 100
good_rows = row_cov >= 25
matrix_g = matrix_f[good_rows]
months_g = [all_months[i] for i in range(len(all_months)) if good_rows[i]]

# Impute (forward fill, back fill, median)
for j in range(matrix_g.shape[1]):
    last = np.nan
    for i in range(matrix_g.shape[0]):
        if np.isnan(matrix_g[i,j]): matrix_g[i,j] = last
        else: last = matrix_g[i,j]
for j in range(matrix_g.shape[1]):
    first = np.nan
    for i in range(matrix_g.shape[0]):
        if not np.isnan(matrix_g[i,j]): first = matrix_g[i,j]; break
    for i in range(matrix_g.shape[0]):
        if np.isnan(matrix_g[i,j]): matrix_g[i,j] = first
        else: break
for j in range(matrix_g.shape[1]):
    mask = np.isnan(matrix_g[:,j])
    if mask.any():
        med = np.nanmedian(matrix_g[:,j])
        matrix_g[mask,j] = med if not np.isnan(med) else 0.5

print(f"Matrix: {matrix_g.shape}, months: {months_g[0]} ~ {months_g[-1]}")

# Normalize
qt = QuantileTransformer(n_quantiles=min(1000, len(months_g)), output_distribution='uniform', random_state=42)
X_norm = qt.fit_transform(matrix_g)
time_vals = np.linspace(0, 1, len(months_g)).reshape(-1, 1)
X_full = np.hstack([X_norm, time_vals])

print(f"\nFull normalized matrix: {X_full.shape}")

# === DIMENSION SWEEP ===
print("\n" + "="*70)
print("  PCA DIMENSION SWEEP — finding optimal H2 detection range")
print("="*70)

# Full PCA to get explained variance
pca_full = PCA(n_components=min(X_full.shape[0]-1, X_full.shape[1]-1))
pca_full.fit(X_full)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

# Test specific dimension counts
test_dims = [8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 70, 100]
test_dims = [d for d in test_dims if d < min(X_full.shape)]

print(f"\n{'Dim':>4} | {'Var%':>6} | {'H0':>4} | {'H1':>4} | {'H1>p75':>7} | {'H2':>4} | {'H2>p75':>7} | {'Time':>6} | Notes")
print("-"*85)

results = []
for n_dim in test_dims:
    var_pct = cumvar[n_dim-1] * 100

    X_pca = pca_full.transform(X_full)[:, :n_dim]

    t0 = time.time()
    result = ripser(X_pca, maxdim=2, do_cocycles=True)
    dt = time.time() - t0

    dgms = result['dgms']
    cocycles = result['cocycles']

    # H0
    dgm0 = dgms[0]
    finite0 = dgm0[dgm0[:,1] < np.inf]
    h0_count = len(finite0)

    # H1
    dgm1 = dgms[1]
    finite1 = dgm1[dgm1[:,1] < np.inf]
    h1_count = len(finite1)
    h1_long = 0
    if h1_count > 0:
        pers1 = finite1[:,1] - finite1[:,0]
        h1_long = int((pers1 > np.percentile(pers1, 75)).sum()) if h1_count >= 4 else h1_count

    # H2
    dgm2 = dgms[2]
    finite2 = dgm2[dgm2[:,1] < np.inf]
    h2_count = len(finite2)
    h2_long = 0
    if h2_count > 0:
        pers2 = finite2[:,1] - finite2[:,0]
        h2_long = int((pers2 > np.percentile(pers2, 75)).sum()) if h2_count >= 4 else h2_count

    # Check if any H1/H2 touch 2024-2026
    h1_recent = 0
    h2_recent = 0

    for dim_level, dgm_finite, cyc_list in [(1, finite1, cocycles[1] if len(cocycles)>1 else []),
                                              (2, finite2, cocycles[2] if len(cocycles)>2 else [])]:
        if len(dgm_finite) == 0: continue
        pers = dgm_finite[:,1] - dgm_finite[:,0]
        for idx in np.argsort(-pers)[:min(20, len(pers))]:
            if idx >= len(cyc_list): continue
            cc = cyc_list[idx]
            pts = set()
            for row in cc:
                for col in range(min(3, row.shape[0])):
                    pts.add(int(row[col]))
            involved_months = [months_g[i] for i in pts if i < len(months_g)]
            has_recent = any(m[:4] in ['2024','2025','2026'] for m in involved_months)
            if has_recent:
                if dim_level == 1: h1_recent += 1
                else: h2_recent += 1

    notes = []
    if h2_count > 0: notes.append(f"H2!")
    if h1_recent > 0: notes.append(f"H1近期{h1_recent}")
    if h2_recent > 0: notes.append(f"H2近期{h2_recent}")

    print(f"{n_dim:>4} | {var_pct:>5.1f}% | {h0_count:>4} | {h1_count:>4} | {h1_long:>7} | {h2_count:>4} | {h2_long:>7} | {dt:>5.1f}s | {' '.join(notes)}")

    results.append({
        'n_dim': n_dim, 'var_pct': var_pct,
        'h0': h0_count, 'h1': h1_count, 'h1_long': h1_long,
        'h2': h2_count, 'h2_long': h2_long,
        'h1_recent': h1_recent, 'h2_recent': h2_recent,
        'time': dt, 'dgms': dgms, 'cocycles': cocycles,
        'X_pca': X_pca
    })

# === FIND OPTIMAL AND ANALYZE ===
print("\n" + "="*70)
print("  OPTIMAL DIMENSION ANALYSIS")
print("="*70)

# Find best: maximize H2 count while having reasonable variance
h2_dims = [r for r in results if r['h2'] > 0]
if h2_dims:
    # Among dims with H2, pick the one with most H2 long-lived features
    best = max(h2_dims, key=lambda r: (r['h2_long'], r['h2'], -r['n_dim']))
    print(f"\nOptimal: {best['n_dim']}D ({best['var_pct']:.1f}% variance)")
    print(f"  H0={best['h0']}, H1={best['h1']} ({best['h1_long']} long), H2={best['h2']} ({best['h2_long']} long)")

    # Detailed H1 analysis at optimal dim
    dgms = best['dgms']
    cocycles = best['cocycles']
    X_pca = best['X_pca']

    finite1 = dgms[1][dgms[1][:,1] < np.inf]
    if len(finite1) > 0:
        pers1 = finite1[:,1] - finite1[:,0]
        thr1 = np.percentile(pers1, 75) if len(pers1) >= 4 else 0
        print(f"\n  Top H1 cycles (persistence > {thr1:.3f}):")
        for rank, idx in enumerate(np.argsort(-pers1)[:15]):
            b, d = finite1[idx]
            p = pers1[idx]
            if idx < len(cocycles[1]):
                cc = cocycles[1][idx]
                pts = set()
                for row in cc: pts.add(int(row[0])); pts.add(int(row[1]))
                involved = [months_g[i] for i in pts if i < len(months_g)]
                if involved:
                    min_m, max_m = min(involved)[:7], max(involved)[:7]
                    has_2025 = any(m[:4] in ['2025','2026'] for m in involved)
                    has_2024 = any(m[:4] == '2024' for m in involved)
                    flag = " <<< 2025!" if has_2025 else (" << 2024" if has_2024 else "")
                    print(f"    #{rank+1} p={p:.3f} | {min_m}~{max_m} | {len(involved)}pts{flag}")

    finite2 = dgms[2][dgms[2][:,1] < np.inf]
    if len(finite2) > 0:
        pers2 = finite2[:,1] - finite2[:,0]
        print(f"\n  All H2 voids ({len(finite2)}):")
        for rank, idx in enumerate(np.argsort(-pers2)):
            b, d = finite2[idx]
            p = pers2[idx]
            if idx < len(cocycles[2]):
                cc = cocycles[2][idx]
                pts = set()
                for row in cc:
                    for col in range(min(3, row.shape[0])):
                        pts.add(int(row[col]))
                involved = [months_g[i] for i in pts if i < len(months_g)]
                if involved:
                    min_m, max_m = min(involved)[:7], max(involved)[:7]
                    has_2025 = any(m[:4] in ['2025','2026'] for m in involved)
                    has_2024 = any(m[:4] == '2024' for m in involved)
                    flag = " <<< 2025!" if has_2025 else (" << 2024" if has_2024 else "")
                    print(f"    #{rank+1} p={p:.4f} | {min_m}~{max_m} | {len(involved)}pts{flag}")
                    if has_2025 or has_2024:
                        recent = sorted([m for m in involved if m[:4] in ['2024','2025','2026']])
                        print(f"         Recent: {', '.join(m[:7] for m in recent[:10])}")

    # Edge analysis at optimal dimension
    print(f"\n  Crisis edge rates at {best['n_dim']}D:")
    nn = NearestNeighbors(n_neighbors=min(10, len(X_pca)-1))
    nn.fit(X_pca)
    dists, _ = nn.kneighbors(X_pca)
    avg_dists = dists.mean(axis=1)
    edge_thr = np.percentile(avg_dists, 80)

    crises = {
        'GFC 2007-09': ('2007-01-01', '2009-06-01'),
        'Trade War 2018-19': ('2018-03-01', '2019-12-01'),
        'COVID 2020': ('2020-02-01', '2020-12-01'),
        'Inflation 2022': ('2022-01-01', '2022-12-01'),
        'Trump 2.0 2025-26': ('2025-01-01', '2026-12-01'),
    }
    for name, (start, end) in crises.items():
        idxs = [i for i, m in enumerate(months_g) if start <= m <= end]
        if idxs:
            edge_pct = sum(1 for i in idxs if avg_dists[i] > edge_thr) / len(idxs) * 100
            print(f"    {name}: {edge_pct:.0f}% edge ({len(idxs)} months)")

    # Current position
    now_idx = len(months_g) - 1
    print(f"\n  Current ({months_g[now_idx][:7]}): dist={avg_dists[now_idx]:.3f}, threshold={edge_thr:.3f}")
    if avg_dists[now_idx] > edge_thr:
        print(f"    >>> ON THE EDGE")
    else:
        print(f"    >>> INSIDE normal region")

else:
    print("\nNo H2 voids detected at any dimension tested!")
    print("Trying finer sweep in low dimensions...")
    for d in range(5, 16):
        if d >= min(X_full.shape): break
        X_pca = pca_full.transform(X_full)[:, :d]
        r = ripser(X_pca, maxdim=2)
        h2 = len(r['dgms'][2][r['dgms'][2][:,1] < np.inf])
        h1 = len(r['dgms'][1][r['dgms'][1][:,1] < np.inf])
        var_p = cumvar[d-1]*100
        print(f"  {d}D ({var_p:.1f}%): H1={h1}, H2={h2}")

print("\nDone.")
