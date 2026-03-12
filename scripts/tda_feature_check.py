#!/usr/bin/env python3
"""
Check: do TDA topological features (cycles, voids) pass through 2025-2026?
"""
import json, numpy as np
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from ripser import ripser
import unicodedata, warnings
warnings.filterwarnings('ignore')

DATA_BASE = Path('data')
ECON_BASE = DATA_BASE / 'economic'
LEADER_BASE = DATA_BASE / 'leaders'

# === LOAD DATA (same as tda_full.py) ===
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

print(f"Series: {len(all_series)}")

all_months_set = set()
for s in all_series.values(): all_months_set.update(s.keys())
all_months = sorted([m for m in all_months_set if '1964-01-01' <= m <= '2026-12-01'])

econ_cols = sorted(all_series.keys())
matrix = np.full((len(all_months), len(econ_cols)), np.nan)
for j, col in enumerate(econ_cols):
    for i, month in enumerate(all_months):
        if month in all_series[col]: matrix[i, j] = all_series[col][month]

# Filter
coverage = np.sum(~np.isnan(matrix), axis=0) / len(all_months) * 100
good_cols = coverage >= 8
matrix_f = matrix[:, good_cols]
col_names_f = [econ_cols[j] for j in range(len(econ_cols)) if good_cols[j]]

row_cov = np.sum(~np.isnan(matrix_f), axis=1) / matrix_f.shape[1] * 100
good_rows = row_cov >= 25
matrix_g = matrix_f[good_rows]
months_g = [all_months[i] for i in range(len(all_months)) if good_rows[i]]

# Impute
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

# Normalize + PCA
qt = QuantileTransformer(n_quantiles=min(1000, len(months_g)), output_distribution='uniform', random_state=42)
X_norm = qt.fit_transform(matrix_g)
time_vals = np.linspace(0, 1, len(months_g)).reshape(-1, 1)
X_full = np.hstack([X_norm, time_vals])
pca = PCA(n_components=min(30, X_full.shape[1]-1, X_full.shape[0]-1))
X_pca = pca.fit_transform(X_full)
explained = np.cumsum(pca.explained_variance_ratio_)
n_keep = np.searchsorted(explained, 0.95) + 1
X_pca = X_pca[:, :n_keep]
print(f"PCA: {n_keep}D ({explained[n_keep-1]*100:.1f}%)")

# TDA with cocycles
print("\nRunning TDA with cocycles...")
result = ripser(X_pca, maxdim=2, do_cocycles=True)
dgms = result['dgms']
cocycles = result['cocycles']

# === ANALYZE: which features involve 2024-2026? ===
print("\n" + "="*60)
print("  TOPOLOGICAL FEATURES NEAR 2025-2026")
print("="*60)

# H1 cycles
dgm1 = dgms[1]
finite1 = dgm1[dgm1[:,1] < np.inf]
if len(finite1) > 0:
    pers1 = finite1[:,1] - finite1[:,0]
    thr1 = np.percentile(pers1, 75)

    print(f"\nH1 环路 (persistence > {thr1:.3f}, 共{(pers1>thr1).sum()}个长寿命):")

    sorted_idx = np.argsort(-pers1)
    for rank, idx in enumerate(sorted_idx[:15]):
        birth, death = finite1[idx]
        p = pers1[idx]

        # Get cocycle representatives
        if idx < len(cocycles[1]):
            cc = cocycles[1][idx]
            # Cocycle gives pairs of point indices
            involved_pts = set()
            for row in cc:
                involved_pts.add(int(row[0]))
                involved_pts.add(int(row[1]))

            involved_months = [months_g[i] for i in involved_pts if i < len(months_g)]

            has_2025 = any(m[:4] in ['2025', '2026'] for m in involved_months)
            has_2024 = any(m[:4] == '2024' for m in involved_months)
            has_recent = has_2025 or has_2024

            # Date range
            if involved_months:
                min_m = min(involved_months)[:7]
                max_m = max(involved_months)[:7]

                flag = "⚠️ 包含2025!" if has_2025 else ("📌 包含2024" if has_2024 else "")
                print(f"  #{rank+1} p={p:.3f} | {min_m}~{max_m} | {len(involved_months)}个月 {flag}")

                if has_recent:
                    recent = sorted([m for m in involved_months if m[:4] in ['2024','2025','2026']])
                    print(f"       近期月份: {', '.join(m[:7] for m in recent[:10])}")

# H2 voids
dgm2 = dgms[2]
finite2 = dgm2[dgm2[:,1] < np.inf]
if len(finite2) > 0:
    pers2 = finite2[:,1] - finite2[:,0]
    print(f"\nH2 空洞 ({len(finite2)}个, {(pers2>np.percentile(pers2,75)).sum()}个长寿命):")

    sorted_idx2 = np.argsort(-pers2)
    for rank, idx in enumerate(sorted_idx2[:5]):
        birth, death = finite2[idx]
        p = pers2[idx]

        if idx < len(cocycles[2]):
            cc = cocycles[2][idx]
            involved_pts = set()
            for row in cc:
                for col in range(min(3, row.shape[0])):
                    involved_pts.add(int(row[col]))

            involved_months = [months_g[i] for i in involved_pts if i < len(months_g)]
            has_2025 = any(m[:4] in ['2025','2026'] for m in involved_months)
            has_2024 = any(m[:4] == '2024' for m in involved_months)

            if involved_months:
                min_m = min(involved_months)[:7]
                max_m = max(involved_months)[:7]
                flag = "⚠️ 包含2025!" if has_2025 else ("📌 包含2024" if has_2024 else "")
                print(f"  #{rank+1} p={p:.4f} | {min_m}~{max_m} | {len(involved_months)}个月 {flag}")
                if has_2025 or has_2024:
                    recent = sorted([m for m in involved_months if m[:4] in ['2024','2025','2026']])
                    print(f"       近期月份: {', '.join(m[:7] for m in recent[:10])}")

# === Check: distance from current point to nearest H2 void boundary ===
print(f"\n{'='*60}")
print("  当前位置 vs 拓扑特征")
print("="*60)

nn = NearestNeighbors(n_neighbors=min(10, len(X_pca)-1))
nn.fit(X_pca)
dists, _ = nn.kneighbors(X_pca)
avg_dists = dists.mean(axis=1)
edge_thr = np.percentile(avg_dists, 80)

now_idx = len(months_g) - 1
print(f"当前: {months_g[now_idx]} | 拓扑距离: {avg_dists[now_idx]:.3f} | 阈值: {edge_thr:.3f}")

# Check recent trend
print(f"\n最近12个月拓扑距离趋势:")
for i in range(max(0, now_idx-11), now_idx+1):
    d = avg_dists[i]
    bar = "█" * int(d / edge_thr * 20)
    flag = " ⚠️" if d > edge_thr else ""
    print(f"  {months_g[i][:7]} dist={d:.3f} |{bar}{flag}")

# Check: is the current state near any H1 cycle?
print(f"\n当前位置到各环路的最小距离:")
for rank, idx in enumerate(np.argsort(-(finite1[:,1]-finite1[:,0]))[:10]):
    if idx < len(cocycles[1]):
        cc = cocycles[1][idx]
        pts = set()
        for row in cc: pts.add(int(row[0])); pts.add(int(row[1]))
        pts_list = [p for p in pts if p < len(X_pca)]
        if pts_list:
            cycle_center = X_pca[pts_list].mean(axis=0)
            dist_to_cycle = np.linalg.norm(X_pca[now_idx] - cycle_center)
            months_in = [months_g[p] for p in pts_list]
            min_m, max_m = min(months_in)[:7], max(months_in)[:7]
            p = finite1[idx,1] - finite1[idx,0]
            print(f"  环路#{rank+1} (p={p:.3f}, {min_m}~{max_m}): 距离={dist_to_cycle:.3f}")
