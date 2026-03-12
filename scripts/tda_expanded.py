#!/usr/bin/env python3
"""
TDA Expanded Analysis — Economic + Market + Commodities + Crypto
================================================================
Builds on tda_global.py but adds:
- 14 global stock indices (daily→monthly)
- Gold, Silver (daily→monthly)
- BTC, ETH (daily→monthly)
- All available commodity data

Then runs persistent homology + UMAP + Mapper visualization.
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

# ============================================================
# 1. DATA PREPARATION
# ============================================================

print("=" * 70)
print("1. DATA PREPARATION — Expanded Dataset")
print("=" * 70)

DATA_BASE = Path('/home/user/global-financial-sim/data/economic')
OUT_DIR = Path('/home/user/global-financial-sim/output/tda_expanded')
OUT_DIR.mkdir(parents=True, exist_ok=True)

all_series = {}  # key: "category_name", value: {date_str: value}
category_counts = {}

# --- A) Original FRED economic data (6 economies) ---
COUNTRIES = ['us', 'cn', 'eu', 'uk', 'jp', 'ru']
for country in COUNTRIES:
    prefix = country.upper()
    cat = f"FRED_{prefix}"
    category_counts[cat] = 0
    for ftype in ['macro', 'financial']:
        fpath = DATA_BASE / country / f'{ftype}.json'
        if not fpath.exists():
            continue
        with open(fpath) as f:
            data = json.load(f)
        indicators = data.get('indicators', data)
        for name, ind in indicators.items():
            series = ind.get('series', [])
            if not series:
                continue
            col_name = f"{prefix}_{name}"
            date_val = {}
            for pt in series:
                d, v = pt['date'], pt['value']
                if v is not None:
                    date_val[d] = float(v)
            if date_val:
                all_series[col_name] = date_val
                category_counts[cat] += 1

# --- B) Global stock indices (daily→monthly average) ---
def daily_to_monthly(data_points, value_key='close'):
    """Convert daily data points to monthly averages."""
    monthly = {}
    for pt in data_points:
        date_str = pt['date']
        val = pt.get(value_key, pt.get('value'))
        if val is None:
            continue
        ym = date_str[:7]  # "YYYY-MM"
        month_key = ym + "-01"
        if month_key not in monthly:
            monthly[month_key] = []
        monthly[month_key].append(float(val))
    return {k: np.mean(v) for k, v in monthly.items()}

indices_path = DATA_BASE / 'indices' / 'global_indices.json'
if indices_path.exists():
    cat = "INDICES"
    category_counts[cat] = 0
    with open(indices_path) as f:
        indices_data = json.load(f)
    for name, info in indices_data.items():
        data_pts = info.get('data', [])
        if not data_pts:
            continue
        monthly = daily_to_monthly(data_pts)
        if monthly:
            col_name = f"IDX_{name}"
            all_series[col_name] = monthly
            category_counts[cat] += 1

# --- C) Gold & Silver ---
for metal in ['gold', 'silver']:
    fpath = DATA_BASE / 'commodities' / f'{metal}_daily.json'
    if fpath.exists():
        with open(fpath) as f:
            metal_data = json.load(f)
        data_pts = metal_data.get('data', [])
        monthly = daily_to_monthly(data_pts)
        if monthly:
            col_name = f"METAL_{metal}"
            all_series[col_name] = monthly
            if "METALS" not in category_counts:
                category_counts["METALS"] = 0
            category_counts["METALS"] += 1

# --- D) Crypto (BTC, ETH) ---
crypto_path = DATA_BASE / 'crypto' / 'crypto.json'
if crypto_path.exists():
    cat = "CRYPTO"
    category_counts[cat] = 0
    with open(crypto_path) as f:
        crypto_data = json.load(f)
    for name, info in crypto_data.items():
        data_pts = info.get('data', [])
        monthly = daily_to_monthly(data_pts)
        if monthly:
            col_name = f"CRYPTO_{name}"
            all_series[col_name] = monthly
            category_counts[cat] += 1

# --- E) FRED Commodities ---
fred_comm_path = DATA_BASE / 'commodities' / 'fred_commodities.json'
if fred_comm_path.exists():
    cat = "COMMODITIES_FRED"
    category_counts[cat] = 0
    with open(fred_comm_path) as f:
        comm_data = json.load(f)
    for name, info in comm_data.items():
        data_pts = info.get('data', [])
        if not data_pts:
            continue
        freq = info.get('frequency', 'monthly')
        if freq == 'daily':
            monthly = daily_to_monthly(data_pts, value_key='value')
        else:
            monthly = {pt['date']: float(pt['value']) for pt in data_pts if pt.get('value') is not None}
        if monthly:
            col_name = f"COMM_{name}"
            all_series[col_name] = monthly
            category_counts[cat] += 1

# --- F) Yahoo Futures ---
yf_path = DATA_BASE / 'commodities' / 'yahoo_futures.json'
if yf_path.exists():
    cat = "COMMODITIES_YF"
    category_counts[cat] = 0
    with open(yf_path) as f:
        yf_data = json.load(f)
    for name, info in yf_data.items():
        data_pts = info.get('data', [])
        if not data_pts:
            continue
        monthly = daily_to_monthly(data_pts, value_key='value')
        if monthly:
            col_name = f"YF_{name}"
            all_series[col_name] = monthly
            category_counts[cat] += 1

# Summary
print(f"\nTotal indicator columns: {len(all_series)}")
for cat, n in sorted(category_counts.items()):
    print(f"  {cat:25s}: {n}")

# ============================================================
# 2. BUILD STATE MATRIX
# ============================================================

print("\n" + "=" * 70)
print("2. BUILD MONTHLY STATE MATRIX")
print("=" * 70)

# Generate full monthly timeline
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

# Build matrix: rows=months, cols=indicators
col_names = sorted(all_series.keys())
n_months = len(all_months)
n_cols = len(col_names)

print(f"Timeline: {all_months[0]} to {all_months[-1]} ({n_months} months)")
print(f"Indicators: {n_cols}")

matrix = np.full((n_months, n_cols), np.nan)
for j, col in enumerate(col_names):
    series = all_series[col]
    for i, month in enumerate(all_months):
        if month in series:
            matrix[i, j] = series[month]

# Coverage analysis
coverage = np.sum(~np.isnan(matrix), axis=0) / n_months * 100
good_cols = coverage >= 10  # At least 10% coverage (~66 months minimum)
print(f"Columns with >=20% coverage: {good_cols.sum()}/{n_cols}")

# Filter columns
matrix_filtered = matrix[:, good_cols]
col_names_filtered = [col_names[j] for j in range(n_cols) if good_cols[j]]
print(f"Filtered matrix: {matrix_filtered.shape}")

# For each month, compute coverage
row_coverage = np.sum(~np.isnan(matrix_filtered), axis=1) / matrix_filtered.shape[1] * 100

# Only keep months with at least 30% of indicators
good_rows = row_coverage >= 30
matrix_good = matrix_filtered[good_rows]
months_good = [all_months[i] for i in range(n_months) if good_rows[i]]
print(f"Months with >=30% coverage: {len(months_good)}")
print(f"Final matrix before imputation: {matrix_good.shape}")

# Forward-fill NaN (carry last known value)
for j in range(matrix_good.shape[1]):
    last_val = np.nan
    for i in range(matrix_good.shape[0]):
        if np.isnan(matrix_good[i, j]):
            matrix_good[i, j] = last_val
        else:
            last_val = matrix_good[i, j]

# Backward-fill remaining NaNs
for j in range(matrix_good.shape[1]):
    first_val = np.nan
    for i in range(matrix_good.shape[0]):
        if not np.isnan(matrix_good[i, j]):
            first_val = matrix_good[i, j]
            break
    for i in range(matrix_good.shape[0]):
        if np.isnan(matrix_good[i, j]):
            matrix_good[i, j] = first_val
        else:
            break

remaining_nan = np.sum(np.isnan(matrix_good))
print(f"Remaining NaN after imputation: {remaining_nan}")

# Drop any column still all-NaN
valid_cols = ~np.all(np.isnan(matrix_good), axis=0)
matrix_good = matrix_good[:, valid_cols]
col_names_final = [col_names_filtered[j] for j in range(len(col_names_filtered)) if valid_cols[j]]
print(f"Final matrix: {matrix_good.shape}")

# Fill any remaining NaN with column median
for j in range(matrix_good.shape[1]):
    col = matrix_good[:, j]
    mask = np.isnan(col)
    if mask.any():
        median_val = np.nanmedian(col)
        matrix_good[mask, j] = median_val if not np.isnan(median_val) else 0

# ============================================================
# 3. NORMALIZE + ADD TIME DIMENSION
# ============================================================

print("\n" + "=" * 70)
print("3. QUANTILE NORMALIZATION + TIME DIMENSION")
print("=" * 70)

qt = QuantileTransformer(n_quantiles=min(1000, len(months_good)),
                         output_distribution='uniform',
                         random_state=42)
X_norm = qt.fit_transform(matrix_good)

# Add time dimension
time_vals = np.linspace(0, 1, len(months_good)).reshape(-1, 1)
X_full = np.hstack([X_norm, time_vals])
print(f"Normalized + time: {X_full.shape} (last col = time)")

# PCA for dimensionality reduction
n_components = min(30, X_full.shape[1] - 1, X_full.shape[0] - 1)
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_full)
explained = np.cumsum(pca.explained_variance_ratio_)
n_keep = np.searchsorted(explained, 0.95) + 1
X_pca = X_pca[:, :n_keep]
print(f"PCA: {n_components} components, keeping {n_keep} for 95% variance ({explained[n_keep-1]*100:.1f}%)")

# ============================================================
# 4. TDA — PERSISTENT HOMOLOGY
# ============================================================

print("\n" + "=" * 70)
print("4. PERSISTENT HOMOLOGY (ripser)")
print("=" * 70)

# Landmark selection for speed
n_landmarks = min(500, len(months_good))
if len(months_good) > n_landmarks:
    idx = np.linspace(0, len(months_good)-1, n_landmarks, dtype=int)
    X_tda = X_pca[idx]
    months_tda = [months_good[i] for i in idx]
else:
    X_tda = X_pca
    months_tda = months_good
    idx = np.arange(len(months_good))

print(f"Landmark points: {len(X_tda)}")

# Run ripser
result = ripser(X_tda, maxdim=2, do_cocycles=True)
dgms = result['dgms']

for dim in range(len(dgms)):
    dgm = dgms[dim]
    finite = dgm[dgm[:, 1] < np.inf]
    if len(finite) > 0:
        persistences = finite[:, 1] - finite[:, 0]
        threshold = np.percentile(persistences, 75)
        long_lived = persistences > threshold
        print(f"  H{dim}: {len(finite)} features, {long_lived.sum()} long-lived (>{threshold:.3f})")
    else:
        print(f"  H{dim}: {len(dgm)} features (all infinite)")

# ============================================================
# 5. CRISIS BOTTLENECK ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("5. CRISIS BOTTLENECK ANALYSIS")
print("=" * 70)

nn = NearestNeighbors(n_neighbors=min(10, len(X_tda)-1))
nn.fit(X_tda)
dists, _ = nn.kneighbors(X_tda)
avg_dists = dists.mean(axis=1)
edge_threshold = np.percentile(avg_dists, 80)

# Define crisis periods
crises = {
    "Oil Crisis 1973-74": ("1973-10-01", "1974-12-01"),
    "Volcker Shock 1980-82": ("1980-01-01", "1982-12-01"),
    "Black Monday 1987": ("1987-08-01", "1988-03-01"),
    "Japan Bubble 1990-92": ("1990-01-01", "1992-12-01"),
    "Asian Crisis 1997-98": ("1997-07-01", "1998-12-01"),
    "Dot-com 2000-02": ("2000-03-01", "2002-10-01"),
    "GFC 2007-09": ("2007-07-01", "2009-06-01"),
    "Euro Crisis 2010-12": ("2010-05-01", "2012-12-01"),
    "Commodity Crash 2014-16": ("2014-07-01", "2016-02-01"),
    "Trade War 2018-19": ("2018-03-01", "2019-12-01"),
    "COVID 2020": ("2020-02-01", "2020-12-01"),
    "Inflation 2022": ("2022-01-01", "2022-12-01"),
    "Trump Tariffs 2025": ("2025-04-01", "2026-03-01"),
}

for crisis_name, (start, end) in crises.items():
    crisis_idx = [i for i, m in enumerate(months_tda) if start <= m <= end]
    if not crisis_idx:
        print(f"  {crisis_name:35s}: no data points")
        continue
    crisis_dists = avg_dists[crisis_idx]
    edge_pct = np.mean(crisis_dists > edge_threshold) * 100
    mean_dist = np.mean(crisis_dists)
    print(f"  {crisis_name:35s}: {edge_pct:5.1f}% edge, avg_dist={mean_dist:.3f} ({len(crisis_idx)} points)")

# ============================================================
# 6. UMAP VISUALIZATION
# ============================================================

print("\n" + "=" * 70)
print("6. UMAP + MAPPER VISUALIZATION")
print("=" * 70)

# UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_tda)

# Decade coloring
def month_to_decade(m):
    year = int(m[:4])
    if year < 1980: return 0
    elif year < 1990: return 1
    elif year < 2000: return 2
    elif year < 2010: return 3
    elif year < 2020: return 4
    else: return 5

decade_labels = ['1971-79', '1980s', '1990s', '2000s', '2010s', '2020s']
decades = [month_to_decade(m) for m in months_tda]
colors = cm.rainbow(np.linspace(0, 1, 6))

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Plot 1: Color by decade
ax = axes[0]
for d in range(6):
    mask = [i for i, dd in enumerate(decades) if dd == d]
    if mask:
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                   c=[colors[d]], s=15, alpha=0.7, label=decade_labels[d])
ax.set_title('UMAP: Expanded Dataset — Color by Decade', fontsize=14)
ax.legend(fontsize=10)
ax.set_xlabel('UMAP-1')
ax.set_ylabel('UMAP-2')

# Plot 2: Color by topological edge distance
ax = axes[1]
sc = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=avg_dists, cmap='hot_r', s=15, alpha=0.7)
plt.colorbar(sc, ax=ax, label='Avg k-NN Distance (topological edge)')
ax.set_title('UMAP: Topological Edge Distance (bright=bottleneck)', fontsize=14)
ax.set_xlabel('UMAP-1')
ax.set_ylabel('UMAP-2')

# Mark crisis periods
for crisis_name, (start, end) in crises.items():
    crisis_idx = [i for i, m in enumerate(months_tda) if start <= m <= end]
    if crisis_idx:
        cx, cy = np.mean(X_umap[crisis_idx, 0]), np.mean(X_umap[crisis_idx, 1])
        short_name = crisis_name.split(' ')[0] + ' ' + crisis_name.split(' ')[-1]
        axes[1].annotate(short_name, (cx, cy), fontsize=7, alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig(OUT_DIR / 'umap_expanded.png', dpi=150, bbox_inches='tight')
print(f"Saved UMAP plot: {OUT_DIR / 'umap_expanded.png'}")

# ============================================================
# 7. PERSISTENCE DIAGRAM
# ============================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plot_diagrams(dgms, ax=ax, show=False)
ax.set_title(f'Persistence Diagram — Expanded Dataset ({X_full.shape[1]}D → {n_keep}D PCA)', fontsize=14)
plt.tight_layout()
plt.savefig(OUT_DIR / 'persistence_expanded.png', dpi=150, bbox_inches='tight')
print(f"Saved persistence diagram: {OUT_DIR / 'persistence_expanded.png'}")

# ============================================================
# 8. MAPPER NETWORK
# ============================================================

print("\nBuilding Mapper network...")
mapper = km.KeplerMapper(verbose=0)
lens = mapper.fit_transform(X_tda, projection=umap.UMAP(n_components=2, random_state=42))

graph = mapper.map(
    lens, X_tda,
    cover=km.Cover(n_cubes=25, perc_overlap=0.4),
    clusterer=DBSCAN(eps=0.5, min_samples=3)
)

n_nodes = len(graph['nodes'])
n_edges = len(graph['links'])
print(f"Mapper: {n_nodes} nodes, {n_edges} edges")

# Color mapper by time
node_colors = []
for node_id, members in graph['nodes'].items():
    avg_time = np.mean([decades[i] for i in members])
    node_colors.append(avg_time)

html_output = OUT_DIR / 'mapper_expanded.html'
mapper.visualize(graph,
                 path_html=str(html_output),
                 title="Global Financial State Space — Expanded Dataset",
                 color_values=np.array([decades[i] for i in range(len(months_tda))]),
                 color_function_name="Decade",
                 node_color_function=np.array(['mean']))
print(f"Saved Mapper HTML: {html_output}")

# ============================================================
# 9. COMPARISON WITH PREVIOUS RUN
# ============================================================

print("\n" + "=" * 70)
print("9. COMPARISON WITH PREVIOUS (71 indicator) RUN")
print("=" * 70)

print(f"""
Previous run (tda_global.py):
  Indicators: 71 + 1 time = 72D → 25D PCA (97.9% variance)
  H0: 8 long-lived components
  H1: 15 long-lived loops
  H2: 2 long-lived voids

Current expanded run:
  Indicators: {len(col_names_final)} + 1 time = {X_full.shape[1]}D → {n_keep}D PCA ({explained[n_keep-1]*100:.1f}% variance)
  Data: see H0/H1/H2 counts above
  New data sources: stock indices, gold/silver, BTC/ETH, commodities
""")

# Save metadata
meta = {
    "run_date": "2026-03-12",
    "n_indicators": len(col_names_final),
    "n_months": len(months_good),
    "n_landmarks": len(X_tda),
    "pca_dims": int(n_keep),
    "pca_variance_explained": float(explained[n_keep-1]),
    "categories": {k: int(v) for k, v in category_counts.items()},
    "indicators": col_names_final,
    "tda_results": {
        f"H{dim}": {
            "total": len(dgms[dim]),
            "finite": len(dgms[dim][dgms[dim][:, 1] < np.inf]),
        } for dim in range(len(dgms))
    }
}
with open(OUT_DIR / 'metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\nDone! Output in {OUT_DIR}/")
