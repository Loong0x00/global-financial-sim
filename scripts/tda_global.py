#!/usr/bin/env python3
"""
TDA Global Analysis — All 6 Economies Combined
================================================
Persistent homology on the combined global economic state space
with time dimension. US/CN/EU/UK/JP/RU.
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
print("1. DATA PREPARATION — 6 Economies Combined")
print("=" * 70)

DATA_BASE = Path('/home/user/global-financial-sim/data/economic')
OUT_DIR = Path('/home/user/global-financial-sim/output/tda_global')
OUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTRIES = ['us', 'cn', 'eu', 'uk', 'jp', 'ru']

# Load all data
all_series = {}  # key: "US_indicator_name", value: {date: value}
country_dims = {}

for country in COUNTRIES:
    prefix = country.upper()
    country_dims[prefix] = 0
    for ftype in ['macro', 'financial']:
        fpath = DATA_BASE / country / f'{ftype}.json'
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
                country_dims[prefix] += 1

print(f"Total indicator columns loaded: {len(all_series)}")
for c, n in country_dims.items():
    print(f"  {c}: {n} indicators")

# Collect all dates across all series
all_dates_set = set()
for series in all_series.values():
    all_dates_set.update(series.keys())

# Generate a full monthly timeline
def generate_months(start_ym, end_ym):
    months = []
    y, m = start_ym
    while (y, m) <= end_ym:
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months

sorted_dates = sorted(all_dates_set)
start_date = sorted_dates[0]
end_date = sorted_dates[-1]
start_ym = (int(start_date[:4]), int(start_date[5:7]))
end_ym = (int(end_date[:4]), int(end_date[5:7]))
all_months = generate_months(start_ym, end_ym)

print(f"Full timeline: {all_months[0]} to {all_months[-1]} ({len(all_months)} months)")

# Build raw matrix with NaN for missing
col_names = sorted(all_series.keys())
raw_matrix = np.full((len(all_months), len(col_names)), np.nan)
date_idx = {d: i for i, d in enumerate(all_months)}

for j, col in enumerate(col_names):
    for d, v in all_series[col].items():
        if d in date_idx:
            raw_matrix[date_idx[d], j] = v

# Forward fill then median fill
for j in range(raw_matrix.shape[1]):
    col = raw_matrix[:, j]
    # Forward fill
    last_valid = np.nan
    for i in range(len(col)):
        if not np.isnan(col[i]):
            last_valid = col[i]
        elif not np.isnan(last_valid):
            col[i] = last_valid
    # Remaining NaN -> column median
    median_val = np.nanmedian(col)
    if np.isnan(median_val):
        median_val = 0.0
    col[np.isnan(col)] = median_val
    raw_matrix[:, j] = col

# Add time dimension: normalized to [0, 1]
time_vals = np.linspace(0, 1, len(all_months)).reshape(-1, 1)
col_names_with_time = col_names + ['TIME']

# Combine
raw_with_time = np.hstack([raw_matrix, time_vals])

print(f"\nRaw matrix shape (with time): {raw_with_time.shape}")
print(f"Time range: {all_months[0]} to {all_months[-1]}")
print(f"Total dimensions: {raw_with_time.shape[1]} ({raw_with_time.shape[1]-1} indicators + 1 time)")

# Quantile transform
qt = QuantileTransformer(output_distribution='uniform', random_state=42)
X_norm = qt.fit_transform(raw_with_time)

# Check if dimensionality is too high -> PCA
n_dims = X_norm.shape[1]
if n_dims > 50:
    print(f"\nDimensionality {n_dims} > 50, applying PCA to reduce to 25...")
    pca = PCA(n_components=25, random_state=42)
    X_pca = pca.fit_transform(X_norm)
    explained = np.cumsum(pca.explained_variance_ratio_)
    print(f"PCA explained variance: {explained[-1]:.1%} with 25 components")
    print(f"  10 components: {explained[9]:.1%}")
    print(f"  15 components: {explained[14]:.1%}")
    print(f"  20 components: {explained[19]:.1%}")
    X_tda = X_pca
    print(f"Working matrix for TDA: {X_tda.shape}")
else:
    X_tda = X_norm
    print(f"Dimensionality {n_dims} <= 50, no PCA needed.")

print(f"\nFinal matrix: {X_tda.shape[0]} time points x {X_tda.shape[1]} dimensions")

# ============================================================
# 2. TDA PERSISTENT HOMOLOGY
# ============================================================

print("\n" + "=" * 70)
print("2. TDA PERSISTENT HOMOLOGY (Vietoris-Rips)")
print("=" * 70)

# If too many points, subsample
n_points = X_tda.shape[0]
MAX_POINTS = 400
if n_points > MAX_POINTS:
    print(f"\n{n_points} points > {MAX_POINTS}, subsampling for TDA...")
    # Use landmark maxmin subsampling for better coverage
    np.random.seed(42)
    # Start with random point, greedily add farthest
    indices = [np.random.randint(n_points)]
    dists = np.full(n_points, np.inf)
    for _ in range(MAX_POINTS - 1):
        new_dists = np.linalg.norm(X_tda - X_tda[indices[-1]], axis=1)
        dists = np.minimum(dists, new_dists)
        indices.append(np.argmax(dists))
    landmark_idx = np.array(sorted(indices))
    X_ripser = X_tda[landmark_idx]
    print(f"Selected {len(landmark_idx)} landmark points via maxmin")
    # Map for crisis annotation later
    landmark_months = [all_months[i] for i in landmark_idx]
else:
    X_ripser = X_tda
    landmark_idx = np.arange(n_points)
    landmark_months = all_months

print(f"Running ripser on {X_ripser.shape} ...")
result = ripser(X_ripser, maxdim=2, thresh=2.0)
diagrams = result['dgms']

for dim in range(3):
    dgm = diagrams[dim]
    finite = dgm[dgm[:, 1] < np.inf]
    n_total = len(dgm)
    n_finite = len(finite)
    n_inf = n_total - n_finite
    if n_finite > 0:
        lifetimes = finite[:, 1] - finite[:, 0]
        median_life = np.median(lifetimes)
        long_lived = lifetimes > 2 * median_life
        n_long = long_lived.sum()
        print(f"\nH{dim}: {n_total} features ({n_finite} finite, {n_inf} infinite)")
        print(f"  Lifetime: median={median_life:.4f}, max={lifetimes.max():.4f}, mean={lifetimes.mean():.4f}")
        print(f"  Long-lived (>2x median): {n_long}")
    else:
        print(f"\nH{dim}: {n_total} features (all infinite)")

# ============================================================
# 3. CRISIS ANNOTATION
# ============================================================

print("\n" + "=" * 70)
print("3. CRISIS PERIOD ANNOTATION")
print("=" * 70)

CRISES = {
    '1973-74 Oil Crisis':            ('1973-10', '1975-03'),
    '1980-82 Volcker/Global':        ('1980-01', '1982-12'),
    '1987 Black Monday':             ('1987-09', '1988-02'),
    '1990-91 Gulf War':              ('1990-07', '1991-06'),
    '1997-98 Asian Crisis':          ('1997-07', '1998-10'),
    '2001 Dot-com':                  ('2001-03', '2001-11'),
    '2007-09 GFC':                   ('2007-12', '2009-06'),
    '2010-12 Euro Debt':             ('2010-04', '2012-07'),
    '2014-16 Commodity/EM':          ('2014-06', '2016-02'),
    '2020 COVID':                    ('2020-02', '2020-06'),
    '2022 Inflation/Tightening':     ('2022-01', '2022-12'),
}

# Map months to full timeline indices
full_date_idx = {d: i for i, d in enumerate(all_months)}

# For each crisis, find indices in full timeline and in landmark set
crisis_full_idx = {}
crisis_landmark_idx = {}
landmark_set = set(landmark_idx.tolist())
landmark_pos = {orig: pos for pos, orig in enumerate(landmark_idx)}

for name, (start, end) in CRISES.items():
    full_indices = [full_date_idx[m] for m in all_months if start <= m <= end]
    crisis_full_idx[name] = full_indices
    lm_indices = [landmark_pos[i] for i in full_indices if i in landmark_pos]
    crisis_landmark_idx[name] = lm_indices
    print(f"  {name}: {len(full_indices)} months total, {len(lm_indices)} landmark points")

# Labels for full timeline
labels_full = np.zeros(len(all_months), dtype=int)
crisis_list = list(CRISES.keys())
for ci, (name, indices) in enumerate(crisis_full_idx.items()):
    for idx in indices:
        labels_full[idx] = ci + 1

# Labels for landmark set
labels_landmark = np.zeros(len(landmark_idx), dtype=int)
for ci, (name, indices) in enumerate(crisis_landmark_idx.items()):
    for idx in indices:
        labels_landmark[idx] = ci + 1

# ============================================================
# 4. UMAP + DENSITY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("4. UMAP EMBEDDING + DENSITY ANALYSIS")
print("=" * 70)

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                     metric='euclidean', random_state=42)
X_2d = reducer.fit_transform(X_tda)  # full dataset
print(f"UMAP embedding shape: {X_2d.shape}")

# Local density in original space
nn = NearestNeighbors(n_neighbors=10, metric='euclidean')
nn.fit(X_tda)
distances, _ = nn.kneighbors(X_tda)
local_density = 1.0 / (distances.mean(axis=1) + 1e-8)

normal_mask = labels_full == 0
crisis_mask = labels_full > 0
print(f"\nDensity analysis (in {X_tda.shape[1]}D space):")
print(f"  Normal periods: mean={local_density[normal_mask].mean():.2f}, median={np.median(local_density[normal_mask]):.2f}")
if crisis_mask.sum() > 0:
    print(f"  Crisis periods:  mean={local_density[crisis_mask].mean():.2f}, median={np.median(local_density[crisis_mask]):.2f}")
    print(f"  Ratio (crisis/normal): {local_density[crisis_mask].mean() / local_density[normal_mask].mean():.2f}x")

print("\nPer-crisis density ranking:")
crisis_density_info = []
for name, indices in crisis_full_idx.items():
    if indices:
        idx_arr = np.array(indices)
        d = local_density[idx_arr]
        pct_below = np.mean(d < np.median(local_density)) * 100
        crisis_density_info.append((name, d.mean(), pct_below))
        print(f"  {name}: mean_density={d.mean():.2f}, {pct_below:.0f}% below median")

# Sort by % below median (most peripheral first)
crisis_density_info.sort(key=lambda x: -x[2])
print("\nDensity ranking (most peripheral first):")
for i, (name, d, pct) in enumerate(crisis_density_info, 1):
    print(f"  {i}. {name}: {pct:.0f}% below median")

# ============================================================
# 5. VISUALIZATIONS
# ============================================================

print("\n" + "=" * 70)
print("5. GENERATING VISUALIZATIONS")
print("=" * 70)

# --- 5a. Persistence Diagram ---
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plot_diagrams(diagrams, ax=ax, show=False)
ax.set_title('Persistence Diagram — Global Economic State Space (6 Economies)', fontsize=13)
ax.set_xlabel('Birth', fontsize=12)
ax.set_ylabel('Death', fontsize=12)

# Annotate long-lived features
for dim in range(3):
    dgm = diagrams[dim]
    finite = dgm[dgm[:, 1] < np.inf]
    if len(finite) > 0:
        lifetimes = finite[:, 1] - finite[:, 0]
        median_life = np.median(lifetimes)
        for k in range(len(finite)):
            if lifetimes[k] > 3 * median_life:
                ax.annotate(f'H{dim}', (finite[k, 0], finite[k, 1]),
                           fontsize=7, alpha=0.6)

plt.tight_layout()
plt.savefig(OUT_DIR / 'persistence_diagram.png', dpi=150)
plt.close()
print("  Saved persistence_diagram.png")

# --- 5b. State Space — Time-colored ---
fig, ax = plt.subplots(1, 1, figsize=(16, 11))
time_idx = np.arange(len(all_months))
scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=time_idx, cmap='coolwarm',
                     s=6, alpha=0.6)
cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
n_ticks = 8
tick_pos = np.linspace(0, len(all_months)-1, n_ticks).astype(int)
cbar.set_ticks(tick_pos)
cbar.set_ticklabels([all_months[i][:4] for i in tick_pos])
cbar.set_label('Year', fontsize=12)

# Annotate crises on the time-colored map
crisis_colors_list = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990'
]
crisis_markers = ['*', 'D', 'v', '^', 'X', 'P', 's', 'h', '8', 'p', 'd']

for ci, (name, indices) in enumerate(crisis_full_idx.items()):
    if indices:
        idx_arr = np.array(indices)
        ax.scatter(X_2d[idx_arr, 0], X_2d[idx_arr, 1],
                  marker=crisis_markers[ci % len(crisis_markers)],
                  s=50, c=crisis_colors_list[ci % len(crisis_colors_list)],
                  edgecolors='black', linewidth=0.3, label=name, zorder=5)

ax.set_title('Global Economic State Space (UMAP 2D) — Colored by Time', fontsize=14)
ax.set_xlabel('UMAP-1', fontsize=12)
ax.set_ylabel('UMAP-2', fontsize=12)
ax.legend(loc='best', fontsize=7, framealpha=0.9, ncol=2)
plt.tight_layout()
plt.savefig(OUT_DIR / 'state_space_time.png', dpi=150)
plt.close()
print("  Saved state_space_time.png")

# --- 5c. State Space — Crisis highlight ---
fig, ax = plt.subplots(1, 1, figsize=(16, 11))

# Normal in grey
normal_idx = np.where(labels_full == 0)[0]
ax.scatter(X_2d[normal_idx, 0], X_2d[normal_idx, 1], c='lightgrey', s=5, alpha=0.4, label='Normal')

# Highlight each crisis
for ci, (name, indices) in enumerate(crisis_full_idx.items()):
    if indices:
        idx_arr = np.array(indices)
        color = crisis_colors_list[ci % len(crisis_colors_list)]
        marker = crisis_markers[ci % len(crisis_markers)]
        ax.scatter(X_2d[idx_arr, 0], X_2d[idx_arr, 1],
                  marker=marker, s=70, c=color,
                  edgecolors='black', linewidth=0.5, label=name, zorder=5)
        # Convex hull
        if len(idx_arr) >= 3:
            from scipy.spatial import ConvexHull
            pts = X_2d[idx_arr]
            try:
                hull = ConvexHull(pts)
                hull_pts = np.append(hull.vertices, hull.vertices[0])
                ax.plot(pts[hull_pts, 0], pts[hull_pts, 1], '--', c=color, alpha=0.5, linewidth=1)
            except Exception:
                pass

# Density contours
from scipy.stats import gaussian_kde
try:
    kde = gaussian_kde(X_2d.T)
    xmin, xmax = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    ymin, ymax = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contour(xx, yy, zz, levels=8, colors='grey', alpha=0.25, linewidths=0.5)
except Exception:
    pass

ax.set_title('Global Economic State Space — Crisis Topology', fontsize=14)
ax.set_xlabel('UMAP-1', fontsize=12)
ax.set_ylabel('UMAP-2', fontsize=12)
ax.legend(loc='best', fontsize=7, framealpha=0.9, ncol=2)
plt.tight_layout()
plt.savefig(OUT_DIR / 'state_space_crisis.png', dpi=150)
plt.close()
print("  Saved state_space_crisis.png")

# --- 5d. Mapper Graph ---
print("\n  Generating Mapper graph...")
try:
    mapper = km.KeplerMapper(verbose=0)

    # Use UMAP 2D as lens (already computed)
    lens = X_2d

    # Build simplicial complex
    graph = mapper.map(
        lens,
        X=X_tda,
        cover=km.Cover(n_cubes=25, perc_overlap=0.4),
        clusterer=DBSCAN(eps=0.5, min_samples=3)
    )

    # Extract node info for plotting
    node_positions = {}
    node_sizes = {}
    node_colors = {}
    edges = []

    for node_id, members in graph['nodes'].items():
        members = np.array(members)
        # Position = mean of UMAP coords
        pos = X_2d[members].mean(axis=0)
        node_positions[node_id] = pos
        node_sizes[node_id] = len(members)
        # Color = mean time index (normalized)
        node_colors[node_id] = np.mean(members) / len(all_months)

    for node_id, neighbors in graph['links'].items():
        for neighbor in neighbors:
            if node_id < neighbor:
                edges.append((node_id, neighbor))

    if node_positions:
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))

        # Draw edges
        for n1, n2 in edges:
            if n1 in node_positions and n2 in node_positions:
                p1, p2 = node_positions[n1], node_positions[n2]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'grey', alpha=0.3, linewidth=0.8)

        # Draw nodes
        positions = np.array([node_positions[n] for n in node_positions])
        sizes = np.array([node_sizes[n] for n in node_positions]) * 8
        colors = np.array([node_colors[n] for n in node_positions])

        scatter = ax.scatter(positions[:, 0], positions[:, 1],
                           c=colors, cmap='coolwarm', s=sizes,
                           edgecolors='black', linewidth=0.5, alpha=0.8, zorder=5)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Average Time (0=earliest, 1=latest)', fontsize=11)

        # Mark crisis nodes
        for ci, (cname, indices) in enumerate(crisis_full_idx.items()):
            crisis_set = set(indices)
            for node_id, members in graph['nodes'].items():
                overlap = len(crisis_set.intersection(members))
                if overlap > len(members) * 0.3 and node_id in node_positions:
                    pos = node_positions[node_id]
                    ax.annotate(cname.split(' ')[0], pos,
                               fontsize=6, fontweight='bold',
                               color=crisis_colors_list[ci % len(crisis_colors_list)],
                               ha='center', va='bottom',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        ax.set_title(f'Mapper Graph — Global Economic Topology ({len(node_positions)} nodes, {len(edges)} edges)', fontsize=13)
        ax.set_xlabel('UMAP-1', fontsize=12)
        ax.set_ylabel('UMAP-2', fontsize=12)
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'mapper_graph.png', dpi=150)
        plt.close()
        print(f"  Saved mapper_graph.png ({len(node_positions)} nodes, {len(edges)} edges)")
    else:
        print("  Mapper produced no nodes, skipping graph.")

except Exception as e:
    print(f"  Mapper failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 6. ANALYSIS SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("6. ANALYSIS SUMMARY")
print("=" * 70)

print(f"""
DATA:
  Economies: {', '.join(c.upper() for c in COUNTRIES)}
  Total indicator dimensions: {len(col_names)} + 1 (time) = {len(col_names)+1}
  Per-economy breakdown: {', '.join(f'{c}:{n}' for c,n in country_dims.items())}
  Time range: {all_months[0]} to {all_months[-1]} ({len(all_months)} months)
  TDA input: {X_ripser.shape[0]} points x {X_ripser.shape[1]} dimensions
  {'PCA applied: ' + str(n_dims) + 'D -> ' + str(X_tda.shape[1]) + 'D' if n_dims > 50 else 'No PCA needed'}
""")

print("TOPOLOGICAL FEATURES:")
for dim in range(3):
    dgm = diagrams[dim]
    finite = dgm[dgm[:, 1] < np.inf]
    n_inf = len(dgm) - len(finite)
    if len(finite) > 0:
        lifetimes = finite[:, 1] - finite[:, 0]
        median_life = np.median(lifetimes)
        n_long = (lifetimes > 2 * median_life).sum()
        print(f"  H{dim}: {len(dgm)} total, {n_long} long-lived, max_lifetime={lifetimes.max():.4f}")
    else:
        print(f"  H{dim}: {len(dgm)} total ({n_inf} infinite)")

print(f"""
GLOBAL vs US-ONLY COMPARISON:
  US-only (previous run): 15 dims, 581 months, H0=182/H1=455/H2=116 features
  Global (this run): {X_tda.shape[1]} dims, {len(all_months)} months, {len(COUNTRIES)} economies

CRISIS BOTTLENECK RANKING (most peripheral first):""")
for i, (name, d, pct) in enumerate(crisis_density_info, 1):
    tag = "EDGE" if pct > 60 else ("MODERATE" if pct > 40 else "EMBEDDED")
    print(f"  {i}. [{tag}] {name}: {pct:.0f}% below median density")

print(f"""
TOPOLOGICAL INTERPRETATION:
  - H0 (connected components): Number of distinct global economic regimes.
    Multiple long-lived H0 features = regime changes that persist across
    large distance scales. These include e.g. pre/post-2008 regulatory shifts,
    China's WTO entry reshaping global trade topology.

  - H1 (loops/cycles): Global business cycles and feedback loops.
    Cross-country cycles (e.g. US rate hike -> EM capital flight -> commodity drop
    -> US import prices) should appear as persistent H1 features.

  - H2 (voids): Forbidden state combinations across 6 economies.
    E.g. "all economies simultaneously at max growth + low inflation + low rates"
    is topologically impossible — should appear as a void.

CROSS-COUNTRY TRANSMISSION CHANNELS:
  The multi-economy state space should reveal:
  1. Correlated crisis clusters: GFC/Euro Debt/Commodity crises occupy
     overlapping but distinct regions — transmission with time lag.
  2. Contagion pathways: Asian Crisis -> Russia Default -> LTCM visible
     as connected trajectory in state space.
  3. Decoupling: China's 2015 devaluation may form isolated cluster
     if transmission to developed markets was limited.
  4. Synchronization: COVID should show maximum clustering (all economies
     hit simultaneously) vs Asian Crisis (sequential spread).

Output saved to: {OUT_DIR}/
""")
