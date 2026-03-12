#!/usr/bin/env python3
"""
leader_manifold.py
==================
Manifold learning analysis on 38 leader behavioral profiles.

Goal: "How many independent dimensions does it really take to describe a leader's decision behavior?"

Steps:
1. Load all 38 profiles from data/leaders/profiles/*.json
2. Build a leader × dimension matrix (fill NaN for missing params)
3. Impute missing values with column median
4. PCA elbow plot → true dimensionality
5. UMAP + t-SNE 2D projections
6. Hierarchical clustering → natural groupings
7. Autocrat base model — mean/std per dimension
8. Key findings: clusters, distinguishing dims, outliers, similarity matrix
9. Interactive Plotly visualizations

Output: output/leader_manifold/
"""

import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT = Path("/home/user/global-financial-sim")
PROFILES_DIR = PROJECT / "data" / "leaders" / "profiles"
OUTPUT_DIR = PROJECT / "output" / "leader_manifold"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Imports (after venv activated) ──────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import umap

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

print("=" * 70)
print("LEADER MANIFOLD ANALYSIS — Global Financial Sim")
print("=" * 70)

# ─── STEP 1: Load all profiles ────────────────────────────────────────────────
print("\n[1/9] Loading profiles from", PROFILES_DIR)

SKIP_FILES = {"jp_transition_periods.json"}  # collective / aggregate profiles handled separately

def extract_params_from_profile(data: dict) -> dict:
    """
    Extract {param_name: float_value} from a single profile JSON.
    Handles multiple schema variants:
      A) behavior_matrix.vectors[{label, value}]
      B) behavioral_parameters[].parameters[{name, value}]
      C) aggregate_parameters (jp_transition_periods style)
      D) behavior_matrix with sub-keys having .vectors  (nested periods)
    Returns flat dict of {str: float}
    """
    params = {}

    # ----- Variant A: standard behavior_matrix.vectors -----
    bm = data.get("behavior_matrix") or data.get("aggregate_parameters")
    if bm and isinstance(bm, dict):
        vectors = bm.get("vectors", [])
        for v in vectors:
            if isinstance(v, dict) and "label" in v and "value" in v:
                try:
                    params[v["label"]] = float(v["value"])
                except (TypeError, ValueError):
                    pass
        # Variant C: aggregate_parameters dict style (jp_transition)
        for k, v in bm.items():
            if isinstance(v, dict) and "value" in v:
                try:
                    params[k] = float(v["value"])
                except (TypeError, ValueError):
                    pass

    # ----- Variant B: behavioral_parameters[].parameters[{name,value}] -----
    bp_list = data.get("behavioral_parameters", [])
    for section in bp_list:
        if isinstance(section, dict):
            for item in section.get("parameters", []):
                if isinstance(item, dict) and "name" in item and "value" in item:
                    try:
                        params[item["name"]] = float(item["value"])
                    except (TypeError, ValueError):
                        pass

    # ----- Variant D: nested periods (jp_transition_periods) -----
    periods = data.get("periods", [])
    for period in periods:
        ap = period.get("aggregate_parameters", {})
        for k, v in ap.items():
            if isinstance(v, dict) and "value" in v:
                try:
                    # prefix with period name to avoid collision
                    key = f"{period.get('period_name','period')}_{k}"
                    params[key] = float(v["value"])
                except (TypeError, ValueError):
                    pass

    return params


# Determine leader name from JSON
def get_leader_name(data: dict, filename: str) -> str:
    for key in ("leader", "collective_agent", "description"):
        val = data.get(key)
        if val and isinstance(val, str):
            # shorten if too long
            if len(val) > 40:
                val = val[:38] + ".."
            return val
    return filename.replace(".json", "")


profile_files = sorted(PROFILES_DIR.glob("*.json"))
print(f"  Found {len(profile_files)} JSON files")

leaders = []          # list of leader name strings
leader_params = []    # list of dicts {param: value}
file_names = []       # short filename stems

for pf in profile_files:
    if pf.name in SKIP_FILES:
        # Handle jp_transition_periods separately — extract each sub-period
        with open(pf) as f:
            data = json.load(f)
        for period in data.get("periods", []):
            pname = period.get("period_name", "period")
            ap = period.get("aggregate_parameters", {})
            params = {}
            for k, v in ap.items():
                if isinstance(v, dict) and "value" in v:
                    try:
                        params[k] = float(v["value"])
                    except (TypeError, ValueError):
                        pass
            if params:
                leaders.append(f"JP:{pname[:30]}")
                leader_params.append(params)
                file_names.append(pf.stem + "_" + pname)
        continue

    with open(pf) as f:
        data = json.load(f)

    params = extract_params_from_profile(data)
    name = get_leader_name(data, pf.name)
    leaders.append(name)
    leader_params.append(params)
    file_names.append(pf.stem)

print(f"  Loaded {len(leaders)} leader entries")
for i, (n, p) in enumerate(zip(leaders, leader_params)):
    print(f"    [{i+1:02d}] {n:<45}  {len(p)} params")

# ─── STEP 2: Build leader × dimension matrix ─────────────────────────────────
print("\n[2/9] Building leader × dimension matrix")

# Count how many leaders have each parameter
param_counts = defaultdict(int)
for pd_dict in leader_params:
    for k in pd_dict:
        param_counts[k] += 1

# Keep only params that appear in >= 2 profiles
# (Most params are leader-specific; requiring >=3 leaves almost nothing overlapping)
MIN_PROFILES = 2
kept_params = sorted([k for k, c in param_counts.items() if c >= MIN_PROFILES])
print(f"  Total unique params: {len(param_counts)}")
print(f"  Params in >= {MIN_PROFILES} profiles: {len(kept_params)}")

# Build matrix
matrix = np.full((len(leaders), len(kept_params)), np.nan)
for i, pd_dict in enumerate(leader_params):
    for j, param in enumerate(kept_params):
        if param in pd_dict:
            matrix[i, j] = pd_dict[param]

df = pd.DataFrame(matrix, index=leaders, columns=kept_params)
print(f"  Matrix shape: {df.shape}")
coverage = df.notna().mean()
sparsity = df.isna().mean()
print(f"  Mean param coverage per leader: {df.notna().mean(axis=1).mean():.1%}")
print(f"  Mean param coverage per dimension: {coverage.mean():.1%}")

# ─── STEP 3: Impute missing values ───────────────────────────────────────────
print("\n[3/9] Imputing missing values (column median)")

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(matrix)
print(f"  Imputed matrix: {X.shape}  (any NaN remaining: {np.isnan(X).any()})")

# Scale for PCA/UMAP
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─── STEP 4: PCA elbow analysis ──────────────────────────────────────────────
print("\n[4/9] PCA elbow analysis")

max_components = min(len(leaders), len(kept_params))
pca_full = PCA(n_components=max_components)
pca_full.fit(X_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

# Find elbow via second derivative
evr = pca_full.explained_variance_ratio_
d2 = np.diff(np.diff(evr))
elbow_idx = np.argmax(d2) + 2  # offset for double diff

# Also find components needed for 80%, 90%, 95% variance
def n_for_var(cum, threshold):
    idx = np.searchsorted(cum, threshold)
    return min(idx + 1, len(cum))

n_80 = n_for_var(cum_var, 0.80)
n_90 = n_for_var(cum_var, 0.90)
n_95 = n_for_var(cum_var, 0.95)

print(f"  Components for 80% variance: {n_80}")
print(f"  Components for 90% variance: {n_90}")
print(f"  Components for 95% variance: {n_95}")
print(f"  Elbow point (2nd-derivative): {elbow_idx}")
print(f"  Variance at elbow:            {cum_var[elbow_idx-1]:.1%}")

# True dimensionality = elbow point
TRUE_DIM = elbow_idx
print(f"\n  *** TRUE DIMENSIONALITY = {TRUE_DIM} independent dimensions ***")

# ─── STEP 5: PCA loadings — which dimensions matter most ────────────────────
print("\n[5/9] PCA component analysis")

pca_3d = PCA(n_components=min(3, max_components))
X_pca = pca_3d.fit_transform(X_scaled)

pca_true = PCA(n_components=TRUE_DIM)
X_pca_true = pca_true.fit_transform(X_scaled)

print("  Top 5 parameters loading on PC1:")
pc1_loadings = pd.Series(pca_3d.components_[0], index=kept_params)
for name_p, val in pc1_loadings.abs().sort_values(ascending=False)[:5].items():
    print(f"    {name_p:<50} {val:.3f}")

print("  Top 5 parameters loading on PC2:")
pc2_loadings = pd.Series(pca_3d.components_[1], index=kept_params)
for name_p, val in pc2_loadings.abs().sort_values(ascending=False)[:5].items():
    print(f"    {name_p:<50} {val:.3f}")

# ─── STEP 6: UMAP 2D embedding ───────────────────────────────────────────────
print("\n[6/9] UMAP 2D embedding")

n_neighbors = min(10, len(leaders) - 1)
reducer = umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=0.3,
    n_components=2,
    random_state=42,
    metric="euclidean"
)
X_umap = reducer.fit_transform(X_scaled)
print(f"  UMAP shape: {X_umap.shape}")

# ─── STEP 7: t-SNE for comparison ───────────────────────────────────────────
print("\n[7/9] t-SNE 2D projection")

perplexity = min(15, len(leaders) // 3)
tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    random_state=42,
    max_iter=1000,
    learning_rate="auto",
    init="pca"
)
X_tsne = tsne.fit_transform(X_scaled)
print(f"  t-SNE shape: {X_tsne.shape}")

# ─── STEP 8: Hierarchical clustering ─────────────────────────────────────────
print("\n[8/9] Hierarchical clustering")

linkage_matrix = linkage(X_pca_true, method="ward")
n_clusters = min(6, len(leaders) // 4)
cluster_labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")

print(f"  Number of clusters: {n_clusters}")
for c in range(1, n_clusters + 1):
    members = [leaders[i] for i in range(len(leaders)) if cluster_labels[i] == c]
    print(f"  Cluster {c}: {members}")

# ─── STEP 9: Autocrat base model ─────────────────────────────────────────────
print("\n[9/9] Autocrat base model analysis")

AUTOCRAT_NAMES = {
    "Xi Jinping", "Vladimir Putin", "Stalin", "Brezhnev",
    "Khrushchev", "Lenin", "Boris Yeltsin",
    "Ali Khamenei (Supreme Leader of Iran)",
    "Mojtaba Khamenei (Heir Apparent, Iran)",
    "Benjamin Netanyahu"
}

# Map leader display names → indices
def is_autocrat(name: str) -> bool:
    n = name.lower()
    keywords = ["xi jinping", "putin", "stalin", "brezhnev", "khrushchev",
                "lenin", "yeltsin", "khamenei", "netanyahu"]
    return any(k in n for k in keywords)

autocrat_mask = np.array([is_autocrat(n) for n in leaders])
democrat_mask = ~autocrat_mask & ~np.array(["jp:" in n.lower() for n in leaders])

print(f"  Autocrats: {autocrat_mask.sum()}")
print(f"  Others (democrats/technocrats): {democrat_mask.sum()}")
print("  Autocrat leaders:")
for i, n in enumerate(leaders):
    if autocrat_mask[i]:
        print(f"    - {n}")

# Compute centroids in scaled space
auto_centroid = X_scaled[autocrat_mask].mean(axis=0)
demo_centroid = X_scaled[democrat_mask].mean(axis=0)

# Dimensions that diverge most
divergence = np.abs(auto_centroid - demo_centroid)
top_diverge_idx = np.argsort(divergence)[-15:][::-1]
top_diverge_params = [(kept_params[i], divergence[i],
                       auto_centroid[i], demo_centroid[i])
                      for i in top_diverge_idx]

print("\n  Top 15 dimensions diverging most between autocrats vs others:")
print(f"  {'Parameter':<55} {'Divergence':>10} {'Autocrat':>10} {'Other':>10}")
print("  " + "-" * 90)
for param, div, ac, dc in top_diverge_params:
    print(f"  {param:<55} {div:>10.3f} {ac:>10.3f} {dc:>10.3f}")

# Most unique leaders (furthest from their cluster centroid)
cluster_centroids = {}
for c in range(1, n_clusters + 1):
    idx = np.where(cluster_labels == c)[0]
    cluster_centroids[c] = X_pca_true[idx].mean(axis=0)

uniqueness_scores = []
for i in range(len(leaders)):
    c = cluster_labels[i]
    dist = np.linalg.norm(X_pca_true[i] - cluster_centroids[c])
    uniqueness_scores.append((dist, leaders[i]))

uniqueness_scores.sort(reverse=True)
print("\n  Most 'unique' leaders (furthest from their cluster centroid):")
for rank, (dist, name) in enumerate(uniqueness_scores[:8], 1):
    print(f"    {rank}. {name:<45} dist={dist:.3f}")

# Similarity matrix (cosine similarity in full scaled space)
sim_matrix = cosine_similarity(X_scaled)

print("\n  Most similar leader pairs:")
pairs = []
for i in range(len(leaders)):
    for j in range(i + 1, len(leaders)):
        pairs.append((sim_matrix[i, j], leaders[i], leaders[j]))
pairs.sort(reverse=True)
for sim, n1, n2 in pairs[:10]:
    print(f"    {n1:<40} ↔  {n2:<40}  sim={sim:.3f}")

print("\n  Most dissimilar leader pairs:")
pairs.sort()
for sim, n1, n2 in pairs[:5]:
    print(f"    {n1:<40} ↔  {n2:<40}  sim={sim:.3f}")

# ─── VISUALIZATIONS ──────────────────────────────────────────────────────────
print("\n[VIZ] Generating interactive Plotly visualizations...")

CLUSTER_COLORS = px.colors.qualitative.Bold
LEADER_TYPE_COLORS = {
    "autocrat": "#E63946",
    "democrat": "#457B9D",
    "technocrat": "#2A9D8F",
    "jp_period": "#E9C46A",
    "other": "#6D6875",
}

def leader_type(name: str) -> str:
    if is_autocrat(name):
        return "autocrat"
    n = name.lower()
    if "jp:" in n:
        return "jp_period"
    if any(k in n for k in ["powell", "bernanke", "yellen", "greenspan",
                             "draghi", "lagarde", "nabiullina", "zhou"]):
        return "technocrat"
    return "democrat"

leader_types = [leader_type(n) for n in leaders]
type_color_list = [LEADER_TYPE_COLORS[t] for t in leader_types]

# Get top-5 params per leader for hover
def top5_params(i: int) -> str:
    row = matrix[i]
    present = [(kept_params[j], row[j]) for j in range(len(kept_params))
               if not np.isnan(row[j])]
    present.sort(key=lambda x: abs(x[1]), reverse=True)
    lines = [f"{p}: {v:.3f}" for p, v in present[:5]]
    return "<br>".join(lines)

hover_texts = [top5_params(i) for i in range(len(leaders))]
cluster_str = [f"Cluster {c}" for c in cluster_labels]

# ── Figure 1: PCA Elbow Plot ──────────────────────────────────────────────────
fig_elbow = go.Figure()
n_show = min(40, max_components)
x_vals = list(range(1, n_show + 1))

fig_elbow.add_trace(go.Bar(
    x=x_vals,
    y=pca_full.explained_variance_ratio_[:n_show] * 100,
    name="Individual variance %",
    marker_color="#457B9D",
    opacity=0.7,
))
fig_elbow.add_trace(go.Scatter(
    x=x_vals,
    y=cum_var[:n_show] * 100,
    name="Cumulative variance %",
    line=dict(color="#E63946", width=2),
    yaxis="y2",
))

# Mark elbow
fig_elbow.add_vline(x=TRUE_DIM, line_dash="dash", line_color="green",
                    annotation_text=f"Elbow: {TRUE_DIM}D", annotation_position="top right")
fig_elbow.add_hline(y=80, line_dash="dot", line_color="orange", yref="y2",
                    annotation_text="80%", annotation_position="right")
fig_elbow.add_hline(y=90, line_dash="dot", line_color="red", yref="y2",
                    annotation_text="90%", annotation_position="right")

fig_elbow.update_layout(
    title=dict(text=f"PCA Elbow Analysis — True Dimensionality = {TRUE_DIM} independent dimensions",
               font_size=16),
    xaxis=dict(title="Number of Principal Components", dtick=2),
    yaxis=dict(title="Individual Explained Variance (%)", showgrid=False),
    yaxis2=dict(title="Cumulative Explained Variance (%)", overlaying="y",
                side="right", range=[0, 105]),
    legend=dict(x=0.5, y=0.5),
    height=500,
    template="plotly_white",
    annotations=[
        dict(x=TRUE_DIM + 0.5, y=pca_full.explained_variance_ratio_[TRUE_DIM - 1] * 100 + 2,
             text=f"Elbow={TRUE_DIM}D<br>Cumulative={cum_var[TRUE_DIM-1]:.0%}",
             showarrow=False, font_size=11, bgcolor="lightgreen", bordercolor="green")
    ]
)
fig_elbow.write_html(str(OUTPUT_DIR / "01_pca_elbow.html"))
print(f"  Saved: 01_pca_elbow.html")

# ── Figure 2: UMAP scatter (main visualization) ───────────────────────────────
cluster_labels_str = [f"Cluster {c}" for c in cluster_labels]

fig_umap = go.Figure()

# Plot each type separately
for ltype in ["autocrat", "technocrat", "democrat", "jp_period", "other"]:
    idx = [i for i in range(len(leaders)) if leader_types[i] == ltype]
    if not idx:
        continue
    fig_umap.add_trace(go.Scatter(
        x=X_umap[idx, 0],
        y=X_umap[idx, 1],
        mode="markers+text",
        name=ltype.capitalize(),
        text=[leaders[i] for i in idx],
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(
            color=LEADER_TYPE_COLORS[ltype],
            size=12,
            symbol=("diamond" if ltype == "autocrat" else "circle"),
            line=dict(color="white", width=1),
            opacity=0.85,
        ),
        customdata=[[hover_texts[i], cluster_labels_str[i],
                     f"{df.notna().sum(axis=1).iloc[i]} params"]
                    for i in idx],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Type: " + ltype + "<br>"
            "%{customdata[1]}<br>"
            "Coverage: %{customdata[2]}<br>"
            "<br><b>Top parameters:</b><br>%{customdata[0]}<extra></extra>"
        )
    ))

fig_umap.update_layout(
    title=dict(text="UMAP 2D — Leader Decision-Space Map<br>"
               "<sup>Color = leader type. Diamonds = autocrats. "
               "Hover for top parameters.</sup>",
               font_size=15),
    xaxis=dict(title="UMAP-1", showgrid=False, zeroline=False),
    yaxis=dict(title="UMAP-2", showgrid=False, zeroline=False),
    height=700,
    width=1100,
    template="plotly_white",
    legend=dict(title="Leader Type", x=1.02, y=1),
)
fig_umap.write_html(str(OUTPUT_DIR / "02_umap_scatter.html"))
print(f"  Saved: 02_umap_scatter.html")

# ── Figure 3: UMAP colored by cluster ─────────────────────────────────────────
fig_umap_cluster = go.Figure()

for c in range(1, n_clusters + 1):
    idx = [i for i in range(len(leaders)) if cluster_labels[i] == c]
    fig_umap_cluster.add_trace(go.Scatter(
        x=X_umap[idx, 0],
        y=X_umap[idx, 1],
        mode="markers+text",
        name=f"Cluster {c}",
        text=[leaders[i] for i in idx],
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(
            color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
            size=12,
            symbol=["diamond" if is_autocrat(leaders[i]) else "circle" for i in idx],
            line=dict(color="white", width=1),
        ),
        customdata=[[hover_texts[i], leader_types[i]] for i in idx],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Type: %{customdata[1]}<br>"
            "<br><b>Top parameters:</b><br>%{customdata[0]}<extra></extra>"
        )
    ))

fig_umap_cluster.update_layout(
    title=dict(text=f"UMAP 2D — {n_clusters} Natural Clusters (Ward Hierarchical)<br>"
               "<sup>Diamonds = autocrats. Hover for details.</sup>",
               font_size=15),
    xaxis=dict(title="UMAP-1", showgrid=False, zeroline=False),
    yaxis=dict(title="UMAP-2", showgrid=False, zeroline=False),
    height=700,
    width=1100,
    template="plotly_white",
    legend=dict(title="Cluster", x=1.02, y=1),
)
fig_umap_cluster.write_html(str(OUTPUT_DIR / "03_umap_clusters.html"))
print(f"  Saved: 03_umap_clusters.html")

# ── Figure 4: t-SNE scatter ────────────────────────────────────────────────────
fig_tsne = go.Figure()

for ltype in ["autocrat", "technocrat", "democrat", "jp_period", "other"]:
    idx = [i for i in range(len(leaders)) if leader_types[i] == ltype]
    if not idx:
        continue
    fig_tsne.add_trace(go.Scatter(
        x=X_tsne[idx, 0],
        y=X_tsne[idx, 1],
        mode="markers+text",
        name=ltype.capitalize(),
        text=[leaders[i] for i in idx],
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(
            color=LEADER_TYPE_COLORS[ltype],
            size=12,
            symbol=("diamond" if ltype == "autocrat" else "circle"),
            line=dict(color="white", width=1),
            opacity=0.85,
        ),
        hovertemplate="<b>%{text}</b><extra></extra>"
    ))

fig_tsne.update_layout(
    title=dict(text=f"t-SNE 2D (perplexity={perplexity}) — Leader Decision Space<br>"
               "<sup>Cross-check with UMAP: similar clusters = robust structure</sup>",
               font_size=15),
    xaxis=dict(title="t-SNE-1", showgrid=False, zeroline=False),
    yaxis=dict(title="t-SNE-2", showgrid=False, zeroline=False),
    height=700,
    width=1100,
    template="plotly_white",
    legend=dict(title="Leader Type", x=1.02, y=1),
)
fig_tsne.write_html(str(OUTPUT_DIR / "04_tsne_scatter.html"))
print(f"  Saved: 04_tsne_scatter.html")

# ── Figure 5: Autocrat vs Democrat diverging dimensions ──────────────────────
# Use raw (unscaled) space to show actual parameter ranges
auto_means = np.nanmean(matrix[autocrat_mask], axis=0)
demo_means = np.nanmean(matrix[democrat_mask], axis=0)
raw_divergence = np.abs(auto_means - demo_means)

# Pick top 20 diverging (only where both groups have data)
both_have = (~np.isnan(auto_means)) & (~np.isnan(demo_means))
raw_div_masked = np.where(both_have, raw_divergence, 0)
top20_idx = np.argsort(raw_div_masked)[-20:][::-1]

fig_div = go.Figure()

params_top20 = [kept_params[i] for i in top20_idx]
auto_vals = [auto_means[i] for i in top20_idx]
demo_vals = [demo_means[i] for i in top20_idx]
# Shorten labels
short_labels = [p.replace("_", " ")[:45] for p in params_top20]

fig_div.add_trace(go.Bar(
    y=short_labels, x=auto_vals,
    name="Autocrats", orientation="h",
    marker_color="#E63946", opacity=0.8,
))
fig_div.add_trace(go.Bar(
    y=short_labels, x=demo_vals,
    name="Democrats/Technocrats", orientation="h",
    marker_color="#457B9D", opacity=0.8,
))

fig_div.update_layout(
    title=dict(text="Top 20 Dimensions Diverging Between Autocrats vs Others<br>"
               "<sup>Raw parameter values [0–1]. Higher = stronger presence of trait.</sup>",
               font_size=14),
    xaxis=dict(title="Parameter Value [0–1]", range=[0, 1.1]),
    barmode="group",
    height=700,
    template="plotly_white",
    legend=dict(x=0.7, y=0.02),
    margin=dict(l=280),
)
fig_div.write_html(str(OUTPUT_DIR / "05_autocrat_vs_democrat.html"))
print(f"  Saved: 05_autocrat_vs_democrat.html")

# ── Figure 6: Radar chart — Autocrat vs Democrat archetype ───────────────────
# Use the top 10 most diverging dimensions shared by both groups
top10_idx_radar = top20_idx[:10]
radar_params = [kept_params[i] for i in top10_idx_radar]
radar_labels = [p.replace("_", " ")[:35] for p in radar_params]

auto_radar = [float(auto_means[i]) for i in top10_idx_radar]
demo_radar = [float(demo_means[i]) for i in top10_idx_radar]

# Close the radar loop
radar_labels_closed = radar_labels + [radar_labels[0]]
auto_radar_closed = auto_radar + [auto_radar[0]]
demo_radar_closed = demo_radar + [demo_radar[0]]

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=auto_radar_closed,
    theta=radar_labels_closed,
    fill="toself",
    name="Autocrat Archetype",
    line_color="#E63946",
    fillcolor="rgba(230,57,70,0.2)",
))
fig_radar.add_trace(go.Scatterpolar(
    r=demo_radar_closed,
    theta=radar_labels_closed,
    fill="toself",
    name="Democrat/Technocrat",
    line_color="#457B9D",
    fillcolor="rgba(69,123,157,0.2)",
))

fig_radar.update_layout(
    title=dict(text="Autocrat vs Democrat/Technocrat Archetype Radar<br>"
               "<sup>Top 10 most diverging dimensions</sup>",
               font_size=14),
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=True,
    height=600,
    template="plotly_white",
)
fig_radar.write_html(str(OUTPUT_DIR / "06_radar_archetypes.html"))
print(f"  Saved: 06_radar_archetypes.html")

# ── Figure 7: Similarity heatmap ──────────────────────────────────────────────
# Sort leaders by hierarchical cluster for cleaner display
sort_order = np.argsort(cluster_labels)
sorted_names = [leaders[i] for i in sort_order]
sorted_sim = sim_matrix[sort_order][:, sort_order]

# Short names for display
def short_name(n: str, max_len: int = 22) -> str:
    n = n.replace("(Supreme Leader of Iran)", "").replace("(Heir Apparent, Iran)", "").strip()
    return n[:max_len]

sorted_short = [short_name(n) for n in sorted_names]

fig_heat = go.Figure(data=go.Heatmap(
    z=sorted_sim,
    x=sorted_short,
    y=sorted_short,
    colorscale="RdBu",
    zmid=0.5,
    zmin=0,
    zmax=1,
    text=np.round(sorted_sim, 2),
    hovertemplate="%{y} ↔ %{x}<br>Cosine similarity: %{z:.3f}<extra></extra>",
))

fig_heat.update_layout(
    title=dict(text="Leader Similarity Matrix (Cosine Similarity in Behavioral Space)<br>"
               "<sup>Sorted by hierarchical cluster. Red = similar, Blue = different.</sup>",
               font_size=14),
    height=750,
    width=850,
    xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
    yaxis=dict(tickfont=dict(size=9)),
    template="plotly_white",
)
fig_heat.write_html(str(OUTPUT_DIR / "07_similarity_heatmap.html"))
print(f"  Saved: 07_similarity_heatmap.html")

# ── Figure 8: Dimension importance (top params by variance) ──────────────────
param_var = np.nanvar(matrix, axis=0)
top30_var_idx = np.argsort(param_var)[-30:][::-1]
top30_params = [kept_params[i] for i in top30_var_idx]
top30_vals = [param_var[i] for i in top30_var_idx]
top30_labels = [p.replace("_", " ")[:50] for p in top30_params]

fig_importance = go.Figure(go.Bar(
    x=top30_vals,
    y=top30_labels,
    orientation="h",
    marker=dict(
        color=top30_vals,
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="Variance"),
    ),
))
fig_importance.update_layout(
    title=dict(text="Top 30 Dimensions by Cross-Leader Variance<br>"
               "<sup>High variance = high discriminatory power</sup>",
               font_size=14),
    xaxis=dict(title="Cross-Leader Variance"),
    height=750,
    template="plotly_white",
    margin=dict(l=320),
)
fig_importance.write_html(str(OUTPUT_DIR / "08_dimension_importance.html"))
print(f"  Saved: 08_dimension_importance.html")

# ── Figure 9: 3D PCA scatter ──────────────────────────────────────────────────
pca_3d_model = PCA(n_components=3)
X_pca3 = pca_3d_model.fit_transform(X_scaled)

fig_3d = go.Figure()
for ltype in ["autocrat", "technocrat", "democrat", "jp_period"]:
    idx = [i for i in range(len(leaders)) if leader_types[i] == ltype]
    if not idx:
        continue
    fig_3d.add_trace(go.Scatter3d(
        x=X_pca3[idx, 0],
        y=X_pca3[idx, 1],
        z=X_pca3[idx, 2],
        mode="markers+text",
        name=ltype.capitalize(),
        text=[leaders[i] for i in idx],
        marker=dict(
            color=LEADER_TYPE_COLORS[ltype],
            size=7,
            symbol=("diamond" if ltype == "autocrat" else "circle"),
            opacity=0.85,
        ),
        hovertemplate="<b>%{text}</b><extra></extra>"
    ))

var_explained = pca_3d_model.explained_variance_ratio_
fig_3d.update_layout(
    title=dict(text=f"3D PCA — Leader Decision Space<br>"
               f"<sup>PC1={var_explained[0]:.1%} | PC2={var_explained[1]:.1%} | PC3={var_explained[2]:.1%}</sup>",
               font_size=14),
    scene=dict(
        xaxis_title=f"PC1 ({var_explained[0]:.1%})",
        yaxis_title=f"PC2 ({var_explained[1]:.1%})",
        zaxis_title=f"PC3 ({var_explained[2]:.1%})",
    ),
    height=700,
    template="plotly_white",
    legend=dict(x=1.02, y=1),
)
fig_3d.write_html(str(OUTPUT_DIR / "09_pca_3d.html"))
print(f"  Saved: 09_pca_3d.html")

# ── Figure 10: Dashboard — combined overview ──────────────────────────────────
fig_dash = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        f"PCA Elbow (True Dim = {TRUE_DIM})",
        "UMAP 2D by Type",
        "Top Diverging: Autocrat vs Democrat",
        "Dimension Importance (Variance)"
    ),
    specs=[[{}, {}], [{}, {}]],
)

# Panel 1: elbow (bar only)
fig_dash.add_trace(
    go.Bar(x=x_vals, y=pca_full.explained_variance_ratio_[:n_show] * 100,
           marker_color="#457B9D", name="Individual %", showlegend=False),
    row=1, col=1
)
fig_dash.add_trace(
    go.Scatter(x=x_vals, y=cum_var[:n_show] * 100,
               line=dict(color="#E63946", width=2), name="Cumulative %", showlegend=False),
    row=1, col=1
)
fig_dash.add_vline(x=TRUE_DIM, line_dash="dash", line_color="green", row=1, col=1)

# Panel 2: UMAP scatter (simplified)
for ltype in ["autocrat", "technocrat", "democrat"]:
    idx = [i for i in range(len(leaders)) if leader_types[i] == ltype]
    if not idx:
        continue
    fig_dash.add_trace(
        go.Scatter(x=X_umap[idx, 0], y=X_umap[idx, 1],
                   mode="markers+text", text=[leaders[i][:12] for i in idx],
                   textposition="top center", textfont=dict(size=7),
                   name=ltype, marker=dict(color=LEADER_TYPE_COLORS[ltype], size=8,
                                          symbol=("diamond" if ltype == "autocrat" else "circle")),
                   showlegend=False),
        row=1, col=2
    )

# Panel 3: diverging bars (top 10)
top10 = top20_idx[:10]
p10_labels = [kept_params[i].replace("_", " ")[:30] for i in top10]
fig_dash.add_trace(
    go.Bar(y=p10_labels, x=[auto_means[i] for i in top10],
           name="Autocrat", orientation="h", marker_color="#E63946",
           showlegend=True),
    row=2, col=1
)
fig_dash.add_trace(
    go.Bar(y=p10_labels, x=[demo_means[i] for i in top10],
           name="Democrat", orientation="h", marker_color="#457B9D",
           showlegend=True),
    row=2, col=1
)

# Panel 4: top 15 importance
top15_var_idx = np.argsort(param_var)[-15:][::-1]
fig_dash.add_trace(
    go.Bar(y=[kept_params[i].replace("_", " ")[:30] for i in top15_var_idx],
           x=[param_var[i] for i in top15_var_idx],
           orientation="h", marker_color="#2A9D8F", showlegend=False),
    row=2, col=2
)

fig_dash.update_layout(
    title=dict(text="Leader Manifold Analysis — Dashboard Overview", font_size=16),
    height=900,
    width=1400,
    template="plotly_white",
    barmode="group",
    legend=dict(x=0.35, y=0.5),
)
fig_dash.write_html(str(OUTPUT_DIR / "00_dashboard.html"))
print(f"  Saved: 00_dashboard.html")

# ─── FINAL KEY FINDINGS REPORT ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("KEY FINDINGS SUMMARY")
print("=" * 70)

print(f"""
1. TRUE DIMENSIONALITY (PCA elbow)
   ─────────────────────────────────────
   Input matrix:       {len(leaders)} leaders × {len(kept_params)} shared parameters
   Elbow point:        {TRUE_DIM} independent dimensions
   Variance captured:  {cum_var[TRUE_DIM-1]:.1%}
   (80% needs {n_80} dims; 90% needs {n_90} dims; 95% needs {n_95} dims)

   Interpretation: Despite LLMs producing {len(kept_params)} unique parameter names,
   the actual behavioral variance across leaders collapses to ~{TRUE_DIM} independent
   axes. Most parameters are correlated proxy measurements of the same underlying
   behavioral drivers.

2. NATURAL CLUSTERS (Ward hierarchical, {n_clusters} clusters)
   ─────────────────────────────────────""")

cluster_composition = {}
for c in range(1, n_clusters + 1):
    members = [leaders[i] for i in range(len(leaders)) if cluster_labels[i] == c]
    n_auto = sum(1 for m in members if is_autocrat(m))
    n_tech = sum(1 for m in members if leader_type(m) == "technocrat")
    cluster_composition[c] = (members, n_auto, n_tech)
    print(f"   Cluster {c} ({len(members)} leaders, {n_auto} autocrats, {n_tech} technocrats):")
    for m in members:
        marker = "★" if is_autocrat(m) else ("⚙" if leader_type(m) == "technocrat" else "·")
        print(f"     {marker} {m}")

print(f"""
3. AUTOCRAT vs DEMOCRAT — TOP 5 DIVERGING DIMENSIONS
   ─────────────────────────────────────""")
for param, div, ac, dc in top_diverge_params[:5]:
    direction = "↑ autocrats" if ac > dc else "↑ democrats"
    print(f"   {param[:55]}")
    print(f"     autocrat={ac:+.3f}  democrat={dc:+.3f}  ({direction}, Δ={div:.3f})")

print(f"""
4. MOST UNIQUE LEADERS (furthest from cluster centroid)
   ─────────────────────────────────────""")
for rank, (dist, name) in enumerate(uniqueness_scores[:5], 1):
    print(f"   {rank}. {name:<50} (dist={dist:.3f})")

print(f"""
5. MOST SIMILAR LEADER PAIRS (cosine similarity)
   ─────────────────────────────────────""")
pairs_sorted_desc = sorted(pairs, reverse=True)
for sim, n1, n2 in pairs_sorted_desc[:5]:
    print(f"   {n1[:35]:<36} ↔ {n2[:35]:<36} sim={sim:.3f}")

# Save text report
report_lines = [
    "LEADER MANIFOLD ANALYSIS — KEY FINDINGS",
    "=" * 60,
    f"Date: 2026-03-12",
    f"Leaders analyzed: {len(leaders)}",
    f"Parameters (>=3 profiles): {len(kept_params)}",
    "",
    f"TRUE DIMENSIONALITY: {TRUE_DIM}",
    f"  Variance at elbow: {cum_var[TRUE_DIM-1]:.1%}",
    f"  80% variance needs: {n_80} dims",
    f"  90% variance needs: {n_90} dims",
    f"  95% variance needs: {n_95} dims",
    "",
    f"CLUSTERS: {n_clusters}",
]
for c in range(1, n_clusters + 1):
    members, n_auto, n_tech = cluster_composition[c]
    report_lines.append(f"  Cluster {c}: {', '.join(members)}")

report_lines += [
    "",
    "TOP DIVERGING DIMENSIONS (autocrat vs democrat):",
]
for param, div, ac, dc in top_diverge_params[:10]:
    report_lines.append(f"  {param}: autocrat={ac:.3f}, democrat={dc:.3f}, delta={div:.3f}")

report_lines += [
    "",
    "MOST UNIQUE LEADERS:",
]
for rank, (dist, name) in enumerate(uniqueness_scores[:8], 1):
    report_lines.append(f"  {rank}. {name} (dist={dist:.3f})")

report_lines += [
    "",
    "MOST SIMILAR PAIRS:",
]
for sim, n1, n2 in sorted(pairs, reverse=True)[:10]:
    report_lines.append(f"  {n1} <-> {n2}: {sim:.3f}")

with open(OUTPUT_DIR / "findings.txt", "w") as f:
    f.write("\n".join(report_lines))
print(f"\n  Saved: findings.txt")

print(f"\n{'='*70}")
print(f"OUTPUT DIRECTORY: {OUTPUT_DIR}")
print(f"Files generated:")
for fn in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {fn.name}")
print(f"\nAnalysis complete. True dimensionality = {TRUE_DIM} independent behavioral axes.")
print("=" * 70)
