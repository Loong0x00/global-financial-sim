#!/usr/bin/env python3
"""
TDA Forward Projection — 基于拓扑近邻的历史类比推演
====================================================
核心思路：
  1. 找到当前状态在拓扑空间中的位置
  2. 找到历史上最相似的N个时期
  3. 看这些时期之后发生了什么
  4. 用拓扑距离加权，得到未来路径的概率分布

这不是预测，是"如果历史拓扑结构继续成立，从当前位置出发最可能的演化方向"
"""

import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import umap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import unicodedata
import warnings
warnings.filterwarnings('ignore')

DATA_BASE = Path('/home/user/global-financial-sim/data')
ECON_BASE = DATA_BASE / 'economic'
LEADER_BASE = DATA_BASE / 'leaders'
OUT_DIR = Path('/home/user/global-financial-sim/output/tda_forecast')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. LOAD DATA (reuse from viz_interactive.py)
# ============================================================
print("=" * 60)
print("  加载数据...")
print("=" * 60)

all_series = {}

# FRED
COUNTRIES = ['us', 'cn', 'eu', 'uk', 'jp', 'ru']
for c in COUNTRIES:
    fpath = ECON_BASE / 'fred' / f'{c}.json'
    if fpath.exists():
        with open(fpath) as f:
            data = json.load(f)
        for key, series in data.items():
            name = f"{c}_{key}"
            monthly = {}
            for pt in series.get('data', []):
                d = pt.get('date', '')[:7]
                v = pt.get('value')
                if d and v is not None:
                    try:
                        monthly[d + '-01'] = float(v)
                    except:
                        pass
            if monthly:
                all_series[name] = monthly

# Indices
idx_file = ECON_BASE / 'indices' / 'global_indices.json'
if idx_file.exists():
    with open(idx_file) as f:
        idx_data = json.load(f)
    for name, info in idx_data.items():
        monthly = {}
        for pt in info.get('data', []):
            d = pt.get('date', '')[:7]
            v = pt.get('close')
            if d and v is not None:
                key = d + '-01'
                if key not in monthly:
                    monthly[key] = []
                monthly[key].append(float(v))
        all_series[f"idx_{name}"] = {k: np.mean(v) for k, v in monthly.items()}

# Crypto
crypto_file = ECON_BASE / 'crypto' / 'crypto.json'
if crypto_file.exists():
    with open(crypto_file) as f:
        crypto_data = json.load(f)
    for name, info in crypto_data.items():
        monthly = {}
        for pt in info.get('data', []):
            d = pt.get('date', '')[:7]
            v = pt.get('close')
            if d and v is not None:
                key = d + '-01'
                if key not in monthly:
                    monthly[key] = []
                monthly[key].append(float(v))
        all_series[f"crypto_{name}"] = {k: np.mean(v) for k, v in monthly.items()}

# Commodities
comm_dir = ECON_BASE / 'commodities'
for fname in ['fred_commodities.json', 'yahoo_futures.json', 'world_bank_commodities.json']:
    fpath = comm_dir / fname
    if fpath.exists():
        with open(fpath) as f:
            cdata = json.load(f)
        prefix = fname.split('_')[0][:4]
        for name, info in cdata.items():
            monthly = {}
            for pt in info.get('data', []):
                d = pt.get('date', '')[:7]
                v = pt.get('value') or pt.get('close')
                if d and v is not None:
                    key = d + '-01'
                    if key not in monthly:
                        monthly[key] = []
                    if isinstance(monthly[key], list):
                        monthly[key].append(float(v))
                    else:
                        monthly[key] = [monthly[key], float(v)]
            all_series[f"comm_{prefix}_{name}"] = {
                k: np.mean(v) if isinstance(v, list) else v for k, v in monthly.items()
            }

# Gold/Silver
for metal in ['gold_daily.json', 'silver_daily.json']:
    fpath = comm_dir / metal
    if fpath.exists():
        with open(fpath) as f:
            mdata = json.load(f)
        mname = metal.replace('_daily.json', '')
        monthly = {}
        for pt in mdata.get('data', []):
            d = pt.get('date', '')[:7]
            v = pt.get('close') or pt.get('value')
            if d and v is not None:
                key = d + '-01'
                if key not in monthly:
                    monthly[key] = []
                monthly[key].append(float(v))
        all_series[f"metal_{mname}"] = {k: np.mean(v) for k, v in monthly.items()}

print(f"  经济序列: {len(all_series)}")

# Time axis
all_months_set = set()
for s in all_series.values():
    all_months_set.update(s.keys())
all_months = sorted([m for m in all_months_set if '1964-01-01' <= m <= '2026-12-01'])

# Load leaders
profiles = {}
prof_dir = LEADER_BASE / 'profiles'
if prof_dir.exists():
    for f in sorted(prof_dir.glob('*.json')):
        with open(f) as fh:
            pdata = json.load(fh)
        params = {}
        bm = pdata.get('behavior_matrix', {})
        if isinstance(bm, dict):
            for v in bm.get('vectors', []):
                val = v.get('value')
                label = v.get('label', v.get('name', ''))
                if isinstance(val, (int, float)) and label:
                    params[label] = float(val)
        for section in pdata.get('behavioral_parameters', []):
            if isinstance(section, dict):
                for p in section.get('parameters', []):
                    val = p.get('value')
                    if isinstance(val, (int, float)):
                        params[p.get('name', '')] = float(val)
        if params:
            profiles[f.stem] = {'params': params}

print(f"  领导人档案: {len(profiles)}")

# Top dims
all_dims = {}
for pk, pv in profiles.items():
    for d in pv['params']:
        all_dims[d] = all_dims.get(d, 0) + 1
top_dims = sorted(all_dims.keys(), key=lambda x: -all_dims[x])[:40]

# Timeline matching (simplified)
def normalize_name(name):
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    name = name.lower().strip().replace('.', '').replace('-', ' ')
    return '_'.join(name.split())

name_to_profile = {pk: pk for pk in profiles}
name_to_profile.update({
    'hu_jintao': 'hu_wen', 'wen_jiabao': 'hu_wen',
    'leonid_brezhnev': 'brezhnev', 'nikita_khrushchev': 'khrushchev',
    'joseph_stalin': 'stalin', 'vladimir_lenin': 'lenin',
})

ROLE_SLOTS = {
    'US_PRES': ['president'],
    'US_FED': ['fed_chair', 'federal_reserve_chair', 'chairman_of_the_federal_reserve'],
    'CN_LEADER': ['general_secretary', 'paramount_leader', 'chairman_of_cpc', 'president_of_prc'],
    'CN_PREMIER': ['premier_of_state_council', 'premier'],
    'CN_PBOC': ['pboc_governor', 'governor_of_pboc', "governor_of_people's_bank", 'governor,_people'],
    'EU_ECB': ['ecb_president', 'president_of_ecb', 'bundesbank_president'],
    'EU_LEADER': ['german_chancellor', 'french_president'],
    'UK_PM': ['prime_minister'],
    'UK_BOE': ['boe_governor', 'governor_of_bank_of_england'],
    'JP_PM': ['prime_minister'],
    'JP_BOJ': ['governor,_bank_of_japan', 'governor_of_bank_of_japan', 'boj_governor'],
    'RU_LEADER': ['president', 'head_of_state', 'general_secretary', 'general_secretary_of_the_cpsu'],
    'RU_CB': ['chairman,_gosbank', 'governor,_central_bank', 'chairman_of_the_central_bank'],
    'IR_LEADER': ['supreme_leader'],
}

def match_role_slot(role_str, country):
    role_lower = role_str.lower().replace('-', '_').replace(' ', '_')
    for slot, keywords in ROLE_SLOTS.items():
        if not slot.startswith(country):
            continue
        for kw in keywords:
            if kw in role_lower:
                return slot
    return None

leader_periods = []
tl_dir = LEADER_BASE / 'timelines'
if tl_dir.exists():
    for f in sorted(tl_dir.glob('*.json')):
        country = f.stem.upper()
        with open(f) as fh:
            tl_data = json.load(fh)
        entries = []
        if isinstance(tl_data, list):
            entries = tl_data
        elif isinstance(tl_data, dict):
            for key, val in tl_data.items():
                if key == 'roles' and isinstance(val, dict):
                    for rn, rd in val.items():
                        if isinstance(rd, dict) and 'leaders' in rd:
                            for leader in rd['leaders']:
                                if isinstance(leader, dict):
                                    entries.append(leader)
                elif isinstance(val, list) and val and isinstance(val[0], dict):
                    entries.extend(val)
        for entry in entries:
            name = entry.get('name', '')
            role = entry.get('role', entry.get('position', ''))
            start = entry.get('start', entry.get('term_start', ''))
            end = entry.get('end', entry.get('term_end', '2026-12'))
            if not name or not start:
                continue
            if len(start) == 7: start += '-01'
            if len(end) == 7: end += '-01'
            nn = normalize_name(name)
            pk = name_to_profile.get(nn)
            if pk is None:
                last = nn.split('_')[-1]
                for k in profiles:
                    if last in k or k in nn:
                        pk = k
                        break
            if pk and pk in profiles:
                leader_periods.append((pk, start, end, role, country))

# Role assignments
role_assignments = {}
for month in all_months:
    role_assignments[month] = {}
    for pk, start, end, role, country in leader_periods:
        if start <= month <= end:
            slot = match_role_slot(role, country)
            if slot:
                role_assignments[month][slot] = pk

slots_used = sorted(set(
    slot for m in all_months for slot in role_assignments.get(m, {})
))

N_LEADER_DIMS = 15
econ_cols = sorted(all_series.keys())
leader_col_names = []
for slot in slots_used:
    for dim in top_dims[:N_LEADER_DIMS]:
        leader_col_names.append(f"L_{slot}_{dim}")

all_col_names = econ_cols + leader_col_names
matrix = np.full((len(all_months), len(all_col_names)), np.nan)

for j, col in enumerate(econ_cols):
    series = all_series[col]
    for i, month in enumerate(all_months):
        if month in series:
            matrix[i, j] = series[month]

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

# Filter
coverage = np.sum(~np.isnan(matrix), axis=0) / len(all_months) * 100
good_cols = coverage >= 8
matrix_f = matrix[:, good_cols]
col_names_f = [all_col_names[j] for j in range(len(all_col_names)) if good_cols[j]]

row_coverage = np.sum(~np.isnan(matrix_f), axis=1) / matrix_f.shape[1] * 100
good_rows = row_coverage >= 25
matrix_g = matrix_f[good_rows]
months_g = [all_months[i] for i in range(len(all_months)) if good_rows[i]]

# Impute
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
for j in range(matrix_g.shape[1]):
    col = matrix_g[:, j]
    mask = np.isnan(col)
    if mask.any():
        med = np.nanmedian(col)
        matrix_g[mask, j] = med if not np.isnan(med) else 0.5

print(f"  最终矩阵: {matrix_g.shape} ({len(months_g)}个月 × {matrix_g.shape[1]}维)")

# ============================================================
# 2. NORMALIZE + PCA + UMAP
# ============================================================
print("\n" + "=" * 60)
print("  归一化 + 降维...")
print("=" * 60)

qt = QuantileTransformer(n_quantiles=min(1000, len(months_g)),
                         output_distribution='uniform', random_state=42)
X_norm = qt.fit_transform(matrix_g)
time_vals = np.linspace(0, 1, len(months_g)).reshape(-1, 1)
X_full = np.hstack([X_norm, time_vals])

n_comp = min(30, X_full.shape[1] - 1, X_full.shape[0] - 1)
pca = PCA(n_components=n_comp)
X_pca = pca.fit_transform(X_full)
explained = np.cumsum(pca.explained_variance_ratio_)
n_keep = np.searchsorted(explained, 0.95) + 1
X_pca = X_pca[:, :n_keep]
print(f"  PCA: {X_full.shape[1]}D → {n_keep}D ({explained[n_keep-1]*100:.1f}%)")

# UMAP
reducer = umap.UMAP(n_neighbors=20, min_dist=0.15, n_components=2, random_state=42)
X_2d = reducer.fit_transform(X_pca)

# NN for topology
nn = NearestNeighbors(n_neighbors=min(10, len(X_pca)-1))
nn.fit(X_pca)
dists, _ = nn.kneighbors(X_pca)
avg_dists = dists.mean(axis=1)
edge_thr = np.percentile(avg_dists, 80)

# ============================================================
# 3. FORWARD PROJECTION — 核心推演逻辑
# ============================================================
print("\n" + "=" * 60)
print("  正向推演分析...")
print("=" * 60)

# Current position = last point
NOW_IDX = len(months_g) - 1
NOW_MONTH = months_g[NOW_IDX]
NOW_PCA = X_pca[NOW_IDX]
print(f"  当前位置: {NOW_MONTH} (index {NOW_IDX})")
print(f"  当前拓扑距离: {avg_dists[NOW_IDX]:.3f} ({'⚠️ 边缘!' if avg_dists[NOW_IDX] > edge_thr else '正常区域'})")

# Find K nearest historical neighbors in PCA space (excluding last 6 months to avoid self)
EXCLUDE_RECENT = 6  # Don't count the most recent 6 months as "historical"
K_NEIGHBORS = 20

historical_pca = X_pca[:NOW_IDX - EXCLUDE_RECENT]
historical_months = months_g[:NOW_IDX - EXCLUDE_RECENT]

nn_hist = NearestNeighbors(n_neighbors=K_NEIGHBORS)
nn_hist.fit(historical_pca)
neighbor_dists, neighbor_idxs = nn_hist.kneighbors(NOW_PCA.reshape(1, -1))
neighbor_dists = neighbor_dists[0]
neighbor_idxs = neighbor_idxs[0]

print(f"\n  与当前状态最相似的{K_NEIGHBORS}个历史时期:")
print(f"  {'排名':>4s}  {'月份':>10s}  {'拓扑距离':>8s}  {'事件':>30s}")
print(f"  {'─'*4}  {'─'*10}  {'─'*8}  {'─'*30}")

# Crisis lookup
crises = {
    "石油危机": ("1973-10", "1974-12"),
    "沃尔克紧缩": ("1980-01", "1982-12"),
    "黑色星期一": ("1987-08", "1988-03"),
    "日本泡沫": ("1990-01", "1992-12"),
    "亚洲金融危机": ("1997-07", "1998-12"),
    "互联网泡沫": ("2000-03", "2002-10"),
    "全球金融危机": ("2007-07", "2009-06"),
    "欧债危机": ("2010-05", "2012-12"),
    "大宗商品危机": ("2014-07", "2016-02"),
    "中美贸易战": ("2018-03", "2019-12"),
    "新冠疫情": ("2020-02", "2020-12"),
    "全球通胀": ("2022-01", "2022-12"),
}

def get_crisis(m):
    m7 = m[:7]
    for cname, (s, e) in crises.items():
        if s <= m7 <= e:
            return cname
    return ""

for rank, (ni, nd) in enumerate(zip(neighbor_idxs, neighbor_dists)):
    m = historical_months[ni]
    crisis = get_crisis(m)
    print(f"  {rank+1:>4d}  {m[:7]:>10s}  {nd:>8.3f}  {crisis:>30s}")

# ============================================================
# 4. TRACE FORWARD FROM EACH NEIGHBOR
# ============================================================
print("\n" + "=" * 60)
print("  从每个类比时期向前追踪...")
print("=" * 60)

FORWARD_MONTHS = 24  # Look 24 months ahead from each analog

# For each neighbor, trace what happened in the next 24 months
forward_paths = []
for ni, nd in zip(neighbor_idxs, neighbor_dists):
    start_idx = ni
    path = []
    for delta in range(FORWARD_MONTHS + 1):
        future_idx = start_idx + delta
        if future_idx < len(months_g):
            path.append({
                'delta': delta,
                'month': months_g[future_idx],
                'pca': X_pca[future_idx],
                'umap': X_2d[future_idx],
                'dist': avg_dists[future_idx],
                'crisis': get_crisis(months_g[future_idx]),
            })
    if path:
        weight = 1.0 / (nd + 0.01)  # Inverse distance weighting
        forward_paths.append({
            'start_month': historical_months[ni],
            'start_dist': nd,
            'weight': weight,
            'path': path,
        })

# Analyze outcomes
print(f"\n  追踪 {len(forward_paths)} 条历史类比路径，每条 {FORWARD_MONTHS} 个月:")

# At each future delta, what's the weighted average topological distance?
# (higher = more likely approaching crisis)
for delta in [3, 6, 12, 18, 24]:
    crisis_count = 0
    total_weight = 0
    weighted_dist = 0
    crisis_names = {}

    for fp in forward_paths:
        w = fp['weight']
        for pt in fp['path']:
            if pt['delta'] == delta:
                total_weight += w
                weighted_dist += w * pt['dist']
                if pt['crisis']:
                    crisis_count += 1
                    crisis_names[pt['crisis']] = crisis_names.get(pt['crisis'], 0) + 1
                break

    if total_weight > 0:
        avg_d = weighted_dist / total_weight
        pct_edge = avg_d / edge_thr * 100
        crisis_str = ", ".join(f"{k}({v})" for k, v in crisis_names.items()) if crisis_names else "无"
        print(f"  +{delta:2d}个月: 加权拓扑距离={avg_d:.3f} (阈值的{pct_edge:.0f}%), "
              f"历史类比中的危机: {crisis_str}")

# ============================================================
# 5. SCENARIO ANALYSIS — 路径聚类
# ============================================================
print("\n" + "=" * 60)
print("  情景分析 — 历史类比路径分类")
print("=" * 60)

# Classify paths by outcome: did they enter a crisis within 24 months?
scenarios = {
    'crisis': [],      # Entered a major crisis
    'stress': [],      # Elevated but no crisis
    'normal': [],      # Remained normal
}

for fp in forward_paths:
    max_dist = max(pt['dist'] for pt in fp['path'])
    any_crisis = any(pt['crisis'] for pt in fp['path'])

    if any_crisis:
        scenarios['crisis'].append(fp)
    elif max_dist > edge_thr:
        scenarios['stress'].append(fp)
    else:
        scenarios['normal'].append(fp)

total_w = sum(fp['weight'] for fp in forward_paths)
for sname, paths in scenarios.items():
    sw = sum(fp['weight'] for fp in paths)
    pct = sw / total_w * 100 if total_w > 0 else 0
    names = {
        'crisis': '🔴 危机情景',
        'stress': '🟡 压力情景',
        'normal': '🟢 正常情景',
    }

    print(f"\n  {names[sname]}: {len(paths)}条路径, 加权概率={pct:.1f}%")

    if paths:
        for fp in paths[:5]:
            crises_hit = set()
            for pt in fp['path']:
                if pt['crisis']:
                    crises_hit.add(pt['crisis'])
            crisis_str = "+".join(crises_hit) if crises_hit else "拓扑边缘但无命名危机"
            print(f"    类比起点: {fp['start_month'][:7]} (距离={fp['start_dist']:.3f})"
                  f" → {crisis_str}")

# ============================================================
# 6. KEY INDICATORS — 哪些指标在类比时期最重要
# ============================================================
print("\n" + "=" * 60)
print("  关键驱动因子分析")
print("=" * 60)

# Compare current state vs historical average to find anomalous dims
current_state = X_norm[NOW_IDX]
historical_mean = np.mean(X_norm[:NOW_IDX - EXCLUDE_RECENT], axis=0)
historical_std = np.std(X_norm[:NOW_IDX - EXCLUDE_RECENT], axis=0) + 1e-10

z_scores = (current_state - historical_mean) / historical_std

# Map back to column names
econ_col_names_f = [c for c in col_names_f if not c.startswith('L_')]
leader_col_names_f = [c for c in col_names_f if c.startswith('L_')]

print(f"\n  当前状态 vs 历史均值 — 最异常的经济指标 (|z| > 1.5):")
anomalies = []
for j, col in enumerate(col_names_f):
    if j < len(z_scores) and abs(z_scores[j]) > 1.5:
        anomalies.append((col, z_scores[j], current_state[j]))

anomalies.sort(key=lambda x: -abs(x[1]))
for col, z, val in anomalies[:20]:
    direction = "↑ 异常高" if z > 0 else "↓ 异常低"
    is_leader = "👤" if col.startswith('L_') else "📊"
    print(f"    {is_leader} {col:45s}  z={z:+6.2f}  ({direction})")

# ============================================================
# 7. INTERACTIVE FORECAST VISUALIZATION
# ============================================================
print("\n" + "=" * 60)
print("  生成交互式推演图...")
print("=" * 60)

# ERA colors
era_colors = {
    '1970s': '#8B4513', '1980s': '#FF6347', '1990s': '#FFD700',
    '2000s': '#32CD32', '2010s': '#4169E1', '2020s': '#9400D3'
}
def get_era(m):
    y = int(m[:4])
    if y < 1980: return '1970s'
    elif y < 1990: return '1980s'
    elif y < 2000: return '1990s'
    elif y < 2010: return '2000s'
    elif y < 2020: return '2010s'
    else: return '2020s'

# Main figure
fig = go.Figure()

# Background: all historical points, faded
eras = [get_era(m) for m in months_g]
for era_name, color in era_colors.items():
    mask = [i for i, e in enumerate(eras) if e == era_name]
    if not mask:
        continue
    fig.add_trace(go.Scatter(
        x=X_2d[mask, 0], y=X_2d[mask, 1],
        mode='markers',
        marker=dict(size=3, color=color, opacity=0.2),
        name=era_name,
        hovertext=[f"{months_g[i][:7]} | {get_crisis(months_g[i]) or '正常'}" for i in mask],
        hoverinfo='text',
    ))

# Time trajectory
fig.add_trace(go.Scatter(
    x=X_2d[:, 0], y=X_2d[:, 1],
    mode='lines',
    line=dict(color='rgba(128,128,128,0.1)', width=0.5),
    showlegend=False, hoverinfo='skip',
))

# Highlight nearest neighbors
nn_x = X_2d[neighbor_idxs, 0]
nn_y = X_2d[neighbor_idxs, 1]
nn_texts = [
    f"<b>类比#{rank+1}: {historical_months[ni][:7]}</b><br>"
    f"拓扑距离: {nd:.3f}<br>"
    f"事件: {get_crisis(historical_months[ni]) or '正常时期'}"
    for rank, (ni, nd) in enumerate(zip(neighbor_idxs, neighbor_dists))
]
fig.add_trace(go.Scatter(
    x=nn_x, y=nn_y,
    mode='markers',
    marker=dict(size=12, color='#00FF00', symbol='circle',
               line=dict(width=2, color='black'), opacity=0.8),
    name='🟢 历史类比点',
    text=nn_texts, hoverinfo='text',
))

# Draw forward paths from each neighbor
scenario_colors = {'crisis': 'red', 'stress': 'orange', 'normal': 'green'}

for sname, paths in scenarios.items():
    color = scenario_colors[sname]
    names_cn = {'crisis': '🔴 危机路径', 'stress': '🟡 压力路径', 'normal': '🟢 正常路径'}

    first = True
    for fp in paths:
        path_x = [pt['umap'][0] for pt in fp['path']]
        path_y = [pt['umap'][1] for pt in fp['path']]

        hover = [
            f"{pt['month'][:7]} (+{pt['delta']}月)<br>"
            f"拓扑距离: {pt['dist']:.3f}<br>"
            f"事件: {pt['crisis'] or '正常'}"
            for pt in fp['path']
        ]

        fig.add_trace(go.Scatter(
            x=path_x, y=path_y,
            mode='lines+markers',
            line=dict(color=color, width=2, dash='dot'),
            marker=dict(size=4, color=color, opacity=0.6),
            name=names_cn[sname] if first else None,
            showlegend=first,
            text=hover, hoverinfo='text',
            opacity=0.5,
        ))
        first = False

        # Arrow at end of path
        if len(path_x) > 1:
            fig.add_annotation(
                x=path_x[-1], y=path_y[-1],
                ax=path_x[-2], ay=path_y[-2],
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=2, arrowsize=1.5,
                arrowcolor=color, arrowwidth=1.5, opacity=0.5,
            )

# Current position (big star)
fig.add_trace(go.Scatter(
    x=[X_2d[NOW_IDX, 0]], y=[X_2d[NOW_IDX, 1]],
    mode='markers+text',
    marker=dict(size=25, color='red', symbol='star',
               line=dict(width=2, color='gold')),
    name='⭐ 你在这里 (2026.03)',
    text=['<b>你在这里</b>'],
    textposition='top center',
    textfont=dict(size=16, color='red'),
    hovertext=(
        f"<b>当前: {NOW_MONTH[:7]}</b><br>"
        f"拓扑距离: {avg_dists[NOW_IDX]:.3f}<br>"
        f"{'⚠️ 拓扑边缘!' if avg_dists[NOW_IDX] > edge_thr else '正常区域'}"
    ),
    hoverinfo='text',
))

# Compute scenario probabilities
total_w = sum(fp['weight'] for fp in forward_paths)
crisis_pct = sum(fp['weight'] for fp in scenarios['crisis']) / total_w * 100 if total_w else 0
stress_pct = sum(fp['weight'] for fp in scenarios['stress']) / total_w * 100 if total_w else 0
normal_pct = sum(fp['weight'] for fp in scenarios['normal']) / total_w * 100 if total_w else 0

# Layout
fig.update_layout(
    title=dict(
        text=(
            f'<b>全球金融状态空间 — 前瞻推演</b><br>'
            f'<sub>基于{K_NEIGHBORS}个最相似历史时期的拓扑类比 | '
            f'向前追踪{FORWARD_MONTHS}个月<br>'
            f'🔴 危机情景: {crisis_pct:.0f}%  '
            f'🟡 压力情景: {stress_pct:.0f}%  '
            f'🟢 正常情景: {normal_pct:.0f}%</sub>'
        ),
        font=dict(size=17),
    ),
    width=1400, height=900,
    template='plotly_white',
    legend=dict(
        yanchor="top", y=0.99, xanchor="right", x=0.99,
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='gray', borderwidth=1,
    ),
    xaxis=dict(title='UMAP-1', showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
    yaxis=dict(title='UMAP-2', showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
)

fig.write_html(OUT_DIR / 'forecast_paths.html')
print(f"  保存: {OUT_DIR / 'forecast_paths.html'}")

# ============================================================
# 8. TIMELINE FORECAST CHART — 未来24个月拓扑距离预测
# ============================================================
print("  生成时间线预测图...")

fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                     subplot_titles=('拓扑距离预测 (加权历史类比)', '历史类比：危机概率'),
                     vertical_spacing=0.12)

# Compute weighted forecast per delta month
deltas = list(range(FORWARD_MONTHS + 1))
forecast_mean = []
forecast_p10 = []
forecast_p90 = []
crisis_prob = []

for delta in deltas:
    dists_at_delta = []
    weights_at_delta = []
    n_crisis = 0
    total_w_d = 0

    for fp in forward_paths:
        for pt in fp['path']:
            if pt['delta'] == delta:
                dists_at_delta.append(pt['dist'])
                weights_at_delta.append(fp['weight'])
                total_w_d += fp['weight']
                if pt['crisis']:
                    n_crisis += fp['weight']
                break

    if dists_at_delta:
        dists_arr = np.array(dists_at_delta)
        w_arr = np.array(weights_at_delta)
        w_arr = w_arr / w_arr.sum()

        mean_d = np.average(dists_arr, weights=w_arr)
        # Weighted percentiles (approximate)
        sorted_idx = np.argsort(dists_arr)
        sorted_d = dists_arr[sorted_idx]
        sorted_w = np.cumsum(w_arr[sorted_idx])
        p10 = sorted_d[np.searchsorted(sorted_w, 0.1)]
        p90 = sorted_d[np.searchsorted(sorted_w, 0.9, side='right')]

        forecast_mean.append(mean_d)
        forecast_p10.append(p10)
        forecast_p90.append(p90)
        crisis_prob.append(n_crisis / total_w_d * 100 if total_w_d > 0 else 0)
    else:
        forecast_mean.append(np.nan)
        forecast_p10.append(np.nan)
        forecast_p90.append(np.nan)
        crisis_prob.append(0)

# Future month labels
import datetime
base = datetime.datetime(2026, 3, 1)
future_labels = [(base + datetime.timedelta(days=30*d)).strftime('%Y-%m') for d in deltas]

# Plot 1: Topological distance forecast
fig2.add_trace(go.Scatter(
    x=future_labels, y=forecast_p90,
    mode='lines', line=dict(width=0), showlegend=False,
    hoverinfo='skip',
), row=1, col=1)

fig2.add_trace(go.Scatter(
    x=future_labels, y=forecast_p10,
    mode='lines', line=dict(width=0),
    fill='tonexty', fillcolor='rgba(100,149,237,0.2)',
    name='10-90%区间',
    hoverinfo='skip',
), row=1, col=1)

fig2.add_trace(go.Scatter(
    x=future_labels, y=forecast_mean,
    mode='lines+markers',
    line=dict(color='#4169E1', width=3),
    marker=dict(size=6),
    name='加权均值',
    hovertext=[
        f"<b>{future_labels[i]}</b><br>"
        f"拓扑距离: {forecast_mean[i]:.3f}<br>"
        f"10%分位: {forecast_p10[i]:.3f}<br>"
        f"90%分位: {forecast_p90[i]:.3f}"
        for i in range(len(deltas))
    ],
    hoverinfo='text',
), row=1, col=1)

# Danger threshold
fig2.add_hline(y=edge_thr, line_dash="dash", line_color="red",
               annotation_text="⚠️ 拓扑瓶颈阈值",
               annotation_position="top left",
               row=1, col=1)

# Plot 2: Crisis probability
bar_colors = ['#FF0000' if p > 30 else '#FF8C00' if p > 10 else '#4169E1' for p in crisis_prob]
fig2.add_trace(go.Bar(
    x=future_labels, y=crisis_prob,
    marker_color=bar_colors,
    name='危机概率',
    hovertext=[
        f"<b>{future_labels[i]}</b><br>"
        f"历史类比危机概率: {crisis_prob[i]:.1f}%"
        for i in range(len(deltas))
    ],
    hoverinfo='text',
), row=2, col=1)

fig2.update_layout(
    title=dict(
        text=(
            f'<b>未来{FORWARD_MONTHS}个月推演 — 基于拓扑历史类比</b><br>'
            f'<sub>从{K_NEIGHBORS}个最相似历史时期的后续演化推断 | '
            f'逆距离加权 | 阴影=10-90%置信区间</sub>'
        ),
        font=dict(size=16),
    ),
    width=1200, height=700,
    template='plotly_white',
    showlegend=True,
)

fig2.update_yaxes(title_text='拓扑距离', row=1, col=1)
fig2.update_yaxes(title_text='危机概率 (%)', row=2, col=1)
fig2.update_xaxes(title_text='月份', row=2, col=1)

fig2.write_html(OUT_DIR / 'forecast_timeline.html')
print(f"  保存: {OUT_DIR / 'forecast_timeline.html'}")

# ============================================================
# 9. SCENARIO DETAIL TABLE
# ============================================================
print("\n" + "=" * 60)
print("  情景详细报告")
print("=" * 60)

# Build HTML report
report_lines = [
    '<!DOCTYPE html><html><head><meta charset="utf-8">',
    '<title>TDA前瞻推演报告</title>',
    '<style>',
    'body { font-family: "Microsoft YaHei", "Segoe UI", sans-serif; max-width: 1000px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }',
    'h1 { color: #333; border-bottom: 3px solid #4169E1; padding-bottom: 10px; }',
    'h2 { color: #4169E1; margin-top: 30px; }',
    '.card { background: white; border-radius: 10px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }',
    '.crisis { border-left: 5px solid #FF0000; }',
    '.stress { border-left: 5px solid #FF8C00; }',
    '.normal { border-left: 5px solid #32CD32; }',
    '.prob { font-size: 2em; font-weight: bold; }',
    '.red { color: #FF0000; } .orange { color: #FF8C00; } .green { color: #32CD32; }',
    'table { border-collapse: collapse; width: 100%; margin: 10px 0; }',
    'th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }',
    'th { background: #4169E1; color: white; }',
    'tr:nth-child(even) { background: #f9f9f9; }',
    '.indicator { display: inline-block; padding: 3px 8px; border-radius: 4px; margin: 2px; font-size: 0.85em; }',
    '.high { background: #FFE0E0; color: #CC0000; }',
    '.low { background: #E0F0FF; color: #0066CC; }',
    '</style></head><body>',
    f'<h1>TDA 前瞻推演报告</h1>',
    f'<p>生成时间: 2026-03-12 | 当前位置: {NOW_MONTH[:7]} | 推演窗口: {FORWARD_MONTHS}个月</p>',
    '',
    '<h2>一、情景概率</h2>',
    '<div style="display:flex; gap:20px;">',
    f'<div class="card crisis" style="flex:1;text-align:center;"><div class="prob red">{crisis_pct:.0f}%</div><div>🔴 危机情景<br>24个月内进入已知危机区域</div></div>',
    f'<div class="card stress" style="flex:1;text-align:center;"><div class="prob orange">{stress_pct:.0f}%</div><div>🟡 压力情景<br>拓扑距离升高但未触发命名危机</div></div>',
    f'<div class="card normal" style="flex:1;text-align:center;"><div class="prob green">{normal_pct:.0f}%</div><div>🟢 正常情景<br>保持在正常拓扑区域内</div></div>',
    '</div>',
    '',
    '<h2>二、最相似历史时期 (TOP 10)</h2>',
    '<div class="card">',
    '<table><tr><th>排名</th><th>月份</th><th>拓扑距离</th><th>当时事件</th><th>之后发生</th></tr>',
]

for rank, (ni, nd) in enumerate(zip(neighbor_idxs[:10], neighbor_dists[:10])):
    m = historical_months[ni]
    crisis_now = get_crisis(m) or "正常时期"
    # What happened next?
    future_crises = set()
    for delta in range(1, min(25, len(months_g) - ni)):
        fc = get_crisis(months_g[ni + delta])
        if fc:
            future_crises.add(fc)
    future_str = ", ".join(future_crises) if future_crises else "持续正常"

    report_lines.append(
        f'<tr><td>{rank+1}</td><td>{m[:7]}</td><td>{nd:.3f}</td>'
        f'<td>{crisis_now}</td><td>{future_str}</td></tr>'
    )

report_lines.extend([
    '</table></div>',
    '',
    '<h2>三、当前异常指标</h2>',
    '<div class="card">',
    '<p>以下指标显著偏离历史均值 (|z-score| > 1.5):</p>',
])

for col, z, val in anomalies[:15]:
    cls = 'high' if z > 0 else 'low'
    direction = '偏高' if z > 0 else '偏低'
    icon = '📊' if not col.startswith('L_') else '👤'
    report_lines.append(
        f'<span class="indicator {cls}">{icon} {col}: z={z:+.2f} ({direction})</span>'
    )

report_lines.extend([
    '</div>',
    '',
    '<h2>四、关键时间窗口</h2>',
    '<div class="card">',
    '<table><tr><th>时间</th><th>拓扑距离(均值)</th><th>距阈值%</th><th>危机概率</th><th>评估</th></tr>',
])

for i, delta in enumerate([3, 6, 12, 18, 24]):
    if delta < len(forecast_mean):
        d = forecast_mean[delta]
        pct = d / edge_thr * 100
        cp = crisis_prob[delta]
        if cp > 30:
            assessment = '🔴 高风险'
        elif cp > 10 or pct > 80:
            assessment = '🟡 需关注'
        else:
            assessment = '🟢 正常'

        report_lines.append(
            f'<tr><td>+{delta}个月 ({future_labels[delta]})</td>'
            f'<td>{d:.3f}</td><td>{pct:.0f}%</td>'
            f'<td>{cp:.1f}%</td><td>{assessment}</td></tr>'
        )

report_lines.extend([
    '</table></div>',
    '',
    '<h2>五、方法论说明</h2>',
    '<div class="card">',
    '<p><b>这不是预测，是拓扑历史类比。</b></p>',
    '<ul>',
    '<li>在73维状态空间（经济数据+领导人行为参数）中找到当前状态的20个最近邻</li>',
    '<li>追踪这些历史类比点之后24个月的实际演化路径</li>',
    '<li>用拓扑距离的倒数加权（越相似的历史时期权重越大）</li>',
    '<li>危机概率 = 历史类比路径中进入已知危机区域的加权比例</li>',
    '</ul>',
    '<p><b>局限性：</b></p>',
    '<ul>',
    '<li>假设拓扑结构（状态空间的形状）未来仍然成立</li>',
    '<li>不包含未见过的黑天鹅事件（但历史类比自然包含极端情景）</li>',
    '<li>领导人行为参数是静态的（未来可能变化）</li>',
    '<li>这是Phase 1的初步推演，完整Monte Carlo引擎待Phase 3实现</li>',
    '</ul>',
    '</div>',
    '</body></html>',
])

report_path = OUT_DIR / 'forecast_report.html'
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))
print(f"  保存: {report_path}")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
print(f"  推演完成！输出文件:")
print(f"  1. forecast_paths.html    — 状态空间上的推演路径图")
print(f"  2. forecast_timeline.html — 未来24个月拓扑距离+危机概率")
print(f"  3. forecast_report.html   — 完整推演报告")
print(f"  ")
print(f"  情景概率:")
print(f"    🔴 危机: {crisis_pct:.0f}%")
print(f"    🟡 压力: {stress_pct:.0f}%")
print(f"    🟢 正常: {normal_pct:.0f}%")
print(f"{'='*60}")
