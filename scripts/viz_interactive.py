#!/usr/bin/env python3
"""
Interactive TDA Visualization — 交互式全球金融状态空间可视化
=============================================================
用 Plotly 生成可缩放、悬停查看细节的交互式图表。
中文标注，直观易懂。
"""

import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import umap
import warnings
warnings.filterwarnings('ignore')

DATA_BASE = Path('/home/user/global-financial-sim/data')
ECON_BASE = DATA_BASE / 'economic'
LEADER_BASE = DATA_BASE / 'leaders'
OUT_DIR = Path('/home/user/global-financial-sim/output/tda_full')
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'plotly', '-q'])
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px

# ============================================================
# Reuse data loading from tda_full.py (run it first!)
# ============================================================
meta_file = OUT_DIR / 'metadata.json'
if meta_file.exists():
    with open(meta_file) as f:
        metadata = json.load(f)
    print(f"Loaded metadata: {metadata.get('final_shape', 'unknown')}")

# ============================================================
# 1. LOAD ALL DATA (same logic as tda_full.py)
# ============================================================
print("Loading data...")

import unicodedata

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

# Indices (daily → monthly avg)
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
            all_series[f"comm_{prefix}_{name}"] = {k: np.mean(v) if isinstance(v, list) else v for k, v in monthly.items()}

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

print(f"Loaded {len(all_series)} economic series")

# Months
all_months_set = set()
for s in all_series.values():
    all_months_set.update(s.keys())
all_months = sorted([m for m in all_months_set if '1964-01-01' <= m <= '2026-12-01'])
print(f"Time range: {all_months[0]} to {all_months[-1]} ({len(all_months)} months)")

# Economic matrix
econ_cols = sorted(all_series.keys())
econ_matrix = np.full((len(all_months), len(econ_cols)), np.nan)
for j, col in enumerate(econ_cols):
    for i, month in enumerate(all_months):
        if month in all_series[col]:
            econ_matrix[i, j] = all_series[col][month]

# ============================================================
# 2. LOAD LEADERS (same logic as tda_full.py)
# ============================================================
print("Loading leaders...")

profiles = {}
prof_dir = LEADER_BASE / 'profiles'
if prof_dir.exists():
    for f in sorted(prof_dir.glob('*.json')):
        with open(f) as fh:
            pdata = json.load(fh)
        params = {}
        # Try multiple structures
        # Structure 1: behavior_matrix.vectors[].{label, value}
        bm = pdata.get('behavior_matrix', {})
        if isinstance(bm, dict):
            for v in bm.get('vectors', []):
                val = v.get('value')
                label = v.get('label', v.get('name', ''))
                if isinstance(val, (int, float)) and label:
                    params[label] = float(val)
        # Structure 2: behavioral_parameters[].parameters[].{name, value}
        for section in pdata.get('behavioral_parameters', []):
            if isinstance(section, dict):
                for p in section.get('parameters', []):
                    val = p.get('value')
                    if isinstance(val, (int, float)):
                        params[p.get('name', '')] = float(val)
        if params:
            profiles[f.stem] = {'params': params, 'data': pdata}

print(f"Loaded {len(profiles)} profiles")

# Count dims
all_dims = {}
for pk, pv in profiles.items():
    for d in pv['params']:
        all_dims[d] = all_dims.get(d, 0) + 1

top_dims = sorted(all_dims.keys(), key=lambda x: -all_dims[x])[:40]

# Timelines and role matching
def normalize_name(name):
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    name = name.lower().strip().replace('.', '').replace('-', ' ')
    parts = name.split()
    return '_'.join(parts)

name_to_profile = {}
for pk in profiles:
    name_to_profile[pk] = pk

# Special mappings
special_mappings = {
    'hu_jintao': 'hu_wen', 'wen_jiabao': 'hu_wen',
    'leonid_brezhnev': 'brezhnev', 'nikita_khrushchev': 'khrushchev',
    'joseph_stalin': 'stalin', 'vladimir_lenin': 'lenin',
}
name_to_profile.update(special_mappings)

# Load timelines
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
    role_lower = role_str.lower().replace('-', '_').replace(' ', '_')
    for slot, keywords in ROLE_SLOTS.items():
        slot_country = slot.split('_')[0]
        if country != slot_country:
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
                    for role_name, role_data in val.items():
                        if isinstance(role_data, dict) and 'leaders' in role_data:
                            for leader in role_data['leaders']:
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

            if len(start) == 7:
                start += '-01'
            if len(end) == 7:
                end += '-01'

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

print(f"Leader periods: {len(leader_periods)}")

# Role assignments
role_assignments = {}
for month in all_months:
    role_assignments[month] = {}
    for pk, start, end, role, country in leader_periods:
        if start <= month <= end:
            slot = match_role_slot(role, country)
            if slot:
                role_assignments[month][slot] = pk

slots_used = set()
for month in all_months:
    for slot in role_assignments[month]:
        slots_used.add(slot)
slots_used = sorted(slots_used)

N_LEADER_DIMS = 15
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

# Filter columns
coverage = np.sum(~np.isnan(matrix), axis=0) / len(all_months) * 100
good_cols = coverage >= 8
matrix_f = matrix[:, good_cols]
col_names_f = [all_col_names[j] for j in range(len(all_col_names)) if good_cols[j]]

# Filter rows
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

print(f"Final matrix: {matrix_g.shape} ({len(months_g)} months × {matrix_g.shape[1]} dims)")

# ============================================================
# 3. NORMALIZE + UMAP
# ============================================================
print("Normalizing + UMAP...")

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

# UMAP to 2D
reducer = umap.UMAP(n_neighbors=20, min_dist=0.15, n_components=2, random_state=42, metric='euclidean')
X_2d = reducer.fit_transform(X_pca)

# Also 3D for extra depth
reducer_3d = umap.UMAP(n_neighbors=20, min_dist=0.15, n_components=3, random_state=42, metric='euclidean')
X_3d = reducer_3d.fit_transform(X_pca)

# NN distances for bottleneck detection
nn = NearestNeighbors(n_neighbors=min(10, len(X_pca)-1))
nn.fit(X_pca)
dists, _ = nn.kneighbors(X_pca)
avg_dists = dists.mean(axis=1)
edge_thr = np.percentile(avg_dists, 80)

print(f"UMAP complete. Edge threshold: {edge_thr:.3f}")

# ============================================================
# 4. PREPARE HOVER DATA
# ============================================================

# Crisis labels
crises = {
    "石油危机 1973-74": ("1973-10", "1974-12"),
    "沃尔克紧缩 1980-82": ("1980-01", "1982-12"),
    "黑色星期一 1987": ("1987-08", "1988-03"),
    "日本泡沫 1990-92": ("1990-01", "1992-12"),
    "亚洲金融危机 1997-98": ("1997-07", "1998-12"),
    "互联网泡沫 2000-02": ("2000-03", "2002-10"),
    "全球金融危机 2007-09": ("2007-07", "2009-06"),
    "欧债危机 2010-12": ("2010-05", "2012-12"),
    "大宗商品危机 2014-16": ("2014-07", "2016-02"),
    "中美贸易战 2018-19": ("2018-03", "2019-12"),
    "新冠疫情 2020": ("2020-02", "2020-12"),
    "全球通胀 2022": ("2022-01", "2022-12"),
    "特朗普2.0 2025": ("2025-01", "2026-03"),
}

# Assign crisis label to each month
crisis_label = []
for m in months_g:
    m7 = m[:7]
    label = "正常时期"
    for cname, (s, e) in crises.items():
        if s <= m7 <= e:
            label = cname
            break
    crisis_label.append(label)

# Leader info per month
leader_info = []
for m in months_g:
    assignments = role_assignments.get(m, {})
    parts = []
    slot_names_cn = {
        'US_PRES': '美总统', 'US_FED': '美联储', 'CN_LEADER': '中国领导人',
        'CN_PREMIER': '中国总理', 'CN_PBOC': '中国央行', 'EU_ECB': '欧央行',
        'EU_LEADER': '欧洲领导', 'UK_PM': '英首相', 'UK_BOE': '英央行',
        'JP_PM': '日首相', 'JP_BOJ': '日央行', 'RU_LEADER': '俄领导人',
        'RU_CB': '俄央行', 'IR_LEADER': '伊朗领袖', 'IR_PRES': '伊朗总统',
    }
    for slot in ['US_PRES', 'US_FED', 'CN_LEADER', 'EU_ECB', 'JP_PM', 'RU_LEADER']:
        if slot in assignments:
            pk = assignments[slot]
            cn_slot = slot_names_cn.get(slot, slot)
            parts.append(f"{cn_slot}: {pk}")
    leader_info.append('<br>'.join(parts) if parts else '无数据')

# Decade colors
def get_era(m):
    y = int(m[:4])
    if y < 1980: return '1970s'
    elif y < 1990: return '1980s'
    elif y < 2000: return '1990s'
    elif y < 2010: return '2000s'
    elif y < 2020: return '2010s'
    else: return '2020s'

eras = [get_era(m) for m in months_g]

# ============================================================
# 5. INTERACTIVE 2D VISUALIZATION
# ============================================================
print("Building interactive 2D visualization...")

# Color map for eras
era_colors = {
    '1970s': '#8B4513', '1980s': '#FF6347', '1990s': '#FFD700',
    '2000s': '#32CD32', '2010s': '#4169E1', '2020s': '#9400D3'
}

# Main figure
fig = go.Figure()

# Plot by era (background)
for era_name in ['1970s', '1980s', '1990s', '2000s', '2010s', '2020s']:
    mask = [i for i, e in enumerate(eras) if e == era_name]
    if not mask:
        continue

    hover_texts = []
    for i in mask:
        m = months_g[i]
        yr_month = m[:7]
        dist_val = avg_dists[i]
        is_edge = "⚠️ 拓扑边缘!" if dist_val > edge_thr else "正常区域"
        cr = crisis_label[i]
        li = leader_info[i]
        hover_texts.append(
            f"<b>{yr_month}</b><br>"
            f"状态: {cr}<br>"
            f"拓扑距离: {dist_val:.3f} ({is_edge})<br>"
            f"<b>在位领导人:</b><br>{li}"
        )

    fig.add_trace(go.Scatter(
        x=X_2d[mask, 0], y=X_2d[mask, 1],
        mode='markers',
        marker=dict(size=5, color=era_colors[era_name], opacity=0.6),
        name=era_name,
        text=hover_texts,
        hoverinfo='text',
    ))

# Time trajectory (connect consecutive points with line)
fig.add_trace(go.Scatter(
    x=X_2d[:, 0], y=X_2d[:, 1],
    mode='lines',
    line=dict(color='rgba(128,128,128,0.15)', width=0.5),
    name='时间轨迹',
    hoverinfo='skip',
    showlegend=False,
))

# Highlight crises
crisis_colors = {
    "全球金融危机 2007-09": "#FF0000",
    "新冠疫情 2020": "#FF4500",
    "中美贸易战 2018-19": "#FF8C00",
    "大宗商品危机 2014-16": "#FFD700",
    "全球通胀 2022": "#FF69B4",
    "互联网泡沫 2000-02": "#00CED1",
    "欧债危机 2010-12": "#1E90FF",
    "亚洲金融危机 1997-98": "#9370DB",
    "特朗普2.0 2025": "#FF1493",
}

for cname, color in crisis_colors.items():
    mask = [i for i, c in enumerate(crisis_label) if c == cname]
    if not mask:
        continue

    hover_texts = []
    for i in mask:
        m = months_g[i]
        dist_val = avg_dists[i]
        is_edge = "⚠️ 拓扑边缘!" if dist_val > edge_thr else ""
        li = leader_info[i]
        hover_texts.append(
            f"<b>{m[:7]} — {cname}</b><br>"
            f"拓扑距离: {dist_val:.3f} {is_edge}<br>"
            f"<b>在位领导人:</b><br>{li}"
        )

    fig.add_trace(go.Scatter(
        x=X_2d[mask, 0], y=X_2d[mask, 1],
        mode='markers',
        marker=dict(size=10, color=color, symbol='diamond', opacity=0.9,
                   line=dict(width=1, color='black')),
        name=f'🔴 {cname}',
        text=hover_texts,
        hoverinfo='text',
    ))

    # Label at center
    cx = np.mean(X_2d[mask, 0])
    cy = np.mean(X_2d[mask, 1])
    fig.add_annotation(
        x=cx, y=cy,
        text=f"<b>{cname}</b>",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
        font=dict(size=11, color=color),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor=color, borderwidth=1, borderpad=3,
    )

# Leader transitions
transitions = []
for slot in slots_used:
    prev_leader = None
    for i, month in enumerate(months_g):
        current = role_assignments.get(month, {}).get(slot)
        if current != prev_leader and prev_leader is not None and current is not None:
            transitions.append((i, month, slot, prev_leader, current))
        prev_leader = current

slot_cn = {
    'US_PRES': '美总统', 'US_FED': '美联储', 'CN_LEADER': '中国领导人',
    'CN_PREMIER': '中国总理', 'CN_PBOC': '中国央行', 'EU_ECB': '欧央行',
    'EU_LEADER': '欧洲领导', 'UK_PM': '英首相', 'UK_BOE': '英央行',
    'JP_PM': '日首相', 'JP_BOJ': '日央行', 'RU_LEADER': '俄领导人',
    'RU_CB': '俄央行', 'IR_LEADER': '伊朗领袖',
}

trans_x, trans_y, trans_text, trans_colors = [], [], [], []
for idx, month, slot, old, new in transitions:
    dist = avg_dists[idx]
    is_edge = dist > edge_thr
    color = 'red' if is_edge else 'rgba(0,100,255,0.7)'
    edge_str = "⚠️ 在拓扑瓶颈处!" if is_edge else "正常过渡"

    trans_x.append(X_2d[idx, 0])
    trans_y.append(X_2d[idx, 1])
    trans_colors.append(color)
    trans_text.append(
        f"<b>领导人交接 {month[:7]}</b><br>"
        f"{slot_cn.get(slot, slot)}: {old} → {new}<br>"
        f"拓扑距离: {dist:.3f}<br>"
        f"<b>{edge_str}</b>"
    )

fig.add_trace(go.Scatter(
    x=trans_x, y=trans_y,
    mode='markers',
    marker=dict(size=14, color=trans_colors, symbol='triangle-up',
               line=dict(width=2, color='black')),
    name='▲ 领导人交接',
    text=trans_text,
    hoverinfo='text',
))

# "You are here" marker (latest point)
fig.add_trace(go.Scatter(
    x=[X_2d[-1, 0]], y=[X_2d[-1, 1]],
    mode='markers+text',
    marker=dict(size=20, color='red', symbol='star',
               line=dict(width=2, color='gold')),
    name='⭐ 现在位置',
    text=['<b>你在这里<br>(2026.03)</b>'],
    textposition='top center',
    textfont=dict(size=14, color='red'),
    hoverinfo='text',
    hovertext=f"<b>当前位置: {months_g[-1][:7]}</b><br>"
              f"拓扑距离: {avg_dists[-1]:.3f}<br>"
              f"{leader_info[-1]}",
))

fig.update_layout(
    title=dict(
        text='<b>全球金融状态空间 — 经济数据 + 领导人行为参数 TDA</b><br>'
             f'<sub>{matrix_g.shape[1]}维状态空间 → {n_keep}维PCA → UMAP投影 | '
             f'{len(months_g)}个月 ({months_g[0][:7]} ~ {months_g[-1][:7]}) | '
             f'{len(profiles)}个领导人档案</sub>',
        font=dict(size=18),
    ),
    width=1400, height=900,
    template='plotly_white',
    legend=dict(
        yanchor="top", y=0.99, xanchor="right", x=0.99,
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='gray', borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(title='UMAP-1', showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
    yaxis=dict(title='UMAP-2', showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
    hoverlabel=dict(bgcolor='white', font_size=12),
)

fig.write_html(OUT_DIR / 'state_space_interactive.html')
print(f"Saved: {OUT_DIR / 'state_space_interactive.html'}")

# ============================================================
# 6. 3D INTERACTIVE (extra depth)
# ============================================================
print("Building 3D visualization...")

fig3d = go.Figure()

for era_name in ['1970s', '1980s', '1990s', '2000s', '2010s', '2020s']:
    mask = [i for i, e in enumerate(eras) if e == era_name]
    if not mask:
        continue
    hover_texts = []
    for i in mask:
        m = months_g[i]
        cr = crisis_label[i]
        dist_val = avg_dists[i]
        hover_texts.append(f"{m[:7]} | {cr} | dist={dist_val:.3f}")

    fig3d.add_trace(go.Scatter3d(
        x=X_3d[mask, 0], y=X_3d[mask, 1], z=X_3d[mask, 2],
        mode='markers',
        marker=dict(size=3, color=era_colors[era_name], opacity=0.5),
        name=era_name,
        text=hover_texts,
        hoverinfo='text',
    ))

# Crisis points
for cname, color in crisis_colors.items():
    mask = [i for i, c in enumerate(crisis_label) if c == cname]
    if not mask:
        continue
    hover_texts = [f"{months_g[i][:7]} | {cname}" for i in mask]
    fig3d.add_trace(go.Scatter3d(
        x=X_3d[mask, 0], y=X_3d[mask, 1], z=X_3d[mask, 2],
        mode='markers',
        marker=dict(size=6, color=color, symbol='diamond', opacity=0.9),
        name=f'{cname}',
        text=hover_texts,
        hoverinfo='text',
    ))

# Time trajectory
fig3d.add_trace(go.Scatter3d(
    x=X_3d[:, 0], y=X_3d[:, 1], z=X_3d[:, 2],
    mode='lines',
    line=dict(color='rgba(128,128,128,0.2)', width=1),
    name='时间轨迹',
    hoverinfo='skip',
    showlegend=False,
))

# You are here
fig3d.add_trace(go.Scatter3d(
    x=[X_3d[-1, 0]], y=[X_3d[-1, 1]], z=[X_3d[-1, 2]],
    mode='markers+text',
    marker=dict(size=10, color='red', symbol='diamond'),
    name='⭐ 现在',
    text=['你在这里'],
    hoverinfo='text',
))

fig3d.update_layout(
    title=dict(
        text='<b>全球金融状态空间 — 3D视图</b><br>'
             '<sub>可拖拽旋转。危机点用菱形标注。灰线是时间轨迹。</sub>',
        font=dict(size=16),
    ),
    width=1200, height=800,
    scene=dict(
        xaxis_title='UMAP-1',
        yaxis_title='UMAP-2',
        zaxis_title='UMAP-3',
    ),
)

fig3d.write_html(OUT_DIR / 'state_space_3d.html')
print(f"Saved: {OUT_DIR / 'state_space_3d.html'}")

# ============================================================
# 7. BOTTLENECK BAR CHART (简洁的危机对比图)
# ============================================================
print("Building bottleneck chart...")

crisis_stats = []
for cname, (s, e) in crises.items():
    mask = [i for i, m in enumerate(months_g) if s <= m[:7] <= e]
    if mask:
        edge_pct = np.mean([avg_dists[i] > edge_thr for i in mask]) * 100
        mean_dist = np.mean([avg_dists[i] for i in mask])
        crisis_stats.append({'name': cname, 'edge_pct': edge_pct, 'mean_dist': mean_dist, 'n': len(mask)})

crisis_stats.sort(key=lambda x: -x['edge_pct'])

fig_bar = go.Figure()
colors_bar = ['#FF0000' if s['edge_pct'] > 50 else '#FF8C00' if s['edge_pct'] > 20 else '#4169E1'
              for s in crisis_stats]

fig_bar.add_trace(go.Bar(
    y=[s['name'] for s in crisis_stats],
    x=[s['edge_pct'] for s in crisis_stats],
    orientation='h',
    marker_color=colors_bar,
    text=[f"{s['edge_pct']:.0f}% ({s['n']}个月)" for s in crisis_stats],
    textposition='outside',
    hovertext=[f"{s['name']}<br>拓扑边缘率: {s['edge_pct']:.1f}%<br>"
               f"平均拓扑距离: {s['mean_dist']:.3f}<br>"
               f"持续: {s['n']}个月" for s in crisis_stats],
    hoverinfo='text',
))

fig_bar.update_layout(
    title=dict(
        text='<b>历次危机的拓扑异常程度对比</b><br>'
             '<sub>拓扑边缘率 = 该时期有多少比例的月份落在状态空间边缘。越高 = 越异常。</sub>',
        font=dict(size=16),
    ),
    xaxis=dict(title='拓扑边缘率 (%)', range=[0, 100]),
    yaxis=dict(autorange='reversed'),
    width=900, height=500,
    template='plotly_white',
    margin=dict(l=200),
)

fig_bar.write_html(OUT_DIR / 'crisis_bottleneck.html')
print(f"Saved: {OUT_DIR / 'crisis_bottleneck.html'}")

# ============================================================
# 8. LEADER TRANSITION IMPACT (领导人交接冲击图)
# ============================================================
print("Building leader transition chart...")

trans_data = []
for idx, month, slot, old, new in transitions:
    dist = avg_dists[idx]
    trans_data.append({
        'month': month[:7],
        'slot': slot_cn.get(slot, slot),
        'old': old,
        'new': new,
        'dist': dist,
        'is_edge': dist > edge_thr,
        'label': f"{old} → {new}",
    })

trans_data.sort(key=lambda x: -x['dist'])

fig_trans = go.Figure()
fig_trans.add_trace(go.Bar(
    y=[f"{t['month']} {t['slot']}" for t in trans_data],
    x=[t['dist'] for t in trans_data],
    orientation='h',
    marker_color=['#FF0000' if t['is_edge'] else '#4169E1' for t in trans_data],
    text=[t['label'] for t in trans_data],
    textposition='outside',
    textfont=dict(size=10),
    hovertext=[f"<b>{t['month']} {t['slot']}</b><br>"
               f"{t['old']} → {t['new']}<br>"
               f"拓扑距离: {t['dist']:.3f}<br>"
               f"{'⚠️ 在拓扑瓶颈处!' if t['is_edge'] else '正常过渡'}"
               for t in trans_data],
    hoverinfo='text',
))

fig_trans.add_vline(x=edge_thr, line_dash="dash", line_color="red",
                    annotation_text="瓶颈阈值", annotation_position="top right")

fig_trans.update_layout(
    title=dict(
        text='<b>领导人交接对状态空间的冲击</b><br>'
             '<sub>红色 = 交接发生在拓扑瓶颈处（=危机时刻）。蓝色 = 正常过渡。</sub>',
        font=dict(size=16),
    ),
    xaxis=dict(title='拓扑距离（越大=状态空间跳跃越大）'),
    yaxis=dict(autorange='reversed'),
    width=1000, height=max(400, len(trans_data) * 35),
    template='plotly_white',
    margin=dict(l=200),
)

fig_trans.write_html(OUT_DIR / 'leader_transitions.html')
print(f"Saved: {OUT_DIR / 'leader_transitions.html'}")

# ============================================================
# DONE
# ============================================================
print(f"\n{'='*60}")
print(f"  所有可视化已保存到: {OUT_DIR}/")
print(f"  1. state_space_interactive.html — 2D交互式状态空间图")
print(f"  2. state_space_3d.html          — 3D可旋转状态空间图")
print(f"  3. crisis_bottleneck.html       — 危机拓扑异常度对比")
print(f"  4. leader_transitions.html      — 领导人交接冲击图")
print(f"{'='*60}")
print(f"用浏览器打开 .html 文件即可交互查看！")
