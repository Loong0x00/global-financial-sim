#!/usr/bin/env python3
"""
TDA Feature Analysis — 分析 H1 环路和 H2 空洞的经济含义
=========================================================
对全球 6 经济体 71 指标的 TDA 结果进行深度解读：
- 提取每个拓扑特征涉及的数据点（通过 ripser cocycles）
- 将数据点映射回具体月份和经济状态
- 解读每个 H1 环路和 H2 空洞的经济含义
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from ripser import ripser
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 数据准备（复用 tda_global.py 的逻辑）
# ============================================================

print("=" * 70)
print("TDA 特征分析 — H1 环路与 H2 空洞的经济解读")
print("=" * 70)

DATA_BASE = Path('/home/user/global-financial-sim/data/economic')
OUT_DIR = Path('/home/user/global-financial-sim/output/tda_global')
OUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTRIES = ['us', 'cn', 'eu', 'uk', 'jp', 'ru']

# 加载所有数据
all_series = {}
country_dims = {}
# 保存人类可读的指标名称映射
INDICATOR_LABELS = {
    'US_gdp_growth': '美国GDP增长率', 'US_cpi_yoy': '美国CPI同比', 'US_core_cpi_yoy': '美国核心CPI同比',
    'US_fed_funds_rate': '美联储基准利率', 'US_unemployment': '美国失业率',
    'US_nonfarm_payrolls_change': '美国非农就业变化', 'US_pce_yoy': '美国PCE同比',
    'US_core_pce_yoy': '美国核心PCE同比', 'US_consumer_sentiment': '美国消费者信心',
    'US_manufacturing_employment': '美国制造业就业', 'US_industrial_production': '美国工业生产',
    'US_retail_sales': '美国零售销售', 'US_home_price_index': '美国房价指数',
    'US_sp500': '标普500', 'US_treasury_10y': '美国10年国债', 'US_treasury_2y': '美国2年国债',
    'US_vix': 'VIX恐慌指数', 'US_usd_index': '美元指数', 'US_credit_spread': '信用利差',
    'US_m2_money_supply': '美国M2', 'US_fed_total_assets': '美联储总资产',
    'US_yield_curve_10y2y': '美国收益率曲线(10Y-2Y)',
    'CN_gdp_growth': '中国GDP增长率', 'CN_gdp_growth_alt': '中国GDP增长率(替代)',
    'CN_cpi_yoy': '中国CPI同比', 'CN_cpi_index': '中国CPI指数',
    'CN_discount_rate': '中国贴现率', 'CN_unemployment_alt': '中国失业率(替代)',
    'CN_m2': '中国M2', 'CN_m1': '中国M1', 'CN_exports': '中国出口',
    'CN_imports': '中国进口', 'CN_population': '中国人口',
    'CN_current_account_gdp': '中国经常账户/GDP', 'CN_usd_cny': '美元/人民币',
    'CN_fx_reserves': '中国外汇储备', 'CN_short_rate': '中国短期利率',
    'CN_share_prices': '中国股市', 'CN_reer': '人民币实际有效汇率',
    'EU_gdp_growth': '欧元区GDP增长率', 'EU_hicp_all_items': '欧元区HICP通胀',
    'EU_interest_rate': '欧央行利率', 'EU_unemployment': '欧元区失业率',
    'EU_m3_money_supply': '欧元区M3', 'EU_industrial_production': '欧元区工业生产',
    'EU_eur_usd': '欧元/美元', 'EU_germany_10y_bond': '德国10年国债',
    'UK_gdp_growth': '英国GDP增长率', 'UK_cpi': '英国CPI', 'UK_interest_rate': '英央行利率',
    'UK_unemployment': '英国失业率', 'UK_industrial_production': '英国工业生产',
    'UK_gbp_usd': '英镑/美元', 'UK_m2_money_supply': '英国M2', 'UK_uk_10y_bond': '英国10年国债',
    'JP_gdp_growth': '日本GDP增长率', 'JP_cpi': '日本CPI', 'JP_interest_rate': '日本央行利率',
    'JP_industrial_production': '日本工业生产', 'JP_m2': '日本M2',
    'JP_exchange_rate_usd_jpy': '美元/日元', 'JP_unemployment': '日本失业率',
    'JP_govt_bond_10y': '日本10年国债', 'JP_nikkei225': '日经225',
    'RU_gdp': '俄罗斯GDP', 'RU_cpi': '俄罗斯CPI', 'RU_interest_rate': '俄央行利率',
    'RU_unemployment': '俄罗斯失业率', 'RU_industrial_production': '俄罗斯工业生产',
    'RU_m2': '俄罗斯M2', 'RU_exchange_rate_usd_rub': '美元/卢布',
}

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

print(f"总指标数: {len(all_series)}")

# 生成完整月度时间线
all_dates_set = set()
for series in all_series.values():
    all_dates_set.update(series.keys())

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
print(f"时间线: {all_months[0]} ~ {all_months[-1]} ({len(all_months)} 个月)")

# 构建矩阵
col_names = sorted(all_series.keys())
raw_matrix = np.full((len(all_months), len(col_names)), np.nan)
date_idx = {d: i for i, d in enumerate(all_months)}

for j, col in enumerate(col_names):
    for d, v in all_series[col].items():
        if d in date_idx:
            raw_matrix[date_idx[d], j] = v

# 前向填充 + 中位数填充
for j in range(raw_matrix.shape[1]):
    col = raw_matrix[:, j]
    last_valid = np.nan
    for i in range(len(col)):
        if not np.isnan(col[i]):
            last_valid = col[i]
        elif not np.isnan(last_valid):
            col[i] = last_valid
    median_val = np.nanmedian(col)
    if np.isnan(median_val):
        median_val = 0.0
    col[np.isnan(col)] = median_val
    raw_matrix[:, j] = col

# 保存原始矩阵用于后续解读
raw_matrix_backup = raw_matrix.copy()

# 加时间维度
time_vals = np.linspace(0, 1, len(all_months)).reshape(-1, 1)
raw_with_time = np.hstack([raw_matrix, time_vals])

# 分位数变换
qt = QuantileTransformer(output_distribution='uniform', random_state=42)
X_norm = qt.fit_transform(raw_with_time)

# PCA
n_dims = X_norm.shape[1]
pca = None
if n_dims > 50:
    pca = PCA(n_components=25, random_state=42)
    X_tda = pca.fit_transform(X_norm)
    explained = np.cumsum(pca.explained_variance_ratio_)
    print(f"PCA: {n_dims}D -> 25D (解释方差: {explained[-1]:.1%})")
else:
    X_tda = X_norm

print(f"TDA 输入矩阵: {X_tda.shape}")

# ============================================================
# 2. Landmark 子采样 + Ripser（带 cocycles）
# ============================================================

print("\n" + "=" * 70)
print("2. 运行 Ripser（带 cocycles）")
print("=" * 70)

n_points = X_tda.shape[0]
MAX_POINTS = 400
if n_points > MAX_POINTS:
    np.random.seed(42)
    indices = [np.random.randint(n_points)]
    dists = np.full(n_points, np.inf)
    for _ in range(MAX_POINTS - 1):
        new_dists = np.linalg.norm(X_tda - X_tda[indices[-1]], axis=1)
        dists = np.minimum(dists, new_dists)
        indices.append(np.argmax(dists))
    landmark_idx = np.array(sorted(indices))
    X_ripser = X_tda[landmark_idx]
    landmark_months = [all_months[i] for i in landmark_idx]
    print(f"Maxmin 子采样: {len(landmark_idx)} 个 landmark 点")
else:
    X_ripser = X_tda
    landmark_idx = np.arange(n_points)
    landmark_months = list(all_months)

print(f"运行 ripser（maxdim=2, do_cocycles=True）...")
result = ripser(X_ripser, maxdim=2, thresh=2.0, do_cocycles=True)
diagrams = result['dgms']
cocycles = result['cocycles']

# 打印基本统计
for dim in range(3):
    dgm = diagrams[dim]
    finite = dgm[dgm[:, 1] < np.inf]
    if len(finite) > 0:
        lifetimes = finite[:, 1] - finite[:, 0]
        median_life = np.median(lifetimes)
        n_long = (lifetimes > 2 * median_life).sum()
        print(f"  H{dim}: {len(dgm)} 特征, {n_long} 个长寿命 (>2x median)")

# ============================================================
# 3. 提取长寿命特征的 cocycle 代表元
# ============================================================

print("\n" + "=" * 70)
print("3. 提取拓扑特征的顶点信息")
print("=" * 70)

def get_long_lived_features(dim, min_multiplier=2.0):
    """获取长寿命特征，返回 [(birth, death, persistence, cocycle_vertices), ...]"""
    dgm = diagrams[dim]
    finite_mask = dgm[:, 1] < np.inf
    finite = dgm[finite_mask]
    if len(finite) == 0:
        return []

    lifetimes = finite[:, 1] - finite[:, 0]
    median_life = np.median(lifetimes)
    threshold = min_multiplier * median_life

    dim_cocycles = cocycles[dim] if dim < len(cocycles) else []

    # ripser cocycles 与 dgm 是 1:1 对应的（按顺序）
    # dgm[i] 对应 dim_cocycles[i]
    features = []
    for i in range(len(dgm)):
        if dgm[i, 1] >= np.inf:
            continue
        birth, death = dgm[i, 0], dgm[i, 1]
        persistence = death - birth
        if persistence > threshold and i < len(dim_cocycles):
            cocycle = dim_cocycles[i]
            if dim == 1:
                vertices = set(cocycle[:, 0].tolist() + cocycle[:, 1].tolist())
            elif dim == 2:
                vertices = set(cocycle[:, 0].tolist() + cocycle[:, 1].tolist() + cocycle[:, 2].tolist())
            else:
                vertices = set()
            features.append((birth, death, persistence, sorted(vertices), cocycle))

    features.sort(key=lambda x: -x[2])
    return features


def get_month_from_landmark(vertex_idx):
    """将 ripser 的顶点索引映射回月份"""
    if vertex_idx < len(landmark_idx):
        full_idx = landmark_idx[vertex_idx]
        if full_idx < len(all_months):
            return all_months[full_idx]
    return "Unknown"


def get_raw_values_at_month(month):
    """获取某月的原始经济指标值"""
    if month not in date_idx:
        return {}
    idx = date_idx[month]
    values = {}
    for j, col in enumerate(col_names):
        values[col] = raw_matrix_backup[idx, j]
    return values


def find_most_changed_indicators(vertices, top_n=8):
    """
    找到在环路涉及的数据点之间变化最大的指标。
    使用标准化后的变化范围（相对于该指标的全局标准差）。
    """
    months = [get_month_from_landmark(v) for v in vertices]
    months = [m for m in months if m != "Unknown" and m in date_idx]
    if not months:
        return []

    month_indices = [date_idx[m] for m in months]
    changes = []
    for j, col in enumerate(col_names):
        vals = raw_matrix_backup[month_indices, j]
        col_std = np.std(raw_matrix_backup[:, j])
        if col_std > 0:
            val_range = np.max(vals) - np.min(vals)
            normalized_change = val_range / col_std
            changes.append((col, normalized_change, np.min(vals), np.max(vals)))

    changes.sort(key=lambda x: -x[1])
    return changes[:top_n]


def analyze_economic_context(months):
    """分析一组月份的经济状态和关键事件"""
    # 已知经济事件
    EVENTS = {
        ('1973-10', '1975-03'): '第一次石油危机',
        ('1979-01', '1980-06'): '第二次石油危机',
        ('1980-01', '1982-12'): '沃尔克紧缩/全球衰退',
        ('1987-09', '1988-02'): '黑色星期一',
        ('1990-07', '1991-06'): '海湾战争/衰退',
        ('1994-01', '1995-03'): '美联储意外加息/墨西哥危机',
        ('1997-07', '1998-10'): '亚洲金融危机/俄罗斯违约',
        ('2000-03', '2002-10'): '互联网泡沫破裂',
        ('2007-12', '2009-06'): '全球金融危机',
        ('2010-04', '2012-07'): '欧债危机',
        ('2014-06', '2016-02'): '大宗商品崩盘/新兴市场危机',
        ('2018-01', '2018-12'): '中美贸易战/全球紧缩',
        ('2020-02', '2020-06'): 'COVID-19 冲击',
        ('2021-01', '2022-12'): '后疫情通胀/全球紧缩',
    }

    matched_events = []
    for (start, end), event_name in EVENTS.items():
        for m in months:
            if start <= m <= end:
                matched_events.append(event_name)
                break

    return matched_events


def interpret_h1_loop(vertices, changed_indicators, months, events):
    """
    基于涉及的月份、变化最大的指标和历史事件，推断 H1 环路的经济含义。
    """
    if not months:
        return "数据点不足，无法解读"

    start_year = int(months[0][:4])
    end_year = int(months[-1][:4])
    span_years = end_year - start_year

    # 分析主导指标的国家分布
    country_counts = {}
    for col, change, _, _ in changed_indicators:
        country = col.split('_')[0]
        country_counts[country] = country_counts.get(country, 0) + change

    dominant_countries = sorted(country_counts.items(), key=lambda x: -x[1])

    # 分析指标类型
    rate_indicators = [c for c, _, _, _ in changed_indicators if 'rate' in c.lower() or 'interest' in c.lower() or 'fed_funds' in c.lower()]
    inflation_indicators = [c for c, _, _, _ in changed_indicators if 'cpi' in c.lower() or 'pce' in c.lower() or 'hicp' in c.lower()]
    growth_indicators = [c for c, _, _, _ in changed_indicators if 'gdp' in c.lower() or 'industrial' in c.lower() or 'employment' in c.lower()]
    fx_indicators = [c for c, _, _, _ in changed_indicators if 'usd' in c.lower() or 'eur' in c.lower() or 'gbp' in c.lower() or 'exchange' in c.lower() or 'reer' in c.lower()]
    market_indicators = [c for c, _, _, _ in changed_indicators if 'sp500' in c.lower() or 'nikkei' in c.lower() or 'vix' in c.lower() or 'share' in c.lower()]

    interpretations = []

    if events:
        interpretations.append(f"覆盖历史事件: {', '.join(set(events))}")

    if rate_indicators and inflation_indicators:
        interpretations.append("利率-通胀周期: 央行加息抑制通胀→经济减速→降息→通胀回升的循环")
    elif rate_indicators and growth_indicators:
        interpretations.append("货币政策-经济增长周期: 紧缩→减速→宽松→复苏的反馈环")
    elif fx_indicators and rate_indicators:
        interpretations.append("利率-汇率传导环: 利差变化→资本流动→汇率调整→贸易再平衡")
    elif growth_indicators and market_indicators:
        interpretations.append("经济-市场反身性环: 增长预期→市场上涨→财富效应→更多增长")

    if len(dominant_countries) >= 2:
        top2 = [c[0] for c in dominant_countries[:2]]
        interpretations.append(f"主要涉及经济体: {' + '.join(top2)}")
        if 'US' in top2 and 'CN' in top2:
            interpretations.append("中美经济联动/对抗周期")
        elif 'US' in top2 and 'EU' in top2:
            interpretations.append("大西洋两岸经济协同/分化")
        elif 'US' in top2 and 'JP' in top2:
            interpretations.append("美日利差驱动的资本流动周期")

    if span_years > 10:
        interpretations.append(f"长周期结构 ({span_years}年跨度): 可能反映结构性经济转型")
    elif span_years > 5:
        interpretations.append(f"中周期结构 ({span_years}年跨度): 典型的经济/信贷周期")
    elif span_years > 2:
        interpretations.append(f"短周期结构 ({span_years}年跨度): 可能是政策响应周期")

    return "; ".join(interpretations) if interpretations else "混合型拓扑特征，涉及多个经济维度的联合波动"


# ============================================================
# 4. H1 环路分析
# ============================================================

print("\n" + "=" * 70)
print("4. H1 环路分析")
print("=" * 70)

h1_features = get_long_lived_features(1, min_multiplier=2.0)
print(f"  长寿命 H1 环路: {len(h1_features)} 个")

# 如果太少，降低阈值
if len(h1_features) < 5:
    h1_features = get_long_lived_features(1, min_multiplier=1.5)
    print(f"  降低阈值后: {len(h1_features)} 个")

h1_reports = []
for i, (birth, death, persistence, vertices, cocycle) in enumerate(h1_features):
    months = sorted([get_month_from_landmark(v) for v in vertices])
    months = [m for m in months if m != "Unknown"]
    changed = find_most_changed_indicators(vertices)
    events = analyze_economic_context(months)
    interpretation = interpret_h1_loop(vertices, changed, months, events)

    report = {
        'rank': i + 1,
        'birth': birth,
        'death': death,
        'persistence': persistence,
        'n_vertices': len(vertices),
        'n_edges': len(cocycle),
        'months': months,
        'time_range': f"{months[0]} ~ {months[-1]}" if months else "N/A",
        'changed_indicators': changed,
        'events': events,
        'interpretation': interpretation,
    }
    h1_reports.append(report)

# ============================================================
# 5. H2 空洞分析
# ============================================================

print("\n" + "=" * 70)
print("5. H2 空洞分析")
print("=" * 70)

h2_features = get_long_lived_features(2, min_multiplier=2.0)
print(f"  长寿命 H2 空洞: {len(h2_features)} 个")

if len(h2_features) < 1:
    h2_features = get_long_lived_features(2, min_multiplier=1.5)
    print(f"  降低阈值后: {len(h2_features)} 个")

if len(h2_features) < 1:
    h2_features = get_long_lived_features(2, min_multiplier=1.0)
    print(f"  进一步降低阈值后: {len(h2_features)} 个")

h2_reports = []
for i, (birth, death, persistence, vertices, cocycle) in enumerate(h2_features):
    months = sorted([get_month_from_landmark(v) for v in vertices])
    months = [m for m in months if m != "Unknown"]

    # 对于 H2 空洞，分析边界数据点的经济状态
    # 空洞"内部"是不存在的状态组合
    boundary_values = {}
    for m in months:
        vals = get_raw_values_at_month(m)
        for k, v in vals.items():
            if k not in boundary_values:
                boundary_values[k] = []
            boundary_values[k].append(v)

    # 找出在边界上取值范围最广的指标（空洞在这些维度上"围出"了空间）
    void_dims = []
    for col, vals in boundary_values.items():
        if len(vals) >= 2:
            col_std = np.std(raw_matrix_backup[:, col_names.index(col)])
            if col_std > 0:
                val_range = max(vals) - min(vals)
                normalized = val_range / col_std
                void_dims.append((col, normalized, min(vals), max(vals), np.mean(vals)))
    void_dims.sort(key=lambda x: -x[1])

    # 解读空洞
    # 空洞意味着：边界数据点围成了一个"壳"，但壳内没有实际数据
    # 即某些状态组合在理论上被边界点"包围"，但在历史中从未出现

    top_dims = void_dims[:8]
    # 推断空洞内部的状态
    interior_states = []
    for col, norm_range, vmin, vmax, vmean in top_dims:
        label = INDICATOR_LABELS.get(col, col)
        interior_states.append(f"{label}: {vmean:.2f} (边界范围 {vmin:.2f}~{vmax:.2f})")

    # 分析为什么空洞内部状态不可能
    impossible_reasons = []
    rate_cols = [(c, vmin, vmax, vmean) for c, _, vmin, vmax, vmean in top_dims if 'rate' in c.lower() or 'interest' in c.lower() or 'fed_funds' in c.lower()]
    growth_cols = [(c, vmin, vmax, vmean) for c, _, vmin, vmax, vmean in top_dims if 'gdp' in c.lower()]
    inflation_cols = [(c, vmin, vmax, vmean) for c, _, vmin, vmax, vmean in top_dims if 'cpi' in c.lower() or 'pce' in c.lower() or 'hicp' in c.lower()]
    fx_cols = [(c, vmin, vmax, vmean) for c, _, vmin, vmax, vmean in top_dims if 'usd' in c.lower() or 'exchange' in c.lower() or 'eur' in c.lower()]
    employment_cols = [(c, vmin, vmax, vmean) for c, _, vmin, vmax, vmean in top_dims if 'unemployment' in c.lower() or 'employment' in c.lower()]
    yield_curve_cols = [(c, vmin, vmax, vmean) for c, _, vmin, vmax, vmean in top_dims if 'yield_curve' in c.lower()]
    market_cols = [(c, vmin, vmax, vmean) for c, _, vmin, vmax, vmean in top_dims if 'sp500' in c.lower() or 'nikkei' in c.lower() or 'vix' in c.lower() or 'share' in c.lower()]
    sentiment_cols = [(c, vmin, vmax, vmean) for c, _, vmin, vmax, vmean in top_dims if 'sentiment' in c.lower() or 'confidence' in c.lower() or 'consumer' in c.lower()]

    involved_countries = set(c.split('_')[0] for c, _, _, _, _ in top_dims)

    if yield_curve_cols and growth_cols:
        yc = yield_curve_cols[0]
        gc = growth_cols[0]
        impossible_reasons.append(
            f"收益率曲线({INDICATOR_LABELS.get(yc[0],yc[0])}: {yc[1]:.2f}~{yc[2]:.2f})与GDP增长"
            f"({INDICATOR_LABELS.get(gc[0],gc[0])}: {gc[1]:.2f}~{gc[2]:.2f})的某些中间组合"
            f"在历史中不存在——收益率曲线是经济预期的领先指标，其特定斜率只在特定增长阶段出现，"
            f"空洞内部对应的是\"收益率曲线发出某种信号但增长走势与信号矛盾\"的状态")

    if sentiment_cols and market_cols:
        sc = sentiment_cols[0]
        mc = market_cols[0]
        impossible_reasons.append(
            f"消费者信心({INDICATOR_LABELS.get(sc[0],sc[0])}: {sc[1]:.2f}~{sc[2]:.2f})与"
            f"市场指标({INDICATOR_LABELS.get(mc[0],mc[0])}: {mc[1]:.2f}~{mc[2]:.2f})存在反身性约束——"
            f"某些\"市场涨但信心低\"或\"市场跌但信心高\"的中间状态无法持续")

    if rate_cols and inflation_cols:
        impossible_reasons.append("利率与通胀受泰勒规则约束，央行对通胀的系统性反应使某些利率-通胀组合不稳定（会被政策迅速纠正）")

    if fx_cols and rate_cols:
        impossible_reasons.append("利率平价关系约束了利差与汇率的联动，某些利差-汇率组合无法持续")

    if len(involved_countries) >= 3:
        impossible_reasons.append(
            f"涉及 {len(involved_countries)} 个经济体 ({', '.join(involved_countries)}) 的联合约束: "
            f"全球经济的联动性使得某些跨国状态组合在结构上不可能——"
            f"例如美欧日中不可能同时处于某些指标的中间值组合")

    if market_cols and len(market_cols) >= 2:
        impossible_reasons.append("多个市场（股市/VIX/信用利差）的某些同时取值组合不可能——"
                                  "例如VIX高（恐慌）但股市也高且信用利差窄的状态是自相矛盾的")

    if not impossible_reasons:
        impossible_reasons.append("边界数据点围成的高维空间内部对应的经济状态组合在55年历史中从未出现，反映了宏观经济变量之间的结构性约束")

    report = {
        'rank': i + 1,
        'birth': birth,
        'death': death,
        'persistence': persistence,
        'n_vertices': len(vertices),
        'n_triangles': len(cocycle),
        'months': months,
        'time_range': f"{months[0]} ~ {months[-1]}" if months else "N/A",
        'top_void_dims': top_dims,
        'interior_states': interior_states,
        'impossible_reasons': impossible_reasons,
    }
    h2_reports.append(report)

# ============================================================
# 6. 补充分析：通过距离矩阵找环路路径
# ============================================================

print("\n" + "=" * 70)
print("6. 补充: 环路路径重建")
print("=" * 70)

# 对于每个 H1 环路，用 cocycle 的边信息重建环路路径顺序
def reconstruct_loop_path(cocycle):
    """从 cocycle 的边列表重建近似环路路径"""
    edges = [(int(cocycle[k, 0]), int(cocycle[k, 1])) for k in range(len(cocycle))]

    if len(edges) == 0:
        return []

    # 构建邻接表
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    # 尝试找一条从某顶点出发的路径
    # 贪心：从度最小的点出发，优先走度小的邻居
    start = min(adj.keys(), key=lambda x: len(adj[x]))
    path = [start]
    visited = {start}

    current = start
    while True:
        neighbors = [n for n in adj.get(current, []) if n not in visited]
        if not neighbors:
            # 检查是否能回到起点
            if start in adj.get(current, []) and len(path) > 2:
                path.append(start)
            break
        # 选距离当前月份最近的（时间连续性）
        next_node = neighbors[0]
        path.append(next_node)
        visited.add(next_node)
        current = next_node

    return path


for report in h1_reports:
    idx = report['rank'] - 1
    if idx < len(h1_features):
        cocycle = h1_features[idx][4]
        path = reconstruct_loop_path(cocycle)
        path_months = [get_month_from_landmark(v) for v in path]
        path_months = [m for m in path_months if m != "Unknown"]
        report['loop_path_months'] = path_months

# ============================================================
# 7. 输出详细报告
# ============================================================

print("\n" + "=" * 70)
print("=" * 70)
print("              TDA 拓扑特征经济解读报告")
print("              全球 6 经济体 71 指标")
print("=" * 70)
print("=" * 70)

print(f"\n数据: US/CN/EU/UK/JP/RU, {len(col_names)} 指标 + 1 时间维度")
print(f"时间范围: {all_months[0]} ~ {all_months[-1]} ({len(all_months)} 个月)")
print(f"TDA 输入: {X_ripser.shape[0]} 点 x {X_ripser.shape[1]} 维")

# ---------- H1 ----------
print("\n")
print("=" * 70)
print("                     H1 环路分析")
print("   (环路 = 状态空间中的闭合路径 = 经济周期/反馈环)")
print("=" * 70)

if not h1_reports:
    print("\n  未提取到长寿命 H1 特征的 cocycle 信息。")
    print("  可能原因: ripser cocycles 与 persistence diagram 的索引不匹配。")
else:
    for report in h1_reports:
        print(f"\n{'─' * 60}")
        print(f"环路 #{report['rank']}（persistence = {report['persistence']:.4f}，"
              f"birth = {report['birth']:.4f}，death = {report['death']:.4f}）")
        print(f"{'─' * 60}")
        print(f"  cocycle 边数: {report['n_edges']}，涉及顶点数: {report['n_vertices']}")
        print(f"  涉及时间: {report['time_range']}")

        if report.get('loop_path_months'):
            print(f"  环路路径: {' → '.join(report['loop_path_months'][:15])}", end="")
            if len(report['loop_path_months']) > 15:
                print(f" ... (共 {len(report['loop_path_months'])} 个月)")
            else:
                print()

        months_list = report['months']
        if len(months_list) > 10:
            shown = months_list[:5] + ['...'] + months_list[-5:]
            print(f"  涉及月份: {', '.join(shown)} (共 {len(months_list)} 个)")
        elif months_list:
            print(f"  涉及月份: {', '.join(months_list)}")

        if report['events']:
            print(f"  对应历史事件: {', '.join(set(report['events']))}")

        print(f"  关键指标变化（标准化变化幅度从大到小）:")
        for col, change, vmin, vmax in report['changed_indicators']:
            label = INDICATOR_LABELS.get(col, col)
            print(f"    {label}: {vmin:.2f} → {vmax:.2f} (标准化变幅 {change:.2f})")

        print(f"  经济解读: {report['interpretation']}")

# ---------- H2 ----------
print("\n\n")
print("=" * 70)
print("                     H2 空洞分析")
print("   (空洞 = 状态空间中被包围但内部为空的区域 = 不可能的状态组合)")
print("=" * 70)

if not h2_reports:
    print("\n  未提取到长寿命 H2 特征的 cocycle 信息。")
    print("  可能原因: H2 特征数量少或 cocycle 提取未成功。")

    # 备用: 直接分析 H2 persistence diagram
    dgm_h2 = diagrams[2]
    finite_h2 = dgm_h2[dgm_h2[:, 1] < np.inf]
    if len(finite_h2) > 0:
        lifetimes_h2 = finite_h2[:, 1] - finite_h2[:, 0]
        top_h2 = np.argsort(-lifetimes_h2)[:5]
        print(f"\n  H2 persistence diagram 中最长寿命的 {min(5, len(top_h2))} 个特征:")
        for rank, idx in enumerate(top_h2):
            b, d = finite_h2[idx]
            p = d - b
            print(f"    #{rank+1}: birth={b:.4f}, death={d:.4f}, persistence={p:.4f}")

        # 用近似方法分析
        print(f"\n  使用近似方法分析 H2 空洞...")
        print(f"  方法: 在 birth 半径附近找密度最低的区域，推断空洞位置")

        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=10, metric='euclidean')
        nn.fit(X_ripser)
        distances, nn_indices = nn.kneighbors(X_ripser)
        local_density = 1.0 / (distances.mean(axis=1) + 1e-8)

        for rank, idx in enumerate(top_h2[:2]):
            b, d = finite_h2[idx]
            # 找 birth 半径附近的点对
            dist_matrix = np.linalg.norm(X_ripser[:, None] - X_ripser[None, :], axis=2)
            # 找距离接近 birth 半径的点对
            near_birth = np.abs(dist_matrix - b) < 0.1 * b
            involved_points = set()
            for ii in range(len(X_ripser)):
                for jj in range(ii+1, len(X_ripser)):
                    if near_birth[ii, jj]:
                        involved_points.add(ii)
                        involved_points.add(jj)

            if involved_points:
                involved = sorted(involved_points)[:30]  # 取前30个点
                involved_months = sorted([get_month_from_landmark(v) for v in involved])
                involved_months = [m for m in involved_months if m != "Unknown"]

                print(f"\n  空洞 #{rank+1}（persistence = {d-b:.4f}）")
                print(f"    birth 半径附近涉及约 {len(involved_points)} 个数据点")
                if involved_months:
                    print(f"    时间范围: {involved_months[0]} ~ {involved_months[-1]}")
                    events = analyze_economic_context(involved_months)
                    if events:
                        print(f"    对应事件: {', '.join(set(events))}")

                    # 分析空洞维度
                    changed = find_most_changed_indicators(involved)
                    print(f"    空洞最大维度（边界指标变化）:")
                    for col, change, vmin, vmax in changed[:6]:
                        label = INDICATOR_LABELS.get(col, col)
                        print(f"      {label}: {vmin:.2f} ~ {vmax:.2f} (标准化变幅 {change:.2f})")

                    # 推断空洞内部状态
                    mid_vals = {}
                    for v in involved:
                        m = get_month_from_landmark(v)
                        if m != "Unknown":
                            vals = get_raw_values_at_month(m)
                            for k, val in vals.items():
                                if k not in mid_vals:
                                    mid_vals[k] = []
                                mid_vals[k].append(val)

                    print(f"    空洞内部对应的不可达状态:")
                    print(f"      边界数据点围成的多维区域内部——")
                    print(f"      即历史上从未同时出现过的经济状态组合")

                    # 分析具体约束
                    if changed:
                        top_col_countries = set(c.split('_')[0] for c, _, _, _ in changed[:6])
                        if len(top_col_countries) >= 3:
                            print(f"      涉及 {len(top_col_countries)} 个经济体的联合约束: {', '.join(top_col_countries)}")
                            print(f"      解释: 这些经济体的指标不可能同时取到某些极端组合值")
                            print(f"              反映了全球经济的结构性互联约束")

else:
    for report in h2_reports:
        print(f"\n{'─' * 60}")
        print(f"空洞 #{report['rank']}（persistence = {report['persistence']:.4f}，"
              f"birth = {report['birth']:.4f}，death = {report['death']:.4f}）")
        print(f"{'─' * 60}")
        print(f"  cocycle 三角形数: {report['n_triangles']}，涉及顶点数: {report['n_vertices']}")
        print(f"  边界数据点时间: {report['time_range']}")

        months_list = report['months']
        if len(months_list) > 10:
            shown = months_list[:5] + ['...'] + months_list[-5:]
            print(f"  涉及月份: {', '.join(shown)} (共 {len(months_list)} 个)")
        elif months_list:
            print(f"  涉及月份: {', '.join(months_list)}")

        print(f"  空洞最大维度（边界上变化最大的指标）:")
        for col, norm_range, vmin, vmax, vmean in report['top_void_dims'][:8]:
            label = INDICATOR_LABELS.get(col, col)
            print(f"    {label}: {vmin:.2f} ~ {vmax:.2f} (边界均值 {vmean:.2f}, 标准化变幅 {norm_range:.2f})")

        print(f"  空洞内部对应的不可达状态:")
        for state in report['interior_states'][:6]:
            print(f"    {state}")

        print(f"  为什么不可能:")
        for reason in report['impossible_reasons']:
            print(f"    - {reason}")

# ============================================================
# 8. 总结
# ============================================================

print("\n\n")
print("=" * 70)
print("                        总结")
print("=" * 70)

print(f"""
H1 环路总数: {len(h1_reports)} 个长寿命特征
  环路代表状态空间中的闭合路径——经济系统出发、经历一系列变化后
  回到相似状态的周期性模式。persistence 越大，说明该周期结构
  越稳健、跨越的状态空间距离越大。

H2 空洞总数: {len(h2_reports)} 个长寿命特征
  空洞代表状态空间中被数据点"包围"但内部为空的区域——
  即某些经济状态组合在理论上被相邻状态包围，但在 55 年历史中
  从未实际出现过。这反映了宏观经济变量之间的深层结构性约束。

方法论说明:
  - 顶点索引通过 ripser cocycle representatives 获取
  - cocycle 中的边（H1）/三角形（H2）的顶点直接对应 landmark 数据点
  - landmark 点通过 maxmin 子采样选取，保证状态空间覆盖均匀
  - 指标变化使用标准化变幅（除以全局标准差），跨指标可比
""")

print(f"输出目录: {OUT_DIR}/")
print("分析完成。")
