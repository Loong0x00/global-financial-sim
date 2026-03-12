# Data Format Specification

## 目录结构

```
data/
├── economic/              # 经济数据（硬数据）
│   ├── us/                # 按国家/经济体分目录
│   │   ├── macro.json     # 宏观经济指标时间序列
│   │   └── financial.json # 金融市场数据
│   ├── cn/
│   ├── eu/
│   ├── jp/
│   └── ru/
├── leaders/               # 领导人数据
│   ├── profiles/          # LLM 编译的行为矩阵
│   │   ├── powell.json    # 鲍威尔行为档案
│   │   ├── xi.json
│   │   └── ...
│   ├── decisions/         # 决策时间线（训练数据核心）
│   │   ├── powell_decisions.json
│   │   └── ...
│   └── archetypes/        # 独裁者通用底模
│       ├── clusters.json  # 聚类结果
│       └── priors.json    # 各类型先验分布
├── events/                # 结构化事件数据（GDELT等）
│   └── global_events.json
├── transmission/          # 传导关系数据
│   ├── trade_matrix.json  # 贸易依赖矩阵
│   └── financial_exposure.json
models/                    # 训练好的模型
├── decision_functions/    # 各领导人的小 Transformer
│   ├── powell.pt
│   └── ...
└── manifold/              # 流形降维结果
    ├── leader_embedding.npy
    └── elbow_analysis.json
output/                    # 模拟输出
├── topology/              # TDA 拓扑结构
├── monte_carlo/           # MC 路径数据
└── visualizations/        # 可视化图表
```

## 经济数据格式 (economic/*.json)

```json
{
  "entity": "US",
  "source": "FRED",
  "last_updated": "2026-03-12",
  "indicators": {
    "gdp_growth": {
      "description": "Real GDP Growth Rate (quarterly, annualized)",
      "unit": "percent",
      "frequency": "quarterly",
      "series": [
        {"date": "2022-01", "value": 2.1, "confidence": "official"},
        {"date": "2022-04", "value": -0.6, "confidence": "official"}
      ]
    },
    "cpi_yoy": {
      "description": "CPI Year-over-Year Change",
      "unit": "percent",
      "frequency": "monthly",
      "series": [
        {"date": "2022-01", "value": 7.5, "confidence": "official"}
      ]
    },
    "fed_funds_rate": {
      "description": "Federal Funds Effective Rate",
      "unit": "percent",
      "frequency": "monthly",
      "series": [
        {"date": "2022-01", "value": 0.08, "confidence": "official"}
      ]
    },
    "unemployment": {
      "description": "Unemployment Rate (U-3)",
      "unit": "percent",
      "frequency": "monthly",
      "series": [
        {"date": "2022-01", "value": 4.0, "confidence": "official"}
      ]
    }
  }
}
```

### confidence 字段

数据可信度分级，对应 Design Principle #5（数据源分级）：
- `"official"` — 官方发布，直接使用
- `"official_unreliable"` — 官方发布但已知不可靠（如中国GDP），需修正
- `"satellite"` — 卫星/遥感数据修正值
- `"cross_validated"` — 贸易对手交叉验证值
- `"proxy"` — 代理指标推算（如李克强指数）
- `"estimated"` — 模型估计值

## 领导人决策时间线格式 (leaders/decisions/*.json)

这是决策函数训练数据的核心：每条记录 = (当时的状态, 做了什么决策)

```json
{
  "leader": "Jerome Powell",
  "role": "Federal Reserve Chair",
  "entity": "US",
  "tenure": {"start": "2018-02-05", "end": null},
  "leader_type": "technocrat_central_banker",
  "decisions": [
    {
      "date": "2022-03-16",
      "context": {
        "economic_state": {
          "cpi_yoy": 7.9,
          "unemployment": 3.8,
          "gdp_growth": 2.1,
          "fed_funds_rate": 0.08,
          "sp500_yoy_change": -8.2,
          "oil_price": 98.5,
          "usd_index": 98.4
        },
        "political_state": {
          "geopolitical_tension": 0.8,
          "domestic_pressure": 0.6,
          "election_proximity": 0.3,
          "description": "Russia invaded Ukraine 3 weeks prior. Inflation at 40-year high. Political pressure to act."
        }
      },
      "decision": {
        "category": "monetary_policy",
        "action": "rate_hike",
        "magnitude": 25,
        "unit": "bp",
        "forward_guidance": "hawkish",
        "description": "First rate hike since 2018. +25bp to 0.25-0.50%. Signaled more hikes coming."
      },
      "outcome": {
        "market_reaction": "S&P500 +2.2% on day, yield curve flattened",
        "description": "Market relieved by only 25bp (some expected 50bp)"
      },
      "sources": [
        "FOMC Statement 2022-03-16",
        "FOMC Press Conference 2022-03-16"
      ]
    }
  ]
}
```

## 领导人行为矩阵格式 (leaders/profiles/*.json)

LLM 编译输出 — 高维行为矩阵，维度不限

```json
{
  "leader": "Jerome Powell",
  "compiled_by": "claude-sonnet-4-6",
  "compiled_date": "2026-03-12",
  "compilation_sources": [
    "FOMC statements and minutes 2018-2026",
    "Congressional testimonies",
    "Jackson Hole speeches",
    "Press conference transcripts"
  ],
  "behavior_matrix": {
    "description": "High-dimensional behavior vector. Dimensions are not predefined — LLM outputs as many as it deems necessary. Manifold learning will discover the true dimensionality.",
    "dimensions": 150,
    "labels": [
      "inflation_sensitivity",
      "employment_sensitivity",
      "market_crash_response_speed",
      "political_pressure_resistance",
      "forward_guidance_consistency",
      "dovish_bias_base",
      "hawkish_threshold_cpi",
      "gradualism_preference",
      "data_dependency_weight",
      "financial_stability_concern",
      "...更多维度由 LLM 自由定义..."
    ],
    "values": [0.85, 0.72, 0.65, 0.78, 0.82, 0.35, 0.68, 0.88, 0.91, 0.55]
  },
  "red_lines": [
    {
      "condition": "cpi_yoy > 8.0 AND unemployment < 5.0",
      "forced_action": "aggressive_hike",
      "confidence": 0.9,
      "evidence": "2022 behavior: accelerated to 75bp hikes when CPI exceeded 8%"
    }
  ],
  "information_fidelity": 0.95,
  "information_fidelity_reasoning": "Fed has extensive data collection. Minor fidelity loss from political noise and market narratives.",
  "special_notes": "Powell shows strong 'gradualism' preference — prefers series of small moves over single large move. Exception: extreme inflation (2022 H2 75bp hikes)."
}
```

## 状态快照格式

Monte Carlo 每一步的世界状态，可序列化

```json
{
  "timestamp": "2024-06-01",
  "time_normalized": 0.73,
  "entities": {
    "US": {
      "economic": {"gdp_growth": 0.65, "cpi": 0.42, "rate": 0.92, "unemployment": 0.25},
      "note": "values are quantile-transformed [0,1]"
    },
    "CN": {
      "economic": {"gdp_growth": 0.35, "cpi": 0.18, "rate": 0.45, "unemployment": 0.60}
    }
  },
  "leaders": {
    "powell": {"factors": [0.85, 0.72, 0.65], "note": "manifold-reduced dimensions"},
    "xi": {"factors": [0.45, 0.88, 0.32]}
  },
  "topology_position": {
    "nearest_node_id": 15234,
    "distance_to_nearest_bottleneck": 0.15,
    "current_connected_component": 0
  }
}
```

## 关键约定

1. **所有数值型经济数据最终归一化到 [0,1]**（分位数变换），原始值保留在 economic/*.json 中
2. **日期格式**: `YYYY-MM-DD` 或 `YYYY-MM`（月度数据）
3. **confidence 字段必填**: 每个数据点标注可信度
4. **sources 字段必填**: 每条决策记录标注数据来源
5. **领导人行为矩阵维度不限**: LLM 自由输出，流形学习自动发现真实维度
6. **所有 JSON 使用 UTF-8 编码**，支持中文描述字段
