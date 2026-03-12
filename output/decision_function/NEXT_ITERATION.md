# Decision Function v2 — 对手方感知模型

## 核心改进：加入对手方/环境领导人参数

v1 (当前): input = [economic_state | my_params]
v2 (下一步): input = [economic_state | my_params | opponent_params | opponent_recent_actions]

## 具体实现

### Powell 的例子
- 2018-19 (Trump): trade_war_escalation=0.85, pressure_on_fed=0.90 → Powell 被迫转鸽
- 2021-22 (Biden): fiscal_stimulus=0.80, no_fed_pressure=0.10 → Powell 自主决策
- 同样的通胀数据，不同的总统参数 → 完全不同的利率路径

### Xi 的例子
- 面对 Trump 关税: retaliation_probability 取决于 trump.escalation_willingness
- 面对 Biden 芯片禁令: technology_self_reliance 取决于 biden.decoupling_commitment

### 输入向量结构
```
[经济指标 ~30D]
[我的行为参数 ~15D]
[对手方1参数 ~10D]  ← 最相关对手（如Powell→总统）
[对手方2参数 ~10D]  ← 次要对手（如Powell→ECB行长）
[对手最近动作 ~5D]  ← 最近3个月的关键决策编码
= ~70D total
```

### 为什么这很重要
- 之前试点 66.7% 准确率，转折点全错
- 转折点几乎都和对手方变化有关（Trump施压/贸易战升级/疫情政策）
- 缺了这个维度，模型看到的是"通胀一样但决策不同"→认为是噪声
- 加上对手方参数，"通胀一样+总统不同→决策不同"变得可解释

### 博弈论框架
- 不是 agent 独立决策，是 multi-agent 互动
- 每个领导人的决策函数输入包含其他领导人的状态
- 形成反馈环：A决策→改变经济状态→B感知到→B决策→改变经济状态→A感知到
- TDA 的 H1 环路可能就是在捕捉这些反馈环
