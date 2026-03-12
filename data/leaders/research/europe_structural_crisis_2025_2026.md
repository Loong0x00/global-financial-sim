# 欧洲结构性危机：移民+高福利+财政紧张
## Monte Carlo 模拟参数研究报告

**编制日期:** 2026-03-13
**覆盖范围:** 欧洲主要经济体（德/法/意/西/荷/瑞典/波/匈），2020-2026
**用途:** 为 european_unity_index 及欧洲领导人决策空间提供参数基础

---

## 1. 移民数据

### 1.1 EU 庇护申请趋势（2020-2025）

| 年份 | 首次庇护申请（约） | 同比变化 | 备注 |
|------|-------------------|----------|------|
| 2020 | ~470,000 | -33% | COVID 大幅抑制 |
| 2021 | ~630,000 | +34% | COVID 后反弹 |
| 2022 | ~960,000 | +52% | 含乌克兰临时保护外的常规申请 |
| 2023 | ~1,140,000 | +19% | 历史高峰 |
| 2024 | ~1,050,000 | -8% | 开始下降 |
| 2025(1-11月) | ~750,000(推算全年~820,000) | -23% | 显著下降 |

**关键细节：**
- 2025年上半年EU+国家收到约399,000份申请，同比下降23%
- 2025年10月：62,010份首次申请，同比降28%
- 2025年11月：54,825份首次申请，同比降26%
- 叙利亚阿萨德政权倒台后，叙利亚人不再是最大申请群体，委内瑞拉人取而代之，阿富汗人第二

### 1.2 净移民数据（EU整体）

| 年份 | EU净移民 | 备注 |
|------|---------|------|
| 2020 | ~1,066,000 | COVID 基线 |
| 2021 | ~1,024,000 | 略降 |
| 2022 | ~4,207,000 | 乌克兰危机导致暴增310% |
| 2023 | ~1,304,000 | 乌克兰人部分回流，但仍高于2019 |
| 2024 | 数据待更新 | 乌克兰>115万人回国 |

**主要目的地国（2023年移民总流入）：**
- 德国：1,271,200
- 西班牙：1,251,000
- 意大利：439,700
- 法国：417,600

### 1.3 公众态度变化

- 70%的欧洲人同意融合是新来者和东道社区的共同责任
- **认知偏差严重**：68%的受访者高估移民占比，平均认为16%人口是非EU公民，实际不到7%
- 移民是意大利（66%）、马耳他（65%）、匈牙利（62%）选民最关心的议题
- 但在整个EU层面，移民排在经济和通胀之后，约31%认为是优先挑战

### 1.4 对模型的含义

庇护申请在2025年**已显著下降**（叙利亚因素消除、EU政策收紧），但公众感知存在滞后——**选民对移民的不满情绪并未因实际数字下降而缓解**。这意味着：
- 移民作为政治变量的驱动力 > 实际移民数据的变化
- 右翼政党仍可利用"移民焦虑"获得选票，即使实际流入已减少
- 模型中 `migration_pressure` 应区分 `actual_flow` 和 `perceived_pressure`

---

## 2. 财政状况

### 2.1 债务/GDP 和赤字/GDP（2024-2026）

| 国家 | 债务/GDP(2024) | 赤字/GDP(2024) | 2026展望 | 风险等级 |
|------|---------------|---------------|---------|---------|
| **意大利** | ~138% | 3.4% | 债务升至~140% | 高（第二高债务国） |
| **法国** | ~114% | 5.8% | 预计升至118-119% | 极高（赤字远超3%限制） |
| **德国** | ~62% | <3% | 500亿基建基金开支 | 低（但财政扩张中） |
| **西班牙** | ~105% | ~3.5% | 改善中 | 中 |
| **希腊** | 下降中(-8.9pp) | 改善 | 持续改善 | 中低（逆转趋势） |
| **波兰** | 上升(+5.0pp) | 恶化 | 军费压力 | 中高 |
| **芬兰** | 上升(+4.7pp) | 恶化 | 军费+福利双压力 | 中 |
| **罗马尼亚** | 上升(+5.5pp) | 恶化 | EU最大债务增幅 | 高 |

**欧元区整体：**
- 2025年赤字/GDP预计3.2%，2026年3.3%（**整体超过马约3%红线**）
- 2025年Q3债务/GDP：欧元区88.5%，EU整体82.1%

### 2.2 各国详情

#### 法国：政治+财政双重危机
- 2024赤字5.8%（从2023年5.4%恶化），远超EU 3%限制
- **政府接连倒台**：Barnier使用49.3条款强推紧缩预算→不信任案→倒台→Bayrou继任→再次倒台
- 公共债务预计从113%升至118-119%
- IMF：法国承诺2029年前赤字降至3%以下，但市场持怀疑态度
- **关键风险**：左翼和右翼都反对紧缩，任何政府都无法通过削减福利的预算

#### 意大利：表面改善，底层脆弱
- BTP-Bund利差收窄至<100bp（2021年以来最低），市场信心改善
- S&P上调评级至BBB+
- **但**：2025年起实际借贷成本转正（2020年以来首次），增长预期极低
- 债务/GDP ~135-138%，仍居欧元区第二
- Meloni政府政治稳定性有助于短期改善，但结构性问题未解

#### 德国：历史性财政转向
- 2025年3月通过宪法修正案，设立**5000亿欧元基建基金**（GDP的11.6%）
- 国防支出超过GDP 1%的部分豁免于"债务刹车"规则
- 各州每年可新增GDP 0.35%的净借贷
- 2026年预算：从基建基金拨489亿，从气候转型基金拨217亿
- 预计到2029年，年投资水平维持约1200亿
- **经济影响**：GDP预计到2029年额外增长约1.25%，到2035年增长约2.5%
- **历史意义**：德国从"财政鹰派"转向"财政扩张"，对欧洲政治经济的影响巨大

### 2.3 EU新财政规则

- 新《稳定与增长公约》要求各国在2025年开始的4-7年调整期内，抵消人口老龄化带来的财政成本
- Eurogroup认定2026年预算合规的国家：奥地利、意大利、斯洛伐克、法国、塞浦路斯、爱沙尼亚、芬兰、德国、希腊、爱尔兰、拉脱维亚、卢森堡、葡萄牙
- **国家逃逸条款**：因国防开支增加，新规允许在不立即削减其他支出的情况下适应

### 2.4 军费增长 vs 福利支出的零和博弈

**军费扩张：**
- EU成员国国防支出2024年达3430亿欧元，2025年预计3810亿（+11%）
- 2024年达GDP的1.9%，2025年预计2.1%
- **2025年6月北约峰会**：商定2035年前达到GDP 3.5%的目标 + 额外1.5%用于国防基础设施
- **这意味着从2025年的2.1%到2035年的5%**——每年需额外增加数千亿欧元

**零和博弈的现实：**
- 欧洲国家以强社会安全网著称，而非军事力量
- 中期内，公共财政需要再平衡以吸收扩大的国防开支
- 在已超过3%赤字限制的国家（法国、意大利），增加军费只能来自削减福利或增加债务
- **政治上几乎不可能**：任何削减养老金或医疗支出的政府都会被选民抛弃

### 2.5 人口老龄化的财政冲击

- 到2050年，65岁以上人口增加3200万，20岁以下减少2100万
- 工作年龄人口（20-64岁）预计下降21%
- 养老金、医疗、长期护理支出持续增加
- **结构性矛盾**：需要增加军费+老龄化支出增加+赤字已超标+增长低迷

---

## 3. 右转趋势量化

### 3.1 各国右翼政党支持率

**历史性突破——三大经济体右翼同时领先民调（2025年中）：**

| 国家 | 右翼/民粹政党 | 支持率 | 执政党/主要对手 | 支持率 |
|------|-------------|--------|---------------|--------|
| **德国** | AfD | 26% | CDU/CSU（执政） | 24% |
| **英国** | Reform UK | 27.3% | 保守党 | 20.0% |
| **法国** | RN (国民联盟) | 领先 | 执政联盟碎片化 | — |

**已执政或参与联合政府的右翼政党：**
- 意大利（Meloni的兄弟党）
- 荷兰（自由党PVV, Wilders）
- 瑞典（瑞典民主党支持少数政府）
- 芬兰（正统芬兰人党）
- 匈牙利（Fidesz, Orban）
- 斯洛伐克（Smer, Fico）
- 克罗地亚
- 比利时

### 3.2 已执政的右翼政府表现

#### 意大利 — Meloni：最大的"说一套做一套"案例
- **竞选承诺**：海军封锁阻止移民，大规模驱逐非法移民
- **实际执行**：
  - 批准了2000年代初以来最慷慨的劳工移民法令——3年内45万合法移民配额
  - 2018-2019年每年不到31,000临时工→2023-2025年45万+
  - 旗舰"阿尔巴尼亚方案"反复被法院否决——24名送往阿尔巴尼亚的庇护者全部未留下
  - **超过70%的意大利人认为Meloni在移民问题上"做得比承诺少"**
- **为什么右转后不能解决问题**：
  - 意大利经济需要移民劳动力（人口下降+老龄化）
  - 法律框架（欧洲人权法院、EU法律）限制极端行动
  - 务实派在政府内部压倒了意识形态派

#### 瑞典 — 从开放到收紧
- 中右翼政府+瑞典民主党支持，大幅收紧移民
- 2024年仅批准约6250份庇护相关居留许可——比执政时下降42%，**1985年以来最低**
- 取消了"轨道转换"通道（庇护转工签）
- **结果**：确实减少了移民数量，但代价是劳动力短缺加剧

#### 荷兰 — PVV 联合政府
- 引入紧急法律终止庇护者永久居留权
- 更严格的入籍要求和筛查
- 但对技术移民通道基本未动（经济需要）
- **政府已倒台**

#### 匈牙利 — Orban 面临罕见困境
- 190亿欧元EU资金被冻结（占GDP约16%）
- 2024年底永久失去10亿欧元（未在期限内满足条件）
- 2025年底前不解冻将再失去10亿
- ECJ罚款：2亿欧元一次性罚款+每日100万欧元（截至2025年2月已累计4.43亿）
- **2026年4月大选**：Fidesz首次在民调中排第二

### 3.3 西班牙：欧洲右转浪潮中的逆行者

西班牙在整个欧洲走向限制移民的趋势中选择了相反方向：
- 2026年1月宣布向数十万无证移民授予合法身份
- 条件：2025年12月31日前到达、在西班牙居住5个月以上、无犯罪记录
- **经济逻辑**：移民被视为填补劳动力缺口、推动西班牙"经济复兴"的关键
- 但Vox（右翼）支持率在缓慢上升

### 3.4 "右转但不解决问题"的模式总结

| 承诺 | 现实 | 原因 |
|------|------|------|
| 大规模驱逐 | 法律障碍重重 | 欧洲人权公约、EU法、国内法院 |
| 减少移民 | 劳工移民反而增加 | 经济需要（老龄化+劳动力短缺） |
| 削减福利给移民 | 有限调整 | 福利体系是普惠性的，难以针对性削减 |
| 退出EU/欧元 | 无一兑现 | 经济依赖度太高 |
| 降低赤字 | 反而恶化 | 军费增加+老龄化+低增长 |

**核心悖论**：右翼靠反移民上台→上台后发现经济需要移民+法律限制行动→无法兑现承诺→选民更加不满→更极端的右翼崛起

---

## 4. 对模型的参数建议

### 4.1 european_unity_index

**初始值建议：0.45**（范围 0.35-0.55）

驱动函数组件：

```python
# 欧洲团结指数驱动函数
european_unity_index = f(
    fiscal_divergence,      # 各国赤字/债务差异（负向）— 法德意三国差距扩大
    migration_pressure_perceived,  # 感知移民压力（负向）— 与实际流入脱钩
    defense_burden_sharing,  # 军费分摊争议（负向）— 搭便车问题
    ecb_policy_fit,         # ECB政策对各国适配度（负向）— 一刀切问题
    external_threat_level,   # 外部威胁（正向）— 俄罗斯威胁促团结
    eu_budget_transfers,     # EU内部转移支付争议（负向）
    right_wing_gov_count,    # 右翼政府数量（非线性，少量温和右转不一定分裂）

    # 时间效应
    election_cycle_europe,   # 选举周期（大选年团结度下降）
    eu_institutional_reform  # EU机构改革进展（正向但缓慢）
)
```

**初始值分解：**
- `fiscal_divergence` = 0.65（法德意差距大：法114%赤字5.8% vs 德62%赤字<3%）
- `migration_pressure_perceived` = 0.55（实际下降但感知滞后）
- `defense_burden_sharing` = 0.50（东欧>3% vs 西欧刚到2%）
- `ecb_policy_fit` = 0.40（法国需宽松，德国能承受紧缩）
- `external_threat_level` = 0.70（俄罗斯威胁是团结的最大粘合剂）
- `eu_budget_transfers` = 0.45（匈牙利资金冻结加剧对立）

### 4.2 各国领导人决策空间约束

#### Merz（德国总理，2025.3起）
```
fiscal_space: HIGH (债务仅62%, 5000亿基建基金)
political_space: MEDIUM (联合政府, AfD 26%施压)
policy_options:
  - 大规模基建投资 ✓
  - 军费扩张 ✓
  - 福利维持 ✓（短期内三者可并行，因债务空间大）
  - 难民政策收紧 ✓（回应AfD压力）
constraint: 宪法修正案已通过，执行空间大
time_horizon: 2025-2029任期
```

#### 法国（当前无稳定政府）
```
fiscal_space: VERY LOW (赤字5.8%, 债务114%, 无法通过预算)
political_space: MINIMAL (任何总理都面临不信任案威胁)
policy_options:
  - 紧缩 ✗（左翼+右翼同时反对）
  - 刺激 ✗（赤字已超标）
  - 改革 ✗（政治僵局）
  - 维持现状 ✓（唯一可行选项，但赤字继续恶化）
constraint: 第五共和国制度性瘫痪，Macron到2027年不能提前选举
time_horizon: 极短（几个月级别），下一次政府倒台随时可能
```

#### Meloni（意大利）
```
fiscal_space: LOW (债务138%, 但赤字收窄+利差收窄给了喘息空间)
political_space: MEDIUM-HIGH (目前政治稳定，但移民承诺未兑现)
policy_options:
  - 温和紧缩 ✓（S&P已上调评级）
  - 移民政策象征性收紧 ✓
  - 实质性移民减少 ✗（经济需要+法律障碍）
constraint: 务实派已压倒意识形态派，但民意转向是风险
time_horizon: 2022-2027任期，目前过半
```

### 4.3 "下一届领导人"不确定性量化

| 国家 | 下次大选 | 当前最可能继任 | 右翼概率 | 政策不连续性风险 |
|------|---------|--------------|---------|----------------|
| **法国** | 2027 | Le Pen / RN候选人 | 55-65% | 极高（可能退出部分EU承诺） |
| **德国** | 2029 | AfD进一步壮大 | 30%（联合执政）| 中高（AfD不太可能单独执政） |
| **意大利** | 2027 | Meloni连任 | 70% | 低（已务实化） |
| **英国** | 2029 | Reform UK | 35% | 高（可能重新定义与EU关系） |
| **荷兰** | ~2026-2027 | 不确定 | 50% | 中 |
| **匈牙利** | 2026.4 | Orban可能输 | 50%（反对派） | 中高（如Orban下台则亲EU转向） |
| **瑞典** | 2026 | 右翼继续 | 60% | 低（已完成政策转向） |

---

## 5. 与Monte Carlo的接口

### 5.1 状态变量建议

```python
# 欧洲状态空间（每月更新）
europe_state = {
    # 财政维度
    'fr_deficit_gdp': float,      # 法国赤字/GDP（当前5.8%）
    'it_debt_gdp': float,         # 意大利债务/GDP（当前138%）
    'de_investment_pace': float,   # 德国基建基金执行速度
    'eu_avg_deficit': float,       # EU平均赤字（当前3.2%）
    'btp_bund_spread': float,      # 意德利差（当前~90bp）
    'oat_bund_spread': float,      # 法德利差（关键监测）

    # 军费维度
    'nato_spending_gdp': float,    # 欧洲NATO成员平均军费/GDP（当前2.1%）
    'defense_vs_welfare_ratio': float,  # 军费vs福利支出比变化

    # 移民维度
    'asylum_applications_monthly': float,  # 月度庇护申请
    'migration_perception_gap': float,     # 感知vs实际移民差距

    # 政治维度
    'right_wing_vote_share_avg': float,    # 欧洲右翼平均支持率
    'government_stability_index': float,   # 政府稳定性（法国最低）
    'european_unity_index': float,         # 综合团结指数

    # 宏观维度
    'ecb_rate': float,              # ECB利率
    'eu_gdp_growth': float,         # EU GDP增长
    'eu_unemployment': float,       # EU失业率
    'eu_cpi': float,                # EU通胀
}
```

### 5.2 阈值定义（触发事件边界）

#### 财政崩溃阈值
```python
thresholds = {
    # 法国：赤字持续>6% + 利差>150bp = 财政危机
    'fr_fiscal_crisis': {
        'trigger': 'fr_deficit_gdp > 6.5 AND oat_bund_spread > 150',
        'probability_given_trigger': 0.60,
        'consequence': 'EU紧急财政干预 / ECB TPI激活',
        'time_to_materialize': '3-6个月'
    },

    # 意大利：利差突破250bp = 债务危机
    'it_debt_crisis': {
        'trigger': 'btp_bund_spread > 250 AND it_debt_gdp > 145',
        'probability_given_trigger': 0.40,
        'consequence': 'ECB TPI / 再次讨论退出欧元',
        'time_to_materialize': '1-3个月（金融市场反应快）'
    },

    # 移民危机升级：月度申请突然翻倍
    'migration_surge': {
        'trigger': 'asylum_applications_monthly > 120000',
        'probability_given_trigger': 0.30,
        'consequence': '边境管制恢复 / 申根暂停 / 政府倒台',
        'time_to_materialize': '1-2个月'
    },

    # 政府倒台（法国特别高）
    'government_collapse': {
        'france': {
            'trigger': 'government_stability_index < 0.2',
            'probability_per_quarter': 0.35,  # 法国当前每季度35%概率
            'consequence': '预算停摆 / 赤字进一步恶化'
        },
        'other_eu': {
            'trigger': 'government_stability_index < 0.3',
            'probability_per_quarter': 0.10,
            'consequence': '政策不确定性上升'
        }
    },

    # EU碎片化
    'eu_fragmentation': {
        'trigger': 'european_unity_index < 0.25',
        'probability_given_trigger': 0.20,
        'consequence': '财政规则实质失效 / 各自为政',
        'time_to_materialize': '6-12个月'
    }
}
```

### 5.3 时间尺度分析

| 变量 | 变化速度 | 突然恶化可能性 | 模型处理 |
|------|---------|-------------|---------|
| 财政赤字 | **慢变量**（年度预算周期） | 低（除非债务危机） | 月度线性插值，年度跳跃 |
| 债务/GDP | **慢变量**（累积效应） | 极低 | 年度更新 |
| BTP-Bund利差 | **快变量**（市场情绪驱动） | **高**（可在数日内翻倍） | 日度/周度波动，月度均值 |
| 庇护申请 | **中速变量**（季节性+事件驱动） | 中等（战争/灾难可突然推高） | 月度更新，事件触发跳跃 |
| 右翼支持率 | **慢变量**（结构性趋势） | 低（选举才兑现） | 季度更新 |
| 政府稳定性 | **中速→突然跳跃** | **高**（法国已证明） | 月度更新+事件触发 |
| 军费支出 | **慢变量**（预算周期） | 低 | 年度更新 |
| EU团结指数 | **慢变量** | 中（极端事件可快速改变） | 月度更新 |
| 人口老龄化 | **极慢变量** | 零 | 年度更新，确定性趋势 |

### 5.4 关键不确定性参数（Monte Carlo采样分布）

```python
monte_carlo_params = {
    # 法国赤字路径（高度不确定）
    'fr_deficit_2026': {
        'distribution': 'normal',
        'mean': 5.5,
        'std': 0.8,
        'bounds': [4.0, 7.5],
        'rationale': '政府承诺5%以下但反复失败'
    },

    # 意大利增长（低且不确定）
    'it_gdp_growth_2026': {
        'distribution': 'normal',
        'mean': 0.7,
        'std': 0.4,
        'bounds': [-0.5, 1.5],
        'rationale': '低增长是意大利结构性问题'
    },

    # 德国基建乘数效应
    'de_fiscal_multiplier': {
        'distribution': 'uniform',
        'low': 0.5,
        'high': 1.5,
        'rationale': '取决于执行效率和供给侧瓶颈'
    },

    # 欧洲军费达标速度
    'nato_spending_annual_increase_pp': {
        'distribution': 'triangular',
        'low': 0.1,
        'mode': 0.2,
        'high': 0.4,
        'rationale': '从2.1%到3.5%需每年增0.14pp最低'
    },

    # 右翼选举表现（下一轮选举）
    'rw_vote_share_shift': {
        'distribution': 'normal',
        'mean': +3.0,  # 百分点
        'std': 4.0,
        'bounds': [-5, +15],
        'rationale': '趋势上升但可能逆转'
    },

    # EU团结指数年度变化
    'unity_index_annual_change': {
        'distribution': 'normal',
        'mean': -0.03,  # 缓慢下降趋势
        'std': 0.08,
        'bounds': [-0.20, +0.15],
        'rationale': '趋势向下但外部冲击可逆转'
    }
}
```

### 5.5 关键情景路径

**路径A：缓慢衰退（基准，概率~50%）**
- 法国持续政治僵局但不崩溃
- 意大利利差稳定在100bp以下
- 德国基建拉动温和增长
- 右翼缓慢扩大但不颠覆
- european_unity_index: 0.45→0.38（3年内）

**路径B：法国财政危机（概率~20%）**
- 法国赤字超6%→利差飙升→ECB被迫干预
- 触发EU范围的财政规则重新谈判
- 右翼借机攻击EU体制
- european_unity_index: 0.45→0.25（1-2年内）

**路径C：外部威胁团结（概率~15%）**
- 俄罗斯军事升级→欧洲紧急团结
- 军费快速增加，财政规则实质暂停
- 移民问题被安全议题压过
- european_unity_index: 0.45→0.55（短期反弹）

**路径D：右翼浪潮（概率~10%）**
- Le Pen 2027当选法国总统
- AfD进入德国联合政府
- 多国同时右转→EU决策机制瘫痪
- european_unity_index: 0.45→0.20（2-3年内）

**路径E：意大利债务危机（概率~5%）**
- 全球利率意外上升+意大利增长停滞
- 利差突破300bp→欧元区存续性被质疑
- european_unity_index: 0.45→0.15（6-12个月急剧下降）

---

## 6. 数据来源

### 移民数据
- [Eurostat Migration 2025 Edition](https://ec.europa.eu/eurostat/web/interactive-publications/migration-2025)
- [EU Asylum Trends Shift in 2025](https://etias.com/articles/eu-migration-trends-shift-in-2025-asylum-claims-down,-border-crossings-fall)
- [EUAA Latest Asylum Trends](https://euaa.europa.eu/latest-asylum-trends)
- [EUAA: Asylum Applications Down 23% H1 2025](https://www.euaa.europa.eu/news-events/asylum-applications-down-23-first-half-2025)
- [EU Atlas of Migration 2026](https://home-affairs.ec.europa.eu/news/latest-atlas-migration-shows-decrease-first-asylum-applications-eu-while-displacement-rises-2026-01-30_en)
- [Eurostat Asylum November 2025](https://ec.europa.eu/eurostat/web/products-eurostat-news/w/ddn-20260217-1)
- [EU Net Migration - Macrotrends](https://www.macrotrends.net/global-metrics/countries/euu/european-union/net-migration)

### 财政数据
- [Eurogroup Statement on 2026 Budgetary Plans](https://www.consilium.europa.eu/en/press/press-releases/2025/12/11/eurogroup-statement-on-the-draft-budgetary-plans-for-2026/)
- [Eurostat Government Debt Q3 2025](https://ec.europa.eu/eurostat/web/products-euro-indicators/w/2-22012026-ap)
- [EU Debt Map 2025](https://www.eudebtmap.com)
- [European Commission Fiscal Forecast](https://economy-finance.ec.europa.eu/document/download/016d9065-1679-4d50-9f3a-e758dc9ae859_en)

### 法国
- [IMF France 2025 Article IV](https://www.imf.org/en/news/articles/2025/05/22/cs-france-2025)
- [France Government Collapse - TIME](https://time.com/7315503/france-government-collapse-emmanuel-macron-francois-bayrou-budget-national-rally/)
- [France Budget Crisis - CNBC](https://www.cnbc.com/2025/08/26/french-pm-takes-confidence-vote-gamble-over-budget-woes.html)
- [Barnier's Budget Plan - Bruegel](https://www.bruegel.org/first-glance/barniers-balancing-act-skilful-precarious-budget-plan)
- [France Budget - ING](https://think.ing.com/articles/frances-budget-headache-back-in-the-news/)

### 意大利
- [Italy BTP Spread 2026 - Il Sole 24 ORE](https://en.ilsole24ore.com/art/government-debt-assumption-2026-record-spread-area-50-investors-watch-italy-and-spain-AIfPaGg)
- [Italy BTP Yields 2026 - Morningstar](https://global.morningstar.com/en-eu/bonds/why-btp-yields-will-not-soar-with-2026-italian-budget)
- [Italy Government Bond Yield - Trading Economics](https://tradingeconomics.com/italy/government-bond-yield)
- [Amundi View on Italy Debt](https://research-center.amundi.com/article/view-italy-and-its-government-debt)
- [OECD Global Debt Report 2026](https://www.oecd.org/en/publications/2026/03/global-debt-report-2026_59d2d627/full-report.html)

### 德国
- [Germany Infrastructure Fund Q&A - Clean Energy Wire](https://www.cleanenergywire.org/factsheets/qa-germanys-eu500-bln-infrastructure-fund-whats-it-climate-and-energy)
- [EC Economic Impact of German Fiscal Reform](https://economy-finance.ec.europa.eu/economic-forecast-and-surveys/economic-forecasts/spring-2025-economic-forecast-moderate-growth-amid-global-economic-uncertainty/potential-economic-impact-reform-germanys-fiscal-framework_en)
- [Breaking the German Debt Brake - NEAM](https://www.neamgroup.com/insights/breaking-the-german-debt-brake)
- [Germany 2026 Federal Budget](https://www.bundesfinanzministerium.de/Content/EN/Pressemitteilungen/2025/2025-07-30-government-draft-2026-federal-budget.html)
- [German Debt Brake Reform for Europe - Bruegel](https://www.bruegel.org/newsletter/what-does-german-debt-brake-reform-mean-europe)

### 军费与国防
- [NATO Defence Expenditure 2014-2025 (PDF)](https://www.nato.int/content/dam/nato/webready/documents/finance/def-exp-2025-en.pdf)
- [NATO 5% Commitment](https://www.nato.int/en/what-we-do/introduction-to-nato/defence-expenditures-and-natos-5-commitment)
- [EU Defence in Numbers - Consilium](https://www.consilium.europa.eu/en/policies/defence-numbers/)
- [European Defense by the Numbers - McKinsey](https://www.mckinsey.com/industries/aerospace-and-defense/our-insights/european-defense-by-the-numbers)
- [Price of Security - WEF](https://www.weforum.org/stories/2025/07/europe-defence-recall-difficult-past-nato/)

### 右翼趋势
- [Far-Right Top Polls in Germany, France, Britain - NBC](https://www.nbcnews.com/world/europe/far-right-populists-top-polls-germany-france-britain-first-time-rcna224706)
- [Far-Right Lead in Europe's Four Largest Economies - Axios](https://www.axios.com/2025/08/22/europe-far-right-afd-germany-france-uk)
- [European Radical Right in Age of Trump 2.0 - Carnegie](https://carnegieendowment.org/research/2025/09/the-european-radical-right-in-the-age-of-trump-20?lang=en)
- [Creeping Integration of Far-Right - SWP Berlin](https://www.swp-berlin.org/en/publication/the-creeping-integration-of-far-right-parties-in-europe)
- [Pew: How People View Political Parties 2025](https://www.pewresearch.org/short-reads/2025/09/15/how-people-in-24-countries-feel-about-their-political-parties/)

### Meloni/移民政策
- [Meloni Immigration Paradox - Springer](https://link.springer.com/article/10.1186/s40878-025-00438-y)
- [Meloni's Struggle with Migration - Illiberalism.org](https://www.illiberalism.org/only-the-pragmatic-survive-the-meloni-governments-struggle-with-migration/)
- [Two Years of Anti-Immigrant Policy - InfoMigrants](https://www.infomigrants.net/en/post/60758/two-years-of-antiimmigrant-policy-in-giorgia-melonis-italy)
- [Spain Embracing Immigration - IR Review](https://www.irreview.org/articles/2026/3/2/spains-bet-on-the-future-embracing-immigration-in-an-increasingly-restrictive-europe)
- [Spain Grants Legal Status - NPR](https://www.npr.org/2026/01/28/g-s1-107636/spain-legal-status-immigrants)
- [Sweden Immigration Laws 2025 - Dispatches Europe](https://dispatcheseurope.com/what-expats-need-to-know-how-swedens-immigration-laws-are-changing-in-2025/)

### EU资金/规则
- [Freezing EU Funds - CER](https://www.cer.eu/insights/freezing-eu-funds-effective-tool-enforce-rule-law)
- [Hungary Loses EU Aid >1B - European Newsroom](https://europeannewsroom.com/hungary-loses-right-to-eu-aid-worth-more-than-e1-billion/)
- [Hungary Rule of Law 2025 - eucrim](https://eucrim.eu/news/hungary-rule-of-law-developments-in-2025/)

### 老龄化/福利
- [EU 2024 Ageing Report](https://economy-finance.ec.europa.eu/publications/2024-ageing-report-economic-and-budgetary-projections-eu-member-states-2022-2070_en)
- [Demographic Change and Debt Sustainability - Bruegel](https://www.bruegel.org/policy-brief/how-demographic-change-will-hit-debt-sustainability-european-union-countries)
- [Fiscal Impact of Population Ageing - OECD](https://oecdecoscope.blog/2025/11/07/the-fiscal-impact-of-population-ageing-how-can-we-afford-getting-older/)
- [ECB Macroeconomic Impact of Ageing](https://www.ecb.europa.eu/pub/pdf/scpops/ecb.op296~aaf209ffe5.en.pdf)
