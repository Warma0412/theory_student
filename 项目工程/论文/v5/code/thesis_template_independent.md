---
title: 基于去随机化Model-X Knockoff的电商GMV多粒度受控维度选择研究
subtitle: 基于去随机化Model-X Knockoff、组级稳健性与可解释机器学习
author: |
  学校：【待填写】  
  学院：【待填写】  
  专业：【待填写】  
  研究生：吴优  
  学号：2025213385  
  指导教师：张吕欧
date: 二〇二六年九月
lang: zh-CN
---

# 原创性与学术诚信说明

本文以公开数据为研究对象，数据处理、统计计算、模型训练和图表生成均保留可复核的程序与机器可读结果。所有实证数字均由本地全量数据运行得到，不使用示例数字。作者应根据所在学校的正式模板替换本页，补充原创性声明、论文使用授权及签名信息，并按照学校规定披露程序和人工智能工具的辅助范围。论文作者对研究问题、模型设定、参考文献、结果解释和最终文字承担学术责任。

# 摘要

电商经营看板通常同时包含价格、交易规模、供给、支付、履约、口碑、地域和营销等大量指标。若仅依据边际相关、单变量显著性或单次机器学习重要性筛选监控维度，高度相关的代理变量容易被重复保留，入选集合中的误选比例也缺乏明确约束。本文研究在多个候选经营维度中筛选对下一期核心指标仍具有增量预测信息的变量，并将错误发现率控制、预测验证与可调展示数量纳入统一框架。

研究使用Olist巴西电商公开数据，在订单、商品明细、支付、评价、商品、卖家、顾客、地理位置和营销漏斗的自然粒度上先聚合后连接。月度主分析形成{{PANEL_ROWS}}个卖家-月观测、{{PANEL_SELLERS}}个卖家和{{PANEL_MONTHS}}个月；周度稳健性分析形成{{WEEK_ROWS}}个卖家-周观测、{{WEEK_SELLERS}}个卖家和{{WEEK_COUNT}}周。本文构造{{CANDIDATE_COUNT}}项候选维度，排除一项零方差变量后，使用31项本期特征预测下一期`log(1+GMV)`。该目标避免件数、订单数和均价对同周期GMV的机械解释。

方法上，本文采用随机化秩高斯Copula处理混合边际，以Ledoit-Wolf收缩协方差和最小方差重构准则生成二阶Model-X Knockoff，并以Lasso真实变量与仿制变量系数绝对值之差构造反对称统计量。每个主分析重复生成60组Knockoff，设置单轮水平`α_kn=0.10`、最终水平`α_eBH=0.20`，通过e-value聚合和e-BH实施去随机化选择。在确认集合内，使用训练期XGBoost和验证集TreeSHAP建立可调参数$K$的展示路径。研究进一步实施卖家均值—离差分层、十二业务组Group Knockoff、深度生成器交换性诊断、GRIP2式轨迹重要性、双向固定效应和自然时间外预测。

月度主分析在最终FDR水平0.20下入选{{STRICT_COUNT}}项，其中{{STABLE_COUNT}}项在单轮程序中的入选频率不低于90%。基于验证集贡献与预测误差的联合规则，默认推荐$K={{V3_DEFAULT_K}}$；对应测试RMSE为{{V3_RMSE}}，与全31维XGBoost基本一致。周度分析在相同统计流程下入选{{WEEK_STRICT_COUNT}}项，与月度集合共同入选{{OVERLAP_COUNT}}项，Jaccard系数为{{JACCARD}}。周度时间外最佳模型为{{WEEK_BEST_MODEL}}，RMSE为{{WEEK_BEST_RMSE}}；只使用周度入选集合的XGBoost RMSE为{{WEEK_SELECTED_RMSE}}。月度与周度卖家内离差层均未形成严格发现，表明当前数据中的主要信号更多来自卖家间长期差异。

研究结果表明，月度聚合能够降低低频卖家的零膨胀与偶发订单噪声，适合作为确认性主分析；周度聚合增加时间点，但稀疏性和共线性更强，适合作为粒度稳健性和短期预警补充。本文识别的是条件预测信息，不将Knockoff、SHAP或固定效应结果解释为因果效应。研究为高维经营看板提供了“受控确认—参数化展示—多粒度复核”的可审计维度治理路径。

**关键词：** 电商GMV；Model-X Knockoff；错误发现率；多粒度面板；可解释机器学习

# Abstract

E-commerce dashboards commonly contain a large number of correlated indicators covering transaction scale, price, assortment, payment, fulfillment, reputation, geography, and marketing. Selecting monitoring dimensions solely by marginal correlation, individual significance, or a single machine-learning importance ranking may retain redundant proxies and provides no explicit control over the proportion of false discoveries. This study develops an integrated framework for identifying variables with incremental predictive information for next-period gross merchandise value while controlling the false discovery rate and allowing a configurable display size.

The empirical analysis uses the public Brazilian Olist e-commerce data. Source tables are aggregated at their natural grains before joining to prevent many-to-many multiplication of item values. The monthly primary panel contains {{PANEL_ROWS}} seller-month observations for {{PANEL_SELLERS}} sellers over {{PANEL_MONTHS}} months. The weekly robustness panel contains {{WEEK_ROWS}} seller-week observations for {{WEEK_SELLERS}} sellers over {{WEEK_COUNT}} weeks. Thirty-two candidate dimensions are constructed; one constant variable is excluded, and 31 current-period features are used to predict next-period `log(1+GMV)`.

A randomized rank-Gaussian copula is used for mixed marginals. Second-order Model-X knockoffs are generated from a Ledoit-Wolf covariance estimate under the minimum-variance-reconstructability criterion. Antisymmetric statistics are defined by the absolute coefficient difference between original and knockoff variables in a joint Lasso model. Sixty independent knockoff realizations are aggregated through e-values, with `α_kn=0.10` and a final e-BH level of `α_eBH=0.20`. Within the confirmatory set, a training-period XGBoost model and validation-period TreeSHAP values define a user-controlled top-$K$ display path. Clustered mean-deviation knockoffs, business-group knockoffs, deep exchangeability diagnostics, trajectory-integrated nonlinear importance, two-way fixed effects, and out-of-time prediction provide complementary evidence.

The monthly analysis selects {{STRICT_COUNT}} dimensions at the final FDR level of 0.20, of which {{STABLE_COUNT}} are selected in at least 90% of individual runs. The prespecified rule recommends $K={{V3_DEFAULT_K}}$, with a test RMSE of {{V3_RMSE}}. The weekly analysis selects {{WEEK_STRICT_COUNT}} dimensions; {{OVERLAP_COUNT}} overlap with the monthly set, yielding a Jaccard index of {{JACCARD}}. The best weekly predictor is {{WEEK_BEST_MODEL}} with an RMSE of {{WEEK_BEST_RMSE}}, while the XGBoost model using only weekly selected variables obtains an RMSE of {{WEEK_SELECTED_RMSE}}. Neither the monthly nor weekly within-seller layer yields a strict discovery, indicating that most identifiable signals arise from persistent differences across sellers.

The evidence supports monthly aggregation as the confirmatory design for low-frequency sellers and weekly aggregation as a robustness and short-horizon monitoring layer. The selected variables represent conditional predictive information rather than causal effects. The proposed framework provides an auditable route from controlled discovery to configurable dashboard presentation and multi-granularity verification.

**Key words:** e-commerce GMV; Model-X knockoff; false discovery rate; multi-granularity panel; interpretable machine learning

# 第1章 绪论

本章围绕电商经营指标过载与维度准入缺乏统计约束的问题展开。首先说明研究背景与实践意义，随后界定条件预测型维度选择问题，梳理相关研究的能力边界，最后给出研究问题、主要工作、创新点和全文结构。

## 1.1 研究背景与意义

数字平台能够持续记录获客、下单、支付、发货、送达和评价等经营过程。记录能力的提升使企业可以构建覆盖多个业务环节的监控看板，但也带来指标冗余：订单数、件数和顾客数共同反映交易规模；商品价格、运费和物理属性可能共同反映商品结构；履约时长指标之间也存在明显关联。若不断增加字段而缺乏准入与退出机制，看板将难以支持快速判断。

传统筛选方式主要包括经验判断、相关系数、逐项显著性检验和机器学习重要性排序。经验判断依赖人员知识且难以复核；边际相关和单变量检验没有控制其余候选变量；树模型重要性能够处理非线性，却通常只给出相对排序。更重要的是，当候选变量较多时，多次检验会累积误报，而相关代理变量可能因共同解释同一信号而重复进入结果。

错误发现率（False Discovery Rate，FDR）关注全部入选变量中错误发现比例的期望。与严格控制任一错误的族错误率相比，FDR允许在整体错误比例受控的条件下保留更多真实信号，适合需要兼顾探索效率和结果可信度的指标治理场景[1]。Model-X Knockoff通过为每个真实变量构造保持特征依赖结构的仿制变量，使真实变量与负对照变量在同一模型中竞争，从而把变量筛选转化为可审计的受控发现问题[2-3]。

## 1.2 问题定义与识别边界

本文研究的业务问题是：给定卖家在当前周期的多个经营维度，哪些变量在控制其余候选信息后，仍包含下一周期GMV的增量预测信息，并值得进入持续监控体系。月度主分析的条件零假设定义为

$$
H_{0j}:Y_{s,t+1}\perp X_{j,s,t}\mid X_{-j,s,t},
$$

其中，$Y_{s,t+1}$为卖家$s$在次月的对数GMV，$X_{j,s,t}$为本月第$j$个候选维度。周度分析将$t$替换为自然周$w$，其余定义保持一致。

拒绝该零假设表示变量包含其他候选变量未覆盖的条件预测信息。该结论不表示人为改变变量一定会改变GMV，也不等价于结构因果效应。订单数和顾客数更接近状态变量；履约、内容和定价变量即使具备可行动性，也仍需要随机实验或准实验识别干预效应。

## 1.3 相关研究与研究空白

多重检验研究从单项第一类错误扩展到发现集合的整体质量控制。Benjamini和Hochberg提出FDR控制程序[1]；Wang和Ramdas进一步发展基于e-value的FDR控制[6]。Lasso和Elastic Net能够形成稀疏模型[7-8]，但交叉验证选择的正则参数以预测误差为目标，并不自动约束错误发现比例。

Model-X Knockoff不要求正确指定响应条件分布，而把关键假设放在特征联合分布与成对可交换性上[3]。二阶高斯方法具有清晰的矩结构，MVR准则通过降低真变量与仿制变量的可重构性提高功效[9-10]。针对非高斯和复杂依赖，Deep Knockoffs、DDLK及依赖正则方法尝试以神经网络学习仿制变量[4,33-35]。然而，边际分布接近不等于联合交换性成立，深度生成器仍需独立的两样本诊断。

在预测和解释方面，随机森林、梯度提升、XGBoost和多层感知机能够拟合非线性关系[11-13]，TreeSHAP可以分解模型预测贡献[14]。这些方法适合评价入选集合的时间外预测价值，但模型重要性不等于条件独立检验，更不等于因果效应。现有电商研究通常关注销售预测、用户分群或推荐，较少把FDR控制、随机稳定性、多粒度复核和看板展示数量纳入同一框架。

据此，本文聚焦三个研究空白：其一，混合分布且高度相关的经营维度缺乏可控误选的准入机制；其二，随机生成仿制变量造成的结果波动需要正式聚合，而非任意频率阈值；其三，月度与周度聚合可能改变信号强度和稀疏结构，维度结论需要跨粒度验证。

## 1.4 研究问题、研究内容与贡献

围绕上述研究空白，本文回答四个问题。第一，如何在多表电商数据中构造不重复计算GMV的卖家面板；第二，如何在31项混合且相关的候选变量中控制错误发现率；第三，如何在不改变统计确认集合的前提下，通过参数$K$控制最终展示数量；第四，月度结论在周度面板、组级竞争、卖家内变化和非线性模型下是否保持稳定。

本文的主要工作包括以下四项。

1. 在原始表自然粒度上先聚合后连接，分别构建卖家-月与卖家-周面板，并采用下一周期GMV避免同周期定义泄漏。
2. 构建随机化秩高斯Copula-MVR Knockoff、Lasso反对称统计量、60轮e-value聚合和e-BH组成的确认性选择流程。
3. 在确认集合内使用验证集TreeSHAP构造$K\in[1,14]$的参数化展示路径，并以自然时间外预测检验压缩后的信息保留程度。
4. 通过卖家均值—离差分层、业务组Knockoff、深度交换性诊断、轨迹重要性和周度完整重跑，检验结论对相关结构、模型形式和时间粒度的敏感性。

本文的贡献不在于提出新的Knockoff理论，而在于形成一套面向经营指标治理的可复核应用框架：统计确认、展示数量、模型解释和稳健性证据分别承担不同结论权限，避免把预测重要性、稳定频率或失败的深度生成结果混写为正式发现。

## 1.5 论文结构

全文共七章。第1章说明研究背景、问题、研究空白和贡献；第2章介绍FDR、Model-X Knockoff、e-value、相关结构处理及机器学习解释；第3章说明数据来源、面板构造、变量和评价设计；第4章给出统一方法体系；第5章报告月度主分析与各类稳健性结果；第6章讨论跨粒度解释、看板治理和适用边界；第7章总结研究结论、贡献、局限与后续方向。

# 第2章 理论基础与相关方法

第1章将研究目标界定为受控的条件预测型维度选择。本章进一步说明该目标所依赖的统计概念和模型工具，重点讨论FDR、Model-X Knockoff、去随机化e-value、面板相关结构、深度交换性诊断及可解释机器学习之间的分工。

## 2.1 多重检验与错误发现率

设$R$为全部发现数，$V$为其中错误发现数，则FDR定义为

$$
\mathrm{FDR}=\mathbb{E}\left[\frac{V}{R\vee1}\right].
$$

Bonferroni方法控制族错误率，在候选变量较多或信号相关时可能过于保守。FDR允许有限探索，同时约束发现集合的平均错误比例，更适合指标准入。需要强调，FDR是重复抽样意义下的目标控制水平，不是对某一次具体入选集合中错误比例的确定性描述。

## 2.2 Model-X Knockoff与反对称统计量

Model-X Knockoff为真实特征$X$构造仿制特征$\widetilde X$。理想仿制变量满足

$$
(X,\widetilde X)_{\mathrm{swap}(S)}
\overset{d}{=}(X,\widetilde X),\qquad
\widetilde X\perp Y\mid X,
$$

其中$\mathrm{swap}(S)$表示交换任意变量子集中的真实变量和对应仿制变量。仿制变量不是简单打乱，也不是在原变量上添加独立噪声，而是在保留特征依赖结构的同时提供响应无关的负对照。

将真实变量和仿制变量共同输入预测器，得到重要性$Z_j$和$\widetilde Z_j$后，可定义

$$
W_j=|Z_j|-|\widetilde Z_j|.
$$

交换$X_j$和$\widetilde X_j$时，$W_j$应改变符号。零变量的统计量符号在有效Knockoff条件下具有对称性，因而可用负向统计量估计正向发现中的错误数量。

## 2.3 去随机化Knockoff与e-value

单次Knockoff生成具有随机性。若仅报告某个随机种子或以任意入选频率聚合，多次运行不会自动继承FDR保证。本文采用去随机化Knockoff方法[5]。第$m$轮中第$j$个变量的e-value定义为

$$
e_j^{(m)}
=p\frac{\mathbf{1}\{W_j^{(m)}\ge T^{(m)}\}}
{1+\sum_{k=1}^{p}\mathbf{1}\{W_k^{(m)}\le -T^{(m)}\}},
$$

其中$T^{(m)}$为Knockoff+阈值。对$M$轮e-value取平均，

$$
\bar e_j=\frac{1}{M}\sum_{m=1}^{M}e_j^{(m)},
$$

再对$\bar e_j$实施e-BH。本文区分最终水平$\alpha_{\mathrm{eBH}}$与单轮水平$\alpha_{\mathrm{kn}}$，主设定分别为0.20和0.10。单轮入选频率仅作为随机稳定性描述，不承担正式FDR声明。

## 2.4 相关结构、深度诊断与组级选择

卖家面板同时包含跨卖家差异和同一卖家随时间变化。直接把所有观测视为独立可能低估组内相关风险。本文将时变特征分解为卖家均值和卖家内离差，分别检验长期位置差异与短期动态信息。对于高度共线的代理变量，进一步按价格成本、商品物理、商品内容、支付、履约、交易规模、市场覆盖和营销等业务概念构造Group Knockoff，以组内真实和仿制系数的$L_2$范数差定义组级统计量[20]。

深度Knockoff能够表达复杂非线性联合分布，但模型名称不能替代交换性验证。本文同时使用边际KS、协方差误差、交叉协方差非对称度、固定核MMD、深度核MMD和分类器两样本检验。深度核在训练子集学习表示，在独立检验子集执行置换，避免在同一数据上学习并检验。未通过预设诊断的生成器不进入FDR结论。

## 2.5 可解释机器学习与面板模型

XGBoost、Extra Trees和多层感知机用于自然时间外预测，评价指标包括对数RMSE、MAE、$R^2$和原始金额WAPE。TreeSHAP用于解释模型在验证或测试样本上的平均绝对贡献[14]，其作用是对确认集合排序，而不是独立决定显著性。

双向固定效应模型控制卖家不随时间变化的特征和共同周期冲击，标准误按卖家聚类[17-18]。该模型能够提供同一卖家随时间变化的条件关联，但仍可能受到促销、曝光、库存和竞争等时变遗漏变量影响。因此，固定效应与SHAP均作为补充证据，不转写为因果效应。

# 第3章 数据来源与研究设计

第2章明确了不同统计工具的结论权限。本章转向数据构造，说明原始表、连接顺序、分析单位、目标变量、候选维度和时间切分。数据处理的核心原则是保持业务金额可对账，并确保模型只比较定义明确的候选信息。

## 3.1 数据来源与自然粒度连接

主数据来自Olist发布的Brazilian E-Commerce Public Dataset[27]，营销信息来自Marketing Funnel by Olist[28]。实际读取的10张CSV及其用途见表3-1。

**表3-1 原始数据表及实际读取规模**

{{SOURCE_TABLE}}

支付和评价表可能一单多行，商品明细也可能一单多件。若直接连接全部明细后再汇总，商品价格和运费会被支付行或评价行重复。本文先在各表自然粒度上聚合：支付和评价聚合至订单，地理位置按邮编前缀取经纬度中位数，商品明细保留卖家和商品粒度；随后再按卖家与周期连接。

顾客数量使用`customer_unique_id`去重，而非订单级`customer_id`。样本包含{{DELIVERED_ORDERS}}笔已送达订单和{{DELIVERED_ITEMS}}行已送达商品明细。处理后的GMV与原始商品明细逐笔对账一致，月度和周度面板主键重复数均为零。

## 3.2 月度与周度分析单位

月度主分析使用卖家$s$在月份$t$的经营状态预测次月GMV：

$$
Y_{s,t+1}=\log(1+\mathrm{GMV}_{s,t+1}).
$$

特征期为{{PANEL_START}}至{{PANEL_END}}，得到{{PANEL_ROWS}}个卖家-月、{{PANEL_SELLERS}}个卖家和{{PANEL_MONTHS}}个月。若卖家本月活跃而次月无已送达订单，则次月GMV记为零，零值率为{{ZERO_RATE}}。

周度分析以周一至周日为一个自然周，使用卖家$s$在周$w$的经营状态预测下一周GMV：

$$
Y_{s,w+1}=\log(1+\mathrm{GMV}_{s,w+1}).
$$

周度特征期为{{WEEK_START}}至{{WEEK_END}}，得到{{WEEK_ROWS}}个卖家-周、{{WEEK_SELLERS}}个卖家和{{WEEK_COUNT}}周。下一周零值率为{{WEEK_ZERO_RATE}}。周度增加时间点，但每个卖家活跃周数中位数仅为{{WEEK_MEDIAN_ACTIVE}}，因此稀疏性高于月度。

**表3-2 月度与周度研究口径**

{{WEEK_GRAIN_TABLE}}

## 3.3 候选维度与数据质量

本文构造{{CANDIDATE_COUNT}}个候选维度，覆盖价格、运费、商品物理属性、内容丰富度、支付、口碑、履约、交易规模、市场覆盖、卖家地域和营销。申报月收入对数在面板中恒为零，按照事前零方差规则排除，最终{{ANALYSIS_COUNT}}项进入模型。

**表3-3 候选维度及处理状态**

{{FEATURE_TABLE}}

月度面板最大特征缺失率为{{MISSING_MAX}}。物流时长中的负值和明显超界值设为缺失；距离超过5000公里的异常记录也设为缺失。所有缩尾点、插补值、偏态变换和标准化参数只在相应训练样本或当前推断样本的特征矩阵上估计，不使用未来响应。

## 3.4 描述统计与分布特征

关键变量的描述统计见表3-4。订单数、件数、顾客数和买家州数量具有明显右偏；支付方式占比和营销标记包含大量0或1；物流时长和距离存在长尾。这些分布特征说明直接使用原始高斯假设可能造成边际失配。

**表3-4 关键变量描述统计**

{{KEY_DESC_TABLE}}

![图3-1 月度GMV变化](figures/fig01_monthly_gmv.png)

![图3-2 次月对数GMV分布](figures/fig02_target_distribution.png)

月度GMV在{{PEAK_MONTH}}达到样本期峰值{{PEAK_GMV_THOUSAND}}千雷亚尔，平台规模随时间呈明显变化。时间趋势可能同时推高多个变量，因此预测评估采用自然时间切分，动态关联补充采用周期固定效应。

## 3.5 训练验证测试划分与评价指标

月度预测将2017-01至2017-12作为训练期，2018-01至2018-04作为验证期，2018-05至2018-07作为测试期，对应样本量分别为{{TRAIN_N}}、{{VALID_N}}和{{TEST_N}}。周度预测使用同样的自然时间边界：2017-01-02至2017-12-25为训练期，2018-01-01至2018-04-30为验证期，2018-05-07至2018-07-30为测试期。

模型调参与参数$K$推荐只使用训练期和验证期；测试期在规则确定后用于一次性评价。对数RMSE和MAE衡量典型预测误差，$R^2$衡量相对均值基线的解释比例，WAPE保留原始GMV金额尺度。本文不以单一预测指标替代统计选择，而是检验入选集合是否保留足够的时间外信息。

# 第4章 多粒度可控维度选择方法

第3章构建了月度和周度面板，并说明了混合边际、共线性和稀疏性。针对这些特征，本章建立由确认性选择、参数化展示、面板结构检验、深度交换性诊断和跨粒度复核组成的统一方法体系。

## 4.1 Copula-MVR Knockoff与确认性选择

预处理首先对连续变量实施1%和99%分位缩尾，对非负偏态变量取`log(1+x)`，随后以中位数插补并标准化。为处理离散值和并列秩，第$j$个变量通过固定随机种子打散并列值，再映射为近似标准正态：

$$
Z_{ij}=\Phi^{-1}\left(\frac{r_{ij}-0.5}{n}\right).
$$

在潜在高斯空间中，使用Ledoit-Wolf收缩协方差$\widehat\Sigma$，并按MVR准则确定对角矩阵$S$。条件高斯形式为

$$
\widetilde Z\mid Z
\sim \mathcal N\left(
Z-Z\widehat\Sigma^{-1}S,\,
2S-S\widehat\Sigma^{-1}S
\right).
$$

将$[Z,\widetilde Z]$共同输入Lasso，得到真实和仿制变量系数，构造$W_j=|\widehat\beta_j|-|\widehat\beta_{j+p}|$。每轮按Knockoff+计算阈值并形成e-value，60轮平均后实施e-BH。月度主分析在$\alpha_{\mathrm{eBH}}=0.20$下承担正式逐变量结论；0.10和0.30仅用于阈值敏感性。

## 4.2 参数$K$与时间外预测验证

确认性程序解决“哪些变量获得受控证据”，参数$K$解决“页面最终显示多少项”。设月度确认集合为$S_{\mathrm{FDR}}$，训练期XGBoost在验证集上得到平均绝对TreeSHAP排序$\pi$，则

$$
S(K)=\{\pi_1,\ldots,\pi_K\},\qquad K\in\{1,\ldots,|S_{\mathrm{FDR}}|\}.
$$

默认$K$取满足以下两个条件的最小值：累计验证SHAP贡献至少90%，且验证RMSE不高于全部候选$K$中最优值的100.5%。测试集不参与排序或默认值确定。$K$只改变展示长度，不重新计算FDR，也不把排名前列解释为更强的因果效应。

预测模型包括Ridge、Lasso、Extra Trees、XGBoost和MLP，并与本期GMV直接外推的朴素基线比较。TreeSHAP、置换重要性和双向固定效应提供解释性证据；它们与Knockoff的统计目标不同，因此分别报告。

## 4.3 面板分层与业务组稳健性

为区分长期卖家差异与同一卖家短期变化，对每个时变特征实施

$$
X_{s,t}=\bar X_s+(X_{s,t}-\bar X_s).
$$

卖家间层以每个卖家的时间均值建模，卖家内层以离差建模。两个层级分别拟合Copula-MVR采样器、选择Lasso惩罚并重复40轮。时间不变特征在离差层成为零方差并自动排除。

逐变量竞争可能把同一业务概念拆分到多个相关代理。本文事先把31项变量划分为12组，并以组内真实系数与仿制系数的$L_2$范数差构造组级$W_g$。组级分析重复60轮并报告0.20和0.30两个水平，用于评价业务模块是否整体包含信息。

## 4.4 深度生成诊断与轨迹重要性

噪声条件深度生成器以真实特征和同维高斯噪声为输入，联合优化多比例swap-MMD、边际、协方差、交叉对称与可重构性损失。进一步设置可微最坏swap对手，在每批次搜索可区分性较大的交换子集。经验边际校准只在生成器训练子集拟合，再应用于验证子集。

生成器准入依赖多指标联合诊断。边际KS衡量单变量匹配，协方差误差与交叉非对称度衡量二阶结构；固定核MMD、深度核MMD与Extra Trees分类器检验联合交换差异。199次核置换的最小可达$p$值为0.005，99次分类器置换的最小可达$p$值为0.010。任一关键联合诊断明显失败时，该生成器的选择结果不作FDR声明。

非线性重要性使用配对竞争MLP。网络为每个真实/仿制变量设置成对入口权重，并通过共享下游结构保持交换反对称性。轨迹版本在4个$L_1$强度与4个组正则强度上循环采样，沿优化轨迹累计真实与仿制入口活跃度之差。数值交换测试用于确认统计量在交换后变号。

## 4.5 周度粒度稳健性

周度分析完整复用31项候选维度、Copula-MVR、Lasso竞争、60轮e-value和e-BH流程，并重跑高斯生成、同周GMV、截尾窗口、下一周客单价、测试期前样本和XGBoost统计量。聚类分层、业务组、轨迹重要性、固定效应和预测模型也在周度面板上重新估计。

该设计用于隔离时间聚合粒度的影响，而不是把月度和周度结果合并为一个更大的确认集合。月周重合程度使用交集数量和Jaccard系数评价：

$$
J(S_m,S_w)=\frac{|S_m\cap S_w|}{|S_m\cup S_w|}.
$$

两种粒度均采用“最终已送达订单按购买周期归属”的回溯口径，以保证统计比较的一致性。物流和评价信息在周期结束时未必已经完全可见，因此周度实验不是严格的在线部署回放；实际部署需按事件真实可见时间归属或设置滞后。

# 第5章 实证结果

第4章给出了统一方法体系。本章按照“数据质量与生成诊断—月度确认性结果—预测与参数化展示—结构和深度稳健性—周度粒度复核”的顺序报告结果。每项分析均区分正式发现、稳定性描述和探索性证据。

## 5.1 数据质量与Knockoff诊断

月度面板包含{{PANEL_ROWS}}行，样本GMV为{{GMV_MILLION}}百万雷亚尔，最大缺失率为{{MISSING_MAX}}。面板主键无重复，处理后GMV与原始已送达商品明细一致。月度面板SHA-256为`{{PANEL_SHA256}}`。

Copula生成器的平均边际KS为{{COPULA_MEAN_KS}}，最大KS为{{COPULA_MAX_KS}}，协方差相对误差为{{COPULA_COV_ERROR}}，交叉协方差非对称度为{{COPULA_CROSS_ASYM}}。原始高斯生成器的平均边际KS为{{GAUSSIAN_MEAN_KS}}。Copula明显改善混合边际匹配，因此确认性分析采用Copula-MVR。

**表5-1 深度与基线生成器的联合诊断**

{{JOINT_DIAGNOSTIC_TABLE}}

基础深度生成器和最坏swap生成器虽然取得较小的边际KS，但深度核MMD在全部交换比例上拒绝，分类器AUC也明显高于0.5。两类生成器均未通过预设准入，故其形式上的选择集合只作为失败诊断，不进入FDR结论。该结果表明，边际匹配不能替代联合交换性。

![图5-1 生成器联合交换性诊断](figures/fig13_exchangeability.png)

## 5.2 月度确认集合与参数化展示

月度主分析在最终FDR水平0.20下入选{{STRICT_COUNT}}项：{{STRICT_LIST}}。其中{{STABLE_COUNT}}项的单轮入选频率不低于90%：{{STABLE_LIST}}。

**表5-2 月度主分析前20项变量**

{{PRIMARY_TABLE}}

![图5-2 月度主分析平均e-value](figures/fig03_knockoff_evalues.png)

![图5-3 月度主分析单轮入选频率](figures/fig04_selection_frequency.png)

在0.10水平下，严格发现为{{Q10_COUNT}}项；在0.30水平下，严格发现为{{Q30_COUNT}}项。阈值变化见表5-3。结果从0项跳至14项，反映$p=31$时Knockoff+的离散门槛，不能通过临时降低FDR水平稳定地获得任意长度的集合。

**表5-3 月度FDR水平敏感性**

{{SENSITIVITY_TABLE}}

确认集合内部的验证集TreeSHAP排序形成参数$K$路径。默认规则推荐$K={{V3_DEFAULT_K}}$，对应维度为{{V3_SHORTLIST_LIST}}。

**表5-4 默认参数K的维度与证据**

{{K_SHORTLIST_COMPACT_TABLE}}

**表5-5 代表性K档位**

{{K_PATH_COMPACT_TABLE}}

默认$K={{V3_DEFAULT_K}}$的测试RMSE为{{V3_RMSE}}，全31维XGBoost为{{V3_FULL_RMSE}}，相对变化{{V3_RMSE_CHANGE}}。结果说明在确认集合内进一步压缩展示数量，不会明显损失时间外预测性能。

## 5.3 时间外预测、SHAP与固定效应

月度时间外预测结果见表5-6。最佳模型为{{BEST_MODEL}}，测试RMSE为{{BEST_RMSE}}、$R^2$为{{BEST_R2}}，相对本月GMV朴素基线改善{{RMSE_IMPROVEMENT}}。树模型与线性模型误差接近，深度MLP未形成稳定优势，说明当前样本的主要信息可由较平滑的表格模型捕获。

**表5-6 月度时间外预测表现**

{{V2_PREDICTIVE_TABLE}}

![图5-4 月度时间外预测比较](figures/fig10_predictive_models.png)

测试期XGBoost的SHAP前五项为{{TOP_SHAP_5}}。Knockoff确认集合与SHAP前十项的交集为{{CORE_LIST}}，这些变量同时获得条件选择和非线性预测证据；其余确认变量为{{SECONDARY_LIST}}，适合在一级指标异常后下钻。

**表5-7 XGBoost SHAP重要性**

{{SHAP_TABLE}}

![图5-5 XGBoost SHAP重要性](figures/fig06_xgboost_shap.png)

双向固定效应控制卖家和月份效应，标准误按卖家聚类。5%水平显著的动态变量为{{SIGNIFICANT_FE}}。固定效应系数只描述同一卖家随时间变化的条件关联，不能排除未观测的促销、曝光、库存和竞争变化。

**表5-8 月度双向固定效应中p值最低的15项**

{{FIXED_TABLE}}

## 5.4 面板结构、业务组与非线性稳健性

月度卖家均值—离差分层结果见表5-9。卖家间层在0.20水平入选{{V4_BETWEEN_COUNT}}项，卖家内层为{{V4_WITHIN_COUNT}}项。整表信号主要对应卖家长期经营位置差异；当前19个月的面板不足以支持非空的卖家内严格集合。

**表5-9 月度卖家间与卖家内分层结果**

{{V4_CLUSTER_TABLE}}

十二业务组在0.20水平入选{{V4_GROUP_Q20_COUNT}}组，在0.30水平入选{{V4_GROUP_Q30_COUNT}}组：{{V4_GROUP_Q30_LIST}}。低组数设计中的Knockoff+离散门槛限制了0.20水平功效，因此0.30结果只作为业务模块敏感性证据。

**表5-10 月度业务组Knockoff结果**

{{V4_GROUP_TABLE}}

![图5-6 聚类分层与业务组稳定性](figures/fig12_cluster_group.png)

轨迹聚合统计量的交换反对称最大误差为{{V4_GRIP_ERROR}}。在0.20水平入选{{V4_GRIP_Q20_COUNT}}项，在0.30水平入选{{V4_GRIP_Q30_COUNT}}项：{{V4_GRIP_Q30_LIST}}。该结果说明二维正则轨迹能够在宽松水平恢复部分非线性信号，但不能改写0.20水平的确认结论。

**表5-11 轨迹聚合重要性**

{{V4_GRIP_TABLE}}

![图5-7 轨迹聚合平均W](figures/fig14_grip_importance.png)

真实特征矩阵上的稀疏模拟进一步显示，去随机化e-BH的平均FDP为{{SIM_FDP}}、平均功效为{{SIM_POWER}}；单轮Knockoff+的平均FDP为{{BASE_SIM_FDP}}、功效为{{BASE_SIM_POWER}}。当仅设置5个真信号时，聚合程序因离散门槛得到零发现。本文保留该不利结果，以说明方法在低维且极稀疏任务中的功效边界。

## 5.5 周度粒度稳健性

周度面板包含{{WEEK_ROWS}}行、{{WEEK_SELLERS}}个卖家和{{WEEK_COUNT}}周，GMV合计{{WEEK_GMV}}百万雷亚尔。下一周零成交率为{{WEEK_ZERO_RATE}}，明显高于月度。周度Copula平均边际KS为{{WEEK_COPULA_KS}}，原始高斯为{{WEEK_GAUSSIAN_KS}}，因此周度主分析同样采用Copula-MVR。

![图5-8 周度GMV变化](figures/fig15_weekly_gmv.png)

在0.20水平，周度严格入选{{WEEK_STRICT_COUNT}}项：{{WEEK_STRICT_LIST}}。其中{{WEEK_STABLE_COUNT}}项的单轮频率不低于90%。

**表5-12 周度严格入选变量**

{{WEEK_SELECTION_TABLE}}

![图5-9 周度单轮入选频率](figures/fig16_weekly_selection.png)

月度14项与周度18项共同入选{{OVERLAP_COUNT}}项，Jaccard系数为{{JACCARD}}。共同集合为{{OVERLAP_LIST}}；其中同时满足周度频率不低于90%的跨粒度高稳定核心有{{STABLE_CROSS_COUNT}}项：{{STABLE_CROSS_LIST}}。月度特有{{MONTH_ONLY_COUNT}}项为{{MONTH_ONLY_LIST}}；周度特有{{WEEK_ONLY_COUNT}}项为{{WEEK_ONLY_LIST}}。

**表5-13 月度与周度入选集合对照**

{{WEEK_OVERLAP_TABLE}}

![图5-10 月度与周度入选频率](figures/fig17_monthly_weekly_frequency.png)

周度敏感性结果见表5-14。同周GMV对照入选29项，说明件数、订单数和价格参与GMV定义会造成近乎全选。测试期前样本入选14项，与完整周度主集重合13项；XGBoost统计量入选16项，与主集重合15项；下一周客单价情景入选15项且全部属于周度主集。

**表5-14 周度敏感性结果**

{{WEEK_SENSITIVITY_TABLE}}

周度卖家间层入选{{WEEK_BETWEEN_COUNT}}项，卖家内层为{{WEEK_WITHIN_COUNT}}项；十二业务组在0.20水平入选{{WEEK_GROUP_COUNT}}组；轨迹统计量在0.20水平为{{WEEK_GRIP_Q20_COUNT}}项，在0.30水平为{{WEEK_GRIP_Q30_COUNT}}项。提高时间频率没有形成卖家内严格发现，表明短序列和零膨胀仍限制动态识别。

**表5-15 周度卖家间与卖家内分层结果**

{{WEEK_CLUSTER_TABLE}}

周度最佳预测模型为{{WEEK_BEST_MODEL}}，测试RMSE为{{WEEK_BEST_RMSE}}，相对本周GMV朴素基线改善{{WEEK_RMSE_IMPROVEMENT}}。只使用周度18项的XGBoost RMSE为{{WEEK_SELECTED_RMSE}}，相对全31维XGBoost变化{{WEEK_SELECTED_DELTA}}，表明维度压缩后预测性能基本保留。

**表5-16 周度时间外预测表现**

{{WEEK_METRIC_TABLE}}

![图5-11 周度时间外预测比较](figures/fig18_weekly_prediction.png)

综合来看，月度能够降低低频卖家的偶发噪声，适合承担确认性选择；周度提供更多时间点和短期变化，但零膨胀与相关代理更强。月周共同11项构成跨粒度证据，周度特有7项只能作为观察变量，不能自动升级为确认指标。

# 第6章 讨论与经营应用

第5章给出了月度、周度、结构分层和非线性分支的实证结果。本章不再重复表格，而是讨论这些证据如何共同回答研究问题，并将统计结论转化为可执行的指标治理规则。

## 6.1 条件信息、稳定性与粒度差异

月度确认集合覆盖交易规模、价格成本、履约、商品内容和市场覆盖。订单数、件数和独立顾客数同时入选，说明它们在当前候选集下仍各自保留增量预测信息；但三者高度相关，产品展示时不宜平铺为三个同等级卡片。统计候选保留与业务界面合并属于不同决策层。

月周共同集合集中在规模、价格、承运准备、标题和区域覆盖。该重合说明这些信号不依赖单一时间聚合口径。月度特有的送达时长、预计提前量和描述长度变化较慢；周度特有的商品物理属性、支付结构和营销标记更易受单周订单构成影响。当前证据不能确定差异来源于真实时间机制还是聚合噪声，因此周度特有变量应进入观察池而非确认层。

## 6.2 参数$K$与看板维度治理

统计层先通过e-BH形成14项月度确认池，展示层再通过参数$K$控制页面容量。默认$K=10$兼顾验证贡献和预测误差；$K=4$适合极简经营总览，$K=8$适合常规首页，$K=14$适合完整确认页。该设计避免为了获得更短列表而事后调整FDR水平。

看板可按五类业务问题组织。交易规模模块以订单数为主，并下钻件数和独立顾客数；价格成本模块包含平均商品价格和运费；履约模块包含审核至承运、送达时长和预计偏差；供给内容模块包含品类与文本完整度；市场覆盖模块包含买家州数量和地域基准。每个指标同时记录统计证据、稳定频率、模型排名和更新时间。

## 6.3 深度方法与负结果的解释

深度生成器将边际KS降低到与Copula接近的水平，但深度核MMD和分类器仍能识别交换后的联合样本。该结果表明，复杂模型能够拟合单变量分布，却未必自动满足Knockoff所需的联合可交换性。生成器未通过准入并非计算失败，而是诊断体系发挥了否决作用。

轨迹聚合重要性满足反对称性，并在0.30水平恢复部分信号，但在0.20水平仍无发现。深度方法在本研究中的价值主要体现在检验非线性可能性和暴露生成器缺陷，而不是取得优于统计基线的结论。预测分支中MLP也未超过树模型，说明在当前样本规模和表格结构下，模型复杂度本身不能保证更好的泛化。

## 6.4 预测证据与因果决策

Knockoff、TreeSHAP和固定效应回答不同问题。Knockoff识别给定其他变量后的条件预测信息；TreeSHAP描述已拟合模型如何使用变量；固定效应衡量控制卖家与周期效应后的动态关联。三类结果均不能排除全部时变混杂，因此不能直接推出干预收益。

对于履约时长、商品内容和运费策略等可行动变量，应通过A/B实验、分层随机实验或准实验验证。订单数、件数和顾客数更适合作为状态与预警变量，而非干预处方。将预测信号与因果杠杆区分开，是指标治理能够支持决策而不产生误导的必要条件。

## 6.5 应用边界与实施建议

标准Model-X理论依赖观测独立和有效可交换性，而卖家面板存在组内相关，Copula-MVR也只近似混合联合分布。因此，0.20应解释为模型假设成立下的目标FDR水平，不是具体集合错误比例的确定承诺。卖家内层无严格发现进一步提示，当前结果更适合跨卖家风险分层，而非卖家自身短期变化归因。

周度面板沿用最终已送达订单按购买周归属的回溯口径。实际部署时，应按支付发生时间、承运时间和评价创建时间重构特征，并进行滚动时间外回测。建议每季度固定候选集和阈值重新运行，连续多个窗口缺乏确认、稳定和预测证据的指标可退出常驻看板。

# 第7章 结论与展望

本章围绕绪论提出的四个研究问题总结主要发现，并说明研究贡献、局限与后续工作。结论只概括正文已经报告的证据，不引入新的模型或实证结果。

## 7.1 主要结论

本文以Olist全量公开数据为对象，建立了面向电商GMV监控的多粒度受控维度选择框架，得到以下结论。

1. 自然粒度先聚合后连接能够避免支付、评价和商品明细多对多连接造成的金额膨胀。月度主面板包含{{PANEL_ROWS}}个卖家-月，周度面板包含{{WEEK_ROWS}}个卖家-周；下一周期目标避免了同周期GMV定义关系。
2. 随机化秩高斯Copula显著改善混合边际匹配。60轮Knockoff的e-value聚合在0.20水平形成14项月度确认集合，参数化展示规则推荐$K={{V3_DEFAULT_K}}$，测试RMSE为{{V3_RMSE}}。
3. 周度分析形成18项集合，其中11项与月度重合，Jaccard系数为{{JACCARD}}。周度18项XGBoost的RMSE为{{WEEK_SELECTED_RMSE}}，与全31维模型基本一致。月度适合确认，周度适合稳健性和短期预警。
4. 月度与周度卖家内层均无严格发现；深度生成器未通过联合交换性诊断；轨迹重要性只在较宽松水平恢复信号。上述负结果限定了结论适用范围，避免把模型复杂度或跨卖家差异误写为可靠的短期动态机制。

## 7.2 研究贡献

本文的贡献定位为应用统计研究设计创新，而非提出新的Knockoff定理。正式推断只有一条主线：Copula-MVR Model-X Knockoff、反对称统计量、e-value聚合与e-BH。其余模型均受限于展示、诊断或稳健性角色。在此基础上形成三项相互闭合的贡献。第一，将去随机化Model-X Knockoff用于经营看板维度准入，在多轮随机生成下控制目标错误发现率。第二，把确认性统计选择与参数$K$展示路径分离，使统计风险偏好不受页面容量反向影响。第三，通过卖家间/内分层、业务组竞争、深度联合诊断和周度完整重跑建立多层证据体系，并规定未通过交换性诊断的生成器不得形成FDR结论。

## 7.3 研究局限

研究存在四方面限制。第一，公开数据缺少曝光、广告、库存、促销和竞争强度等关键时变变量。第二，面板组内相关与Copula高阶近似使有限样本FDR保证依赖模型条件。第三，样本仅覆盖19个月，周度卖家活跃序列中位数也较短，不适合复杂长序列模型。第四，周度履约和评价特征采用最终结果回溯归属，不能直接视为周期结束时可见的信息。

## 7.4 后续研究

后续研究可从三个方向推进。其一，在更长企业面板上使用滚动Knockoff、在线e-value和簇级交叉拟合，重点提高卖家内动态识别能力。其二，使用显式离散似然、条件normalizing flow或diffusion生成器改善混合变量联合建模，并保持独立交换性准入。其三，对履约、商品内容和价格策略开展随机实验或准实验，将条件预测信息进一步转化为可干预效应。

# 参考文献

{{REFERENCES_TEXT}}

# 附录A 全部变量描述统计

本附录列出31项建模变量的完整描述统计。正文只展示关键变量，完整表用于复核样本量、缺失率、尺度和与下一期GMV的边际相关。

{{APPENDIX_DESC_TABLE}}

# 附录B 可复现环境与数据指纹

本研究固定随机种子，并记录软件版本和月度、周度面板指纹。表B-1用于确认数据和计算环境是否发生变化。

**表B-1 可复现环境**

{{REPRODUCIBILITY_TABLE}}

原始输入、月度面板和周度面板已统一存放于项目的“论文实际使用数据汇总”目录，并附文件级SHA-256清单。每轮Knockoff统计量、选择结果、预测结果和诊断元数据均保留为CSV或JSON。

# 附录C 核心算法伪代码

本附录以伪代码概括确认性选择和参数化展示流程，便于在其他业务数据上复现。

```text
输入：特征矩阵 X，下一期目标 y，最终水平 q，重复次数 M
输出：FDR确认集合 S_FDR，参数化展示集合 S(K)

1. 对 X 完成缩尾、偏态变换、插补和标准化
2. 对并列值随机打散并映射到秩高斯空间 Z
3. 估计 Ledoit-Wolf 协方差并构造 MVR Knockoff
4. 对 m = 1,...,M：
   4.1 生成仿制矩阵 Z_tilde
   4.2 联合拟合 [Z, Z_tilde] 的 Lasso
   4.3 计算 W_j = |beta_j| - |beta_tilde_j|
   4.4 以 alpha_kn = q/2 计算 Knockoff+ 阈值
   4.5 将本轮选择转换为 e-value
5. 平均 M 轮 e-value，并在水平 q 上实施 e-BH
6. 得到确认集合 S_FDR
7. 在训练期拟合 XGBoost，在验证集计算 S_FDR 内的TreeSHAP排序
8. 根据累计贡献与验证RMSE规则推荐 K，并输出前 K 项
```

# 附录D 提交前检查清单

本清单用于在学校正式提交前完成格式与学术边界复核。

1. 按学校最新模板补充封面、原创性声明、授权书、分类号、密级和答辩信息。
2. 核对中文摘要字数、英文摘要对应关系、关键词数量和术语一致性。
3. 逐项核对参考文献原文、DOI、页码及学校要求的GB/T 7714版本。
4. 在Word中更新目录、图表题注、公式编号和交叉引用域。
5. 确认盲审版删除作者、导师、致谢、项目来源及其他身份信息。
6. 核对所有结果数字与CSV/JSON一致，并保留数据指纹和运行日志。
7. 由导师确认FDR水平、预测而非因果的表述以及业务建议边界。
