---
title: 面向电商GMV监控的可控错误发现维度选择研究
subtitle: Olist可控参数维度选择精简介绍版（v3）
author: |
  学校：【待填写】  
  学院：【待填写】  
  专业：【待填写】  
  研究生：【待填写】  
  指导教师：【待填写】
date: 二〇二六年八月
lang: zh-CN
---

# 摘要

电商经营看板通常同时包含价格、交易规模、供给、支付、履约、口碑、地域和营销等大量维度。若仅依据相关系数、单变量显著性或机器学习重要性筛选字段，容易把高度相关的代理变量重复当作独立发现，也无法约束入选指标中的假发现比例。本文使用 Olist 巴西电商公开数据，构建卖家-月份面板，并以次月 `log(1+GMV)` 为目标，研究哪些本月经营维度在控制其他候选信息后仍具有增量预测价值。

研究在各原始表的自然粒度上先聚合再连接，避免商品明细、支付和评价多对多连接造成 GMV 重复计算。最终样本覆盖 {{PANEL_START}} 至 {{PANEL_END}}，包含 {{PANEL_ROWS}} 个卖家-月、{{PANEL_SELLERS}} 个卖家和 {{PANEL_MONTHS}} 个月，已送达商品 GMV 合计 {{GMV_MILLION}} 百万巴西雷亚尔。共构造 {{CANDIDATE_COUNT}} 个候选维度，其中申报月收入因零方差排除，{{ANALYSIS_COUNT}} 个变量进入分析。

方法上，本文采用随机化秩高斯 Copula、Ledoit-Wolf 收缩协方差和 MVR 二阶 Model-X Knockoff，并以 Lasso 真变量与仿制变量系数差构造统计量。主分析重复生成 60 组 Knockoff，设置最终 `α_eBH=0.20`、单轮 `α_kn=0.10`，通过 e-value 聚合与 e-BH 控制错误发现率。结果显示，Copula 生成器平均边际 KS 距离为 {{COPULA_MEAN_KS}}，显著低于原始高斯基线的 {{GAUSSIAN_MEAN_KS}}。严格 e-BH 入选 {{STRICT_COUNT}} 个维度：{{STRICT_LIST}}。

V2 进一步实跑噪声条件深度 Knockoff、配对竞争 MLP 和残差表格 MLP。深度生成器边际校准后的平均 KS 为 {{V2_DEEP_KS}}，但 swap 分类器 AUC 为 {{V2_DEEP_AUC}}，未通过预设联合交换性门槛，因此该分支不作 FDR 声明。配对 MLP 的反对称性最大误差为 {{V2_ANTISYMMETRY_MAX_ERROR}}；残差 MLP 的时间外 RMSE 为 {{V2_MLP_RMSE}}。V2 由此把深度生成、生成诊断、深度重要性和预测验证分开，保留失败结果而不扩大结论。

V3 不改变上述 14 项正式发现，而是增加参数 $K\in[1,14]$：在确认集内按验证集 TreeSHAP 贡献排序，用户选择前 $K$ 项。系统提供 K=4 极简、K=8 均衡、K=14 完整等档位，并按“累计贡献≥90%、验证RMSE距最优≤0.5%”推荐默认 `K={{V3_DEFAULT_K}}`。默认模型测试 RMSE 为 {{V3_RMSE}}，相对全维模型变化 {{V3_RMSE_CHANGE}}。

时间外测试中，最佳模型为 {{BEST_MODEL}}，对数尺度 RMSE 为 {{BEST_RMSE}}，较直接使用本月 GMV 的朴素基线下降 {{RMSE_IMPROVEMENT}}。SHAP 结果与 Knockoff 在交易规模、价格、履约和市场覆盖维度形成交叉证据。本文据此提出“一级监控、二级诊断、专题分析”的看板分层方案，同时强调所有结果属于条件预测关系，不直接代表因果效应。

**关键词：** 电商 GMV；Model-X Knockoff；错误发现率；可控参数；维度选择路径；XGBoost

# 第1章 研究背景与问题

## 1.1 研究背景

数字平台能够完整记录订单、商品、顾客、支付、履约和评价过程，但指标数量也会随业务扩张持续增加。一个 GMV 看板可能同时展示订单数、件数、顾客数、均价、运费、品类、支付方式、物流时长、评价和地域覆盖。大量字段并列展示会产生三类问题：

1. 订单数、件数和顾客数等高度相关变量重复表达经营规模；
2. 逐项显著性检验会随候选变量增多而累积误报；
3. 机器学习重要性可以排序，却不能直接说明入选集合中可能有多少错误发现。

错误发现率（False Discovery Rate，FDR）定义为全部发现中错误发现比例的期望：

$$
\mathrm{FDR}=\mathbb{E}\left[\frac{V}{R\vee 1}\right],
$$

其中 $V$ 是错误发现数，$R$ 是总发现数。FDR 允许在整体错误比例受控的前提下保留较多有效信号，适合需要兼顾探索效率与结果可信度的指标治理。

## 1.2 研究问题

本文的核心问题是：在本月可观测的卖家经营信息中，哪些维度在控制其余候选变量后，仍包含次月 GMV 的增量预测信息，并值得进入常驻看板？

相应的 Model-X 零假设为：

$$
H_{0j}:Y_{s,t+1}\perp X_{j,s,t}\mid X_{-j,s,t}.
$$

拒绝零假设表示变量具有条件预测信息，不表示人为改变该变量一定会改变 GMV。全文将预测发现、机器学习解释和因果结论严格区分。

## 1.3 研究思路

研究包括四个步骤：

1. 在订单、商品明细、支付、评价、地理和营销等自然粒度上分别聚合，构造卖家-月面板；
2. 使用 Copula-MVR Model-X Knockoff 生成负对照变量，通过重复 Knockoff 和 e-BH 控制 FDR；
3. 使用 Ridge、Lasso、Extra Trees、XGBoost 和 MLP 进行自然时间外预测；
4. 结合 SHAP、双向固定效应和稳健性分析，将结果转化为看板分层建议。

# 第2章 数据与研究设计

## 2.1 数据来源与样本

数据来自 Olist Brazilian E-Commerce Public Dataset 和 Marketing Funnel by Olist，共使用订单、商品明细、支付、评价、商品、卖家、顾客、地理、品类翻译及营销漏斗 11 张 CSV。原始订单为 99,441 笔，商品明细为 112,650 行；主分析只保留已送达订单。

支付表和评价表可能一单多行，不能直接与商品明细连接。本文先把支付方式、支付金额和评价聚合到订单级，再聚合到卖家-月；地理表先按邮编前缀取经纬度中位数；商品明细按卖家-月汇总 GMV、价格、运费和商品属性。该流程使处理后 GMV 与原始已送达商品明细逐笔对账一致。

顾客数量使用 `customer_unique_id` 去重，而非订单级 `customer_id`。不合理的负物流时长和明显超界时间设为缺失，随后在模型训练样本中以中位数插补。

最终面板包含 {{PANEL_ROWS}} 个卖家-月、{{PANEL_SELLERS}} 个卖家和 {{PANEL_MONTHS}} 个月。样本 GMV 为 {{GMV_MILLION}} 百万雷亚尔，次月零成交观测占 {{ZERO_RATE}}。

![图2-1 Olist主分析期月度GMV](figures/fig01_monthly_gmv.png)

## 2.2 目标变量

主目标定义为：

$$
Y_{s,t+1}=\log(1+\mathrm{GMV}_{s,t+1}).
$$

若卖家本月有交易而次月没有已送达订单，则次月 GMV 记为零。使用次月目标可以避免同月件数、订单数和均价对同月 GMV 的机械解释，更符合经营预警场景。

## 2.3 候选维度

{{CANDIDATE_COUNT}} 个候选维度覆盖以下类别：

- 价格与运费：平均商品价格、平均运费；
- 商品结构：重量、长宽高、图片数、标题和描述长度；
- 支付：分期数、支付序列数及四类支付方式占比；
- 口碑与履约：评价、审核时长、承运时长、送达时长和预计提前量；
- 经营规模：件数、订单数、品类数和独立顾客数；
- 地域覆盖：买家州数量、第一大州占比、跨州比例和平均距离；
- 卖家与营销：卖家州频率、营销来源及是否匹配营销成交。

申报月收入在分析面板中恒为零，因此在查看响应结果前按零方差规则排除，最终 {{ANALYSIS_COUNT}} 个变量进入模型。最大特征缺失率为 {{MISSING_MAX}}。

## 2.4 描述统计

**表2-1 关键变量描述统计**

{{KEY_DESC_TABLE}}

边际相关显示，订单数、独立顾客数、件数和市场覆盖与次月 GMV 相关较高，但边际相关不能区分独立信息与相关代理，因此只作为描述。

# 第3章 方法

## 3.1 Copula Model-X Knockoff

Model-X Knockoff 为每个真实变量 $X_j$ 构造仿制变量 $\widetilde X_j$。理想仿制变量满足成对可交换性，并在给定真实特征后不再包含响应信息：

$$
(X,\widetilde X)_{\mathrm{swap}(S)}
\overset{d}{=}(X,\widetilde X),\qquad
\widetilde X\perp Y\mid X.
$$

Olist 变量同时包含金额、计数、比例、二元值和大量并列值。本文先对连续变量进行 1% 与 99% 分位缩尾、偏态变换、中位数插补和标准化，再使用固定随机种子打散并列秩，并映射为近似标准正态：

$$
Z_{ij}=\Phi^{-1}\left(\frac{r_{ij}-0.5}{n}\right).
$$

随后使用 Ledoit-Wolf 收缩协方差与 MVR 准则生成二阶 Knockoff。与直接假定原始变量服从高斯分布相比，Copula 方法更适合混合边际，同时保留了明确的诊断指标。

## 3.2 竞争统计量与e-value

将真实变量和仿制变量共同输入 Lasso，构造：

$$
W_j=|\widehat\beta_j|-|\widehat\beta_{j+p}|.
$$

若真实变量显著胜过仿制变量，$W_j$ 较大且为正。每次 Knockoff 按 `α_kn=0.10` 计算 Knockoff+ 阈值，并转换为 e-value：

$$
e_j^{(m)}
=p\frac{\mathbf{1}\{W_j^{(m)}\ge T^{(m)}\}}
{1+\sum_k\mathbf{1}\{W_k^{(m)}\le -T^{(m)}\}}.
$$

主分析重复 60 次，对 e-value 取平均后，在 `α_eBH=0.20` 下实施 e-BH。最终 FDR 水平与单轮阈值分开设置，遵循去随机化 Knockoff 的推荐方案。

本文还报告单轮入选频率。严格 e-BH 集合承担 FDR 声明，频率至少 90% 的集合只描述生成稳定性，二者不能混用。

## 3.3 预测与解释

按自然月份划分训练、验证和测试集：

- 训练期：2017-01 至 2017-12，{{TRAIN_N}} 行；
- 验证期：2018-01 至 2018-04，{{VALID_N}} 行；
- 测试期：2018-05 至 2018-07，{{TEST_N}} 行。

比较模型包括朴素本月 GMV、Ridge、Lasso、Extra Trees、XGBoost、MLP 以及只使用测试期前 Knockoff 入选变量的 XGBoost。评价指标为对数 RMSE、MAE、$R^2$ 和原始金额 WAPE。XGBoost 使用 TreeSHAP 解释测试期预测。

由于单卖家最多只有 19 个特征月份，且多数序列更短，本文没有强行使用参数量较大的 TFT，而使用 XGBoost、传统 MLP 和残差表格 MLP 完成非线性 AI 验证。

## 3.4 V2深度学习扩展

深度生成器以真实特征和同维高斯噪声为输入，使用 4 层宽度 192 的 LayerNorm-GELU 网络生成仿制变量。训练损失联合 20%、50%、80% swap 比例的混合 RBF-MMD、边际、协方差、交叉协方差和可重构性。保留集同时报告 KS、覆盖率、相关误差、MMD permutation test 与分类器两样本检验。

配对竞争 MLP 使用对称输入层和共享下游范数构造 $W_j$，保证交换 $(X_j,\widetilde X_j)$ 后统计量变号；正式实验重复 {{V2_PAIRED_REPETITIONS}} 次。预测分支新增 3 个残差块的 PyTorch 表格 MLP，与 V1 模型使用完全相同的自然月份切分。

## 3.5 V3参数化选择规则

第一阶段保留 FDR 0.20 下的14项确认池；第二阶段按训练期模型在验证集上的SHAP贡献排序；第三阶段由参数K截取前K项。测试集不参与排序或默认K选择。参数K控制展示长度，不被表述为新的FDR结果。

# 第4章 核心实证结果

## 4.1 Knockoff生成质量

Copula 生成器的平均边际 KS 距离为 {{COPULA_MEAN_KS}}，最大值为 {{COPULA_MAX_KS}}，协方差相对误差为 {{COPULA_COV_ERROR}}，交叉协方差非对称度为 {{COPULA_CROSS_ASYM}}。原始高斯基线平均边际 KS 为 {{GAUSSIAN_MEAN_KS}}。Copula 的边际误差明显较小，因此主结果采用 Copula-MVR。

**表4-1 V2多指标生成诊断**

{{V2_DIAGNOSTIC_TABLE}}

表中统一采用 V2 的 20% 验证子集，故 Copula KS 与 V1 全样本口径略有差异。深度生成器的平均 KS 经校准降至 {{V2_DEEP_KS}}，但协方差误差为 {{V2_DEEP_COV_ERROR}}、swap 分类器 AUC 为 {{V2_DEEP_AUC}}。按预设规则判定为“{{V2_DEEP_DECISION}}”。这说明边际一致不能替代联合交换性。

## 4.2 主变量选择

在 `α_eBH=0.20`、`α_kn=0.10` 下，严格入选 {{STRICT_COUNT}} 个变量：{{STRICT_LIST}}。

**表4-2 严格e-BH入选变量**

{{STRICT_COMPACT_TABLE}}

其中 {{STABLE_COUNT}} 个变量的单轮入选频率达到至少 90%：{{STABLE_LIST}}。严格集合与稳定集合可能不同，因为 e-BH 使用每轮负向统计量数量加权，而 90% 频率只是人为设置的描述阈值。

![图4-1 主分析平均e-value](figures/fig03_knockoff_evalues.png)

严格变量可归为四组：

1. 经营规模：订单数、件数、独立顾客数、活跃品类数；
2. 价格成本：平均商品价格、平均运费；
3. 履约：审核至承运、下单至送达和预计提前量；
4. 商品与覆盖：标题、描述、商品高度、买家州数量和卖家州频率。

这些结果表明，次月 GMV 既有经营规模延续，也包含价格带、履约状态、商品结构和市场覆盖的条件信息。

## 4.3 阈值与时间稳健性

**表4-3 不同FDR水平的严格结果**

{{SENSITIVITY_TABLE}}

在更严格的 0.10 水平下，严格发现为 {{Q10_COUNT}} 项；在 0.30 水平下为 {{Q30_COUNT}} 项。测试期开始前样本独立选出的 {{PRETEST_COUNT}} 个变量为：{{PRETEST_LIST}}。该集合用于精简 XGBoost，避免测试数据进入变量筛选。

## 4.4 时间外预测

**表4-4 时间外测试性能**

{{V2_PREDICTIVE_TABLE}}

最佳模型为 {{BEST_MODEL}}，测试集对数 RMSE 为 {{BEST_RMSE}}、$R^2$ 为 {{BEST_R2}}。残差 MLP 的 RMSE 为 {{V2_MLP_RMSE}}、$R^2$ 为 {{V2_MLP_R2}}，{{V2_MLP_COMPARISON}}。复杂模型没有形成压倒性优势。

![图4-2 V2时间外预测模型比较](figures/fig_v2_predictive_models.png)

## 4.5 深度重要性与事后路径

配对 MLP 的反对称性测试结论为“{{V2_ANTISYMMETRY_DECISION}}”，0.20 水平得到 {{V2_PAIRED_COUNT}} 项：{{V2_PAIRED_LIST}}。它使用通过诊断的 Copula Knockoff，作为非线性稳健性证据，不替代预先指定的 Copula-Lasso 主结果。

固定 V1 的 60 轮 e-value 后，仅改变最终 e-BH 阈值得到：

{{V2_POSTHOC_TABLE}}

## 4.6 V3可调维度路径

**表4-5 默认K={{V3_DEFAULT_K}}的维度及证据**

{{V3_SHORTLIST_TABLE}}

默认维度为{{V3_SHORTLIST_LIST}}。正式14项中未进入默认层的{{V3_REMAINDER_COUNT}}项为{{V3_REMAINDER_LIST}}。

**表4-6 K=4、8、默认值与14的选择路径**

{{V3_PATH_TABLE}}

默认K={{V3_DEFAULT_K}}的RMSE为{{V3_RMSE}}，全维模型为{{V3_FULL_RMSE}}，相对变化{{V3_RMSE_CHANGE}}。用户可根据页面容量选择K，不必重新调整FDR水平。

## 4.7 SHAP与固定效应

SHAP 前五项为{{TOP_SHAP_5}}。Knockoff 严格集合与 SHAP 前十项的交集为{{CORE_LIST}}，可作为优先监控候选；其余严格项目为{{SECONDARY_LIST}}，更适合二级诊断。

![图4-3 XGBoost SHAP重要性](figures/fig06_xgboost_shap.png)

双向固定效应控制卖家和月份效应，标准误按卖家聚类。5% 水平显著的动态变量为{{SIGNIFICANT_FE}}。固定效应只能提供同一卖家随时间变化的关联证据，仍可能受到促销、曝光、库存和竞争等遗漏因素影响。

## 4.8 模拟校准的边界

模拟在真实特征矩阵上设置 5 个真信号。去随机化 e-BH 的平均 FDP 为 {{SIM_FDP}}、平均功效为 {{SIM_POWER}}；单轮 Knockoff+ 的平均 FDP 为 {{BASE_SIM_FDP}}、平均功效为 {{BASE_SIM_POWER}}。

聚合程序在该稀疏模拟中零功效，原因是 `p={{ANALYSIS_COUNT}}`、`α_kn=0.10` 时 Knockoff+ 存在离散门槛，只有 5 个真信号很难在不引入假发现的情况下达到阈值。这一结果说明方法在低维、极稀疏任务中可能保守。本文保留该不利结果，而不通过事后调参制造非空发现。

# 第5章 看板应用建议

## 5.1 三级维度治理

**一级监控层**默认采用K={{V3_DEFAULT_K}}：{{V3_SHORTLIST_LIST}}；也可切换K=4极简、K=8均衡或K=14完整。

**二级诊断层**包含确认集内未进入短名单的 6 项：{{V3_REMAINDER_LIST}}。

**专题分析层**包含未进入严格集合、但具有明确业务用途的支付、评价、距离和营销变量。未入选不等于永远无用，只表示在当前候选集和次月 GMV 目标下缺乏足够条件证据。

## 5.2 展示方式

看板不应把所有入选变量平铺成卡片，而应按经营问题组织：

- 交易规模模块：订单数、件数和独立顾客数；
- 价格成本模块：平均商品价格和平均运费；
- 履约模块：承运准备、送达时长和承诺偏差；
- 供给内容模块：品类数、标题与描述；
- 市场覆盖模块：买家州数量和地域基准。

每个模块保留一个主要状态指标和少量下钻指标，以减少代理变量重复展示。

## 5.3 从预测到实验

订单数和顾客数首先是经营状态变量，不能把“提高订单数”写成可执行因果建议。履约时长、商品内容和运费策略更接近可干预变量，但仍需 A/B 实验、分层实验或准实验确认。Knockoff 的作用是缩小候选范围，不是替代因果识别。

# 第6章 结论与局限

## 6.1 主要结论

本文使用 Olist 全量公开数据构建了 {{PANEL_ROWS}} 个卖家-月观测，并以次月 GMV 为前瞻目标。随机化秩高斯 Copula 显著改善了混合变量的 Knockoff 边际匹配。主分析在最终 FDR 水平 0.20 下严格入选 {{STRICT_COUNT}} 个维度，覆盖经营规模、价格、履约、商品内容和市场覆盖。

时间外测试中，{{BEST_MODEL}} 的 RMSE 为 {{BEST_RMSE}}，较朴素基线改善 {{RMSE_IMPROVEMENT}}。Knockoff、SHAP、预测误差和固定效应构成互补证据，其中只有预先设定的 e-BH 程序承担 FDR 声明。

## 6.2 研究贡献

第一，将 FDR 控制引入电商看板维度治理。第二，采用自然粒度聚合和次月目标，规避多表膨胀与同月定义泄漏。第三，明确区分严格发现、深度生成准入、随机稳定性、预测解释和因果结论。第四，以反对称性测试约束深度重要性。第五，保留失败生成器、完整代码、每轮 W、结果表和随机种子，支持复核。

## 6.3 主要局限

1. 公开数据缺少曝光、广告、库存、促销和竞争强度；
2. 卖家-月面板存在同一卖家内部相关，标准 Model-X 独立条件只能近似满足；
3. Copula 二阶方法不能保证刻画全部高阶和尾部依赖，深度生成器则未通过联合交换性准入；
4. 只有 19 个特征月份，不适合复杂长序列深度模型；
5. 固定效应与 SHAP 不能替代随机实验；
6. 低维稀疏场景下 Knockoff+ 可能因离散阈值而功效不足。

## 6.4 后续方向

未来可在更长企业面板上使用滚动 Knockoff 和在线 e-value；按交易、履约、商品和地域定义 group knockoff；使用离散似然、normalizing flow 或 diffusion 改进深度生成；在更长序列中比较 TFT；对可行动变量开展实验和因果机器学习分析。

# 参考文献

[1] Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing[J]. Journal of the Royal Statistical Society: Series B, 1995, 57(1): 289-300.

[2] Barber R F, Candès E J. Controlling the false discovery rate via knockoffs[J]. The Annals of Statistics, 2015, 43(5): 2055-2085.

[3] Candès E, Fan Y, Janson L, Lv J. Panning for gold: Model-X knockoffs for high-dimensional controlled variable selection[J]. Journal of the Royal Statistical Society: Series B, 2018, 80(3): 551-577.

[4] Romano Y, Sesia M, Candès E. Deep knockoffs[J]. Journal of the American Statistical Association, 2020, 115(532): 1861-1872.

[5] Ren Z, Barber R F. Derandomised knockoffs: leveraging e-values for false discovery rate control[J]. Journal of the Royal Statistical Society Series B, 2024, 86(1): 122-154.

[6] Wang R, Ramdas A. False discovery rate control with e-values[J]. Journal of the Royal Statistical Society Series B, 2022, 84(3): 822-852.

[7] Tibshirani R. Regression shrinkage and selection via the lasso[J]. Journal of the Royal Statistical Society: Series B, 1996, 58(1): 267-288.

[8] Ledoit O, Wolf M. A well-conditioned estimator for large-dimensional covariance matrices[J]. Journal of Multivariate Analysis, 2004, 88(2): 365-411.

[9] Spector A, Janson L. Powerful knockoffs via minimizing reconstructability[J]. The Annals of Statistics, 2022, 50(1): 252-276.

[10] Breiman L. Random forests[J]. Machine Learning, 2001, 45: 5-32.

[11] Chen T, Guestrin C. XGBoost: a scalable tree boosting system[C]//Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2016: 785-794.

[12] Lundberg S M, Lee S I. A unified approach to interpreting model predictions[C]//Advances in Neural Information Processing Systems. 2017, 30.

[13] Lim B, Arık S Ö, Loeff N, Pfister T. Temporal fusion transformers for interpretable multi-horizon time series forecasting[J]. International Journal of Forecasting, 2021, 37(4): 1748-1764.

[14] Wooldridge J M. Econometric analysis of cross section and panel data[M]. 2nd ed. Cambridge, MA: MIT Press, 2010.

[15] Olist. Brazilian E-Commerce Public Dataset by Olist[DB/OL]. Kaggle, 2018.

[16] Olist. Marketing Funnel by Olist[DB/OL]. Kaggle, 2018.

[17] Sudarshan M, Tansey W, Ranganath R. Deep direct likelihood knockoffs[C]//NeurIPS. 2020.

[18] Shen A, Yan Y, Zhao P. Deep dependency regularized knockoff for controlling false discovery rates[C]//ICLR. 2024.

[19] Hollmann N, Müller S, Purucker L, et al. Accurate predictions on small data with a tabular foundation model[J]. Nature, 2025, 637(8045): 319-326.

# 版本说明

本文件为 v3 精简介绍版。V3 完整继承 V1/V2 结果，并新增K=1至14的选择路径、模型默认K和各档真实时间外预测。
