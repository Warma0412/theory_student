---
title: 基于去随机化Model-X Knockoff的电商GMV多粒度受控维度选择研究
subtitle: 去随机化Model-X Knockoff、参数化展示与周度稳健性
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

# 摘要

本文研究电商经营看板中的维度准入问题：在多个相关候选指标中，筛选对下一期GMV仍有增量预测信息的变量，并控制入选集合的目标错误发现率。研究使用Olist全量公开数据，在自然粒度上聚合订单、商品明细、支付、评价、商品、卖家、顾客、地理和营销信息，分别形成{{PANEL_ROWS}}个卖家-月和{{WEEK_ROWS}}个卖家-周观测。32项候选维度中，一项因零方差排除，31项进入分析。

本文使用随机化秩高斯Copula、Ledoit-Wolf收缩协方差和MVR准则生成Model-X Knockoff，以Lasso真实变量与仿制变量系数差构造统计量。主分析重复60轮，设置单轮`α_kn=0.10`、最终`α_eBH=0.20`，经e-value聚合和e-BH形成确认集合。在确认集合内，训练期XGBoost与验证集TreeSHAP构成参数$K$的展示路径。卖家均值—离差分层、业务组Knockoff、深度联合交换性诊断、轨迹重要性、固定效应和周度完整重跑用于检验稳健性。研究还在统一选维窗口、12项预算、预测器和测试期下比较五种方法，并对边际变换、S矩阵、统计量和重复次数进行消融。

月度主分析入选{{STRICT_COUNT}}项，其中{{STABLE_COUNT}}项的单轮频率不低于90%。模型默认推荐$K={{V3_DEFAULT_K}}$，测试RMSE为{{V3_RMSE}}。同预算基准中，{{BEST_BENCHMARK_METHOD}}的测试RMSE最低，为{{BEST_BENCHMARK_RMSE}}；去随机化Knockoff为{{KNOCKOFF_BENCHMARK_RMSE}}。在30组配对的70%卖家子样本中，Elastic Net和去随机化Knockoff的平均集合稳定性分别为{{ELASTIC_BENCHMARK_STABILITY}}和{{KNOCKOFF_BENCHMARK_STABILITY}}，两者差异区间包含0。20轮模拟中，去随机化与单次Knockoff的平均FDP分别为{{SIM_DERAND_FDP}}和{{SIM_SINGLE_FDP}}。周度分析入选{{WEEK_STRICT_COUNT}}项，与月度集合共同入选{{OVERLAP_COUNT}}项，Jaccard={{JACCARD}}。结果支持月度承担确认性选择，周度承担粒度稳健性和短期预警。全部结果属于条件预测信息，不作因果解释。

**关键词：** 电商GMV；Model-X Knockoff；错误发现率；多粒度面板；可解释机器学习

# 第1章 研究问题

电商平台能够记录从获客、下单、支付到履约和评价的完整链路，但指标数量随业务扩张持续增加。订单数、件数和顾客数等变量高度相关；逐项显著性检验会累积误报；机器学习重要性虽然能够排序，却不能直接控制发现集合中的错误比例。

本文将研究问题定义为：给定卖家本期31项经营维度，哪些变量在控制其余候选信息后仍包含下一期GMV的增量预测信息，并值得进入持续监控体系。相应的条件零假设为

$$
H_{0j}:Y_{s,t+1}\perp X_{j,s,t}\mid X_{-j,s,t}.
$$

本文完成五项工作：构造可对账的月度和周度卖家面板；建立去随机化Model-X Knockoff选择流程；在确认集合内建立参数$K$的展示路径；通过面板分层、业务组、深度诊断和跨粒度重跑检验结论边界；通过统一预算基准、组件消融和已知真值模拟区分预测精度、选择稳定性与错误率控制。

# 第2章 数据与研究设计

数据来自Olist Brazilian E-Commerce Public Dataset和Marketing Funnel by Olist。支付与评价先聚合到订单级，地理信息先按邮编前缀聚合，再与商品明细连接，从而避免多对多连接造成GMV重复计算。独立顾客数使用`customer_unique_id`去重。

**表2-1 月度与周度口径**

{{WEEK_GRAIN_TABLE}}

月度主目标为次月`log(1+GMV)`，周度目标为下一周`log(1+GMV)`。若卖家当前期活跃而下一期没有已送达订单，则目标记为零。月度目标零值率为{{ZERO_RATE}}，周度为{{WEEK_ZERO_RATE}}。下一期目标避免订单数、件数和价格对同周期GMV的机械解释。

31项建模变量覆盖价格成本、商品物理属性、内容、支付、评价、履约、交易规模、市场覆盖、卖家地域和营销。月度预测按2017年训练、2018年1月至4月验证、2018年5月至7月测试；周度预测使用相同自然时间边界。

![图2-1 月度GMV变化](figures/fig01_monthly_gmv.png)

![图2-2 周度GMV变化](figures/fig15_weekly_gmv.png)

# 第3章 方法

## 3.1 去随机化Model-X Knockoff

连续变量先进行分位缩尾、偏态变换、中位数插补和标准化，再通过随机化秩高斯变换处理离散值和并列值。使用Ledoit-Wolf协方差与MVR准则生成二阶Knockoff，将真实变量与仿制变量共同输入Lasso：

$$
W_j=|\widehat\beta_j|-|\widehat\beta_{j+p}|.
$$

每轮以$\alpha_{\mathrm{kn}}=0.10$计算Knockoff+阈值并形成e-value，60轮平均后在$\alpha_{\mathrm{eBH}}=0.20$下实施e-BH。单轮频率只描述随机稳定性，不等同于正式FDR保证。

## 3.2 参数$K$与预测验证

月度确认集合先由e-BH固定，再按训练期XGBoost在验证集上的平均绝对TreeSHAP排序。参数$K$截取前$K$项。默认值取累计验证贡献不低于90%、且验证RMSE距最优值不超过0.5%的最小$K$。测试集只用于最终评价。

## 3.3 结构与非线性稳健性

卖家均值—离差分层用于区分长期卖家差异和同一卖家的动态变化。31项变量进一步划为12个业务组，按组内真实与仿制系数的$L_2$范数差实施Group Knockoff。

深度生成器使用多比例swap-MMD、边际和协方差等联合损失。生成器是否有效由边际KS、协方差误差、深度核MMD和分类器检验共同决定。未通过准入的生成器不进入FDR结论。轨迹重要性在二维正则网格上累计配对竞争结果，并单独验证交换反对称性。

## 3.4 基准比较与组件消融

所有方法只使用2018年4月及以前数据选维，以12项为统一预算，再将各自集合输入相同参数的XGBoost并在2018年5月至7月测试。比较方法包括去随机化Knockoff、Elastic Net、稳定性选择、影子变量树模型和XGBoost-SHAP。集合稳定性使用30组完全相同的70%卖家子样本配对评估，Knockoff在每个子样本内聚合60轮。组件消融分别替换Copula边际、MVR/SDP/等相关S矩阵、Lasso/XGBoost统计量以及1至60轮聚合；真实协方差模拟报告FDP、Power和发现数。

# 第4章 月度实证结果

## 4.1 确认集合

Copula生成器平均边际KS为{{COPULA_MEAN_KS}}，低于原始高斯的{{GAUSSIAN_MEAN_KS}}。在最终FDR水平0.20下，月度主分析入选{{STRICT_COUNT}}项：{{STRICT_LIST}}。

**表4-1 月度严格入选变量**

{{STRICT_COMPACT_TABLE}}

![图4-1 月度主分析平均e-value](figures/fig03_knockoff_evalues.png)

## 4.2 参数化展示与预测

默认$K={{V3_DEFAULT_K}}$的维度为{{V3_SHORTLIST_LIST}}。默认模型测试RMSE为{{V3_RMSE}}，全31维XGBoost为{{V3_FULL_RMSE}}，相对变化{{V3_RMSE_CHANGE}}。

**表4-2 代表性K档位**

{{K_PATH_COMPACT_TABLE}}

月度时间外最佳模型为{{BEST_MODEL}}，RMSE为{{BEST_RMSE}}，较朴素基线改善{{RMSE_IMPROVEMENT}}。SHAP前五项为{{TOP_SHAP_5}}，与Knockoff共同支持交易规模、价格、履约和市场覆盖等信息。

## 4.3 结构与深度结果

卖家间层在0.20水平入选{{V4_BETWEEN_COUNT}}项，卖家内层为{{V4_WITHIN_COUNT}}项。十二业务组在0.20水平入选{{V4_GROUP_Q20_COUNT}}组，在0.30水平入选{{V4_GROUP_Q30_COUNT}}组。轨迹统计量在0.20水平入选{{V4_GRIP_Q20_COUNT}}项，在0.30水平入选{{V4_GRIP_Q30_COUNT}}项。

**表4-3 生成器联合诊断**

{{JOINT_DIAGNOSTIC_TABLE}}

深度生成器的边际KS较小，但深度核MMD和分类器仍能区分交换样本，因此不作FDR声明。该负结果说明联合交换性必须独立验证，不能由模型复杂度或单一边际指标替代。

## 4.4 统一基准、消融与模拟

**表4-4 同预算方法比较**

{{BENCHMARK_COMPACT_TABLE}}

{{BEST_BENCHMARK_METHOD}}的测试RMSE最低，为{{BEST_BENCHMARK_RMSE}}；去随机化Knockoff为{{KNOCKOFF_BENCHMARK_RMSE}}。统一子抽样下，Elastic Net、去随机化Knockoff和XGBoost-SHAP的平均稳定性分别为{{ELASTIC_BENCHMARK_STABILITY}}、{{KNOCKOFF_BENCHMARK_STABILITY}}和{{XGB_BENCHMARK_STABILITY}}。Elastic Net与Knockoff差异较小且置信区间重叠，预测最优与稳定受控选择是不同目标。

**表4-5 关键组件消融**

{{ABLATION_COMPACT_TABLE}}

重复次数由1增至60时，严格发现数依次发生变化，只有60轮与正式14项集合完全一致。原始高斯边际与替代S矩阵均改变集合；XGBoost竞争统计量在当前设定下没有严格发现。

**表4-6 已知真值模拟**

{{SIMULATION_COMPARISON_TABLE}}

去随机化Knockoff平均FDP为{{SIM_DERAND_FDP}}，低于单次Knockoff的{{SIM_SINGLE_FDP}}，二者Power均为{{SIM_DERAND_POWER}}。Elastic Net和影子变量树模型的高Power伴随较高FDP；稳定性选择在当前场景表现较好，但不具有相同的Model-X有限样本目标保证。

# 第5章 周度稳健性与应用

## 5.1 周度选择结果

周度Copula平均边际KS为{{WEEK_COPULA_KS}}，原始高斯为{{WEEK_GAUSSIAN_KS}}。在0.20水平，周度入选{{WEEK_STRICT_COUNT}}项，其中{{WEEK_STABLE_COUNT}}项单轮频率不低于90%。

**表5-1 周度严格入选变量**

{{WEEK_SELECTION_TABLE}}

## 5.2 月周交集

月度14项和周度18项共同入选{{OVERLAP_COUNT}}项，Jaccard={{JACCARD}}。共同集合为{{OVERLAP_LIST}}。月度特有{{MONTH_ONLY_COUNT}}项为{{MONTH_ONLY_LIST}}；周度特有{{WEEK_ONLY_COUNT}}项为{{WEEK_ONLY_LIST}}。

![图5-1 月度与周度入选频率](figures/fig17_monthly_weekly_frequency.png)

交集集中于交易规模、价格、承运准备、商品标题和市场覆盖。月度特有项更偏慢变化的履约与内容信息；周度特有项更易受单周商品组合、支付和营销构成波动影响。周度特有变量只进入观察层。

## 5.3 周度模型与结构检验

**表5-2 周度时间外预测**

{{WEEK_METRIC_TABLE}}

最佳模型{{WEEK_BEST_MODEL}}的RMSE为{{WEEK_BEST_RMSE}}；周度18项XGBoost的RMSE为{{WEEK_SELECTED_RMSE}}，相对全31维模型变化{{WEEK_SELECTED_DELTA}}。

周度卖家间层入选{{WEEK_BETWEEN_COUNT}}项，卖家内层为{{WEEK_WITHIN_COUNT}}项；Group Knockoff在0.20水平入选{{WEEK_GROUP_COUNT}}组；轨迹统计量在0.20水平为{{WEEK_GRIP_Q20_COUNT}}项。提高时间频率未恢复卖家内严格发现，短序列和零膨胀仍限制动态识别。

## 5.4 看板治理建议

月度层承担统计确认和参数$K$展示，周度层承担短期异常监测。月周共同11项可标记为跨粒度核心，其中周度高稳定的10项优先用于预警。看板按交易规模、价格成本、履约、供给内容和市场覆盖组织，每个模块保留一个主要状态指标和少量下钻指标。

实际周度部署必须按字段真实可见时间重构物流和评价特征。Knockoff、SHAP和固定效应均识别预测关系；对履约、内容和价格策略的可行动建议仍需随机实验或准实验确认。

# 第6章 结论与展望

本文构建了受控确认、参数化展示和多粒度复核组成的GMV维度选择框架。月度主分析在0.20水平形成14项确认集合，默认$K=10$在压缩展示数量的同时保留预测性能。周度分析形成18项集合，与月度共同入选11项；周度入选集合的XGBoost RMSE为{{WEEK_SELECTED_RMSE}}。月度降低低频噪声，适合正式确认；周度增加时间点，适合作为稳健性和短期预警。

研究的主要贡献包括：以多轮e-value聚合降低Knockoff随机性；将统计确认与参数$K$展示分离；通过面板分层、业务组、深度联合诊断和周度重跑明确结论边界；用统一信息窗口、变量预算、预测器和测试期完成公平基准，并以组件消融说明主流程各环节的作用。主要限制包括公开数据缺少曝光、库存和促销信息，面板存在组内相关，深度生成器未通过联合准入，周度回溯口径不等同于实时部署，20轮模拟不足以覆盖所有稀疏度与相关结构。

后续研究可在更长企业面板上使用滚动Knockoff和在线e-value，采用显式离散似然或条件生成模型改善联合分布，并对可行动变量开展实验或因果机器学习分析。

# 参考文献

{{REFERENCES_TEXT}}
