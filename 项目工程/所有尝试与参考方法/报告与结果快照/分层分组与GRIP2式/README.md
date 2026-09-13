# Olist 电商 GMV 维度选择论文（v4）

V4保留V1确认主线、V2深度诊断与V3参数K，新增：

- 卖家均值/卖家内离差的聚类分层Knockoff；
- 十二业务组Group Knockoff；
- DDLK式最坏swap深度生成器；
- 仅训练子集拟合的经验边际校准；
- 训练/检验分离的深度核MMD；
- GRIP2式二维正则轨迹聚合重要性；
- 置换次数与最小p值的明确记录。

主要结果：

- q=0.20：卖家间层16项，卖家内层0项；
- 组级q=0.20为0组，q=0.30为10组；
- V4最坏swap生成器KS约0.019，但深度核5/5拒绝、分类器AUC约0.937，准入失败；
- GRIP2式反对称误差为0，q=0.20为0项，q=0.30为5项；
- V1的14项确认集与V3默认K=10保持不变。

## 复现

```bash
论文/v1/.venv/bin/python 论文/v4/code/run_v4_analysis.py
论文/v1/.venv/bin/python 论文/v4/code/generate_v4_thesis.py
论文/v1/.venv/bin/python 论文/v4/code/generate_v4_condensed_thesis.py
论文/v1/.venv/bin/python 论文/v4/code/generate_v4_ultra_condensed_thesis.py
```
