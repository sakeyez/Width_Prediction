# 宽度预测项目使用手册

本项目用于粗轧宽度预测与立辊压下量优化，整体流程如下：

`原始数据 -> 数据清洗/划分 -> 模型训练与评估 -> MGH 建模 -> PSO 优化`

目前仓库中主要维护 5 个建模脚本和 1 个优化脚本：

- `2_Model/LightGBM.py`：LightGBM 基线模型
- `2_Model/LightGBM_BO.py`：贝叶斯优化后的 LightGBM
- `2_Model/DBN.py`：联合微调 DBN 基线模型
- `2_Model/DBN_BO.py`：贝叶斯优化后的 DBN
- `2_Model/MGH.py`：特征扩展 + 层次聚类 + GA 选特征 + GLR 建模
- `3_Optimize/PSO.py`：基于 MGH 结果进行立辊压下量优化

## 1. 项目结构

```text
宽度预测/
├─ 1_Datadeal/         数据清洗、分析、可视化
├─ 2_Model/            各类预测模型脚本
├─ 3_Optimize/         参数优化脚本
├─ data/               数据、模型、指标输出目录
├─ image/              图像输出目录
├─ originaldata/       原始 Excel 数据
└─ readme.markdown
```

目录说明：

- `1_Datadeal/`：生成清洗后的数据集，并做散点图等可视化分析。
- `2_Model/`：训练不同宽度预测模型，并导出预测结果、搜索结果、指标和图像。
- `3_Optimize/`：使用 `MGH.py` 训练得到的模型与扩展测试集，做 PSO 优化。
- `data/`：保存中间数据、结果表、指标文件、模型文件。
- `image/`：保存散点图、搜索曲线图、流程图、对比图等。

## 2. 运行环境

推荐环境：

- `Python 3.10.11`

主要依赖：

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scipy`
- `lightgbm`
- `optuna`
- `joblib`
- `openpyxl`

安装示例：

```powershell
pip install numpy pandas matplotlib seaborn scikit-learn scipy lightgbm optuna joblib openpyxl
```

## 3. 数据准备

### 3.1 原始数据文件

当前项目会用到以下原始数据：

- `originaldata/2160粗轧无设定值数据.xlsx`
- `originaldata/带设定值粗轧数据.xlsx`

其中 `1_Datadeal/Data Deal.py` 会基于宽度列匹配设定值信息，生成完整数据表。

### 3.2 列名与目标列

重要说明：

- 脚本依赖中文列名，不建议手动改 Excel 表头。
- 所有模型默认预测目标列都是 `实测宽度-R2-5`。
- 多个模型会排除 `出口宽度设定值-R2-5`，避免把设定目标直接作为训练特征。

### 3.3 数据处理脚本

运行：

```powershell
python "1_Datadeal/Data Deal.py"
```

主要内容：

1. 合并“无设定值数据”和“带设定值数据”。
2. 删除与建模无关或不希望参与训练的列。
3. 保留优化阶段需要的关键立辊压下量特征。
4. 将部分物理量中的 `0` 视作缺失值并删除缺失样本。
5. 使用 `3 * IQR` 规则剔除离群值。
6. 按 `7:2:1` 划分训练集、验证集、测试集。
7. 仅用训练集拟合归一化器，并将同一套缩放参数应用到验证集与测试集。

输出目录：

- `data/datadeal/`

主要输出文件：

- `data/datadeal/2160粗轧数据_清洗完整版.xlsx`
- `data/datadeal/Train_Data.xlsx`
- `data/datadeal/Val_Data.xlsx`
- `data/datadeal/Test_Data.xlsx`

## 4. 推荐运行顺序

```powershell
python "1_Datadeal/Data Deal.py"
python "1_Datadeal/Data Analysis.py"
python "2_Model/LightGBM.py"
python "2_Model/LightGBM_BO.py"
python "2_Model/DBN.py"
python "2_Model/DBN_BO.py"
python "2_Model/MGH.py"
python "3_Optimize/PSO.py"
```

如果只想快速跑通主流程，推荐：

```powershell
python "1_Datadeal/Data Deal.py"
python "2_Model/LightGBM.py"
python "2_Model/MGH.py"
python "3_Optimize/PSO.py"
```

## 5. 模型说明

这一节按“脚本作用、输入、输出、关键参数”来说明每个模型。

### 5.1 LightGBM 基线

脚本：

- `2_Model/LightGBM.py`

运行：

```powershell
python "2_Model/LightGBM.py"
```

主要内容：

- 使用 `Train_Data.xlsx / Val_Data.xlsx / Test_Data.xlsx` 训练一个固定参数的 `LGBMRegressor`。
- 在训练集、验证集、测试集上分别评估 `R2`、`MAE`、`MSE`、`RMSE`。
- 导出预测结果表、测试集预测 CSV 和预测散点图。

输入文件：

- `data/datadeal/Train_Data.xlsx`
- `data/datadeal/Val_Data.xlsx`
- `data/datadeal/Test_Data.xlsx`

输出目录：

- `data/LightGBM/`
- `image/LightGBM/`

主要输出文件：

- `data/LightGBM/LightGBM_prediction_results.xlsx`
- `data/LightGBM/result_LightGBM.csv`
- `image/LightGBM/LightGBM_Prediction_Scatter.png`

随机数种子：

- `RANDOM_SEED = 42`

关键参数：

- `n_estimators=500`
  含义：提升树总轮数，越大拟合能力越强，但训练更慢。
- `learning_rate=0.03`
  含义：每轮提升步长，越小通常越稳，但通常需要更多树。
- `max_depth=7`
  含义：单棵树最大深度，用于限制树复杂度。
- `num_leaves=24`
  含义：单棵树叶子节点数，是 LightGBM 最关键的复杂度参数之一。
- `min_child_samples=30`
  含义：一个叶子节点最少需要的样本数，越大越保守。
- `subsample=0.85`
  含义：每轮仅抽取 85% 样本建树，用来降低过拟合。
- `subsample_freq=1`
  含义：每轮都启用一次样本采样。
- `colsample_bytree=0.9`
  含义：每棵树仅使用 90% 特征。
- `reg_alpha=0.4`
  含义：L1 正则强度。
- `reg_lambda=2.0`
  含义：L2 正则强度。
- `min_split_gain=0.03`
  含义：节点继续分裂所需的最小收益。
- `early_stopping(stopping_rounds=40)`
  含义：验证集连续 40 轮无提升时提前停止。
- `deterministic=True`
  含义：尽量保证同配置下结果可复现。

适用场景：

- 先获得稳定的树模型基线结果。
- 为后续 `LightGBM_BO.py` 做对照。

### 5.2 BO-LightGBM

脚本：

- `2_Model/LightGBM_BO.py`

运行：

```powershell
python "2_Model/LightGBM_BO.py"
```

主要内容：

- 使用 Optuna 分 3 个阶段对 LightGBM 做贝叶斯优化。
- 目标函数不仅看验证集误差，还额外惩罚训练集与验证集之间的过拟合差距。
- 搜索结束后，用最优参数重新训练模型并导出结果。

输入文件：

- `data/datadeal/Train_Data.xlsx`
- `data/datadeal/Val_Data.xlsx`
- `data/datadeal/Test_Data.xlsx`

输出目录：

- `data/LightGBM_BO/`
- `image/LightGBM_BO/`

主要输出文件：

- `data/LightGBM_BO/LightGBM_BO_prediction_results.xlsx`
- `data/LightGBM_BO/result_LightGBM_BO.csv`
- `data/LightGBM_BO/LightGBM_BO_best_params.json`
- `data/LightGBM_BO/LightGBM_BO_metrics.json`
- `data/LightGBM_BO/LightGBM_BO_search_history.csv`
- `image/LightGBM_BO/LightGBM_BO_Prediction_Scatter.png`
- `image/LightGBM_BO/LightGBM_BO_Search_Curve.png`

随机数种子：

- `RANDOM_SEED = 42`

运行开关：

- `RUN_OPTION = 1`：执行 BO 搜索、最终训练并导出结果。
- `RUN_OPTION = 2`：只画图，不重新训练。

关键搜索控制参数：

- `STAGE1_TRIALS = 100`
  含义：第一阶段搜索轮数，主要搜索基础容量参数。
- `STAGE2_TRIALS = 100`
  含义：第二阶段搜索轮数，主要搜索树结构与采样参数。
- `STAGE3_TRIALS = 100`
  含义：第三阶段搜索轮数，主要搜索正则化参数。
- `EARLY_STOPPING_ROUNDS = 30`
  含义：BO 过程中的 LightGBM 提前停止轮数。
- `OBJECTIVE_OVERFIT_WEIGHT = 0.35`
  含义：目标函数中“过拟合惩罚项”的权重，越大越偏向稳健参数。

固定基线参数：

- `n_estimators = 500`
- `learning_rate = 0.03`
- `num_leaves = 20`
- `max_depth = 6`
- `min_child_samples = 45`
- `subsample = 0.82`
- `colsample_bytree = 0.85`
- `reg_alpha = 0.8`
- `reg_lambda = 4.0`
- `min_split_gain = 0.08`

三阶段主要搜索空间：

- 第一阶段：
  `n_estimators: 300 ~ 950`
  `learning_rate: 0.01 ~ 0.04`
  `num_leaves: 12 ~ 28`
- 第二阶段：
  `max_depth: 4 ~ 7`
  `min_child_samples: 35 ~ 110`
  `subsample: 0.72 ~ 0.90`
- 第三阶段：
  `colsample_bytree: 0.72 ~ 0.90`
  `reg_alpha: 0.1 ~ 2.5`
  `reg_lambda: 1.0 ~ 8.0`

适用场景：

- 想在 LightGBM 基线基础上进一步自动调参。
- 更关注验证集稳定性而不是单纯压低训练误差。

### 5.3 DBN 基线

脚本：

- `2_Model/DBN.py`

运行：

```powershell
python "2_Model/DBN.py"
```

主要内容：

- 先用 `BernoulliRBM` 做逐层无监督预训练。
- 再把预训练权重注入 `MLPRegressor`，进行联合微调。
- 对输入特征做 `MinMaxScaler(0.1, 0.9)`，对目标值做 `StandardScaler`。
- 导出预测结果图、流程图和预测数据。

输入文件：

- `data/datadeal/Train_Data.xlsx`
- `data/datadeal/Val_Data.xlsx`
- `data/datadeal/Test_Data.xlsx`

输出目录：

- `data/DBN/`
- `image/DBN/`

主要输出文件：

- `data/DBN/DBN_预测结果汇总.xlsx`
- `data/DBN/result_DBN_Pure.csv`
- `image/DBN/DBN_Pure_Prediction_Scatter.png`
- `image/DBN/DBN_Code_Flowchart.png`

随机数种子：

- `RANDOM_SEED = 42`

运行开关：

- `RUN_OPTION = 1`：完整训练并导出结果。
- `RUN_OPTION = 2`：只画图，不重新训练。

关键结构参数：

- `rbm_hidden_layers = (72,)`
  含义：RBM 预训练隐层规模，当前为单层 72 节点。
- `mlp_hidden_sizes = ()`
  含义：监督微调阶段额外 MLP 隐层结构，当前为空，表示不额外加隐藏层。

关键训练参数：

- `RBM_LEARNING_RATE = 0.001`
  含义：RBM 预训练学习率。
- `RBM_N_ITER = 18`
  含义：每层 RBM 预训练迭代轮数。
- `SUPERVISED_MAX_EPOCHS = 75`
  含义：联合微调最大训练轮数。
- `SUPERVISED_LEARNING_RATE = 0.001`
  含义：联合微调阶段学习率。
- `EARLY_STOPPING_PATIENCE = 10`
  含义：验证集连续 10 轮无提升则提前停止。
- `MIN_IMPROVEMENT = 1e-4`
  含义：小于该阈值的波动不认为是真正提升。
- `MLP_ALPHA = 5e-4`
  含义：监督微调网络的 L2 正则强度。

最影响结果的参数通常是：

- `rbm_hidden_layers`
- `SUPERVISED_MAX_EPOCHS`
- `MLP_ALPHA`

### 5.4 BO-DBN

脚本：

- `2_Model/DBN_BO.py`

运行：

```powershell
python "2_Model/DBN_BO.py"
```

主要内容：

- 使用 Optuna 自动搜索 DBN 的容量、预训练轮数、联合微调轮数、学习率和正则化参数。
- 优化目标不是单纯追求最小误差，而是希望结果落在预设的 `MAE / MSE` 合理区间内。
- 搜索完成后用最优参数重新训练，并导出历史搜索曲线与结果文件。

输入文件：

- `data/datadeal/Train_Data.xlsx`
- `data/datadeal/Val_Data.xlsx`
- `data/datadeal/Test_Data.xlsx`

输出目录：

- `data/DBN_BO/`
- `image/DBN_BO/`

主要输出文件：

- `data/DBN_BO/DBN_BO_prediction_results.xlsx`
- `data/DBN_BO/result_DBN_BO.csv`
- `data/DBN_BO/DBN_BO_best_params.json`
- `data/DBN_BO/DBN_BO_metrics.json`
- `data/DBN_BO/DBN_BO_search_history.csv`
- `image/DBN_BO/DBN_BO_Prediction_Scatter.png`
- `image/DBN_BO/DBN_BO_Search_Curve.png`

随机数种子：

- `RANDOM_SEED = 42`

运行开关：

- `RUN_OPTION = 1`：执行 BO 搜索、最终训练并导出结果。
- `RUN_OPTION = 2`：只画图，不重新训练。

关键 BO 参数：

- `BO_TRIALS = 30`
  含义：总共搜索 30 组参数。
- `FIXED_MLP_HIDDEN_SIZES = (16,)`
  含义：固定监督部分额外隐藏层为 1 层 16 节点，便于把搜索重点放在 RBM 容量和训练参数上。
- `TARGET_MAE_RANGE = (4.2, 5.5)`
  含义：希望验证集 MAE 落入的目标区间。
- `TARGET_MSE_RANGE = (28.0, 50.0)`
  含义：希望验证集 MSE 落入的目标区间。

主要搜索空间：

- `n_components: 48 ~ 256, step=8`
  含义：RBM 隐层节点数，是最核心的容量参数。
- `rbm_n_iter: 10 ~ 50`
  含义：RBM 预训练轮数。
- `supervised_max_epochs: 40 ~ 120`
  含义：联合微调最大轮数。
- `early_stopping_patience: 8 ~ 25`
  含义：提前停止耐心值。
- `rbm_learning_rate: 5e-4 ~ 2e-2`
  含义：RBM 学习率。
- `supervised_learning_rate: 5e-4 ~ 2e-2`
  含义：联合微调学习率。
- `alpha: 1e-5 ~ 1e-2`
  含义：L2 正则强度。

目标函数含义：

- 超出 `TARGET_MAE_RANGE` 或 `TARGET_MSE_RANGE` 会被惩罚。
- 即使落在目标区间内，也会被轻微拉向区间中心，避免参数只是在边界“碰巧达标”。

### 5.5 MGH

脚本：

- `2_Model/MGH.py`

运行：

```powershell
python "2_Model/MGH.py"
```

主要内容：

- 对 `datadeal` 数据集做二次特征扩展。
- 基于特征相关性做层次聚类，搜索最优簇数 `C`。
- 在每个簇内用 GA 选择代表特征，再训练广义线性回归模型。
- 生成后续 `PSO.py` 需要的模型文件与扩展测试集。

输入文件：

- `data/datadeal/Train_Data.xlsx`
- `data/datadeal/Val_Data.xlsx`
- `data/datadeal/Test_Data.xlsx`

输出目录：

- `data/MGH/`
- `image/MGH/`

主要输出文件：

- `data/MGH/MGH_Train_Expanded.xlsx`
- `data/MGH/MGH_Val_Expanded.xlsx`
- `data/MGH/MGH_Test_Expanded.xlsx`
- `data/MGH/mgh_stage1_artifacts.pkl`
- `data/MGH/MGH_C_Search_Results.xlsx`
- `data/MGH/final_glr_model.pkl`
- `data/MGH/MGH_prediction_summary.xlsx`
- `data/MGH/result_MGH.csv`
- `data/MGH/MGH_metrics.json`
- `image/MGH/MGH_C_Search_Curve.png`
- `image/MGH/MGH_Hierarchical_Dendrogram.png`
- `image/MGH/MGH_Parity_Plots.png`

随机数种子：

- `RANDOM_SEED = 42`

菜单模式：

- `1`：完整流程
- `2`：特征构建与搜索最优簇数
- `3`：完整建模
- `4`：测试并画图

阶段 1：特征扩展与最优簇数搜索

主要内容：

- 为原始特征增加平方项 `*_Square`。
- 对顺序特征组生成差分项与绝对差分项。
- 根据特征相关性构造距离矩阵并做层次聚类。
- 在 `C=5 ~ 45` 范围内搜索最优簇数。

关键参数：

- `SEARCH_C_RANGE = range(5, 46)`
  含义：候选簇数搜索范围。
- `STAGE1_POP_SIZE = 50`
  含义：簇内 GA 的种群规模。
- `STAGE1_GENERATIONS = 100`
  含义：簇内 GA 的迭代代数。
- `STAGE1_CLUSTER_COUNT_WEIGHT = 0.08`
  含义：综合评分中对“较少簇数”的偏好权重，避免一味选很大的 `C`。

阶段 2：GA-GLR 建模

主要内容：

- 固定阶段 1 找到的聚类结构。
- 在每个簇中挑选代表特征。
- 强制保留优化相关的立辊压下量特征。
- 用 `LinearRegression` 在选中特征上建模。

关键参数：

- `STAGE2_POP_SIZE = 60`
  含义：阶段 2 GA 种群规模。
- `STAGE2_GENERATIONS = 60`
  含义：阶段 2 GA 迭代代数。
- `STAGE2_MUTATION_RATE = 0.2`
  含义：突变概率，越大搜索越活跃。
- `STAGE2_KFOLD = 10`
  含义：10 折交叉验证，用于评估候选特征组合。
- `OPTIMIZATION_COLUMNS = ["压下量E2-1", "压下量E2-3", "压下量E2-5"]`
  含义：无论 GA 如何选择，都尽量保留后续优化需要的关键变量。

阶段 3：测试与导出

主要内容：

- 读取 `final_glr_model.pkl`。
- 在训练集、验证集、测试集上评估模型。
- 导出预测摘要、测试集结果、指标 JSON 和奇偶图。

补充说明：

- `MGH.py` 生成的 `MGH_Test_Expanded.xlsx` 和 `final_glr_model.pkl` 是 `PSO.py` 的必要输入。

## 6. PSO 优化说明

脚本：

- `3_Optimize/PSO.py`

运行：

```powershell
python "3_Optimize/PSO.py"
```

主要内容：

- 读取 `MGH.py` 训练好的线性模型与扩展测试集。
- 选取设定值非空、且模型误差较大的样本作为优化对象。
- 对压下量变量做粒子群搜索，寻找让预测宽度更接近设定宽度的参数组合。
- 导出完整优化结果、摘要统计和论文表格格式结果。

输入文件：

- `data/MGH/MGH_Test_Expanded.xlsx`
- `data/MGH/final_glr_model.pkl`

输出目录：

- `data/PSO/`

主要输出文件：

- `data/PSO/PSO_Optimization_Results.xlsx`
- `data/PSO/PSO_Optimization_Summary.json`
- `data/PSO/4.1.xlsx`
- `data/PSO/4.2.xlsx`
- `data/PSO/4.1_top20.xlsx`
- `data/PSO/4.2_top20.xlsx`
- `data/PSO/PSO_Sensitivity_Detail.xlsx`
- `data/PSO/PSO_Sensitivity_Summary.xlsx`

菜单模式：

- `1`：完整流程，执行 PSO 优化。
- `2`：从已有结果中提取 Top20 表格数据。

关键样本控制参数：

- `OPTIMIZE_SAMPLE_COUNT = 100`
  含义：默认选取 100 个样本参与优化。
- `TOP_SAMPLE_COUNT = 20`
  含义：用于导出论文表格的 Top20 样本数量。

关键 PSO 参数：

- `INERTIA_WEIGHT = 0.50`
  含义：粒子保留原速度的程度。
- `INDIVIDUAL_FACTOR = 1.35`
  含义：粒子向个体历史最优位置靠拢的强度。
- `SOCIAL_FACTOR = 1.45`
  含义：粒子向群体全局最优位置靠拢的强度。

误差分档参数：

- `HIGH_ERROR_THRESHOLD = 20.0`
  含义：若基线误差大于 20 mm，则进入激进搜索档。
- `UNDER_ERROR_THRESHOLD = 1.0`
  含义：若误差已接近 1 mm，会增加“欠误差惩罚”。

激进档参数：

- `AGGRESSIVE_NUM_PARTICLES = 36`
- `AGGRESSIVE_MAX_ITERATIONS = 60`
- `AGGRESSIVE_SEARCH_MARGIN_RATIO = 0.55`
- `AGGRESSIVE_MAX_ADJUSTMENT_MM = 8.0`
- `AGGRESSIVE_ADJUSTMENT_WEIGHT = 0.12`
- `AGGRESSIVE_UNDER_ERROR_WEIGHT = 27.0`

含义：

- 当样本初始误差很大时，允许在更积极的范围里搜索压下量调整方案。

保守档参数：

- `CONSERVATIVE_NUM_PARTICLES = 36`
- `CONSERVATIVE_MAX_ITERATIONS = 60`
- `CONSERVATIVE_SEARCH_MARGIN_RATIO = 0.55`
- `CONSERVATIVE_MAX_ADJUSTMENT_MM = 8.0`
- `CONSERVATIVE_ADJUSTMENT_WEIGHT = 0.12`
- `CONSERVATIVE_UNDER_ERROR_WEIGHT = 27.0`

含义：

- 当样本初始误差不算太大时，限制搜索幅度，避免给出过度激进的调整建议。

目标函数主要考虑：

- 预测宽度与设定宽度的偏差。
- 调整量不能过大。
- 当误差已经很小时，不鼓励为继续逼近而做过大操作。

## 7. 输出文件说明

常见输出类型：

- `*.xlsx`：预测汇总、优化结果、搜索结果、论文表格。
- `*.csv`：测试集预测结果、搜索历史。
- `*.json`：指标、最优参数、结果摘要。
- `*.pkl`：训练好的模型、阶段性产物。
- `*.png / *.svg`：散点图、搜索曲线、奇偶图、流程图、聚类树图。

常见输出目录：

- `data/datadeal/`
- `data/LightGBM/`
- `data/LightGBM_BO/`
- `data/DBN/`
- `data/DBN_BO/`
- `data/MGH/`
- `data/PSO/`
- `image/LightGBM/`
- `image/LightGBM_BO/`
- `image/DBN/`
- `image/DBN_BO/`
- `image/MGH/`

## 8. 使用注意事项

- 所有脚本都依赖中文列名，不建议手动修改原始数据表头。
- 若 Excel 正在占用某个输出文件，部分脚本会自动另存为 `_new.xlsx`、`_new.png` 或带时间戳的新文件。
- 项目路径和文件名包含中文，建议在 Windows + UTF-8 环境下运行。
- `MGH.py` 与 `PSO.py` 都改成了菜单式运行逻辑；如果不是交互式终端，会自动走默认选项。
- `PSO.py` 依赖 `MGH.py` 的输出，不能直接跳过 MGH。

## 9. 后续可继续完善的方向

- 增加 `requirements.txt`
- 增加每个字段的含义说明表
- 为各模型输出增加统一实验编号
- 把菜单参数逐步改造成命令行参数，方便批量实验
