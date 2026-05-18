# 宽度预测项目说明

本项目用于粗轧宽度预测与压下量优化，整体流程如下：

`原始数据 -> 数据清洗与切分 -> 模型训练与评估 -> 基于 MGH 的 PSO 优化`

当前仓库里主要维护 4 个模型脚本：

- `LightGBM.py`：树模型基线
- `LightGBM_BO.py`：贝叶斯优化后的 LightGBM
- `DBN.py`：深度置信网络基线
- `DBN_BO.py`：贝叶斯优化后的 DBN
- `MGH.py`：特征扩展 + 聚类搜索 + GA-GLR 建模

以及 1 个优化脚本：

- `PSO.py`：基于 `MGH` 产物对压下量进行粒子群优化

## 1. 项目结构

```text
宽度预测/
├─ 1_Datadeal/         数据清洗、分析与切分
├─ 2_Model/            预测模型
├─ 3_Optimize/         优化脚本（PSO）
├─ data/               表格、模型、指标输出
├─ image/              图像输出
├─ originaldata/       原始 Excel 数据
└─ readme.markdown
```

## 2. 环境与依赖

建议环境：

- `Python 3.10`

常用依赖：

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scipy`
- `lightgbm`
- `optuna`
- `openpyxl`

安装示例：

```powershell
pip install numpy pandas matplotlib seaborn scikit-learn scipy lightgbm optuna openpyxl
```

## 3. 数据准备

### 3.1 原始数据

当前数据处理脚本会同时使用两份原始数据：

- `originaldata/2160粗轧无设定值数据.xlsx`
- `originaldata/带设定值粗轧数据.xlsx`

`1_Datadeal/Data Deal.py` 会用下面三列做匹配：

- `实测宽度-R2-1`
- `实测宽度-R2-3`
- `实测宽度-R2-5`

把带设定值表里的 `出口宽度设定值-R2-5` 回填到无设定值底表，生成：

- `originaldata/完整版数据.xlsx`

### 3.2 处理逻辑

`Data Deal.py` 的主要处理过程是：

1. 先生成 `完整版数据.xlsx`
2. 删除与任务无关或不希望参与训练的列
3. 将物理量中的 `0` 视作缺失值并删除缺失样本
4. 用 `3 * IQR` 剔除离群值
5. 按 `7:2:1` 划分训练集、验证集、测试集
6. 仅用训练集拟合 `MinMaxScaler`
7. 用同一套归一化参数变换训练集、验证集、测试集

输出文件：

- `data/datadeal/2160粗轧数据_清洗完整版.xlsx`
- `data/datadeal/Train_Data.xlsx`
- `data/datadeal/Val_Data.xlsx`
- `data/datadeal/Test_Data.xlsx`

## 4. 推荐运行顺序

### 第一步：数据清洗与切分

```powershell
python "1_Datadeal/Data Deal.py"
```

### 第二步：数据分析与画图

```powershell
python "1_Datadeal/Data Analysis.py"
```

当前 `Data Analysis.py` 已使用相对路径，默认读取：

- `originaldata/2160粗轧无设定值数据.xlsx`

### 第三步：训练模型

```powershell
python "2_Model/LightGBM.py"
python "2_Model/LightGBM_BO.py"
python "2_Model/DBN.py"
python "2_Model/DBN_BO.py"
python "2_Model/MGH.py"
```

### 第四步：执行 PSO 优化

```powershell
python "3_Optimize/PSO.py"
```

## 5. 模型说明与关键参数

这一节重点说明每个模型“最值得调的参数”以及它们的作用。

### 5.1 LightGBM 基线

运行：

```powershell
python "2_Model/LightGBM.py"
```

输入：

- `data/datadeal/Train_Data.xlsx`
- `data/datadeal/Val_Data.xlsx`
- `data/datadeal/Test_Data.xlsx`

输出：

- `data/LightGBM/`
- `image/LightGBM/`

当前代码中的完整核心参数为：

```python
n_estimators=500,
learning_rate=0.03,
max_depth=7,
num_leaves=24,
min_child_samples=30,
subsample=0.85,
subsample_freq=1,
colsample_bytree=0.9,
reg_alpha=0.4,
reg_lambda=2.0,
min_split_gain=0.03,
random_state=RANDOM_SEED,
n_jobs=-1,
deterministic=True,
force_col_wise=True,
bagging_seed=RANDOM_SEED,
feature_fraction_seed=RANDOM_SEED,
data_random_seed=RANDOM_SEED,
verbosity=-1
```

这些参数可以分成 4 类来看。

第一类：控制模型容量的参数

- `n_estimators = 500`
  作用：总共训练多少棵树。
  调大后：模型表达能力更强，训练时间更长，若其他约束不够容易过拟合。
  调小后：训练更快，但可能欠拟合。
- `learning_rate = 0.03`
  作用：每棵树对最终预测结果的修正步长。
  调大后：学习更快，但更容易震荡或过拟合。
  调小后：通常更稳，但需要更多树配合。
- `max_depth = 7`
  作用：限制每棵树最多能长多深。
  调大后：树能学到更复杂的规则，但更容易记住噪声。
  调小后：更保守，泛化通常更稳。
- `num_leaves = 24`
  作用：限制一棵树最多有多少叶子节点，是 LightGBM 里非常关键的复杂度参数。
  调大后：拟合能力明显增强，容易把训练误差压得很低。
  调小后：模型更简单，抗过拟合更强。

第二类：控制样本与特征采样的参数

- `min_child_samples = 30`
  作用：一个叶子节点里至少要保留多少样本。
  调大后：模型更保守，不容易把少量异常样本单独切成叶子。
  调小后：更容易切出很细的小叶子，可能提高训练拟合也可能放大噪声。
- `subsample = 0.85`
  作用：每轮训练只随机抽取 85% 的样本参与建树。
  调大后：更接近用全量样本建树，拟合更充分。
  调小后：随机性更强，通常更能抑制过拟合，但太低会损失精度。
- `subsample_freq = 1`
  作用：表示每轮都执行一次样本采样。
  若设为 `0`：则不启用这类 bagging 抽样。
- `colsample_bytree = 0.9`
  作用：每棵树只随机抽取 90% 特征参与训练。
  调大后：每棵树看到的特征更多，拟合能力更强。
  调小后：模型随机性更强，更像随机森林式的降相关思路。

第三类：控制正则化和分裂门槛的参数

- `reg_alpha = 0.4`
  作用：L1 正则化强度。
  调大后：会更强地压缩不重要分支或权重，模型更稀疏、更保守。
  调小后：约束更弱，模型更容易追求极致拟合。
- `reg_lambda = 2.0`
  作用：L2 正则化强度。
  调大后：整体参数会更平滑，能减轻过拟合。
  调小后：模型约束更弱。
- `min_split_gain = 0.03`
  作用：节点继续分裂前，至少要带来多少收益。
  调大后：很多“收益不大”的小分裂会被禁止，树会更短、更稳。
  调小后：树更容易继续细分，可能提高拟合也可能过拟合。

第四类：控制复现性和运行方式的参数

- `random_state = RANDOM_SEED`
  作用：主随机种子，保证多次运行结果尽量一致。
- `bagging_seed = RANDOM_SEED`
  作用：样本采样过程的随机种子。
- `feature_fraction_seed = RANDOM_SEED`
  作用：特征采样过程的随机种子。
- `data_random_seed = RANDOM_SEED`
  作用：数据相关随机过程的种子。
  这几个种子统一设置后，更有利于复现实验结果。
- `n_jobs = -1`
  作用：使用全部 CPU 核心并行训练。
  调成更小的正整数：可以限制 CPU 占用。
- `deterministic = True`
  作用：尽量让训练过程走确定性路径，减少同配置多次运行结果轻微漂移。
- `force_col_wise = True`
  作用：强制 LightGBM 使用按列方式构建直方图，通常在当前环境下更稳，也更容易避免某些线程与内存问题。
- `verbosity = -1`
  作用：关闭冗余训练日志，让终端输出更干净。

另外，当前脚本还有一条很关键的训练控制：

- `early_stopping(stopping_rounds=40)`
  作用：若验证集连续 40 轮没有提升，就提前停止训练。
  调大后：更愿意继续训练，可能更强也可能更容易过拟合。
  调小后：更早停，更保守。

适用场景：

- 想快速得到稳定基线
- 想做后续 BO 结果对照

### 5.2 BO-LightGBM

运行：

```powershell
python "2_Model/LightGBM_BO.py"
```

输出：

- `data/LightGBM_BO/`
- `image/LightGBM_BO/`

当前搜索设置：

- `STAGE1_TRIALS = 100`
- `STAGE2_TRIALS = 100`
- `STAGE3_TRIALS = 100`

含义：

- 第一阶段主要搜基础容量参数
- 第二阶段主要搜树结构和采样参数
- 第三阶段主要搜正则化参数

固定基线参数：

- `n_estimators = 500`
  作用：以 500 棵树作为 BO 搜索的基础起点。
- `learning_rate = 0.03`
  作用：使用偏稳妥的小步长。
- `num_leaves = 20`
  作用：默认限制树的复杂度。
- `max_depth = 6`
  作用：默认比基线更浅一点，抑制过拟合。
- `min_child_samples = 45`
  作用：默认比基线更保守，要求更多样本才能形成叶子。
- `subsample = 0.82`
  作用：提高随机采样程度。
- `colsample_bytree = 0.85`
  作用：略减少每棵树使用的特征比例。
- `reg_alpha = 0.8`
  作用：增强 L1 正则。
- `reg_lambda = 4.0`
  作用：增强 L2 正则。
- `min_split_gain = 0.08`
  作用：提高继续分裂门槛，让树不要长得太激进。

关键搜索空间：

- `n_estimators: 300 ~ 950`
  作用：控制提升轮数。
- `learning_rate: 0.01 ~ 0.04`
  作用：控制学习步长。
- `num_leaves: 12 ~ 28`
  作用：控制树复杂度。
- `max_depth: 4 ~ 7`
  作用：限制树深，防止过拟合。
- `min_child_samples: 35 ~ 110`
  作用：限制小叶子节点。
- `subsample: 0.72 ~ 0.90`
  作用：样本采样比例。
- `colsample_bytree: 0.72 ~ 0.90`
  作用：特征采样比例。
- `reg_alpha: 0.1 ~ 2.5`
  作用：L1 正则搜索范围。
- `reg_lambda: 1.0 ~ 8.0`
  作用：L2 正则搜索范围。

当前防过拟合设置：

- `EARLY_STOPPING_ROUNDS = 30`
  作用：BO 训练时更早停，避免为了追求训练集效果把树不断长大。
- `OBJECTIVE_OVERFIT_WEIGHT = 0.35`
  作用：目标函数不只看验证集误差，还额外考虑训练集和验证集差距。
  调大后：BO 会更偏向“稳健”的参数组合。
  调小后：BO 会更偏向“验证集误差尽量低”的组合。

一句话理解：

- `LightGBM.py` 是人工设定的稳妥基线
- `LightGBM_BO.py` 是自动搜参后的增强版

### 5.3 DBN 基线

运行：

```powershell
python "2_Model/DBN.py"
```

输出：

- `data/DBN/`
- `image/DBN/`

当前代码中的核心参数为：

```python
rbm_hidden_layers = (72,)
mlp_hidden_sizes = ()
RBM_LEARNING_RATE = 0.001
RBM_N_ITER = 18
SUPERVISED_MAX_EPOCHS = 75
SUPERVISED_LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10
MIN_IMPROVEMENT = 1e-4
MLP_ALPHA = 5e-4
```

参数解释：

- `rbm_hidden_layers = (72,)`
  作用：RBM 预训练层的节点数。
  调大后：模型容量变强，更容易学到复杂模式。
  调小后：模型更像保守 baseline。
- `mlp_hidden_sizes = ()`
  作用：联合微调阶段额外 MLP 隐层结构。
  当前为空，表示不额外加隐藏层，让 baseline 不要过强。
- `RBM_LEARNING_RATE = 0.001`
  作用：RBM 预训练学习率。
  调大后：学习更快，但可能不稳定。
  调小后：更稳，但可能学得慢。
- `RBM_N_ITER = 18`
  作用：RBM 每层预训练轮数。
  调大后：预训练更充分。
  调小后：更像弱一点的起步表示。
- `SUPERVISED_MAX_EPOCHS = 75`
  作用：联合微调最多训练多少轮。
  调大后：模型有更多机会把误差继续压低。
  调小后：更保守。
- `SUPERVISED_LEARNING_RATE = 0.001`
  作用：联合微调学习率。
- `EARLY_STOPPING_PATIENCE = 10`
  作用：验证集连续 10 轮无明显提升则停止。
  调大后：更愿意继续训练。
  调小后：更容易提前停。
- `MIN_IMPROVEMENT = 1e-4`
  作用：把很微小的波动过滤掉，只有超过阈值才认定为真正提升。
- `MLP_ALPHA = 5e-4`
  作用：联合微调网络的 L2 正则化强度。
  调大后：更保守，更不容易过拟合。
  调小后：模型更自由。

这些参数里，通常最影响结果的是：

1. `rbm_hidden_layers`
2. `SUPERVISED_MAX_EPOCHS`
3. `MLP_ALPHA`

如果要手调 DBN，优先从这 3 个开始。

### 5.4 BO-DBN

运行：

```powershell
python "2_Model/DBN_BO.py"
```

输出：

- `data/DBN_BO/`
- `image/DBN_BO/`

当前 BO 设置：

- `BO_TRIALS = 30`
  作用：总共尝试 30 组参数。调大后搜索更充分，但耗时更长。
- `FIXED_MLP_HIDDEN_SIZES = (16,)`
  作用：固定一个 16 节点的 MLP 隐层，让 BO 主要关注 RBM 容量和训练超参数。

关键搜索参数：

- `n_components: 48 ~ 256`
  作用：RBM 隐层节点数，是最重要的容量参数。
- `rbm_n_iter: 10 ~ 50`
  作用：RBM 预训练轮数，决定预训练充分程度。
- `supervised_max_epochs: 40 ~ 120`
  作用：联合微调轮数上限。
- `early_stopping_patience: 8 ~ 25`
  作用：提前停止容忍轮数。
- `rbm_learning_rate: 5e-4 ~ 2e-2`
  作用：RBM 预训练学习率。
- `supervised_learning_rate: 5e-4 ~ 2e-2`
  作用：联合微调学习率。
- `alpha: 1e-5 ~ 1e-2`
  作用：正则化强度，越大模型越保守。

额外目标约束：

- `TARGET_MAE_RANGE = (4.2, 5.5)`
- `TARGET_MSE_RANGE = (28.0, 50.0)`

作用：

- BO 不是单纯无脑追最小误差
- 它会更偏向“落在期望区间内”的结果

### 5.5 MGH

运行：

```powershell
python "2_Model/MGH.py"
```

当前 `MGH.py` 已改成终端菜单模式：

1. 完整流程
2. 特征构建与搜索最优簇数
3. 完整建模
4. 进行测试与画图

输出：

- `data/MGH/`
- `image/MGH/`

核心流程说明：

#### 步骤一：特征扩展与最优簇数搜索

关键参数：

- `SEARCH_C_RANGE = range(5, 46)`
  作用：层次聚类时候选簇数从 5 到 45。
- `STAGE1_POP_SIZE = 50`
  作用：步骤一 GA 的种群规模。
- `STAGE1_GENERATIONS = 100`
  作用：步骤一 GA 迭代代数。
- `STAGE1_CLUSTER_COUNT_WEIGHT = 0.08`
  作用：综合评分时，给“较小簇数”一点额外偏好，避免一味选很大的 `C`。

特征工程要点：

- 对所有基础特征增加平方项 `*_Square`
- 对多个顺序特征组生成差分项和绝对差分项
- 当前保留优化相关压下量列：
  - `压下量-E2-1`
  - `压下量-E2-3`
  - `压下量-E2-5`

#### 步骤二：GA-GLR 建模

关键参数：

- `STAGE2_POP_SIZE = 60`
  作用：步骤二 GA 种群规模。
- `STAGE2_GENERATIONS = 60`
  作用：步骤二 GA 代数。
- `STAGE2_MUTATION_RATE = 0.2`
  作用：基因突变概率，越大搜索更活跃。
- `STAGE2_KFOLD = 10`
  作用：10 折交叉验证，评估更稳健。

建模器：

- 当前使用 `LinearRegression`

含义：

- 这里不是复杂非线性模型，而是通过特征扩展 + 特征搜索，把线性回归做强。

#### 步骤三：测试与画图

这一阶段会：

- 载入保存好的 `final_glr_model.pkl`
- 在训练集、验证集、测试集上评估
- 导出 `xlsx/csv/json`
- 生成奇偶图

补充说明：

- `MGH` 会额外保存 `MGH_Test_Expanded.xlsx`
- `PSO.py` 后续就是基于它和 `final_glr_model.pkl` 做优化

## 6. PSO 优化说明

运行：

```powershell
python "3_Optimize/PSO.py"
```

当前 `PSO.py` 也已改成终端菜单模式：

1. 完整流程
2. 提取部分数据

输入依赖：

- `data/MGH/MGH_Test_Expanded.xlsx`
- `data/MGH/final_glr_model.pkl`

输出目录：

- `data/PSO/`

关键参数：

- `OPTIMIZE_SAMPLE_COUNT = 100`
  作用：默认抽取 100 个样本做优化分析。
- `TOP_SAMPLE_COUNT = 20`
  作用：提取前 20 条代表性结果。
- `INERTIA_WEIGHT = 0.50`
  作用：粒子保留原有速度的程度。
- `INDIVIDUAL_FACTOR = 1.35`
  作用：粒子朝自己历史最优位置靠拢的强度。
- `SOCIAL_FACTOR = 1.45`
  作用：粒子朝群体最优位置靠拢的强度。
- `HIGH_ERROR_THRESHOLD = 20.0`
  作用：若基线误差大于 20 mm，就切换到更激进的搜索策略。

搜索强度参数：

- `AGGRESSIVE_NUM_PARTICLES = 36`
- `AGGRESSIVE_MAX_ITERATIONS = 60`
- `CONSERVATIVE_NUM_PARTICLES = 36`
- `CONSERVATIVE_MAX_ITERATIONS = 60`

约束参数：

- `AGGRESSIVE_MAX_ADJUSTMENT_MM = 8.0`
- `CONSERVATIVE_MAX_ADJUSTMENT_MM = 8.0`

作用：

- 限制压下量调整幅度，避免给出过于离谱的建议。

## 7. 输出文件说明

每个模型目录一般会保存这几类文件：

- `*.xlsx`：预测结果、搜索结果、优化结果
- `*.csv`：便于二次分析的数据表
- `*.json`：指标、最优参数、摘要
- `*.png / *.svg`：散点图、收敛图、奇偶图等

常见目录：

- `data/datadeal/`：清洗后数据与切分结果
- `data/LightGBM/`
- `data/LightGBM_BO/`
- `data/DBN/`
- `data/DBN_BO/`
- `data/MGH/`
- `data/PSO/`

## 8. 使用注意事项

- 所有脚本都依赖中文列名，不建议手改表头。
- 当前模型脚本均不应把 `出口宽度设定值-R2-5` 作为训练输入特征。
- 若 Excel 正在占用输出文件，部分脚本会自动另存为 `_new.xlsx` 或时间戳版本。
- 项目路径与文件名包含中文，建议在 Windows + UTF-8 环境下运行。
- 若只想快速跑通一遍，推荐顺序：

```powershell
python "1_Datadeal/Data Deal.py"
python "2_Model/LightGBM.py"
python "2_Model/MGH.py"
python "3_Optimize/PSO.py"
```

## 9. 后续可继续完善的方向

- 增加 `requirements.txt`
- 把 BO/DBN 里仍保留的 `RUN_OPTION` 也改成统一菜单或命令行参数
- 补一份“字段含义说明表”
- 给各模型输出增加统一的实验记录编号
