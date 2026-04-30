import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

import pickle
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import random

#随机数种子
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取扩容特征池
mgh_dir = r"F:\asus\Desktop\毕业设计数据\MGH模型"


df_train = pd.read_excel(os.path.join(mgh_dir, "MGH_Train_Expanded.xlsx"))
df_val = pd.read_excel(os.path.join(mgh_dir, "MGH_Val_Expanded.xlsx"))
df_test = pd.read_excel(os.path.join(mgh_dir, "MGH_Test_Expanded.xlsx"))

target = '实测宽度-R2-5'
setting_target = '出口宽度设定值-R2-5'  # 【新增】定义设定值列名

# 【关键修改】：同时把 实测宽度 和 设定值 踢出特征池，绝对不能让设定值参与聚类和训练！
raw_features = [col for col in df_train.columns if col not in [target, setting_target]]


# 剔除方差为 0 的特征
X_train_raw = df_train[raw_features]
std_zero_cols = X_train_raw.columns[X_train_raw.std() == 0].tolist()
features = [f for f in raw_features if f not in std_zero_cols]

if len(std_zero_cols) > 0:
    print(f"自动剔除了 {len(std_zero_cols)} 个方差为 0 的无效差分特征。")

print(f"当前有效全局特征维度: {len(features)} 维！")

X_train, y_train = df_train[features], df_train[target].values
X_val, y_val = df_val[features], df_val[target].values

# 标准化处理
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=features)


# 层次聚类

# 计算特征之间的绝对相关系数矩阵
corr_matrix = X_train_scaled.corr().abs().values
# 转换为距离矩阵 (距离 = 1 - 相关系数)
dist_matrix = 1.0 - corr_matrix
# 强制对角线为 0 并对称化，防止浮点数精度报错
np.fill_diagonal(dist_matrix, 0)
dist_matrix = np.clip((dist_matrix + dist_matrix.T) / 2, 0, 1)

# 将冗余方阵转换为一维压缩距离向量，并使用 ward 最小方差法构建族谱树
condensed_dist = squareform(dist_matrix)
Z = linkage(condensed_dist, method='ward')



#定义 GA 寻优引擎
def run_ga_for_C(C, feature_clusters, X_tr, y_tr, X_v, y_v):
    pop_size = 50  # 种群数
    max_gen = 100   #                                                                                      代数

    def create_individual():
        return [np.random.choice(feature_clusters[i]) for i in range(C)]

    population = [create_individual() for _ in range(pop_size)]

    # 记录该 C 下的全局最优个体
    best_individual = None
    best_mse = float('inf')
    best_mae = float('inf')
    best_r2 = float('-inf')

    for gen in range(max_gen):
        fitness_list = []
        pop_metrics = []

        for ind in population:
            lr = LinearRegression()
            lr.fit(X_tr[ind], y_tr)
            preds = lr.predict(X_v[ind])

            mse = mean_squared_error(y_v, preds)
            mae = mean_absolute_error(y_v, preds)
            r2 = r2_score(y_v, preds)

            # 以 1/MSE 作为纯进化的适应度指导
            fitness = 1.0 / (mse + 1e-8)
            fitness_list.append(fitness)
            pop_metrics.append((mse, mae, r2, ind))

            # 内部更新该 C 值的历史最佳
            if mse < best_mse:
                best_mse, best_mae, best_r2 = mse, mae, r2
                best_individual = ind.copy()

        # 锦标赛选择
        new_pop = []
        for _ in range(pop_size):
            i1, i2 = np.random.choice(pop_size, 2, replace=False)
            winner = population[i1] if fitness_list[i1] > fitness_list[i2] else population[i2]
            new_pop.append(winner.copy())

        # 两点交叉 (Two-point Crossover)
        for i in range(0, pop_size - 1, 2):
            if np.random.rand() < 0.8 and C > 2:
                pts = sorted(np.random.choice(range(1, C), 2, replace=False))
                n1, n2 = pts[0], pts[1]
                temp1 = new_pop[i][:n1] + new_pop[i + 1][n1:n2] + new_pop[i][n2:]
                temp2 = new_pop[i + 1][:n1] + new_pop[i][n1:n2] + new_pop[i + 1][n2:]
                new_pop[i], new_pop[i + 1] = temp1, temp2

        # 变异
        for i in range(pop_size):
            if np.random.rand() < 0.2:
                mut_idx = np.random.randint(0, C)
                new_pop[i][mut_idx] = np.random.choice(feature_clusters[mut_idx])

        population = new_pop

    return best_mse, best_mae, best_r2, best_individual



# 搜索 C=5~45

C_range = range(5, 46)
results = []

print(f"\n开始在 C=5 到 C=50 之间进行全量网格搜索与 GA 寻优...")

for C in C_range:
    # 进行切树
    cluster_labels = fcluster(Z, t=C, criterion='maxclust')

    # 梳理出每个簇包含的特征
    feature_clusters = {}
    for i in range(1, C + 1):
        feature_clusters[i - 1] = [features[j] for j in range(len(features)) if cluster_labels[j] == i]

    # 启动 GA 寻找代表特征
    mse_val, mae_val, r2_val, best_feats = run_ga_for_C(C, feature_clusters, X_train_scaled, y_train, X_val_scaled,
                                                        y_val)

    results.append({
        'C': C, 'MSE': mse_val, 'MAE': mae_val, 'R2': r2_val, 'Features': best_feats
    })


    print(f"   ▶ 切分为 {C:02d} 簇 | GA 寻优得出 -> R²: {r2_val:.4f} | MAE: {mae_val:.4f} | MSE: {mse_val:.4f}")


# 多目标归一化加权评价
df_res = pd.DataFrame(results)

# 1. 提取极值
max_mse, min_mse = df_res['MSE'].max(), df_res['MSE'].min()
max_mae, min_mae = df_res['MAE'].max(), df_res['MAE'].min()
max_r2, min_r2 = df_res['R2'].max(), df_res['R2'].min()

# 2. 归一化
# MSE 和 MAE 本身就是越低越好
df_res['MSE_norm'] = (df_res['MSE'] - min_mse) / (max_mse - min_mse + 1e-8)
df_res['MAE_norm'] = (df_res['MAE'] - min_mae) / (max_mae - min_mae + 1e-8)
# R² 是越高越好，所以反向归一化 (最大值减去当前值)
df_res['R2_norm'] = (max_r2 - df_res['R2']) / (max_r2 - min_r2 + 1e-8)

# 计算综合得分
df_res['Composite_Score'] = 0.4 * df_res['MAE_norm'] + 0.4 * df_res['MSE_norm'] + 0.2 * df_res['R2_norm']

# 找出得分最低的那一行
best_row = df_res.loc[df_res['Composite_Score'].idxmin()]
optimal_C = int(best_row['C'])
optimal_features = best_row['Features']


print(f"最优聚类簇数为：C = {optimal_C}")
print(f"   - R²  : {best_row['R2']:.4f}")
print(f"   - MAE : {best_row['MAE']:.4f} mm")
print(f"   - MSE : {best_row['MSE']:.4f}")


# 绘图与保存
plt.figure(figsize=(12, 6))
plt.plot(df_res['C'], df_res['Composite_Score'], marker='o', color='#8c564b', linewidth=2, markersize=5, alpha=0.9)
plt.axvline(x=optimal_C, color='r', linestyle='--', linewidth=2, label=f'综合最优切分点 C={optimal_C}')
plt.plot(optimal_C, best_row['Composite_Score'], marker='*', color='gold', markersize=20, markeredgecolor='black',
         zorder=5)

plt.title('MGH: 聚类簇数(C) 对 多目标综合加权得分 的影响曲线', fontsize=15, fontweight='bold')
plt.xlabel('特征聚类簇数 (C)', fontsize=12)
plt.ylabel('综合加权惩罚得分 (Composite Cost Score) - 越低越好', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

plot_path = os.path.join(mgh_dir, "MGH_MultiObj_Score_Curve.png")
plt.savefig(plot_path, dpi=300)
print(f"\n综合评分曲线图已保存至: {plot_path}")



mgh_dir = r"F:\asus\Desktop\毕业设计数据\MGH模型"


# 将特征列表和 Z 矩阵打包成字典
clustering_data = {
    'features': features,  # 确保把 120 维特征的名字也传过去
    'Z_matrix': Z          # 族谱树矩阵
}

# 保存这个完整的字典
with open(os.path.join(mgh_dir, 'clustering_Z.pkl'), 'wb') as f:
    pickle.dump(clustering_data, f)

print("族谱树保存为 clustering_Z.pkl")