import pandas as pd
import numpy as np
import os
import pickle
import random
from scipy.cluster.hierarchy import fcluster
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate

# 数据加载
mgh_dir = r"F:\asus\Desktop\毕业设计数据\MGH模型"
train_path = os.path.join(mgh_dir, "MGH_Train_Expanded.xlsx")

df_train = pd.read_excel(train_path)
target = '实测宽度-R2-5'
y_train = df_train[target].values

with open(os.path.join(mgh_dir, 'clustering_Z.pkl'), 'rb') as f:
    clustering_data = pickle.load(f)

features = clustering_data['features']
Z = clustering_data['Z_matrix']
# 【注意】后续提取数据时直接用 df_train 会更安全，包含所有列
# X_train_full = df_train[features]

optimal_C = 33

# 切出 33 个簇
cluster_labels = fcluster(Z, t=optimal_C, criterion='maxclust')
dynamic_C = len(set(cluster_labels))

clusters = []
# 获取所有簇编号
unique_labels = sorted(list(set(cluster_labels)))

# 遍历
for label in unique_labels:
    # 把所有被贴上当前编号标签的特征，全部挑出来放进这个家族
    cluster_features = [features[j] for j in range(len(features)) if cluster_labels[j] == label]
    clusters.append(cluster_features)

print(f"\n将特征动态划分为 {dynamic_C} 个簇。")

# 遗传算法
POP_SIZE = 100  # 种群大小
GENERATIONS = 100  # 迭代代数
MUTATION_RATE = 0.2  # 变异概率
KFOLD = 10  # 交叉验证折数

glr_model = LinearRegression()
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)


def calculate_fitness(chromosome):
    """解码并同时计算 MSE, MAE, R²，加入特征保送机制"""
    # 1. 获取 GA 算法原本抽取的特征
    ga_selected_features = [clusters[i][chromosome[i]] for i in range(dynamic_C)]

    # ==========================================
    # 【核心修改点】：强行“保送”三个控制变量进入模型！
    # ==========================================
    forced_vars = ['压下量-E2-1', '压下量-E2-3', '压下量-E2-5']

    selected_features = []
    # 遍历去重：防止 GA 刚好也抽到了它们导致特征名重复报错
    for f in ga_selected_features:
        if f not in forced_vars:
            selected_features.append(f)

    # 把必须要的调控旋钮强行加在特征列表最后
    selected_features.extend(forced_vars)

    # 【核心修改点】：直接从 df_train 提取这批特征的值，防止缺失
    X_subset = df_train[selected_features].values

    # 定义多个评价指标
    scoring = {
        'mse': 'neg_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'r2': 'r2'
    }

    # 核心：使用 cross_validate 一次性计算所有指标
    scores = cross_validate(glr_model, X_subset, y_train, cv=kf, scoring=scoring)

    mean_mse = np.mean(np.abs(scores['test_mse']))
    mean_mae = np.mean(np.abs(scores['test_mae']))
    mean_r2 = np.mean(scores['test_r2'])

    # 依然严格按照论文：以 1/MSE 作为适应度进行进化
    fitness = 1.0 / mean_mse

    # 将携带了“保送特征”的新列表返回
    return fitness, mean_mse, mean_mae, mean_r2, selected_features


def create_individual():
    return [random.randint(0, len(clusters[i]) - 1) for i in range(dynamic_C)]


# 运行遗传算法主循环
population = [create_individual() for _ in range(POP_SIZE)]
global_best_mse = float('inf')
global_best_mae = float('inf')
global_best_r2 = float('-inf')
global_best_features = []

for gen in range(GENERATIONS):
    fitness_results = []
    for ind in population:
        fit_val, mse_val, mae_val, r2_val, sel_feats = calculate_fitness(ind)
        # 把 MAE 和 R² 也存入结果元组
        fitness_results.append((ind, fit_val, mse_val, mae_val, r2_val, sel_feats))

    # 按适应度 (fit_val) 从高到低排序
    fitness_results.sort(key=lambda x: x[1], reverse=True)

    # 提取当代最强的数据
    current_best_mse = fitness_results[0][2]
    current_best_mae = fitness_results[0][3]
    current_best_r2 = fitness_results[0][4]

    # 更新全局最强记录
    if current_best_mse < global_best_mse:
        global_best_mse = current_best_mse
        global_best_mae = current_best_mae
        global_best_r2 = current_best_r2
        global_best_features = fitness_results[0][5]

    print(
        f"   ▶ 第 {gen + 1:02d} 代 | 最佳 MSE: {current_best_mse:.4f} | MAE: {current_best_mae:.4f} | R²: {current_best_r2:.4f}")

    # 精英保留
    new_population = []
    elites = [item[0] for item in fitness_results[:5]]
    new_population.extend(elites)

    while len(new_population) < POP_SIZE:
        parent1 = random.choice(fitness_results[:20])[0]
        parent2 = random.choice(fitness_results[:20])[0]
        cross_point = random.randint(1, dynamic_C - 1)
        child = parent1[:cross_point] + parent2[cross_point:]

        if random.random() < MUTATION_RATE:
            mut_point = random.randint(0, dynamic_C - 1)
            child[mut_point] = random.randint(0, len(clusters[mut_point]) - 1)

        new_population.append(child)

    population = new_population

print(f" 最终选出模型表现：")
print(f"   - MSE: {global_best_mse:.4f}")
print(f"   - MAE: {global_best_mae:.4f}")
print(f"   - R² : {global_best_r2:.4f}")

for i, f in enumerate(global_best_features):
    print(f"   {i + 1}. {f}")

# 重新训练最终模型并保存结果
print("\n正在训练最终模型与预测数据...")

# 1. 【修改点】：直接从 df_train 取出最终获胜的特征数据
X_train_final = df_train[global_best_features].values

# 2. 用这些特征，对所有训练数据进行一次最终拟合
final_model = LinearRegression()
final_model.fit(X_train_final, y_train)

# 3. 预测训练集的宽度
y_train_pred = final_model.predict(X_train_final)

# 4. 创建一个 DataFrame 来对比真实值与预测值，并算出绝对误差
df_results = pd.DataFrame({
    '真实宽度': y_train,
    '预测宽度': y_train_pred,
    '绝对误差(mm)': np.abs(y_train - y_train_pred)
})

# 保存对比表格到 Excel
results_path = os.path.join(mgh_dir, 'GLR_Train_Predictions.xlsx')
df_results.to_excel(results_path, index=False)

# 将训练好的 GLR 模型，以及特征名一起打包保存
model_path = os.path.join(mgh_dir, 'final_glr_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': final_model,
        'selected_features': global_best_features
    }, f)

print(f"预测值与真实值对比表已保存至: {results_path}")
print(f"最终 GLR 模型及特征花名册已保存至: {model_path}")