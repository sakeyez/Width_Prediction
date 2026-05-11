import os
import pickle
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RANDOM_SEED = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def evaluate_dataset(name, X_data, y_true, model):
    y_pred = model.predict(X_data)

    metrics = {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
    }

    print(f"\n{name}指标：")
    print(f"R方: {metrics['R2']:.4f}")
    print(f"MAE : {metrics['MAE']:.4f}")
    print(f"MSE : {metrics['MSE']:.4f}")

    return y_pred, metrics


def pad_array(values, target_length):
    padded = np.full(target_length, np.nan, dtype=float)
    padded[: len(values)] = values
    return padded


def build_excel_result(y_train, y_val, y_test, train_pred, val_pred, test_pred):
    max_length = max(len(train_pred), len(val_pred), len(test_pred))

    return pd.DataFrame(
        {
            "实验集预测宽度": pad_array(val_pred, max_length),
            "实验集实测宽度": pad_array(y_val, max_length),
            "训练集预测宽度": pad_array(train_pred, max_length),
            "训练集实测宽度": pad_array(y_train, max_length),
            "测试集预测宽度": pad_array(test_pred, max_length),
            "测试集实测宽度": pad_array(y_test, max_length),
        }
    )


def save_excel_with_fallback(dataframe, preferred_path):
    try:
        dataframe.to_excel(preferred_path, index=False)
        return preferred_path
    except PermissionError:
        fallback_path = os.path.join(
            os.path.dirname(__file__), os.path.basename(preferred_path).replace(".xlsx", "_new.xlsx")
        )
        dataframe.to_excel(fallback_path, index=False)
        print(f"\n检测到 {preferred_path} 正在被占用，已改存为: {fallback_path}")
        return fallback_path


def save_csv_with_fallback(dataframe, preferred_path):
    try:
        dataframe.to_csv(preferred_path, index=False, encoding="utf-8-sig")
        return preferred_path
    except PermissionError:
        fallback_path = os.path.join(
            os.path.dirname(__file__), os.path.basename(preferred_path).replace(".csv", "_new.csv")
        )
        dataframe.to_csv(fallback_path, index=False, encoding="utf-8-sig")
        print(f"检测到 {preferred_path} 正在被占用，已改存为: {fallback_path}")
        return fallback_path


def save_pickle_with_fallback(obj, preferred_path):
    try:
        with open(preferred_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        return preferred_path
    except PermissionError:
        fallback_path = os.path.join(
            os.path.dirname(__file__), os.path.basename(preferred_path).replace(".pkl", "_new.pkl")
        )
        with open(fallback_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        print(f"检测到 {preferred_path} 正在被占用，已改存为: {fallback_path}")
        return fallback_path


def resolve_forced_vars(dataframe):
    candidate_groups = [
        ["压下量-E2-1", "压下量-E2-3", "压下量-E2-5"],
        ["压下量-R2-1", "压下量-R2-3", "压下量-R2-5"],
        ["R2-1压下量", "R2-3压下量", "R2-5压下量"],
    ]

    for candidates in candidate_groups:
        if all(column in dataframe.columns for column in candidates):
            return candidates

    fallback_keywords = ["2-1", "2-3", "2-5"]
    matched_columns = []
    for keyword in fallback_keywords:
        column = next(
            (
                col
                for col in dataframe.columns
                if "压下量" in col and keyword in col
            ),
            None,
        )
        if column is None:
            raise KeyError(f"未找到与压下量 {keyword} 对应的特征列。")
        matched_columns.append(column)

    return matched_columns


mgh_dir = r"F:\asus\Desktop\毕业设计数据\MGH模型"
train_path = os.path.join(mgh_dir, "MGH_Train_Expanded.xlsx")
val_path = os.path.join(mgh_dir, "MGH_Val_Expanded.xlsx")
test_path = os.path.join(mgh_dir, "MGH_Test_Expanded.xlsx")

print("正在加载 MGH 训练、验证、测试数据...")
df_train = pd.read_excel(train_path)
df_val = pd.read_excel(val_path)
df_test = pd.read_excel(test_path)

target = "实测宽度-R2-5"
y_train = df_train[target].values
y_val = df_val[target].values
y_test = df_test[target].values
forced_vars = resolve_forced_vars(df_train)

print("强制保送压下量特征：")
for column in forced_vars:
    print(f"- {column}")

with open(os.path.join(mgh_dir, "clustering_Z.pkl"), "rb") as f:
    clustering_data = pickle.load(f)

features = clustering_data["features"]
Z = clustering_data["Z_matrix"]
optimal_C = 33

cluster_labels = fcluster(Z, t=optimal_C, criterion="maxclust")
dynamic_C = len(set(cluster_labels))
unique_labels = sorted(set(cluster_labels))

clusters = []
for label in unique_labels:
    cluster_features = [features[j] for j in range(len(features)) if cluster_labels[j] == label]
    clusters.append(cluster_features)

print(f"\n将特征动态划分为 {dynamic_C} 个簇。")

POP_SIZE = 100
GENERATIONS = 100
MUTATION_RATE = 0.2
KFOLD = 10

glr_model = LinearRegression()
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_SEED)


def calculate_fitness(chromosome):
    ga_selected_features = [clusters[i][chromosome[i]] for i in range(dynamic_C)]

    selected_features = []
    for feature in ga_selected_features:
        if feature not in forced_vars:
            selected_features.append(feature)
    selected_features.extend(forced_vars)

    X_subset = df_train[selected_features].values

    scoring = {
        "mse": "neg_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    scores = cross_validate(glr_model, X_subset, y_train, cv=kf, scoring=scoring)

    mean_mse = np.mean(np.abs(scores["test_mse"]))
    mean_mae = np.mean(np.abs(scores["test_mae"]))
    mean_r2 = np.mean(scores["test_r2"])
    fitness = 1.0 / mean_mse

    return fitness, mean_mse, mean_mae, mean_r2, selected_features


def create_individual():
    return [random.randint(0, len(clusters[i]) - 1) for i in range(dynamic_C)]


population = [create_individual() for _ in range(POP_SIZE)]
global_best_mse = float("inf")
global_best_mae = float("inf")
global_best_r2 = float("-inf")
global_best_features = []

for gen in range(GENERATIONS):
    fitness_results = []
    for individual in population:
        fit_val, mse_val, mae_val, r2_val, selected_features = calculate_fitness(individual)
        fitness_results.append(
            (individual, fit_val, mse_val, mae_val, r2_val, selected_features)
        )

    fitness_results.sort(key=lambda item: item[1], reverse=True)

    current_best_mse = fitness_results[0][2]
    current_best_mae = fitness_results[0][3]
    current_best_r2 = fitness_results[0][4]

    if current_best_mse < global_best_mse:
        global_best_mse = current_best_mse
        global_best_mae = current_best_mae
        global_best_r2 = current_best_r2
        global_best_features = fitness_results[0][5]

    print(
        f"   第 {gen + 1:02d} 代 | 最佳 MSE: {current_best_mse:.4f} | "
        f"MAE: {current_best_mae:.4f} | R方: {current_best_r2:.4f}"
    )

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

print("\n最终选出模型表现：")
print(f"MSE: {global_best_mse:.4f}")
print(f"MAE: {global_best_mae:.4f}")
print(f"R方: {global_best_r2:.4f}")

for index, feature in enumerate(global_best_features, start=1):
    print(f"{index}. {feature}")

print("\n正在训练最终模型并进行三数据集预测...")
X_train_final = df_train[global_best_features].values
X_val_final = df_val[global_best_features].values
X_test_final = df_test[global_best_features].values

final_model = LinearRegression()
final_model.fit(X_train_final, y_train)

train_pred, train_metrics = evaluate_dataset("实验集", X_train_final, y_train, final_model)
val_pred, val_metrics = evaluate_dataset("验证集", X_val_final, y_val, final_model)
test_pred, test_metrics = evaluate_dataset("测试集", X_test_final, y_test, final_model)

result_excel = build_excel_result(
    y_train=y_train,
    y_val=y_val,
    y_test=y_test,
    train_pred=train_pred,
    val_pred=val_pred,
    test_pred=test_pred,
)
results_path = os.path.join(mgh_dir, "MGH_预测结果汇总.xlsx")
results_path = save_excel_with_fallback(result_excel, results_path)

df_test_result = pd.DataFrame({"真实宽度_True": y_test, "预测宽度_MGH": test_pred})
csv_result_path = save_csv_with_fallback(
    df_test_result, os.path.join(mgh_dir, "result_MGH.csv")
)

model_path = os.path.join(mgh_dir, "final_glr_model.pkl")
model_path = save_pickle_with_fallback(
    {"model": final_model, "selected_features": global_best_features},
    model_path,
)

print("\n结果文件已保存：")
print(f"1. {results_path}")
print(f"2. {csv_result_path}")
print(f"3. {model_path}")
0