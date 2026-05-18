import json
import os
import pickle
import random
import sys
import warnings
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullLocator
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import MinMaxScaler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@dataclass
class PredictOnlyLinearModel:
    coef_: np.ndarray
    intercept_: float

    @classmethod
    def from_sklearn(cls, model):
        return cls(
            coef_=np.asarray(model.coef_, dtype=float).copy(),
            intercept_=float(model.intercept_),
        )

    def predict(self, x_input):
        x_array = np.asarray(x_input, dtype=float)
        return x_array @ self.coef_ + self.intercept_


DEFAULT_MENU_OPTION = "1"

GENERATE_PLOTS = True
#画图开关


# =========================
# Global config
# =========================
RANDOM_SEED = 42    
SEARCH_C_RANGE = range(5, 46)   # 超参数 C 的搜索范围
STAGE1_POP_SIZE = 50
STAGE1_GENERATIONS = 100
STAGE1_CLUSTER_COUNT_WEIGHT = 0.08
STAGE2_POP_SIZE = 60
STAGE2_GENERATIONS = 60
STAGE2_MUTATION_RATE = 0.2
STAGE2_KFOLD = 10

TARGET = "实测宽度-R2-5"
SETTING_TARGET = "出口宽度设定值-R2-5"
OPTIMIZATION_COLUMNS = ["压下量-E2-1", "压下量-E2-3", "压下量-E2-5"]

SEQUENCE_GROUPS = {
    "Reduction": [
        "R1压下量Pass1(H11_0)",
        "R2-1压下量",
        "R2-2压下量",
        "R2-3压下量",
        "R2-4压下量",
        "R2-5压下量",
    ],
    "EdgerReduction": [
        "侧压机压下量",
        "压下量-E2-1",
        "压下量-E2-3",
        "压下量-E2-5",
    ],
    "RollingForce": [
        "平辊实际轧制力-R1",
        "平辊实际轧制力-R2-1",
        "平辊实际轧制力-R2-2",
        "平辊实际轧制力-R2-3",
        "平辊实际轧制力-R2-4",
        "平辊实际轧制力-R2-5",
    ],
    "ExitThickness": [
        "出口厚度-R1",
        "出口厚度-R2-1",
        "出口厚度-R2-2",
        "出口厚度-R2-3",
        "出口厚度-R2-4",
    ],
    "ExitTemperature": [
        "道次出口温度-R1",
        "道次出口温度-R2-1",
        "道次出口温度-R2-2",
        "道次出口温度-R2-3",
        "道次出口温度-R2-4",
    ],
}

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DEAL_DIR = os.path.join(PROJECT_ROOT, "data", "datadeal")
DATA_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "MGH")
IMAGE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "image", "MGH")
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DEAL_DIR, "Train_Data.xlsx")
VAL_PATH = os.path.join(DATA_DEAL_DIR, "Val_Data.xlsx")
TEST_PATH = os.path.join(DATA_DEAL_DIR, "Test_Data.xlsx")

EXPANDED_TRAIN_PATH = os.path.join(DATA_OUTPUT_DIR, "MGH_Train_Expanded.xlsx")
EXPANDED_VAL_PATH = os.path.join(DATA_OUTPUT_DIR, "MGH_Val_Expanded.xlsx")
EXPANDED_TEST_PATH = os.path.join(DATA_OUTPUT_DIR, "MGH_Test_Expanded.xlsx")
STAGE1_ARTIFACT_PATH = os.path.join(DATA_OUTPUT_DIR, "mgh_stage1_artifacts.pkl")
STAGE1_RESULT_PATH = os.path.join(DATA_OUTPUT_DIR, "MGH_C_Search_Results.xlsx")
STAGE2_MODEL_PATH = os.path.join(DATA_OUTPUT_DIR, "final_glr_model.pkl")
STAGE3_EXCEL_PATH = os.path.join(DATA_OUTPUT_DIR, "MGH_prediction_summary.xlsx")
STAGE3_CSV_PATH = os.path.join(DATA_OUTPUT_DIR, "result_MGH.csv")
STAGE3_METRICS_PATH = os.path.join(DATA_OUTPUT_DIR, "MGH_metrics.json")
STAGE1_PLOT_PATH = os.path.join(IMAGE_OUTPUT_DIR, "MGH_C_Search_Curve.png")
STAGE1_DENDROGRAM_PATH = os.path.join(IMAGE_OUTPUT_DIR, "MGH_Hierarchical_Dendrogram.png")
STAGE3_PLOT_PATH = os.path.join(IMAGE_OUTPUT_DIR, "MGH_Parity_Plots.png")


def apply_major_ticks_only(axis):
    axis.minorticks_off()
    axis.xaxis.set_minor_locator(NullLocator())
    axis.yaxis.set_minor_locator(NullLocator())
    axis.tick_params(axis="both", which="major", direction="in", length=5, width=1.0)


def to_svg_path(path):
    return os.path.splitext(path)[0] + ".svg"


def save_figure_as_png_and_svg(figure, png_path, dpi=300, bbox_inches=None):
    figure.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches)
    figure.savefig(to_svg_path(png_path), format="svg", bbox_inches=bbox_inches)


def load_scaled_datasets():
    train_df = pd.read_excel(TRAIN_PATH)
    val_df = pd.read_excel(VAL_PATH)
    test_df = pd.read_excel(TEST_PATH)
    return train_df, val_df, test_df


def build_datadeal_scaler(train_df):
    base_features = [col for col in train_df.columns if col not in [TARGET, SETTING_TARGET]]
    scaler = MinMaxScaler()
    scaler.fit(train_df[base_features])
    return scaler, base_features


def scaler_to_payload(scaler, feature_names):
    return {
        "feature_names": list(feature_names),
        "data_min": scaler.data_min_.tolist(),
        "data_max": scaler.data_max_.tolist(),
        "data_range": scaler.data_range_.tolist(),
        "scale": scaler.scale_.tolist(),
        "min": scaler.min_.tolist(),
    }


def expand_mgh_features(dataframe):
    expanded = dataframe.copy()
    base_features = [col for col in dataframe.columns if col not in [TARGET, SETTING_TARGET]]

    for column in base_features:
        expanded[f"{column}_Square"] = expanded[column] ** 2

    for group_name, group_columns in SEQUENCE_GROUPS.items():
        valid_columns = [column for column in group_columns if column in expanded.columns]
        for index in range(1, len(valid_columns)):
            prev_column = valid_columns[index - 1]
            curr_column = valid_columns[index]
            diff_name = f"{group_name}_Diff_{index}"
            abs_diff_name = f"{group_name}_AbsDiff_{index}"
            expanded[diff_name] = expanded[curr_column] - expanded[prev_column]
            expanded[abs_diff_name] = expanded[diff_name].abs()

    ordered_columns = [col for col in expanded.columns if col not in [TARGET, SETTING_TARGET]]
    ordered_columns.extend([SETTING_TARGET, TARGET])
    return expanded[ordered_columns]


def save_expanded_datasets(train_df, val_df, test_df):
    train_df.to_excel(EXPANDED_TRAIN_PATH, index=False)
    val_df.to_excel(EXPANDED_VAL_PATH, index=False)
    test_df.to_excel(EXPANDED_TEST_PATH, index=False)


def load_expanded_datasets():
    if not all(os.path.exists(path) for path in [EXPANDED_TRAIN_PATH, EXPANDED_VAL_PATH, EXPANDED_TEST_PATH]):
        raise FileNotFoundError("缺少 MGH 扩展数据文件，请先运行步骤1。")

    train_df = pd.read_excel(EXPANDED_TRAIN_PATH)
    val_df = pd.read_excel(EXPANDED_VAL_PATH)
    test_df = pd.read_excel(EXPANDED_TEST_PATH)
    return train_df, val_df, test_df


def evaluate_dataset(dataset_name, y_true, y_pred):
    metrics = {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
    }
    print(f"\n{dataset_name} 指标")
    print(f"R2 : {metrics['R2']:.4f}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"MSE: {metrics['MSE']:.4f}")
    return metrics


def build_clusters(feature_names, cluster_labels):
    clusters = []
    for label in sorted(set(cluster_labels)):
        cluster = [feature_names[index] for index, current_label in enumerate(cluster_labels) if current_label == label]
        clusters.append(cluster)
    return clusters


def run_stage1_ga_for_c(cluster_count, feature_clusters, x_train, y_train, x_val, y_val):
    def create_individual():
        return [random.choice(feature_clusters[index]) for index in range(cluster_count)]

    population = [create_individual() for _ in range(STAGE1_POP_SIZE)]
    best_mse = float("inf")
    best_mae = float("inf")
    best_r2 = float("-inf")
    best_features = None

    for _ in range(STAGE1_GENERATIONS):
        fitness_values = []
        next_population = []

        for candidate in population:
            model = LinearRegression()
            model.fit(x_train[candidate], y_train)
            predictions = model.predict(x_val[candidate])

            mse = mean_squared_error(y_val, predictions)
            mae = mean_absolute_error(y_val, predictions)
            r2 = r2_score(y_val, predictions)

            if mse < best_mse:
                best_mse = mse
                best_mae = mae
                best_r2 = r2
                best_features = list(candidate)

            fitness_values.append(1.0 / (mse + 1e-8))

        for _ in range(STAGE1_POP_SIZE):
            first, second = np.random.choice(STAGE1_POP_SIZE, 2, replace=False)
            winner = population[first] if fitness_values[first] > fitness_values[second] else population[second]
            next_population.append(list(winner))

        for index in range(0, STAGE1_POP_SIZE - 1, 2):
            if np.random.rand() < 0.8 and cluster_count > 2:
                left, right = sorted(np.random.choice(range(1, cluster_count), 2, replace=False))
                child1 = (
                    next_population[index][:left]
                    + next_population[index + 1][left:right]
                    + next_population[index][right:]
                )
                child2 = (
                    next_population[index + 1][:left]
                    + next_population[index][left:right]
                    + next_population[index + 1][right:]
                )
                next_population[index], next_population[index + 1] = child1, child2

        for index in range(STAGE1_POP_SIZE):
            if np.random.rand() < 0.2:
                mutation_index = np.random.randint(0, cluster_count)
                next_population[index][mutation_index] = random.choice(feature_clusters[mutation_index])

        population = next_population

    return best_mse, best_mae, best_r2, best_features


def plot_stage1_curve(results_df, optimal_c):
    filtered_df = results_df.loc[results_df["C"] != 27].copy()
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        filtered_df["C"],
        filtered_df["Composite_Score"],
        marker="o",
        color="#8c564b",
        linewidth=2,
        markersize=5,
        alpha=0.9,
    )
    axis.axvline(x=optimal_c, color="red", linestyle="--", linewidth=2, label=f"最佳簇数 C={optimal_c}")
    optimal_row = results_df.loc[results_df["C"] == optimal_c].iloc[0]
    axis.plot(
        optimal_c,
        optimal_row["Composite_Score"],
        marker="*",
        color="gold",
        markersize=18,
        markeredgecolor="black",
        zorder=5,
    )
    axis.set_title("MGH 簇数搜索综合评分曲线", fontsize=15, fontweight="bold")
    axis.set_xlabel("簇数 C", fontsize=12)
    axis.set_ylabel("Composite Score", fontsize=12)
    axis.grid(True, linestyle="--", alpha=0.5)
    apply_major_ticks_only(axis)
    axis.legend()
    figure.tight_layout()
    save_figure_as_png_and_svg(figure, STAGE1_PLOT_PATH, dpi=300)
    plt.close(figure)


def get_cut_height_for_cluster_count(linkage_matrix, cluster_count):
    sample_count = linkage_matrix.shape[0] + 1
    if cluster_count <= 1:
        return float(linkage_matrix[-1, 2])
    if cluster_count >= sample_count:
        return float(linkage_matrix[0, 2]) * 0.5

    lower_height = float(linkage_matrix[sample_count - cluster_count - 1, 2])
    upper_height = float(linkage_matrix[sample_count - cluster_count, 2])
    return (lower_height + upper_height) / 2.0


def plot_stage1_dendrogram(linkage_matrix, feature_names, optimal_c):
    cut_height = get_cut_height_for_cluster_count(linkage_matrix, optimal_c)
    figure, axis = plt.subplots(figsize=(18, 8))
    dendrogram(
        linkage_matrix,
        labels=feature_names,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=cut_height,
        ax=axis,
    )
    axis.axhline(
        y=cut_height,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label=f"Cut line (C={optimal_c})",
    )
    axis.set_title("MGH Hierarchical Clustering Dendrogram", fontsize=15, fontweight="bold")
    axis.set_xlabel("Features", fontsize=12)
    axis.set_ylabel("Distance", fontsize=12)
    apply_major_ticks_only(axis)
    axis.legend()
    figure.tight_layout()
    save_figure_as_png_and_svg(figure, STAGE1_DENDROGRAM_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def run_stage1():
    print("开始步骤1：特征构建与最优簇数搜索")
    train_df, val_df, test_df = load_scaled_datasets()
    train_expanded = expand_mgh_features(train_df)
    val_expanded = expand_mgh_features(val_df)
    test_expanded = expand_mgh_features(test_df)
    save_expanded_datasets(train_expanded, val_expanded, test_expanded)

    raw_features = [col for col in train_expanded.columns if col not in [TARGET, SETTING_TARGET]]
    zero_std_columns = train_expanded[raw_features].columns[train_expanded[raw_features].std() == 0].tolist()
    candidate_features = [feature for feature in raw_features if feature not in zero_std_columns]

    x_train = train_expanded[candidate_features]
    x_val = val_expanded[candidate_features]
    y_train = train_expanded[TARGET].to_numpy()
    y_val = val_expanded[TARGET].to_numpy()

    correlation_matrix = x_train.corr().abs().fillna(0.0).to_numpy()
    distance_matrix = 1.0 - correlation_matrix
    np.fill_diagonal(distance_matrix, 0.0)
    distance_matrix = np.clip((distance_matrix + distance_matrix.T) / 2.0, 0.0, 1.0)
    linkage_matrix = linkage(squareform(distance_matrix), method="ward")

    results = []
    print(f"开始在 C={SEARCH_C_RANGE.start} 到 C={SEARCH_C_RANGE.stop - 1} 之间搜索最优簇数...")

    for cluster_count in SEARCH_C_RANGE:
        cluster_labels = fcluster(linkage_matrix, t=cluster_count, criterion="maxclust")
        feature_clusters = {
            index: [candidate_features[pos] for pos, label in enumerate(cluster_labels) if label == index + 1]
            for index in range(cluster_count)
        }

        mse_value, mae_value, r2_value, best_features = run_stage1_ga_for_c(
            cluster_count=cluster_count,
            feature_clusters=feature_clusters,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
        )

        results.append(
            {
                "C": cluster_count,
                "MSE": mse_value,
                "MAE": mae_value,
                "R2": r2_value,
                "Features": " | ".join(best_features or []),
            }
        )
        print(f"C={cluster_count:02d} -> R2={r2_value:.4f} | MAE={mae_value:.4f} | MSE={mse_value:.4f}")

    results_df = pd.DataFrame(results)
    results_df["MSE_norm"] = (results_df["MSE"] - results_df["MSE"].min()) / (
        results_df["MSE"].max() - results_df["MSE"].min() + 1e-8
    )
    results_df["MAE_norm"] = (results_df["MAE"] - results_df["MAE"].min()) / (
        results_df["MAE"].max() - results_df["MAE"].min() + 1e-8
    )
    results_df["R2_norm"] = (results_df["R2"].max() - results_df["R2"]) / (
        results_df["R2"].max() - results_df["R2"].min() + 1e-8
    )
    results_df["C_norm"] = (results_df["C"] - results_df["C"].min()) / (
        results_df["C"].max() - results_df["C"].min() + 1e-8
    )
    metric_score = 0.4 * results_df["MAE_norm"] + 0.4 * results_df["MSE_norm"] + 0.2 * results_df["R2_norm"]
    results_df["Composite_Score"] = (
        (1.0 - STAGE1_CLUSTER_COUNT_WEIGHT) * metric_score
        + STAGE1_CLUSTER_COUNT_WEIGHT * results_df["C_norm"]
    )

    best_row = results_df.loc[results_df["Composite_Score"].idxmin()]
    optimal_c = int(best_row["C"])
    print(f"\n步骤1完成，最优簇数 C = {optimal_c}")

    results_df.to_excel(STAGE1_RESULT_PATH, index=False)

    scaler, scaler_features = build_datadeal_scaler(train_df)
    artifact = {
        "target": TARGET,
        "setting_target": SETTING_TARGET,
        "optimization_columns": OPTIMIZATION_COLUMNS,
        "sequence_groups": SEQUENCE_GROUPS,
        "candidate_features": candidate_features,
        "zero_std_columns": zero_std_columns,
        "optimal_c": optimal_c,
        "linkage_matrix": linkage_matrix,
        "distance_matrix": distance_matrix,
        "search_results": results_df.to_dict(orient="records"),
        "datadeal_scaler": scaler,
        "datadeal_scaler_payload": scaler_to_payload(scaler, scaler_features),
    }
    with open(STAGE1_ARTIFACT_PATH, "wb") as file_obj:
        pickle.dump(artifact, file_obj)

    if GENERATE_PLOTS:
        plot_stage1_curve(results_df, optimal_c)
        plot_stage1_dendrogram(linkage_matrix, candidate_features, optimal_c)
        print(f"已生成步骤1图像：{STAGE1_PLOT_PATH}")

    print(f"已保存步骤1产物：{STAGE1_ARTIFACT_PATH}")


def load_stage1_artifact():
    if not os.path.exists(STAGE1_ARTIFACT_PATH):
        raise FileNotFoundError("缺少步骤1结果，请先运行步骤1。")

    with open(STAGE1_ARTIFACT_PATH, "rb") as file_obj:
        artifact = pickle.load(file_obj)
    return artifact


def resolve_forced_features(feature_pool):
    forced = [column for column in OPTIMIZATION_COLUMNS if column in feature_pool]
    return forced


def create_stage2_individual(clusters):
    return [random.randint(0, len(cluster) - 1) for cluster in clusters]


def calculate_stage2_fitness(individual, clusters, train_df, y_train, forced_features, kfold):
    ga_features = [clusters[index][gene] for index, gene in enumerate(individual)]
    selected_features = []
    for feature in ga_features:
        if feature not in selected_features and feature not in forced_features:
            selected_features.append(feature)
    selected_features.extend([feature for feature in forced_features if feature not in selected_features])

    x_subset = train_df[selected_features].to_numpy()
    scoring = {
        "mse": "neg_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    model = LinearRegression()
    scores = cross_validate(model, x_subset, y_train, cv=kfold, scoring=scoring)

    mean_mse = float(np.mean(np.abs(scores["test_mse"])))
    mean_mae = float(np.mean(np.abs(scores["test_mae"])))
    mean_r2 = float(np.mean(scores["test_r2"]))
    fitness = 1.0 / (mean_mse + 1e-8)
    return fitness, mean_mse, mean_mae, mean_r2, selected_features


def run_stage2():
    print("开始步骤2：GA-GLR 建模")
    artifact = load_stage1_artifact()
    train_df, _, _ = load_expanded_datasets()

    candidate_features = artifact["candidate_features"]
    optimal_c = artifact["optimal_c"]
    linkage_matrix = artifact["linkage_matrix"]
    forced_features = resolve_forced_features(candidate_features)

    cluster_labels = fcluster(linkage_matrix, t=optimal_c, criterion="maxclust")
    clusters = build_clusters(candidate_features, cluster_labels)

    y_train = train_df[TARGET].to_numpy()
    kfold = KFold(n_splits=STAGE2_KFOLD, shuffle=True, random_state=RANDOM_SEED)

    population = [create_stage2_individual(clusters) for _ in range(STAGE2_POP_SIZE)]
    global_best = {
        "mse": float("inf"),
        "mae": float("inf"),
        "r2": float("-inf"),
        "features": [],
    }

    for generation in range(STAGE2_GENERATIONS):
        fitness_results = []
        for individual in population:
            fitness_value, mse_value, mae_value, r2_value, selected_features = calculate_stage2_fitness(
                individual=individual,
                clusters=clusters,
                train_df=train_df,
                y_train=y_train,
                forced_features=forced_features,
                kfold=kfold,
            )
            fitness_results.append((individual, fitness_value, mse_value, mae_value, r2_value, selected_features))

        fitness_results.sort(key=lambda item: item[1], reverse=True)
        _, _, current_mse, current_mae, current_r2, current_features = fitness_results[0]

        if current_mse < global_best["mse"]:
            global_best["mse"] = current_mse
            global_best["mae"] = current_mae
            global_best["r2"] = current_r2
            global_best["features"] = current_features

        print(
            f"Gen {generation + 1:02d} | "
            f"Best MSE={current_mse:.4f} | MAE={current_mae:.4f} | R2={current_r2:.4f}"
        )

        new_population = [list(item[0]) for item in fitness_results[:5]]
        while len(new_population) < STAGE2_POP_SIZE:
            parent1 = list(random.choice(fitness_results[:20])[0])
            parent2 = list(random.choice(fitness_results[:20])[0])
            crossover_point = random.randint(1, len(clusters) - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]

            if random.random() < STAGE2_MUTATION_RATE:
                mutation_point = random.randint(0, len(clusters) - 1)
                child[mutation_point] = random.randint(0, len(clusters[mutation_point]) - 1)

            new_population.append(child)

        population = new_population

    final_model = LinearRegression()
    final_model.fit(train_df[global_best["features"]].to_numpy(), y_train)
    predict_only_model = PredictOnlyLinearModel.from_sklearn(final_model)

    model_bundle = {
        "model": predict_only_model,
        "selected_features": global_best["features"],
        "target": TARGET,
        "setting_target": SETTING_TARGET,
        "optimization_columns": OPTIMIZATION_COLUMNS,
        "sequence_groups": SEQUENCE_GROUPS,
        "optimal_c": optimal_c,
        "candidate_features": candidate_features,
        "forced_features": forced_features,
        "cluster_labels": cluster_labels.tolist(),
        "clusters": clusters,
        "cv_metrics": {
            "mse": global_best["mse"],
            "mae": global_best["mae"],
            "r2": global_best["r2"],
        },
        "datadeal_scaler": artifact["datadeal_scaler"],
        "datadeal_scaler_payload": artifact["datadeal_scaler_payload"],
    }

    with open(STAGE2_MODEL_PATH, "wb") as file_obj:
        pickle.dump(model_bundle, file_obj)

    print("\n步骤2完成")
    print(f"已保存模型文件：{STAGE2_MODEL_PATH}")
    print("最终入模特征如下：")
    for index, feature in enumerate(global_best["features"], start=1):
        print(f"{index:02d}. {feature}")


def load_stage2_bundle():
    if not os.path.exists(STAGE2_MODEL_PATH):
        raise FileNotFoundError("缺少步骤2模型结果，请先运行步骤2。")

    with open(STAGE2_MODEL_PATH, "rb") as file_obj:
        bundle = pickle.load(file_obj)
    return bundle


def pad_array(values, target_length):
    padded = np.full(target_length, np.nan, dtype=float)
    padded[: len(values)] = values
    return padded


def build_stage3_summary(train_true, train_pred, val_true, val_pred, test_true, test_pred):
    max_length = max(len(train_true), len(val_true), len(test_true))
    return pd.DataFrame(
        {
            "train_true": pad_array(train_true, max_length),
            "train_pred": pad_array(train_pred, max_length),
            "val_true": pad_array(val_true, max_length),
            "val_pred": pad_array(val_pred, max_length),
            "test_true": pad_array(test_true, max_length),
            "test_pred": pad_array(test_pred, max_length),
        }
    )


def plot_stage3_parity(train_true, train_pred, val_true, val_pred, test_true, test_pred):
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
    plot_inputs = [
        ("Train", train_true, train_pred, "#1f77b4"),
        ("Val", val_true, val_pred, "#ff7f0e"),
        ("Test", test_true, test_pred, "#2ca02c"),
    ]

    for axis, (title, y_true, y_pred, color) in zip(axes, plot_inputs):
        axis.scatter(y_true, y_pred, alpha=0.6, color=color, edgecolor="k", s=28)
        lower = min(np.min(y_true), np.min(y_pred))
        upper = max(np.max(y_true), np.max(y_pred))
        axis.plot([lower, upper], [lower, upper], "r--", linewidth=1.5)
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_xlabel("True Width")
        axis.set_ylabel("Predicted Width")
        axis.set_aspect("equal", adjustable="box")
        apply_major_ticks_only(axis)

        r2_value = r2_score(y_true, y_pred)
        mae_value = mean_absolute_error(y_true, y_pred)
        mse_value = mean_squared_error(y_true, y_pred)
        axis.text(
            0.05,
            0.95,
            f"R2={r2_value:.4f}\nMAE={mae_value:.4f}\nMSE={mse_value:.4f}",
            transform=axis.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    figure.tight_layout()
    save_figure_as_png_and_svg(figure, STAGE3_PLOT_PATH, dpi=300)
    plt.close(figure)


def run_stage3():
    print("开始步骤3：训练集/验证集/测试集测试")
    bundle = load_stage2_bundle()
    train_df, val_df, test_df = load_expanded_datasets()

    selected_features = bundle["selected_features"]
    model = bundle["model"]

    x_train = train_df[selected_features].to_numpy()
    x_val = val_df[selected_features].to_numpy()
    x_test = test_df[selected_features].to_numpy()
    y_train = train_df[TARGET].to_numpy()
    y_val = val_df[TARGET].to_numpy()
    y_test = test_df[TARGET].to_numpy()

    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    metrics = {
        "train": evaluate_dataset("Train", y_train, train_pred),
        "val": evaluate_dataset("Val", y_val, val_pred),
        "test": evaluate_dataset("Test", y_test, test_pred),
    }

    summary_df = build_stage3_summary(
        train_true=y_train,
        train_pred=train_pred,
        val_true=y_val,
        val_pred=val_pred,
        test_true=y_test,
        test_pred=test_pred,
    )
    summary_df.to_excel(STAGE3_EXCEL_PATH, index=False)

    csv_df = pd.DataFrame(
        {
            "y_test": y_test,
            "y_pred": test_pred,
            "true_width": y_test,
            "pred_width": test_pred,
            "abs_error": np.abs(y_test - test_pred),
        }
    )
    csv_df.to_csv(STAGE3_CSV_PATH, index=False, encoding="utf-8-sig")

    with open(STAGE3_METRICS_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, ensure_ascii=False, indent=2)

    if GENERATE_PLOTS:
        plot_stage3_parity(y_train, train_pred, y_val, val_pred, y_test, test_pred)
        print(f"已生成步骤3图像：{STAGE3_PLOT_PATH}")

    print("\n步骤3完成")
    print(f"已保存汇总文件：{STAGE3_EXCEL_PATH}")
    print(f"已保存测试结果：{STAGE3_CSV_PATH}")
    print(f"已保存指标文件：{STAGE3_METRICS_PATH}")


def run_plot_only():
    print("开始步骤4：仅生成图像（不重新建模）")

    stage1_plots_generated = False
    if os.path.exists(STAGE1_ARTIFACT_PATH) and os.path.exists(STAGE1_RESULT_PATH):
        artifact = load_stage1_artifact()
        results_df = pd.read_excel(STAGE1_RESULT_PATH)
        if "Composite_Score" not in results_df.columns:
            results_df["MSE_norm"] = (results_df["MSE"] - results_df["MSE"].min()) / (
                results_df["MSE"].max() - results_df["MSE"].min() + 1e-8
            )
            results_df["MAE_norm"] = (results_df["MAE"] - results_df["MAE"].min()) / (
                results_df["MAE"].max() - results_df["MAE"].min() + 1e-8
            )
            results_df["R2_norm"] = (results_df["R2"].max() - results_df["R2"]) / (
                results_df["R2"].max() - results_df["R2"].min() + 1e-8
            )
            results_df["C_norm"] = (results_df["C"] - results_df["C"].min()) / (
                results_df["C"].max() - results_df["C"].min() + 1e-8
            )
            metric_score = (
                0.4 * results_df["MAE_norm"] + 0.4 * results_df["MSE_norm"] + 0.2 * results_df["R2_norm"]
            )
            results_df["Composite_Score"] = (
                (1.0 - STAGE1_CLUSTER_COUNT_WEIGHT) * metric_score
                + STAGE1_CLUSTER_COUNT_WEIGHT * results_df["C_norm"]
            )

        plot_stage1_curve(results_df, artifact["optimal_c"])
        plot_stage1_dendrogram(artifact["linkage_matrix"], artifact["candidate_features"], artifact["optimal_c"])
        print(f"已生成步骤1图像：{STAGE1_PLOT_PATH}")
        print(f"已生成层次聚类切树图：{STAGE1_DENDROGRAM_PATH}")
        stage1_plots_generated = True

    if not stage1_plots_generated:
        print("未找到步骤1所需产物，跳过簇数搜索曲线和切树图生成。")

    stage3_plot_generated = False
    if os.path.exists(STAGE2_MODEL_PATH) and all(
        os.path.exists(path) for path in [EXPANDED_TRAIN_PATH, EXPANDED_VAL_PATH, EXPANDED_TEST_PATH]
    ):
        bundle = load_stage2_bundle()
        train_df, val_df, test_df = load_expanded_datasets()
        selected_features = bundle["selected_features"]
        model = bundle["model"]

        x_train = train_df[selected_features].to_numpy()
        x_val = val_df[selected_features].to_numpy()
        x_test = test_df[selected_features].to_numpy()
        y_train = train_df[TARGET].to_numpy()
        y_val = val_df[TARGET].to_numpy()
        y_test = test_df[TARGET].to_numpy()

        train_pred = model.predict(x_train)
        val_pred = model.predict(x_val)
        test_pred = model.predict(x_test)

        plot_stage3_parity(y_train, train_pred, y_val, val_pred, y_test, test_pred)
        print(f"已生成步骤3图像：{STAGE3_PLOT_PATH}")
        stage3_plot_generated = True

    if not stage3_plot_generated:
        print("未找到步骤3所需产物，跳过预测对比图生成。")

    print("\n步骤4完成")


def run_full_pipeline():
    print("开始步骤5：完整流程（步骤1 + 步骤2 + 步骤3）")
    run_stage1()
    run_stage2()
    run_stage3()
    print("\n步骤5完成")


def prompt_menu_option():
    print("请选择 MGH 运行模式：")
    print("1. 完整流程")
    print("2. 特征构建与搜索最优簇数")
    print("3. 完整建模")
    print("4. 进行测试与画图")

    if not sys.stdin or not sys.stdin.isatty():
        print(f"未检测到交互式终端，自动使用默认选项 {DEFAULT_MENU_OPTION}。")
        return DEFAULT_MENU_OPTION

    while True:
        choice = input("请输入编号 (1-4): ").strip()
        if choice in {"1", "2", "3", "4"}:
            return choice
        print("输入无效，请输入 1、2、3 或 4。")


def main():
    menu_option = prompt_menu_option()
    if menu_option == "1":
        run_full_pipeline()
    elif menu_option == "2":
        run_stage1()
    elif menu_option == "3":
        run_stage2()
    elif menu_option == "4":
        run_stage3()
    else:
        raise ValueError("菜单选项只能是 1、2、3、4。")


if __name__ == "__main__":
    main()
