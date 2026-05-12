import json
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 用户运行开关
# =========================
RUN_OPTION = 2
# 1：只跑步骤1，MI 特征排序与维度评估
# 2：只跑步骤2，评估不同 K 并保存每个专家家族选定的 K
# 3：只跑步骤3，训练专家模型库
# 4：只跑步骤4，PSO 融合并在 Test 集评估
# 5：按 1 -> 4 顺序全部运行


# =========================
# 全局参数配置
# =========================
GENERATE_PLOTS = True
RANDOM_SEED = 42
TARGET = "实测宽度-R2-5"
MODEL_FAMILIES = ["SVM", "GPR", "ANN", "RF"]
FEATURE_DIM_RANGE_START = 6
CLUSTER_K_RANGE = range(1, 7)

# 步骤1特征数按你的要求固定为 37，不再自动搜索最终 N
SELECTED_FEATURE_COUNT = 37
# 如果这里给出具体值，就直接使用；如果设为 None，就根据验证集结果自动选择
SELECTED_BEST_KS = None

# PSO 参数
PSO_NUM_PARTICLES = 30
PSO_MAX_ITER = 100
PSO_W = 0.8
PSO_C1 = 1.5
PSO_C2 = 1.5
KMEANS_N_INIT = 10
MIN_CLUSTER_TRAIN_SAMPLES = 5
GPR_MAX_TRAIN_SAMPLES = 1200


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DEAL_DIR = os.path.join(PROJECT_ROOT, "data", "datadeal")
DATA_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "Hybrid-2")
IMAGE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "image", "Hybrid-2")
MODELS_DIR = os.path.join(DATA_OUTPUT_DIR, "Trained_Expert_Models")

TRAIN_PATH = os.path.join(DATA_DEAL_DIR, "Train_Data.xlsx")
VAL_PATH = os.path.join(DATA_DEAL_DIR, "Val_Data.xlsx")
TEST_PATH = os.path.join(DATA_DEAL_DIR, "Test_Data.xlsx")

STAGE1_TRAIN_PATH = os.path.join(DATA_OUTPUT_DIR, "Hybrid2_Train.xlsx")
STAGE1_VAL_PATH = os.path.join(DATA_OUTPUT_DIR, "Hybrid2_Val.xlsx")
STAGE1_TEST_PATH = os.path.join(DATA_OUTPUT_DIR, "Hybrid2_Test.xlsx")
MI_RESULTS_PATH = os.path.join(DATA_OUTPUT_DIR, "mi_feature_scores.xlsx")
DIMENSION_SCAN_PATH = os.path.join(DATA_OUTPUT_DIR, "dimension_scan_results.xlsx")
SELECTED_FEATURE_INFO_PATH = os.path.join(DATA_OUTPUT_DIR, "selected_features.json")
K_SCAN_RESULTS_PATH = os.path.join(DATA_OUTPUT_DIR, "k_scan_results.xlsx")
SELECTED_BEST_KS_PATH = os.path.join(DATA_OUTPUT_DIR, "selected_best_ks.pkl")
STAGE4_EXCEL_PATH = os.path.join(DATA_OUTPUT_DIR, "Hybrid2_prediction_summary.xlsx")
STAGE4_CSV_PATH = os.path.join(DATA_OUTPUT_DIR, "result_Hybrid2.csv")
STAGE4_METRICS_PATH = os.path.join(DATA_OUTPUT_DIR, "Hybrid2_metrics.json")
STAGE4_PLOT_PATH = os.path.join(IMAGE_OUTPUT_DIR, "Hybrid2_Parity_Plots.png")

os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)


def drop_auxiliary_columns(dataframe):
    removable_columns = [TARGET, "Cluster_Label", "Cluster"]
    return [column for column in dataframe.columns if column not in removable_columns]


def load_raw_datasets():
    train_df = pd.read_excel(TRAIN_PATH)
    val_df = pd.read_excel(VAL_PATH)
    test_df = pd.read_excel(TEST_PATH)
    return train_df, val_df, test_df


def load_stage1_datasets():
    if not all(os.path.exists(path) for path in [STAGE1_TRAIN_PATH, STAGE1_VAL_PATH, STAGE1_TEST_PATH]):
        raise FileNotFoundError("缺少步骤1输出数据，请先运行步骤1。")

    train_df = pd.read_excel(STAGE1_TRAIN_PATH)
    val_df = pd.read_excel(STAGE1_VAL_PATH)
    test_df = pd.read_excel(STAGE1_TEST_PATH)
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


def pad_array(values, target_length):
    padded = np.full(target_length, np.nan, dtype=float)
    padded[: len(values)] = values
    return padded


def build_prediction_summary(train_true, train_pred, val_true, val_pred, test_true, test_pred):
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


def plot_parity(train_true, train_pred, val_true, val_pred, test_true, test_pred):
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

    plt.tight_layout()
    plt.savefig(STAGE4_PLOT_PATH, dpi=300)
    plt.close()


def get_regressor(model_name):
    if model_name == "SVM":
        return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10.0, gamma="scale"))

    if model_name == "GPR":
        kernel = C(1.0) * RBF(1.0)
        return make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=0,
                normalize_y=True,
                random_state=RANDOM_SEED,
            ),
        )

    if model_name == "ANN":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=3000,
                learning_rate_init=0.01,
                early_stopping=True,
                random_state=RANDOM_SEED,
            ),
        )

    if model_name == "RF":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )

    raise ValueError(f"不支持的模型类型: {model_name}")


def fit_regressor(model_name, x_data, y_data):
    train_x = x_data
    train_y = y_data

    if model_name == "GPR" and len(train_y) > GPR_MAX_TRAIN_SAMPLES:
        rng = np.random.RandomState(RANDOM_SEED)
        sampled_index = rng.choice(len(train_y), size=GPR_MAX_TRAIN_SAMPLES, replace=False)
        if hasattr(train_x, "iloc"):
            train_x = train_x.iloc[sampled_index]
        else:
            train_x = train_x[sampled_index]
        train_y = train_y[sampled_index]

    model = get_regressor(model_name)
    model.fit(train_x, train_y)
    return model


def select_best_dimension(dimension_df):
    if SELECTED_FEATURE_COUNT is not None:
        return int(SELECTED_FEATURE_COUNT)

    best_row = dimension_df.sort_values(
        by=["mse", "mae", "r2", "n"],
        ascending=[True, True, False, True],
    ).iloc[0]
    return int(best_row["n"])


def select_best_ks(k_scan_df):
    if SELECTED_BEST_KS is not None:
        return {key: int(value) for key, value in SELECTED_BEST_KS.items()}

    selected = {}
    for model_name in MODEL_FAMILIES:
        best_row = (
            k_scan_df[k_scan_df["model"] == model_name]
            .sort_values(by=["mse", "mae", "r2", "k"], ascending=[True, True, False, True])
            .iloc[0]
        )
        selected[model_name] = int(best_row["k"])
    return selected


def run_stage1():
    print("开始步骤1：MI 特征排序与维度评估")
    train_df, val_df, test_df = load_raw_datasets()

    features = drop_auxiliary_columns(train_df)
    x_train = train_df[features]
    y_train = train_df[TARGET]
    x_val = val_df[features]
    y_val = val_df[TARGET]

    print(f"基础特征数: {len(features)}")
    print("正在计算互信息得分...")
    mi_scores = mutual_info_regression(x_train, y_train, random_state=RANDOM_SEED)
    mi_df = pd.DataFrame({"Feature": features, "MI_Score": mi_scores})
    mi_df = mi_df.sort_values(by="MI_Score", ascending=False).reset_index(drop=True)
    mi_df.to_excel(MI_RESULTS_PATH, index=False)

    print("\n开始维度扫描...")
    dimension_records = []
    for feature_count in range(FEATURE_DIM_RANGE_START, len(features) + 1):
        selected_features = mi_df["Feature"].iloc[:feature_count].tolist()
        model = make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
        )
        model.fit(x_train[selected_features], y_train)
        preds = model.predict(x_val[selected_features])

        dimension_records.append(
            {
                "n": feature_count,
                "r2": float(r2_score(y_val, preds)),
                "mae": float(mean_absolute_error(y_val, preds)),
                "mse": float(mean_squared_error(y_val, preds)),
            }
        )

    dimension_df = pd.DataFrame(dimension_records)
    dimension_df.to_excel(DIMENSION_SCAN_PATH, index=False)
    print(dimension_df.to_string(index=False))

    selected_feature_count = select_best_dimension(dimension_df)
    selected_features = mi_df["Feature"].iloc[:selected_feature_count].tolist()
    columns_to_keep = selected_features + [TARGET]

    train_df[columns_to_keep].to_excel(STAGE1_TRAIN_PATH, index=False)
    val_df[columns_to_keep].to_excel(STAGE1_VAL_PATH, index=False)
    test_df[columns_to_keep].to_excel(STAGE1_TEST_PATH, index=False)

    with open(SELECTED_FEATURE_INFO_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "selected_feature_count": selected_feature_count,
                "selected_features": selected_features,
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n步骤1完成，自动选定特征数 N = {selected_feature_count}。")
    print(f"训练集输出: {STAGE1_TRAIN_PATH}")


def run_stage2():
    print("开始步骤2：评估不同 K 并保存最佳 K")
    train_df, val_df, _ = load_stage1_datasets()

    x_train = train_df.drop(columns=[TARGET])
    x_val = val_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].to_numpy()
    y_val = val_df[TARGET].to_numpy()
    cluster_scaler = StandardScaler()
    x_train_cluster = cluster_scaler.fit_transform(x_train)
    x_val_cluster = cluster_scaler.transform(x_val)

    print(f"当前参与评估的特征数: {x_train.shape[1]}")
    k_scan_records = []

    for model_name in MODEL_FAMILIES:
        print(f"\n--- {model_name} 模型性能评估 ---")
        for cluster_count in CLUSTER_K_RANGE:
            if cluster_count == 1:
                model = fit_regressor(model_name, x_train, y_train)
                preds = model.predict(x_val)
            else:
                kmeans = KMeans(n_clusters=cluster_count, random_state=RANDOM_SEED, n_init=KMEANS_N_INIT)
                train_labels = kmeans.fit_predict(x_train_cluster)
                val_labels = kmeans.predict(x_val_cluster)
                global_model = fit_regressor(model_name, x_train, y_train)
                preds = global_model.predict(x_val)

                for cluster_index in range(cluster_count):
                    train_mask = train_labels == cluster_index
                    val_mask = val_labels == cluster_index
                    if train_mask.sum() > MIN_CLUSTER_TRAIN_SAMPLES:
                        model = fit_regressor(model_name, x_train[train_mask], y_train[train_mask])
                        if val_mask.sum() > 0:
                            preds[val_mask] = model.predict(x_val[val_mask])

            record = {
                "model": model_name,
                "k": cluster_count,
                "r2": float(r2_score(y_val, preds)),
                "mae": float(mean_absolute_error(y_val, preds)),
                "mse": float(mean_squared_error(y_val, preds)),
            }
            k_scan_records.append(record)
            print(
                f"K={cluster_count:<2} | "
                f"R2={record['r2']:.4f} | MAE={record['mae']:.4f} | MSE={record['mse']:.4f}"
            )

    k_scan_df = pd.DataFrame(k_scan_records)
    k_scan_df.to_excel(K_SCAN_RESULTS_PATH, index=False)

    selected_best_ks = select_best_ks(k_scan_df)
    needed_ks = sorted(set(selected_best_ks.values()))
    for cluster_count in needed_ks:
        if cluster_count > 1:
            cluster_scaler = StandardScaler()
            x_train_cluster = cluster_scaler.fit_transform(x_train)
            kmeans = KMeans(n_clusters=cluster_count, random_state=RANDOM_SEED, n_init=KMEANS_N_INIT)
            kmeans.fit(x_train_cluster)
            joblib.dump(
                {"scaler": cluster_scaler, "kmeans": kmeans},
                os.path.join(DATA_OUTPUT_DIR, f"kmeans_k{cluster_count}.pkl"),
            )

    joblib.dump(selected_best_ks, SELECTED_BEST_KS_PATH)
    print(f"\n步骤2完成，自动选定的最佳 K: {selected_best_ks}")


def run_stage3():
    print("开始步骤3：训练专家模型库")
    train_df, _, _ = load_stage1_datasets()
    if not os.path.exists(SELECTED_BEST_KS_PATH):
        raise FileNotFoundError("缺少 selected_best_ks.pkl，请先运行步骤2。")

    selected_best_ks = joblib.load(SELECTED_BEST_KS_PATH)
    x_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].to_numpy()

    for model_name, cluster_count in selected_best_ks.items():
        print(f"\n--- 正在训练 {model_name} 家族 (K={cluster_count}) ---")
        if cluster_count == 1:
            model = fit_regressor(model_name, x_train, y_train)
            joblib.dump(model, os.path.join(MODELS_DIR, f"{model_name}_Global.pkl"))
            print(f"{model_name} 全局模型已保存")
            continue

        kmeans_path = os.path.join(DATA_OUTPUT_DIR, f"kmeans_k{cluster_count}.pkl")
        if not os.path.exists(kmeans_path):
            raise FileNotFoundError(f"缺少聚类模型: {kmeans_path}")

        kmeans_bundle = joblib.load(kmeans_path)
        cluster_scaler = kmeans_bundle["scaler"]
        kmeans = kmeans_bundle["kmeans"]
        labels = kmeans.predict(cluster_scaler.transform(x_train))

        for cluster_index in range(cluster_count):
            cluster_mask = labels == cluster_index
            if cluster_mask.sum() <= MIN_CLUSTER_TRAIN_SAMPLES:
                continue

            model = fit_regressor(model_name, x_train[cluster_mask], y_train[cluster_mask])
            save_path = os.path.join(
                MODELS_DIR,
                f"{model_name}_Cluster_{cluster_index}_of_K{cluster_count}.pkl",
            )
            joblib.dump(model, save_path)
            print(f"{model_name} 专家 {cluster_index} 已保存，样本数 {cluster_mask.sum()}")

    print(f"\n步骤3完成，专家模型目录: {MODELS_DIR}")


def get_expert_predictions(dataframe, selected_best_ks, features):
    preds_matrix = np.zeros((len(dataframe), len(MODEL_FAMILIES)), dtype=float)
    x_data = dataframe[features]

    for model_index, model_name in enumerate(MODEL_FAMILIES):
        cluster_count = selected_best_ks[model_name]
        if cluster_count == 1:
            model = joblib.load(os.path.join(MODELS_DIR, f"{model_name}_Global.pkl"))
            preds_matrix[:, model_index] = model.predict(x_data)
            continue

        kmeans_bundle = joblib.load(os.path.join(DATA_OUTPUT_DIR, f"kmeans_k{cluster_count}.pkl"))
        cluster_scaler = kmeans_bundle["scaler"]
        kmeans = kmeans_bundle["kmeans"]
        labels = kmeans.predict(cluster_scaler.transform(x_data))

        for cluster_index in range(cluster_count):
            data_mask = labels == cluster_index
            if not data_mask.any():
                continue

            model_path = os.path.join(
                MODELS_DIR,
                f"{model_name}_Cluster_{cluster_index}_of_K{cluster_count}.pkl",
            )
            if not os.path.exists(model_path):
                continue

            model = joblib.load(model_path)
            preds_matrix[data_mask, model_index] = model.predict(x_data[data_mask])

    return preds_matrix


def run_stage4():
    print("开始步骤4：集成建模与结果评估")
    train_df, val_df, test_df = load_stage1_datasets()
    if not os.path.exists(SELECTED_BEST_KS_PATH):
        raise FileNotFoundError("缺少 selected_best_ks.pkl，请先运行步骤2。")

    selected_best_ks = joblib.load(SELECTED_BEST_KS_PATH)
    features = [column for column in train_df.columns if column != TARGET]

    preds_train = get_expert_predictions(train_df, selected_best_ks, features)
    preds_val = get_expert_predictions(val_df, selected_best_ks, features)
    preds_test = get_expert_predictions(test_df, selected_best_ks, features)
    y_train = train_df[TARGET].to_numpy()
    y_val = val_df[TARGET].to_numpy()
    y_test = test_df[TARGET].to_numpy()

    print("启动集成权重搜索...")

    def objective_function(weights):
        normalized_weights = weights / np.sum(weights)
        ensemble_pred = np.dot(preds_val, normalized_weights)
        return np.sqrt(mean_squared_error(y_val, ensemble_pred))

    particle_count = PSO_NUM_PARTICLES
    model_count = len(MODEL_FAMILIES)
    particles_pos = np.random.rand(particle_count, model_count)
    particles_vel = np.zeros((particle_count, model_count), dtype=float)

    pbest_pos = particles_pos.copy()
    pbest_scores = np.array([objective_function(position) for position in particles_pos])
    gbest_index = int(np.argmin(pbest_scores))
    gbest_pos = pbest_pos[gbest_index].copy()
    gbest_score = float(pbest_scores[gbest_index])

    for _ in range(PSO_MAX_ITER):
        r1 = np.random.rand(particle_count, model_count)
        r2 = np.random.rand(particle_count, model_count)
        particles_vel = (
            PSO_W * particles_vel
            + PSO_C1 * r1 * (pbest_pos - particles_pos)
            + PSO_C2 * r2 * (gbest_pos - particles_pos)
        )
        particles_pos = particles_pos + particles_vel
        particles_pos = np.clip(particles_pos, 1e-6, 1.0)

        current_scores = np.array([objective_function(position) for position in particles_pos])
        better_mask = current_scores < pbest_scores
        pbest_pos[better_mask] = particles_pos[better_mask]
        pbest_scores[better_mask] = current_scores[better_mask]

        current_best_index = int(np.argmin(pbest_scores))
        if pbest_scores[current_best_index] < gbest_score:
            gbest_pos = pbest_pos[current_best_index].copy()
            gbest_score = float(pbest_scores[current_best_index])

    optimal_weights = gbest_pos / np.sum(gbest_pos)
    y_pred_train = np.dot(preds_train, optimal_weights)
    y_pred_val = np.dot(preds_val, optimal_weights)
    y_pred_test = np.dot(preds_test, optimal_weights)

    metrics = {
        "train": evaluate_dataset("Train", y_train, y_pred_train),
        "val": evaluate_dataset("Val", y_val, y_pred_val),
        "test": evaluate_dataset("Test", y_test, y_pred_test),
    }

    summary_df = build_prediction_summary(
        train_true=y_train,
        train_pred=y_pred_train,
        val_true=y_val,
        val_pred=y_pred_val,
        test_true=y_test,
        test_pred=y_pred_test,
    )
    summary_df.to_excel(STAGE4_EXCEL_PATH, index=False)

    csv_df = pd.DataFrame(
        {
            "y_test": y_test,
            "y_pred": y_pred_test,
            "true_width": y_test,
            "pred_width": y_pred_test,
            "abs_error": np.abs(y_test - y_pred_test),
        }
    )
    csv_df.to_csv(STAGE4_CSV_PATH, index=False, encoding="utf-8-sig")

    with open(STAGE4_METRICS_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, ensure_ascii=False, indent=2)

    if GENERATE_PLOTS:
        plot_parity(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test)
        print(f"已生成预测散点图：{STAGE4_PLOT_PATH}")

    print("\nHybrid-2 结果文件已保存")
    print(f"汇总文件：{STAGE4_EXCEL_PATH}")
    print(f"测试结果：{STAGE4_CSV_PATH}")
    print(f"指标文件：{STAGE4_METRICS_PATH}")


def main():
    if RUN_OPTION == 1:
        run_stage1()
    elif RUN_OPTION == 2:
        run_stage2()
    elif RUN_OPTION == 3:
        run_stage3()
    elif RUN_OPTION == 4:
        run_stage4()
    elif RUN_OPTION == 5:
        run_stage1()
        run_stage2()
        run_stage3()
        run_stage4()
    else:
        raise ValueError("RUN_OPTION 只能是 1、2、3、4、5。")


if __name__ == "__main__":
    main()
