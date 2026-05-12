import json
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 用户运行开关
# =========================
RUN_OPTION = 4
# 1：只跑步骤1，SSA 优化 Tselikov 理论参数
# 2：只跑步骤2，生成虚拟数据并预训练 PR-DNN
# 3：只跑步骤3，微调 TG-DNN 并输出最终结果
# 4：按 1 -> 3 全流程运行


# =========================
# 全局参数配置
# =========================
GENERATE_PLOTS = True
RANDOM_SEED = 42
TARGET = "实测宽度-R2-5"

SSA_SAMPLE_RATIO = 0.3
SSA_POP_SIZE = 30
SSA_MAX_ITER = 100

VIRTUAL_SAMPLE_COUNT = 30000
PR_DNN_HIDDEN_LAYERS = (25, 10)
PR_DNN_MAX_ITER = 500
PR_DNN_LEARNING_RATE = 0.005


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DEAL_DIR = os.path.join(PROJECT_ROOT, "data", "datadeal")
DATA_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "TG-DNN")
IMAGE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "image", "TG-DNN")

TRAIN_PATH = os.path.join(DATA_DEAL_DIR, "Train_Data.xlsx")
VAL_PATH = os.path.join(DATA_DEAL_DIR, "Val_Data.xlsx")
TEST_PATH = os.path.join(DATA_DEAL_DIR, "Test_Data.xlsx")

SSA_COEF_PATH = os.path.join(DATA_OUTPUT_DIR, "SSA_Tselikov_Coef.npy")
SCALER_PATH = os.path.join(DATA_OUTPUT_DIR, "Scaler_X.pkl")
PR_MODEL_PATH = os.path.join(DATA_OUTPUT_DIR, "PR_DNN_Model.pkl")
FINAL_MODEL_PATH = os.path.join(DATA_OUTPUT_DIR, "Final_TGDNN_Model.pkl")
STAGE3_EXCEL_PATH = os.path.join(DATA_OUTPUT_DIR, "TGDNN_prediction_summary.xlsx")
STAGE3_CSV_PATH = os.path.join(DATA_OUTPUT_DIR, "result_TGDNN.csv")
STAGE3_METRICS_PATH = os.path.join(DATA_OUTPUT_DIR, "TGDNN_metrics.json")
STAGE3_PLOT_PATH = os.path.join(IMAGE_OUTPUT_DIR, "TGDNN_Parity_Plots.png")
SSA_PLOT_PATH = os.path.join(IMAGE_OUTPUT_DIR, "SSA_Tselikov_Convergence.png")

os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)


TG_DNN_FEATURES = [
    "板坯宽度实测值(热态)",
    "出口厚度-R1",
    "R2-1压下量",
    "R2轧制速度Pass1(H11_0)",
    "道次出口温度-R2-1",
    "R2工作辊辊径",
    "R2-2压下量",
    "R2轧制速度Pass2(H11_0)",
    "道次出口温度-R2-2",
    "R2工作辊辊径",
    "R2-3压下量",
    "R2轧制速度Pass3(H11_0)",
    "道次出口温度-R2-3",
    "R2工作辊辊径.1",
    "R2-4压下量",
    "R2轧制速度Pass4(H11_0)",
    "道次出口温度-R2-4",
    "R2工作辊辊径.1",
    "R2-5压下量",
    "R2轧制速度Pass5(H11_0)",
    "道次出口温度-R2-4",
    "R2工作辊辊径.2",
]


def load_clean_dataset():
    file_name = next(
        name
        for name in os.listdir(DATA_DEAL_DIR)
        if name.endswith(".xlsx") and name not in {"Train_Data.xlsx", "Val_Data.xlsx", "Test_Data.xlsx"}
    )
    dataframe = pd.read_excel(os.path.join(DATA_DEAL_DIR, file_name))
    dataframe.columns = dataframe.columns.str.strip()
    return dataframe


def load_split_datasets():
    train_df = pd.read_excel(TRAIN_PATH)
    val_df = pd.read_excel(VAL_PATH)
    test_df = pd.read_excel(TEST_PATH)
    return train_df, val_df, test_df


def validate_feature_columns(dataframe):
    missing_columns = [column for column in TG_DNN_FEATURES + [TARGET] if column not in dataframe.columns]
    if missing_columns:
        raise KeyError(f"TG-DNN 缺少必要列: {missing_columns}")


def tselikov_single_pass(B, H, h, dh, R0, v, t, coef):
    a1, a2, a3, a4, k1, k2, eta1, eta2, eta3 = coef
    dh = np.clip(dh, 1e-3, None)
    H = np.clip(H, 1e-3, None)
    epsilon = (H - h) / H
    phi = k1 + k2 * epsilon
    mu = eta1 - eta2 * t - eta3 * v
    term = B / np.sqrt(R0 * dh)
    C_value = a1 * (term - a2) * np.exp(np.clip(a3 - term, -50, 50)) + a4
    return C_value * dh * np.sqrt(R0 / H) * phi * mu


def multi_pass_predict(features_matrix, coef):
    B_in = features_matrix[:, 0]
    H_in = features_matrix[:, 1]

    dh1 = features_matrix[:, 2]
    v1 = features_matrix[:, 3]
    t1 = features_matrix[:, 4]
    R0_1 = features_matrix[:, 5]

    dh2 = features_matrix[:, 6]
    v2 = features_matrix[:, 7]
    t2 = features_matrix[:, 8]
    R0_2 = features_matrix[:, 9]

    dh3 = features_matrix[:, 10]
    v3 = features_matrix[:, 11]
    t3 = features_matrix[:, 12]
    R0_3 = features_matrix[:, 13]

    dh4 = features_matrix[:, 14]
    v4 = features_matrix[:, 15]
    t4 = features_matrix[:, 16]
    R0_4 = features_matrix[:, 17]

    dh5 = features_matrix[:, 18]
    v5 = features_matrix[:, 19]
    t5 = features_matrix[:, 20]
    R0_5 = features_matrix[:, 21]

    h1 = H_in - dh1
    db1 = tselikov_single_pass(B_in, H_in, h1, dh1, R0_1, v1, t1, coef)
    B1 = B_in + db1

    h2 = h1 - dh2
    db2 = tselikov_single_pass(B1, h1, h2, dh2, R0_2, v2, t2, coef)
    B2 = B1 + db2

    h3 = h2 - dh3
    db3 = tselikov_single_pass(B2, h2, h3, dh3, R0_3, v3, t3, coef)
    B3 = B2 + db3

    h4 = h3 - dh4
    db4 = tselikov_single_pass(B3, h3, h4, dh4, R0_4, v4, t4, coef)
    B4 = B3 + db4

    h5 = h4 - dh5
    db5 = tselikov_single_pass(B4, h4, h5, dh5, R0_5, v5, t5, coef)
    return B4 + db5


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
    plt.savefig(STAGE3_PLOT_PATH, dpi=300)
    plt.close()


def plot_ssa_convergence(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history, linewidth=2, color="darkred")
    plt.title("SSA 优化 Tselikov 理论模型收敛曲线", fontsize=15, fontweight="bold")
    plt.xlabel("迭代次数", fontsize=12)
    plt.ylabel("平均绝对误差 MAE", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(SSA_PLOT_PATH, dpi=300)
    plt.close()


def run_stage1():
    print("开始步骤1：SSA 优化 Tselikov 理论参数")
    clean_df = load_clean_dataset()
    validate_feature_columns(clean_df)

    sample_count = int(len(clean_df) * SSA_SAMPLE_RATIO)
    sampled_df = clean_df.sample(n=sample_count, random_state=RANDOM_SEED).reset_index(drop=True)
    x_sample = sampled_df[TG_DNN_FEATURES].to_numpy(dtype=float)
    y_true = sampled_df[TARGET].to_numpy(dtype=float)

    lb = np.array([1.005, 0.1125, 0.1125, 0.375, 0.1035, 0.246, 0.7875, 0.000375, 0.42])
    ub = np.array([1.675, 0.1875, 0.1875, 0.625, 0.1725, 0.410, 1.3125, 0.000625, 0.70])
    dim = 9

    def calc_fitness(population):
        fitness_values = np.zeros(len(population), dtype=float)
        for index, coef in enumerate(population):
            y_pred = multi_pass_predict(x_sample, coef)
            fitness_values[index] = np.mean(np.abs(y_pred - y_true))
        return fitness_values

    population = np.random.uniform(lb, ub, (SSA_POP_SIZE, dim))
    fitness = calc_fitness(population)
    pbest = population.copy()
    pbest_fitness = fitness.copy()
    gbest = population[np.argmin(fitness)].copy()
    gbest_fitness = float(np.min(fitness))

    pd_ratio = 0.2
    sd_ratio = 0.2
    producer_count = int(SSA_POP_SIZE * pd_ratio)
    scout_count = int(SSA_POP_SIZE * sd_ratio)
    history = []

    for iteration in range(SSA_MAX_ITER):
        sort_index = np.argsort(fitness)
        population = population[sort_index]
        fitness = fitness[sort_index]
        worst_index = int(np.argmax(fitness))
        worst_position = population[worst_index].copy()

        random_gate = np.random.rand()
        for index in range(producer_count):
            if random_gate < 0.8:
                population[index] = population[index] * np.exp(
                    -index / (np.random.rand() * SSA_MAX_ITER + 1e-8)
                )
            else:
                population[index] = population[index] + np.random.randn(dim)

        for index in range(producer_count, SSA_POP_SIZE):
            if index > SSA_POP_SIZE / 2:
                population[index] = np.random.randn(dim) * np.exp(
                    (worst_position - population[index]) / (index ** 2 + 1e-8)
                )
            else:
                direction = np.random.choice([-1, 1], size=dim) / dim
                population[index] = population[0] + np.abs(population[index] - population[0]) * direction

        scout_indices = np.random.choice(SSA_POP_SIZE, scout_count, replace=False)
        for index in scout_indices:
            if fitness[index] > gbest_fitness:
                population[index] = gbest + np.random.randn(dim) * np.abs(population[index] - gbest)
            else:
                population[index] = population[index] + (
                    np.random.choice([-1, 1]) * np.abs(population[index] - worst_position)
                ) / (fitness[index] - fitness[worst_index] + 1e-8)

        population = np.clip(population, lb, ub)
        fitness = calc_fitness(population)

        for index in range(SSA_POP_SIZE):
            if fitness[index] < pbest_fitness[index]:
                pbest_fitness[index] = fitness[index]
                pbest[index] = population[index].copy()

        current_best_index = int(np.argmin(pbest_fitness))
        if pbest_fitness[current_best_index] < gbest_fitness:
            gbest_fitness = float(pbest_fitness[current_best_index])
            gbest = pbest[current_best_index].copy()

        history.append(gbest_fitness)
        if (iteration + 1) % 10 == 0:
            print(f"Iter {iteration + 1:03d} | Best MAE={gbest_fitness:.4f}")

    np.save(SSA_COEF_PATH, gbest)
    if GENERATE_PLOTS:
        plot_ssa_convergence(history)
        print(f"已生成收敛曲线：{SSA_PLOT_PATH}")
    print(f"步骤1完成，最优理论参数已保存：{SSA_COEF_PATH}")


def run_stage2():
    print("开始步骤2：生成虚拟数据并预训练 PR-DNN")
    clean_df = load_clean_dataset()
    validate_feature_columns(clean_df)
    if not os.path.exists(SSA_COEF_PATH):
        raise FileNotFoundError("缺少 SSA_Tselikov_Coef.npy，请先运行步骤1。")

    x_real = clean_df[TG_DNN_FEATURES].to_numpy(dtype=float)
    x_min = x_real.min(axis=0)
    x_max = x_real.max(axis=0)
    best_coef = np.load(SSA_COEF_PATH)

    np.random.seed(RANDOM_SEED)
    x_virtual = np.random.uniform(x_min, x_max, (VIRTUAL_SAMPLE_COUNT, len(TG_DNN_FEATURES)))
    y_virtual = multi_pass_predict(x_virtual, best_coef)

    scaler = MinMaxScaler()
    x_virtual_scaled = scaler.fit_transform(x_virtual)
    joblib.dump(scaler, SCALER_PATH)

    pr_dnn = MLPRegressor(
        hidden_layer_sizes=PR_DNN_HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        max_iter=PR_DNN_MAX_ITER,
        learning_rate_init=PR_DNN_LEARNING_RATE,
        warm_start=True,
        random_state=RANDOM_SEED,
    )
    pr_dnn.fit(x_virtual_scaled, y_virtual)
    joblib.dump(pr_dnn, PR_MODEL_PATH)

    print(f"步骤2完成，预训练模型已保存：{PR_MODEL_PATH}")
    print(f"物理特征缩放器已保存：{SCALER_PATH}")


def run_stage3():
    print("开始步骤3：微调 TG-DNN 并输出最终结果")
    train_df, val_df, test_df = load_split_datasets()
    validate_feature_columns(train_df)
    validate_feature_columns(val_df)
    validate_feature_columns(test_df)

    if not os.path.exists(PR_MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("缺少 PR_DNN_Model.pkl 或 Scaler_X.pkl，请先运行步骤2。")

    scaler = joblib.load(SCALER_PATH)
    tg_dnn = joblib.load(PR_MODEL_PATH)

    x_train = train_df[TG_DNN_FEATURES].to_numpy(dtype=float)
    x_val = val_df[TG_DNN_FEATURES].to_numpy(dtype=float)
    x_test = test_df[TG_DNN_FEATURES].to_numpy(dtype=float)
    y_train = train_df[TARGET].to_numpy(dtype=float)
    y_val = val_df[TARGET].to_numpy(dtype=float)
    y_test = test_df[TARGET].to_numpy(dtype=float)

    x_train_scaled = scaler.transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    tg_dnn.fit(x_train_scaled, y_train)
    y_pred_train = tg_dnn.predict(x_train_scaled)
    y_pred_val = tg_dnn.predict(x_val_scaled)
    y_pred_test = tg_dnn.predict(x_test_scaled)
    joblib.dump(tg_dnn, FINAL_MODEL_PATH)

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
    summary_df.to_excel(STAGE3_EXCEL_PATH, index=False)

    csv_df = pd.DataFrame(
        {
            "y_test": y_test,
            "y_pred": y_pred_test,
            "true_width": y_test,
            "pred_width": y_pred_test,
            "abs_error": np.abs(y_test - y_pred_test),
        }
    )
    csv_df.to_csv(STAGE3_CSV_PATH, index=False, encoding="utf-8-sig")

    with open(STAGE3_METRICS_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, ensure_ascii=False, indent=2)

    if GENERATE_PLOTS:
        plot_parity(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test)
        print(f"已生成预测散点图：{STAGE3_PLOT_PATH}")

    print("\nTG-DNN 结果文件已保存")
    print(f"模型文件：{FINAL_MODEL_PATH}")
    print(f"汇总文件：{STAGE3_EXCEL_PATH}")
    print(f"测试结果：{STAGE3_CSV_PATH}")
    print(f"指标文件：{STAGE3_METRICS_PATH}")


def main():
    if RUN_OPTION == 1:
        run_stage1()
    elif RUN_OPTION == 2:
        run_stage2()
    elif RUN_OPTION == 3:
        run_stage3()
    elif RUN_OPTION == 4:
        run_stage1()
        run_stage2()
        run_stage3()
    else:
        raise ValueError("RUN_OPTION 只能是 1、2、3、4。")


if __name__ == "__main__":
    main()
