# -*- coding: utf-8 -*-
import json
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import optuna
except ImportError as exc:
    raise ImportError("Optuna is required for DBN_BO.py. Run: pip install optuna") from exc

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import BernoulliRBM, MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_SEED = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "DBN_BO")
IMAGE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "image", "DBN_BO")
BASE_DIR = os.path.join(PROJECT_ROOT, "data", "datadeal")
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(BASE_DIR, "Train_Data.xlsx")
VAL_PATH = os.path.join(BASE_DIR, "Val_Data.xlsx")
TEST_PATH = os.path.join(BASE_DIR, "Test_Data.xlsx")

TARGET_CANDIDATES = ["实测宽度-R2-5", "瀹炴祴瀹藉害-R2-5"]
SETTING_TARGET = "出口宽度设定值-R2-5"
RUN_OPTION = 1
# 1: BO搜索 + 最终训练 + 导出结果
# 2: 只画图，不重新训练

BO_TRIALS = 60
FIXED_MLP_HIDDEN_SIZES = (16,)
FIXED_SUPERVISED_MAX_EPOCHS = 75
FIXED_EARLY_STOPPING_PATIENCE = 10
FIXED_SUPERVISED_LEARNING_RATE = 0.001
FIXED_ALPHA = 5e-4

RESULT_EXCEL_PATH = os.path.join(DATA_OUTPUT_DIR, "DBN_BO_prediction_results.xlsx")
RESULT_CSV_PATH = os.path.join(DATA_OUTPUT_DIR, "result_DBN_BO.csv")
RESULT_METRICS_PATH = os.path.join(DATA_OUTPUT_DIR, "DBN_BO_metrics.json")
BEST_PARAMS_PATH = os.path.join(DATA_OUTPUT_DIR, "DBN_BO_best_params.json")
BO_HISTORY_PATH = os.path.join(DATA_OUTPUT_DIR, "DBN_BO_search_history.csv")

SCATTER_PLOT_PATH = os.path.join(IMAGE_OUTPUT_DIR, "DBN_BO_Prediction_Scatter.png")
BO_CURVE_PATH = os.path.join(IMAGE_OUTPUT_DIR, "DBN_BO_Search_Curve.png")


class JointFineTunedDBNRegressor:
    def __init__(
        self,
        rbm_hidden_layers,
        mlp_hidden_sizes,
        rbm_learning_rate,
        rbm_n_iter,
        supervised_learning_rate,
        supervised_max_epochs,
        early_stopping_patience,
        alpha=1e-4,
        min_improvement=1e-4,
        random_state=42,
        verbose=False,
    ):
        self.rbm_hidden_layers = tuple(rbm_hidden_layers)
        self.mlp_hidden_sizes = tuple(mlp_hidden_sizes)
        self.rbm_learning_rate = rbm_learning_rate
        self.rbm_n_iter = rbm_n_iter
        self.supervised_learning_rate = supervised_learning_rate
        self.supervised_max_epochs = supervised_max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.alpha = alpha
        self.min_improvement = min_improvement
        self.random_state = random_state
        self.verbose = verbose

        self.rbms = []
        self.regressor = None
        self.best_epoch_ = 0
        self.best_val_rmse_ = np.inf

    def _log(self, message):
        if self.verbose:
            print(message)

    @property
    def full_hidden_layers(self):
        return self.rbm_hidden_layers + self.mlp_hidden_sizes

    def _build_rbms(self):
        self.rbms = []
        for index, hidden_size in enumerate(self.rbm_hidden_layers):
            rbm = BernoulliRBM(
                n_components=hidden_size,
                learning_rate=self.rbm_learning_rate,
                n_iter=self.rbm_n_iter,
                random_state=self.random_state + index,
                verbose=False,
            )
            self.rbms.append(rbm)

    def _pretrain_rbms(self, x_train):
        self._build_rbms()
        transformed = x_train
        for layer_index, rbm in enumerate(self.rbms, start=1):
            self._log(f"开始预训练第 {layer_index} 层 RBM...")
            rbm.fit(transformed)
            transformed = rbm.transform(transformed)
            self._log(f"第 {layer_index} 层 RBM 训练完成，隐藏特征维度 {transformed.shape[1]}")

    def _build_finetune_network(self, x_train, y_train):
        self.regressor = MLPRegressor(
            hidden_layer_sizes=self.full_hidden_layers,
            activation="logistic",
            solver="adam",
            learning_rate_init=self.supervised_learning_rate,
            max_iter=1,
            warm_start=True,
            random_state=self.random_state,
            shuffle=True,
            batch_size=min(32, len(x_train)),
            alpha=self.alpha,
        )

        self.regressor.fit(x_train, y_train)

        for layer_index, rbm in enumerate(self.rbms):
            self.regressor.coefs_[layer_index] = rbm.components_.T.copy()
            self.regressor.intercepts_[layer_index] = rbm.intercept_hidden_.copy()

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        self.best_epoch_ = 0
        self.best_val_rmse_ = np.inf
        self._pretrain_rbms(x_train)
        self._build_finetune_network(x_train, y_train)

        best_weights = [coef.copy() for coef in self.regressor.coefs_]
        best_intercepts = [intercept.copy() for intercept in self.regressor.intercepts_]
        stale_epochs = 0

        self._log("\n开始整网联合微调...")
        for epoch in range(1, self.supervised_max_epochs + 1):
            self.regressor.partial_fit(x_train, y_train)

            if x_val is not None and y_val is not None:
                val_pred = self.regressor.predict(x_val)
                val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
            else:
                train_pred = self.regressor.predict(x_train)
                val_rmse = float(np.sqrt(mean_squared_error(y_train, train_pred)))

            self._log(f"联合微调轮次 {epoch:03d} | 当前 RMSE: {val_rmse:.6f}")

            if val_rmse + self.min_improvement < self.best_val_rmse_:
                self.best_val_rmse_ = val_rmse
                self.best_epoch_ = epoch
                best_weights = [coef.copy() for coef in self.regressor.coefs_]
                best_intercepts = [intercept.copy() for intercept in self.regressor.intercepts_]
                stale_epochs = 0
            else:
                stale_epochs += 1

            if stale_epochs >= self.early_stopping_patience:
                self._log(f"验证集连续 {self.early_stopping_patience} 轮未提升，提前停止联合微调。")
                break

        self.regressor.coefs_ = best_weights
        self.regressor.intercepts_ = best_intercepts
        self._log(f"联合微调完成，最佳轮次 {self.best_epoch_}，最佳 RMSE: {self.best_val_rmse_:.6f}")
        return self

    def predict(self, x_input):
        return self.regressor.predict(x_input)


def compute_metrics(y_true, y_pred):
    mse_value = float(mean_squared_error(y_true, y_pred))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse_value,
        "RMSE": float(np.sqrt(mse_value)),
    }


def evaluate_dataset(name, x_scaled, y_true, model, scaler_y):
    y_pred_scaled = model.predict(x_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    metrics = compute_metrics(y_true, y_pred)

    print(f"\n{name}指标:")
    print(f"R2  : {metrics['R2']:.4f}")
    print(f"MAE : {metrics['MAE']:.4f}")
    print(f"MSE : {metrics['MSE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    return y_pred, metrics


def pad_array(values, target_length):
    padded = np.full(target_length, np.nan, dtype=float)
    padded[: len(values)] = values
    return padded


def build_excel_result(y_train, y_val, y_test, train_pred, val_pred, test_pred):
    max_length = max(len(train_pred), len(val_pred), len(test_pred))
    return pd.DataFrame(
        {
            "val_prediction": pad_array(val_pred, max_length),
            "val_true": pad_array(y_val, max_length),
            "train_prediction": pad_array(train_pred, max_length),
            "train_true": pad_array(y_train, max_length),
            "test_prediction": pad_array(test_pred, max_length),
            "test_true": pad_array(y_test, max_length),
        }
    )


def save_excel_with_fallback(dataframe, preferred_path):
    try:
        dataframe.to_excel(preferred_path, index=False)
        return preferred_path
    except PermissionError:
        fallback_path = preferred_path.replace(".xlsx", "_new.xlsx")
        dataframe.to_excel(fallback_path, index=False)
        print(f"\n检测到 {preferred_path} 正在被占用，已改存为: {fallback_path}")
        return fallback_path


def save_figure_with_fallback(figure, preferred_path, dpi=300):
    try:
        figure.savefig(preferred_path, dpi=dpi, bbox_inches="tight")
        return preferred_path
    except PermissionError:
        fallback_path = preferred_path.replace(".png", "_new.png")
        figure.savefig(fallback_path, dpi=dpi, bbox_inches="tight")
        print(f"\n检测到 {preferred_path} 正在被占用，图像已改存为: {fallback_path}")
        return fallback_path
    finally:
        plt.close(figure)


def plot_prediction_scatter(y_test, test_pred, output_path):
    figure, axis = plt.subplots(figsize=(8, 8))
    axis.scatter(y_test, test_pred, alpha=0.6, color="royalblue", edgecolor="k")
    axis.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        lw=2,
        label="理想拟合线",
    )
    axis.set_title("BO-DBN 宽度预测性能可视化", fontsize=15, fontweight="bold")
    axis.set_xlabel("真实宽度 (mm)", fontsize=12)
    axis.set_ylabel("BO-DBN 预测宽度 (mm)", fontsize=12)
    axis.legend()
    axis.grid(True, linestyle="--", alpha=0.4)
    figure.tight_layout()
    return save_figure_with_fallback(figure, output_path)


def plot_bo_curve(history_df, output_path):
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(history_df["trial"], history_df["best_objective"], marker="o", color="#1f77b4", linewidth=2)
    axis.set_title("BO-DBN 搜索收敛曲线", fontsize=14, fontweight="bold")
    axis.set_xlabel("Trial", fontsize=11)
    axis.set_ylabel("Best Objective", fontsize=11)
    axis.grid(True, linestyle="--", alpha=0.4)
    figure.tight_layout()
    return save_figure_with_fallback(figure, output_path)


def load_datasets():
    df_train = pd.read_excel(TRAIN_PATH)
    df_val = pd.read_excel(VAL_PATH)
    df_test = pd.read_excel(TEST_PATH)

    for dataframe in [df_train, df_val, df_test]:
        dataframe.columns = dataframe.columns.str.strip()

    target = resolve_target_column(df_train.columns)
    features = [column for column in df_train.columns if column not in [target, SETTING_TARGET]]
    x_train = df_train[features].values
    y_train = df_train[target].values
    x_val = df_val[features].values
    y_val = df_val[target].values
    x_test = df_test[features].values
    y_test = df_test[target].values
    return features, x_train, y_train, x_val, y_val, x_test, y_test


def resolve_target_column(columns):
    for candidate in TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
    raise KeyError(f"未找到目标列，可选列名: {TARGET_CANDIDATES}")


def suggest_params(trial):
    n_components = trial.suggest_int("n_components", 48, 256, step=8)
    rbm_n_iter = trial.suggest_int("rbm_n_iter", 10, 50)
    return {
        "rbm_hidden_layers": (n_components,),
        "mlp_hidden_sizes": FIXED_MLP_HIDDEN_SIZES,
        "rbm_learning_rate": trial.suggest_float("rbm_learning_rate", 5e-4, 2e-2, log=True),
        "rbm_n_iter": rbm_n_iter,
        "supervised_learning_rate": FIXED_SUPERVISED_LEARNING_RATE,
        "supervised_max_epochs": FIXED_SUPERVISED_MAX_EPOCHS,
        "early_stopping_patience": FIXED_EARLY_STOPPING_PATIENCE,
        "alpha": FIXED_ALPHA,
    }


def build_model(params, verbose=False):
    return JointFineTunedDBNRegressor(
        rbm_hidden_layers=params["rbm_hidden_layers"],
        mlp_hidden_sizes=params["mlp_hidden_sizes"],
        rbm_learning_rate=params["rbm_learning_rate"],
        rbm_n_iter=params["rbm_n_iter"],
        supervised_learning_rate=params["supervised_learning_rate"],
        supervised_max_epochs=params["supervised_max_epochs"],
        early_stopping_patience=params["early_stopping_patience"],
        alpha=params["alpha"],
        random_state=RANDOM_SEED,
        verbose=verbose,
    )


def evaluate_params(params, x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled, scaler_y, y_val_raw):
    model = build_model(params, verbose=False)
    model.fit(x_train_scaled, y_train_scaled, x_val=x_val_scaled, y_val=y_val_scaled)
    val_pred_scaled = model.predict(x_val_scaled)
    val_pred = scaler_y.inverse_transform(val_pred_scaled.reshape(-1, 1)).ravel()
    metrics = compute_metrics(y_val_raw, val_pred)
    objective = metrics["RMSE"]
    return {
        "objective": float(objective),
        "metrics": metrics,
        "best_epoch": int(model.best_epoch_),
        "best_val_rmse_scaled": float(model.best_val_rmse_),
    }


def run_bo_search(x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled, scaler_y, y_val_raw):
    history_rows = []

    def objective(trial):
        params = suggest_params(trial)
        result = evaluate_params(params, x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled, scaler_y, y_val_raw)

        trial.set_user_attr("params", params)
        trial.set_user_attr("metrics", result["metrics"])
        trial.set_user_attr("best_epoch", result["best_epoch"])
        trial.set_user_attr("best_val_rmse_scaled", result["best_val_rmse_scaled"])
        history_rows.append(
            {
                "trial": trial.number + 1,
                "objective": result["objective"],
                "best_epoch": result["best_epoch"],
                "val_mae": result["metrics"]["MAE"],
                "val_mse": result["metrics"]["MSE"],
                "val_rmse": result["metrics"]["RMSE"],
                "val_r2": result["metrics"]["R2"],
            }
        )

        print(
            f"Trial {trial.number + 1:02d} | "
            f"Val RMSE={result['metrics']['RMSE']:.4f} | "
            f"Val MAE={result['metrics']['MAE']:.4f} | "
            f"Val R2={result['metrics']['R2']:.4f} | "
            f"Objective={result['objective']:.4f}"
        )
        return result["objective"]

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=BO_TRIALS, show_progress_bar=False)

    history_df = pd.DataFrame(history_rows)
    history_df["best_objective"] = history_df["objective"].cummin()
    return study, history_df


def make_json_safe_params(params):
    payload = dict(params)
    payload["rbm_hidden_layers"] = list(payload["rbm_hidden_layers"])
    payload["mlp_hidden_sizes"] = list(payload["mlp_hidden_sizes"])
    return payload


def run_train_and_export():
    print("正在加载数据集...")
    features, x_train, y_train, x_val, y_val, x_test, y_test = load_datasets()
    print(f"数据准备完毕，BO-DBN 接收 {len(features)} 个特征。")
    print(f"训练集 {len(x_train)} 条，验证集 {len(x_val)} 条，测试集 {len(x_test)} 条。")

    scaler_x = MinMaxScaler(feature_range=(0.1, 0.9))
    x_train_scaled = scaler_x.fit_transform(x_train)
    x_val_scaled = scaler_x.transform(x_val)
    x_test_scaled = scaler_x.transform(x_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

    print("\n开始执行 BO 搜索...")
    study, history_df = run_bo_search(
        x_train_scaled,
        y_train_scaled,
        x_val_scaled,
        y_val_scaled,
        scaler_y,
        y_val,
    )

    best_trial = study.best_trial
    best_params = dict(best_trial.user_attrs["params"])

    print("\nBO 搜索完成，最优参数如下:")
    for key, value in best_params.items():
        print(f"{key}: {value}")

    final_model = build_model(best_params, verbose=True)
    final_model.fit(
        x_train_scaled,
        y_train_scaled,
        x_val=x_val_scaled,
        y_val=y_val_scaled,
    )

    print("\n开始在训练集、验证集、测试集上进行预测...")
    train_pred, train_metrics = evaluate_dataset("训练集", x_train_scaled, y_train, final_model, scaler_y)
    val_pred, val_metrics = evaluate_dataset("验证集", x_val_scaled, y_val, final_model, scaler_y)
    test_pred, test_metrics = evaluate_dataset("测试集", x_test_scaled, y_test, final_model, scaler_y)

    result_excel = build_excel_result(y_train, y_val, y_test, train_pred, val_pred, test_pred)
    excel_result_path = save_excel_with_fallback(result_excel, RESULT_EXCEL_PATH)

    pd.DataFrame({"true_width": y_test, "pred_width_dbn_bo": test_pred}).to_csv(
        RESULT_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    history_df.to_csv(BO_HISTORY_PATH, index=False, encoding="utf-8-sig")

    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(make_json_safe_params(best_params), file_obj, ensure_ascii=False, indent=2)

    metrics_payload = {
        "best_params": make_json_safe_params(best_params),
        "bo_best_validation_metrics": best_trial.user_attrs["metrics"],
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_epoch": int(final_model.best_epoch_),
        "best_val_rmse_scaled": float(final_model.best_val_rmse_),
        "best_objective": float(best_trial.value),
    }
    with open(RESULT_METRICS_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(metrics_payload, file_obj, ensure_ascii=False, indent=2)

    scatter_path = plot_prediction_scatter(y_test, test_pred, SCATTER_PLOT_PATH)
    curve_path = plot_bo_curve(history_df, BO_CURVE_PATH)

    print("\n结果文件已保存:")
    print(f"1. {excel_result_path}")
    print(f"2. {RESULT_CSV_PATH}")
    print(f"3. {BEST_PARAMS_PATH}")
    print(f"4. {RESULT_METRICS_PATH}")
    print(f"5. {BO_HISTORY_PATH}")
    print(f"6. {scatter_path}")
    print(f"7. {curve_path}")


def run_plot_only():
    print("只画图模式：不重新训练 BO-DBN。")
    if not os.path.exists(RESULT_CSV_PATH):
        raise FileNotFoundError(f"未找到已有预测结果文件: {RESULT_CSV_PATH}")

    result_df = pd.read_csv(RESULT_CSV_PATH)
    y_test = result_df["true_width"].to_numpy(dtype=float)
    test_pred = result_df["pred_width_dbn_bo"].to_numpy(dtype=float)
    scatter_path = plot_prediction_scatter(y_test, test_pred, SCATTER_PLOT_PATH)

    if os.path.exists(BO_HISTORY_PATH):
        history_df = pd.read_csv(BO_HISTORY_PATH)
        curve_path = plot_bo_curve(history_df, BO_CURVE_PATH)
    else:
        curve_path = "未找到 BO 历史文件，跳过收敛曲线绘制。"

    print("\n图像文件已保存:")
    print(f"1. {scatter_path}")
    print(f"2. {curve_path}")


if __name__ == "__main__":
    if RUN_OPTION == 1:
        run_train_and_export()
    elif RUN_OPTION == 2:
        run_plot_only()
    else:
        raise ValueError(f"不支持的 RUN_OPTION: {RUN_OPTION}")
