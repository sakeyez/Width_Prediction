# -*- coding: utf-8 -*-
import json
import os
import random
from datetime import datetime
import tempfile
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import optuna
except ImportError as exc:
    raise ImportError("Optuna is required for LightGBM_BO.py. Run: pip install optuna") from exc

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_SEED = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(PROJECT_ROOT, "data", "datadeal")


def ensure_writable_directory(preferred_dir, fallback_root):
    os.makedirs(preferred_dir, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=preferred_dir, delete=True):
            pass
        return preferred_dir
    except (PermissionError, OSError):
        fallback_dir = os.path.join(fallback_root, os.path.basename(preferred_dir))
        os.makedirs(fallback_dir, exist_ok=True)
        return fallback_dir


RUNTIME_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "runtime_outputs")
DATA_OUTPUT_DIR = ensure_writable_directory(os.path.join(PROJECT_ROOT, "data", "LightGBM_BO"), RUNTIME_OUTPUT_ROOT)
IMAGE_OUTPUT_DIR = ensure_writable_directory(os.path.join(PROJECT_ROOT, "image", "LightGBM_BO"), RUNTIME_OUTPUT_ROOT)

TRAIN_PATH = os.path.join(BASE_DIR, "Train_Data.xlsx")
VAL_PATH = os.path.join(BASE_DIR, "Val_Data.xlsx")
TEST_PATH = os.path.join(BASE_DIR, "Test_Data.xlsx")

TARGET_CANDIDATES = ["实测宽度-R2-5"]
SETTING_TARGET = "出口宽度设定值-R2-5"
RUN_OPTION = 1
# 1: BO搜索 + 最终训练 + 导出结果

STAGE1_TRIALS = 100
STAGE2_TRIALS = 100
STAGE3_TRIALS = 100
EARLY_STOPPING_ROUNDS = 30
OBJECTIVE_OVERFIT_WEIGHT = 0.35

RESULT_EXCEL_PATH = os.path.join(DATA_OUTPUT_DIR, "LightGBM_BO_prediction_results.xlsx")
RESULT_CSV_PATH = os.path.join(DATA_OUTPUT_DIR, "result_LightGBM_BO.csv")
RESULT_METRICS_PATH = os.path.join(DATA_OUTPUT_DIR, "LightGBM_BO_metrics.json")
BEST_PARAMS_PATH = os.path.join(DATA_OUTPUT_DIR, "LightGBM_BO_best_params.json")
BO_HISTORY_PATH = os.path.join(DATA_OUTPUT_DIR, "LightGBM_BO_search_history.csv")

SCATTER_PLOT_PATH = os.path.join(IMAGE_OUTPUT_DIR, "LightGBM_BO_Prediction_Scatter.png")
BO_CURVE_PATH = os.path.join(IMAGE_OUTPUT_DIR, "LightGBM_BO_Search_Curve.png")


def compute_metrics(y_true, y_pred):
    mse_value = float(mean_squared_error(y_true, y_pred))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse_value,
        "RMSE": float(np.sqrt(mse_value)),
    }


def evaluate_dataset(name, x_data, y_true, model):
    y_pred = model.predict(x_data)
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
        fallback_dir = os.path.dirname(preferred_path)
        fallback_name = os.path.splitext(os.path.basename(preferred_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for suffix in ["_new", f"_{timestamp}"] + [f"_{timestamp}_{index}" for index in range(1, 100)]:
            fallback_path = os.path.join(fallback_dir, f"{fallback_name}{suffix}.xlsx")
            try:
                dataframe.to_excel(fallback_path, index=False)
                print(f"\n检测到 {preferred_path} 正在被占用，已改存为: {fallback_path}")
                return fallback_path
            except PermissionError:
                continue
        raise PermissionError(f"无法写入 Excel 文件，请关闭占用中的文件后重试: {preferred_path}")


def save_figure_with_fallback(figure, preferred_path, dpi=300):
    try:
        figure.savefig(preferred_path, dpi=dpi, bbox_inches="tight")
        return preferred_path
    except PermissionError:
        fallback_dir = os.path.dirname(preferred_path)
        fallback_name = os.path.splitext(os.path.basename(preferred_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for suffix in ["_new", f"_{timestamp}"] + [f"_{timestamp}_{index}" for index in range(1, 100)]:
            fallback_path = os.path.join(fallback_dir, f"{fallback_name}{suffix}.png")
            try:
                figure.savefig(fallback_path, dpi=dpi, bbox_inches="tight")
                print(f"\n检测到 {preferred_path} 正在被占用，图像已改存为: {fallback_path}")
                return fallback_path
            except PermissionError:
                continue
        raise PermissionError(f"无法写入图像文件，请关闭占用中的文件后重试: {preferred_path}")
    finally:
        plt.close(figure)


def plot_prediction_scatter(y_test, test_pred, output_path):
    figure, axis = plt.subplots(figsize=(8, 8))
    axis.scatter(y_test, test_pred, alpha=0.6, color="mediumseagreen", edgecolor="k")
    axis.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        lw=2,
        label="理想拟合线",
    )
    axis.set_title("BO-LightGBM 宽度预测性能可视化", fontsize=15, fontweight="bold")
    axis.set_xlabel("真实宽度 (mm)", fontsize=12)
    axis.set_ylabel("BO-LightGBM 预测宽度 (mm)", fontsize=12)
    axis.legend()
    axis.grid(True, linestyle="--", alpha=0.4)
    figure.tight_layout()
    return save_figure_with_fallback(figure, output_path)


def plot_bo_curve(history_df, output_path):
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(history_df["trial"], history_df["best_objective"], marker="o", color="#2ca25f", linewidth=2)
    axis.set_title("BO-LightGBM 搜索收敛曲线", fontsize=14, fontweight="bold")
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
    x_train = df_train[features].to_numpy()
    y_train = df_train[target].to_numpy()
    x_val = df_val[features].to_numpy()
    y_val = df_val[target].to_numpy()
    x_test = df_test[features].to_numpy()
    y_test = df_test[target].to_numpy()
    return features, x_train, y_train, x_val, y_val, x_test, y_test


def resolve_target_column(columns):
    for candidate in TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
    raise KeyError(f"未找到目标列，可选列名: {TARGET_CANDIDATES}")


BASE_FIXED_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 20,
    "max_depth": 6,
    "min_child_samples": 45,
    "subsample": 0.82,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.8,
    "reg_lambda": 4.0,
    "min_split_gain": 0.08,
}


def suggest_stage1_params(trial, fixed_params):
    params = dict(fixed_params)
    params.update(
        {
            "n_estimators": trial.suggest_int("n_estimators", 300, 950),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.04, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 12, 28),
        }
    )
    return params


def suggest_stage2_params(trial, fixed_params):
    params = dict(fixed_params)
    params.update(
        {
            "max_depth": trial.suggest_int("max_depth", 4, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 35, 110),
            "subsample": trial.suggest_float("subsample", 0.72, 0.90),
        }
    )
    return params


def suggest_stage3_params(trial, fixed_params):
    params = dict(fixed_params)
    params.update(
        {
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.72, 0.90),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 2.5, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 8.0, log=True),
        }
    )
    return params


def build_model(params):
    return lgb.LGBMRegressor(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        min_split_gain=params["min_split_gain"],
        subsample_freq=1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        bagging_seed=RANDOM_SEED,
        feature_fraction_seed=RANDOM_SEED,
        data_random_seed=RANDOM_SEED,
        verbosity=-1,
    )


def evaluate_params(params, x_train, y_train, x_val, y_val):
    model = build_model(params)
    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="l1",
        callbacks=callbacks,
    )
    train_predictions = model.predict(x_train)
    val_predictions = model.predict(x_val)
    train_metrics = compute_metrics(y_train, train_predictions)
    val_metrics = compute_metrics(y_val, val_predictions)
    mse_gap = max(0.0, val_metrics["MSE"] - train_metrics["MSE"])
    rmse_gap = max(0.0, val_metrics["RMSE"] - train_metrics["RMSE"])
    mae_gap = max(0.0, val_metrics["MAE"] - train_metrics["MAE"])
    r2_gap = max(0.0, train_metrics["R2"] - val_metrics["R2"])
    overfit_penalty = 0.12 * mse_gap + 0.70 * rmse_gap + 0.30 * mae_gap + 4.00 * r2_gap
    objective = (1.0 - OBJECTIVE_OVERFIT_WEIGHT) * val_metrics["MSE"] + OBJECTIVE_OVERFIT_WEIGHT * overfit_penalty
    return {
        "objective": float(objective),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "best_iteration": int(model.best_iteration_ or params["n_estimators"]),
    }


def run_stage_search(stage_name, stage_index, suggest_fn, fixed_params, n_trials, x_train, y_train, x_val, y_val):
    history_rows = []

    def objective(trial):
        params = suggest_fn(trial, fixed_params)
        result = evaluate_params(params, x_train, y_train, x_val, y_val)

        trial.set_user_attr("params", params)
        trial.set_user_attr("train_metrics", result["train_metrics"])
        trial.set_user_attr("val_metrics", result["val_metrics"])
        trial.set_user_attr("best_iteration", result["best_iteration"])
        history_rows.append(
            {
                "stage": stage_name,
                "stage_index": stage_index,
                "trial": trial.number + 1,
                "objective": result["objective"],
                "best_iteration": result["best_iteration"],
                "val_mae": result["val_metrics"]["MAE"],
                "val_mse": result["val_metrics"]["MSE"],
                "val_rmse": result["val_metrics"]["RMSE"],
                "val_r2": result["val_metrics"]["R2"],
                "train_rmse": result["train_metrics"]["RMSE"],
            }
        )

        print(
            f"{stage_name} Trial {trial.number + 1:03d} | "
            f"Val RMSE={result['val_metrics']['RMSE']:.4f} | "
            f"Val MAE={result['val_metrics']['MAE']:.4f} | "
            f"Val R2={result['val_metrics']['R2']:.4f} | "
            f"Objective={result['objective']:.4f}"
        )
        return result["objective"]

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED + stage_index)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    history_df = pd.DataFrame(history_rows)
    history_df["best_objective"] = history_df["objective"].cummin()
    best_trial = study.best_trial
    best_params = dict(best_trial.user_attrs["params"])
    best_params["n_estimators"] = int(best_trial.user_attrs["best_iteration"])
    return best_trial, best_params, history_df


def run_bo_search(x_train, y_train, x_val, y_val):
    fixed_params = dict(BASE_FIXED_PARAMS)

    stage1_trial, stage1_params, history_stage1 = run_stage_search(
        stage_name="Stage1",
        stage_index=1,
        suggest_fn=suggest_stage1_params,
        fixed_params=fixed_params,
        n_trials=STAGE1_TRIALS,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
    )
    fixed_params.update(stage1_params)

    stage2_trial, stage2_params, history_stage2 = run_stage_search(
        stage_name="Stage2",
        stage_index=2,
        suggest_fn=suggest_stage2_params,
        fixed_params=fixed_params,
        n_trials=STAGE2_TRIALS,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
    )
    fixed_params.update(stage2_params)

    stage3_trial, stage3_params, history_stage3 = run_stage_search(
        stage_name="Stage3",
        stage_index=3,
        suggest_fn=suggest_stage3_params,
        fixed_params=fixed_params,
        n_trials=STAGE3_TRIALS,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
    )
    fixed_params.update(stage3_params)

    history_df = pd.concat([history_stage1, history_stage2, history_stage3], ignore_index=True)
    final_trial = stage3_trial
    return final_trial, fixed_params, history_df


def run_train_and_export():
    print("正在加载数据集...")
    print(f"数据输出目录: {DATA_OUTPUT_DIR}")
    print(f"图像输出目录: {IMAGE_OUTPUT_DIR}")
    features, x_train, y_train, x_val, y_val, x_test, y_test = load_datasets()
    print(f"数据准备完毕，BO-LightGBM 接收 {len(features)} 个特征。")
    print(f"训练集 {len(x_train)} 条，验证集 {len(x_val)} 条，测试集 {len(x_test)} 条。")

    print("\n开始执行 BO 搜索...")
    best_trial, best_params, history_df = run_bo_search(x_train, y_train, x_val, y_val)

    print("\nBO 搜索完成，最优参数如下:")
    for key, value in best_params.items():
        print(f"{key}: {value}")

    final_model = build_model(best_params)
    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    final_model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="l1",
        callbacks=callbacks,
    )

    print("\n开始在训练集、验证集、测试集上进行预测...")
    train_pred, train_metrics = evaluate_dataset("训练集", x_train, y_train, final_model)
    val_pred, val_metrics = evaluate_dataset("验证集", x_val, y_val, final_model)
    test_pred, test_metrics = evaluate_dataset("测试集", x_test, y_test, final_model)

    result_excel = build_excel_result(y_train, y_val, y_test, train_pred, val_pred, test_pred)
    excel_result_path = save_excel_with_fallback(result_excel, RESULT_EXCEL_PATH)

    pd.DataFrame({"true_width": y_test, "pred_width_lightgbm_bo": test_pred}).to_csv(
        RESULT_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    history_df.to_csv(BO_HISTORY_PATH, index=False, encoding="utf-8-sig")

    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(best_params, file_obj, ensure_ascii=False, indent=2)

    metrics_payload = {
        "best_params": best_params,
        "bo_best_validation_metrics": best_trial.user_attrs["val_metrics"],
        "bo_best_train_metrics": best_trial.user_attrs["train_metrics"],
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_iteration": int(final_model.best_iteration_ or best_params["n_estimators"]),
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
    print("只画图模式：不重新训练 BO-LightGBM。")
    print(f"数据输出目录: {DATA_OUTPUT_DIR}")
    print(f"图像输出目录: {IMAGE_OUTPUT_DIR}")
    if not os.path.exists(RESULT_CSV_PATH):
        raise FileNotFoundError(f"未找到已有预测结果文件: {RESULT_CSV_PATH}")

    result_df = pd.read_csv(RESULT_CSV_PATH)
    y_test = result_df["true_width"].to_numpy(dtype=float)
    test_pred = result_df["pred_width_lightgbm_bo"].to_numpy(dtype=float)
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
