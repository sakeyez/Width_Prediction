# -*- coding: utf-8 -*-
import os
import random
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RANDOM_SEED = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "LightGBM")
IMAGE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "image", "LightGBM")
BASE_DIR = os.path.join(PROJECT_ROOT, "data", "datadeal")

os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(BASE_DIR, "Train_Data.xlsx")
VAL_PATH = os.path.join(BASE_DIR, "Val_Data.xlsx")
TEST_PATH = os.path.join(BASE_DIR, "Test_Data.xlsx")

TARGET_CANDIDATES = ["实测宽度-R2-5", "瀹炴祴瀹藉害-R2-5"]
SETTING_TARGET = "出口宽度设定值-R2-5"


def resolve_target_column(columns):
    for candidate in TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
    raise KeyError(f"Missing target column. Candidates: {TARGET_CANDIDATES}")


def evaluate_dataset(name, x_data, y_true, model):
    y_pred = model.predict(x_data)
    metrics = {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }

    print(f"\n{name}指标：")
    print(f"R方: {metrics['R2']:.4f}")
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


print("正在加载数据集...")
df_train = pd.read_excel(TRAIN_PATH)
df_val = pd.read_excel(VAL_PATH)
df_test = pd.read_excel(TEST_PATH)

for dataframe in [df_train, df_val, df_test]:
    dataframe.columns = dataframe.columns.str.strip()

target = resolve_target_column(df_train.columns)
features = [column for column in df_train.columns if column not in [target, SETTING_TARGET]]

X_train, y_train = df_train[features].to_numpy(), df_train[target].to_numpy()
X_val, y_val = df_val[features].to_numpy(), df_val[target].to_numpy()
X_test, y_test = df_test[features].to_numpy(), df_test[target].to_numpy()

print(f"数据准备完毕，LightGBM 接收 {X_train.shape[1]} 个特征。")
print(f"训练集 {len(X_train)} 条，验证集 {len(X_val)} 条，测试集 {len(X_test)} 条。")

print("\n开始训练 LightGBM 模型...")
model_lgb = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=7,
    num_leaves=24,
    min_child_samples=30,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_alpha=0.4,
    reg_lambda=2.,
    min_split_gain=0.03,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    deterministic=True,
    force_col_wise=True,
    bagging_seed=RANDOM_SEED,
    feature_fraction_seed=RANDOM_SEED,
    data_random_seed=RANDOM_SEED,
    verbosity=-1,
)

callbacks = [
    lgb.early_stopping(stopping_rounds=40, verbose=False),
    lgb.log_evaluation(period=50),
]

model_lgb.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="l1",
    callbacks=callbacks,
)

print(f"LightGBM best_iteration_: {model_lgb.best_iteration_}")

print("\n训练完成，开始在训练集、验证集、测试集上进行预测...")
train_pred, train_metrics = evaluate_dataset("训练集", X_train, y_train, model_lgb)
val_pred, val_metrics = evaluate_dataset("验证集", X_val, y_val, model_lgb)
test_pred, test_metrics = evaluate_dataset("测试集", X_test, y_test, model_lgb)

result_excel = build_excel_result(
    y_train=y_train,
    y_val=y_val,
    y_test=y_test,
    train_pred=train_pred,
    val_pred=val_pred,
    test_pred=test_pred,
)
excel_result_path = save_excel_with_fallback(
    result_excel,
    os.path.join(DATA_OUTPUT_DIR, "LightGBM_prediction_results.xlsx"),
)

csv_result_path = os.path.join(DATA_OUTPUT_DIR, "result_LightGBM.csv")
pd.DataFrame({"true_width": y_test, "pred_width_lightgbm": test_pred}).to_csv(
    csv_result_path,
    index=False,
    encoding="utf-8-sig",
)

plt.figure(figsize=(8, 8))
plt.scatter(y_test, test_pred, alpha=0.6, color="mediumseagreen", edgecolor="k")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    lw=2,
    label="理想拟合线",
)
plt.title("LightGBM 宽度预测性能可视化", fontsize=15, fontweight="bold")
plt.xlabel("真实宽度 (mm)", fontsize=12)
plt.ylabel("LightGBM 预测宽度 (mm)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
image_result_path = os.path.join(IMAGE_OUTPUT_DIR, "LightGBM_Prediction_Scatter.png")
plt.savefig(image_result_path, dpi=300)
plt.close()

print("\n结果文件已保存：")
print(f"1. {excel_result_path}")
print(f"2. {csv_result_path}")
print(f"3. {image_result_path}")
