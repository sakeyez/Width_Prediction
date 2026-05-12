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

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_output_dir = os.path.join(project_root, "data", "LightGBM")
image_output_dir = os.path.join(project_root, "image", "LightGBM")
os.makedirs(data_output_dir, exist_ok=True)
os.makedirs(image_output_dir, exist_ok=True)


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
        fallback_path = preferred_path.replace(".xlsx", "_new.xlsx")
        dataframe.to_excel(fallback_path, index=False)
        print(f"\n检测到 {preferred_path} 正在被占用，已改存为: {fallback_path}")
        return fallback_path


# ==========================================
# 1. 读取数据
# ==========================================
base_dir = os.path.join(project_root, "data", "datadeal")

print("正在加载全局标准化数据集...")
train_path = os.path.join(base_dir, "Train_Data.xlsx")
val_path = os.path.join(base_dir, "Val_Data.xlsx")
test_path = os.path.join(base_dir, "Test_Data.xlsx")

df_train = pd.read_excel(train_path)
df_val = pd.read_excel(val_path)
df_test = pd.read_excel(test_path)

df_train.columns = df_train.columns.str.strip()
df_val.columns = df_val.columns.str.strip()
df_test.columns = df_test.columns.str.strip()

target = "实测宽度-R2-5"
features = [col for col in df_train.columns if col != target]

X_train, y_train = df_train[features].values, df_train[target].values
X_val, y_val = df_val[features].values, df_val[target].values
X_test, y_test = df_test[features].values, df_test[target].values

print(f"数据准备完毕，LightGBM 接收 {X_train.shape[1]} 个特征。")
print(f"训练集 {len(X_train)} 条，验证集 {len(X_val)} 条，测试集 {len(X_test)} 条。")

# ==========================================
# 2. 构建并训练 LightGBM
# ==========================================
print("\n开始训练 LightGBM 模型...")

model_lgb = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    deterministic=True,
    force_col_wise=True,
    bagging_seed=RANDOM_SEED,
    feature_fraction_seed=RANDOM_SEED,
    data_random_seed=RANDOM_SEED,
)

callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=False),
    lgb.log_evaluation(period=100),
]

model_lgb.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="l1",
    callbacks=callbacks,
)

# ==========================================
# 3. 三个数据集分别预测并输出指标
# ==========================================
print("\n训练完成，开始在实验集、验证集、测试集上进行预测...")

train_pred, train_metrics = evaluate_dataset("实验集", X_train, y_train, model_lgb)
val_pred, val_metrics = evaluate_dataset("验证集", X_val, y_val, model_lgb)
test_pred, test_metrics = evaluate_dataset("测试集", X_test, y_test, model_lgb)

# ==========================================
# 4. 导出结果
# ==========================================
result_excel = build_excel_result(
    y_train=y_train,
    y_val=y_val,
    y_test=y_test,
    train_pred=train_pred,
    val_pred=val_pred,
    test_pred=test_pred,
)
excel_result_path = save_excel_with_fallback(
    result_excel, os.path.join(data_output_dir, "LightGBM_预测结果汇总.xlsx")
)

df_lgb_result = pd.DataFrame({"真实宽度_True": y_test, "预测宽度_LightGBM": test_pred})
csv_result_path = os.path.join(data_output_dir, "result_LightGBM.csv")
df_lgb_result.to_csv(csv_result_path, index=False, encoding="utf-8-sig")

# ==========================================
# 5. 测试集可视化
# ==========================================
plt.figure(figsize=(8, 8))
plt.scatter(y_test, test_pred, alpha=0.6, color="mediumseagreen", edgecolor="k")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    lw=2,
    label="完美拟合线",
)
plt.title("LightGBM 宽展预测性能可视化", fontsize=15, fontweight="bold")
plt.xlabel("真实宽度 (mm)", fontsize=12)
plt.ylabel("LightGBM 预测宽度 (mm)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
image_result_path = os.path.join(image_output_dir, "LightGBM_Prediction_Scatter.png")
plt.savefig(image_result_path, dpi=300)

print("\n结果文件已保存：")
print(f"1. {excel_result_path}")
print(f"2. {csv_result_path}")
print(f"3. {image_result_path}")
