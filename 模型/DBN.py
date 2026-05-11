import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import BernoulliRBM, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RANDOM_SEED = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def evaluate_dataset(name, X_scaled, y_true, model, scaler_y):
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

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
base_dir = r"F:\asus\Desktop\毕业设计数据\数据表格"

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

# ==========================================
# 2. 数据预处理
# ==========================================
print("正在进行数据归一化与标准化...")

scaler_X = MinMaxScaler(feature_range=(0.1, 0.9))
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

# ==========================================
# 3. 构建并训练 DBN
# ==========================================
print("\n开始训练 DBN 模型...")

rbm = BernoulliRBM(
    n_components=128,
    learning_rate=0.001,
    n_iter=20,
    random_state=RANDOM_SEED,
    verbose=False,
)

scaler_hidden = StandardScaler()

mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    learning_rate_init=0.005,
    max_iter=500,
    early_stopping=True,
    random_state=RANDOM_SEED,
    shuffle=False,
)

dbn_model = Pipeline(
    steps=[
        ("rbm", rbm),
        ("scaler_hidden", scaler_hidden),
        ("mlp", mlp),
    ]
)

dbn_model.fit(X_train_scaled, y_train_scaled)

# ==========================================
# 4. 三个数据集分别预测并输出指标
# ==========================================
print("\n训练完成，开始在实验集、验证集、测试集上进行预测...")

train_pred, train_metrics = evaluate_dataset(
    "实验集", X_train_scaled, y_train, dbn_model, scaler_y
)
val_pred, val_metrics = evaluate_dataset("验证集", X_val_scaled, y_val, dbn_model, scaler_y)
test_pred, test_metrics = evaluate_dataset(
    "测试集", X_test_scaled, y_test, dbn_model, scaler_y
)

# ==========================================
# 5. 导出结果
# ==========================================
result_excel = build_excel_result(
    y_train=y_train,
    y_val=y_val,
    y_test=y_test,
    train_pred=train_pred,
    val_pred=val_pred,
    test_pred=test_pred,
)
excel_result_path = save_excel_with_fallback(result_excel, "DBN_预测结果汇总.xlsx")

df_test_result = pd.DataFrame({"真实宽度_True": y_test, "预测宽度_DBN": test_pred})
df_test_result.to_csv("result_DBN_Pure.csv", index=False, encoding="utf-8-sig")

# ==========================================
# 6. 测试集可视化
# ==========================================
plt.figure(figsize=(8, 8))
plt.scatter(y_test, test_pred, alpha=0.6, color="mediumpurple", edgecolor="k")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    lw=2,
    label="完美拟合线",
)
plt.title("纯 DBN 模型宽展预测性能可视化", fontsize=15, fontweight="bold")
plt.xlabel("真实宽度 (mm)", fontsize=12)
plt.ylabel("DBN 预测宽度 (mm)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("DBN_Pure_Prediction_Scatter.png", dpi=300)

print("\n结果文件已保存：")
print(f"1. {excel_result_path}")
print("2. result_DBN_Pure.csv")
print("3. DBN_Pure_Prediction_Scatter.png")
