import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import BernoulliRBM, MLPRegressor
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 统一数据底座读取
# ==========================================
base_dir = r"F:\asus\Desktop\毕业设计数据\数据表格"

print("⏳ 正在加载全局标准化数据集...")
train_path = os.path.join(base_dir, "Train_Data.xlsx")
val_path = os.path.join(base_dir, "Val_Data.xlsx")
test_path = os.path.join(base_dir, "Test_Data.xlsx")

df_train = pd.read_excel(train_path)
df_val = pd.read_excel(val_path)
df_test = pd.read_excel(test_path)

# 清洗表头空格
df_train.columns = df_train.columns.str.strip()
df_val.columns = df_val.columns.str.strip()
df_test.columns = df_test.columns.str.strip()

target = '实测宽度-R2-5'
features = [col for col in df_train.columns if col != target]

X_train, y_train = df_train[features].values, df_train[target].values
X_test, y_test = df_test[features].values, df_test[target].values

# ==========================================
# 2. 改进版双向数据归一化 (拯救神经元的关键)
# ==========================================
print("⚖️ 正在实施改良版数据归一化 (X 防坍塌 + Y 防爆炸)...")

# 【关键改进 1】：将输入 X 压缩到 0.1~0.9，防止触发 RBM 的 Sigmoid 死亡区
scaler_X = MinMaxScaler(feature_range=(0.1, 0.9))
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# 将输出 Y 标准化 (均值0，方差1)
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()


# 3. 抢救版 DBN 流水线构建

print("\n🧠 启动改进版纯粹 DBN 架构：带隐层恢复的深度置信网络...")

# 第一层：RBM 预训练 (无监督特征提取)
rbm = BernoulliRBM(
    n_components=128,
    learning_rate=0.001,   # 极小的学习率，防止盲目破坏物理连续性
    n_iter=20,             # 浅尝辄止的预训练，防止特征过度二值化
    random_state=42,
    verbose=False
)

# 【关键改进 2】：隐层拯救者 (心脏起搏器)！
# 在 RBM 破坏了方差之后，强行在流水线中间插入一个 StandardScaler 恢复数据的正态分布特征
scaler_hidden = StandardScaler()

# 第三层：MLP 微调 (有监督回归)
mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=0.005,
    max_iter=500,
    early_stopping=True,
    random_state=42
)

# 组装超级流水线： RBM -> 中间态恢复 -> MLP
dbn_model = Pipeline(steps=[
    ('rbm', rbm),
    ('scaler_hidden', scaler_hidden),
    ('mlp', mlp)
])

# 开始端到端训练
dbn_model.fit(X_train_scaled, y_train_scaled)

# ==========================================
# 4. 预测与反向放大
# ==========================================
print("\n🎯 训练结束！正在 Test 测试集上大考...")

# 预测出的结果是压缩状态的，需要反向放大回真实的 2000mm 尺度
y_pred_scaled = dbn_model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

mae_val = mean_absolute_error(y_test, y_pred)
mse_val = mean_squared_error(y_test, y_pred)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(y_test, y_pred)
mape_val = mean_absolute_percentage_error(y_test, y_pred) * 100

print("\n" + "★"*50)
print(" 🎓 纯数据驱动 DBN (基准对照组) 成绩单")
print("★"*50)
print(f"   - 均方误差 (MSE)      : {mse_val:.4f}")
print(f"   - 均方根误差 (RMSE)   : {rmse_val:.4f} 毫米")
print(f"   - 平均绝对误差 (MAE)  : {mae_val:.4f} 毫米")
print(f"   - 决定系数 (R²)       : {r2_val:.4f}")
print(f"   - 平均百分比误差(MAPE): {mape_val:.4f} %")
print("★"*50)

# ==========================================
# 5. 保存结果与可视化
# ==========================================
df_dbn_result = pd.DataFrame({'真实宽度_True': y_test, '预测宽度_DBN': y_pred})
df_dbn_result.to_csv('result_DBN_Pure.csv', index=False, encoding='utf-8-sig')

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.6, color='mediumpurple', edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='完美拟合线')
plt.title('纯 DBN 模型宽展预测性能可视化 (对照组)', fontsize=15, fontweight='bold')
plt.xlabel('真实宽度 (mm)', fontsize=12)
plt.ylabel('DBN 预测宽度 (mm)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('DBN_Pure_Prediction_Scatter.png', dpi=300)
print("\n✅ 纯净版 DBN 流程圆满结束！散点图与 CSV 已保存！")