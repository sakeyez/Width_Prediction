import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import warnings

# 忽略 LightGBM 的一些底层打印警告
warnings.filterwarnings('ignore')

# 设置中文字体 (方便画出好看的论文图表)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 统一数据底座读取 (保证和 MGH/Hybrid-2 绝对公平)
# ==========================================
base_dir = r"F:\asus\Desktop\毕业设计数据\数据表格"

print("⏳ 正在加载全局标准化数据集...")
train_path = os.path.join(base_dir, "Train_Data.xlsx")
val_path = os.path.join(base_dir, "Val_Data.xlsx")
test_path = os.path.join(base_dir, "Test_Data.xlsx")

df_train = pd.read_excel(train_path)
df_val = pd.read_excel(val_path)
df_test = pd.read_excel(test_path)

# 【防弹装甲】清洗表头隐形空格，防止 KeyError
df_train.columns = df_train.columns.str.strip()
df_val.columns = df_val.columns.str.strip()
df_test.columns = df_test.columns.str.strip()

target = '实测宽度-R2-5'
features = [col for col in df_train.columns if col != target]

# 直接提取二维矩阵 (LightGBM 不需要区分静态动态，直接喂全量特征)
X_train, y_train = df_train[features].values, df_train[target].values
X_val, y_val = df_val[features].values, df_val[target].values
X_test, y_test = df_test[features].values, df_test[target].values

print(f"✅ 数据准备完毕！LightGBM 接收 {X_train.shape[1]} 个混合物理特征。")
print(f"📦 统一试卷切分：训练集 {len(X_train)} 条，验证集 {len(X_val)} 条，测试集 {len(X_test)} 条。")

# ==========================================
# 2. 构建与训练 LightGBM 引擎
# ==========================================
print("\n🚀 启动 LightGBM 模型训练...")

model_lgb = lgb.LGBMRegressor(
    n_estimators=1000,     # 树的数量适当调大，靠 early_stopping 来拦截
    learning_rate=0.05,    # 学习率
    max_depth=7,           # 树的深度
    num_leaves=31,         # 叶子节点数 (LightGBM 核心参数)
    random_state=42,       # 保证复现
    n_jobs=-1              # CPU 火力全开
)

# 设立验证集监控，连续 50 棵树不进步就提前拔电源
callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=False),
    lgb.log_evaluation(period=100) # 每100棵树打印一次进度
]

model_lgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='l1', # l1等价于MAE
    callbacks=callbacks
)

# ==========================================
# 3. 在全新的 Test 集上进行大考评估
# ==========================================
y_pred = model_lgb.predict(X_test)

mae_val = mean_absolute_error(y_test, y_pred)
mse_val = mean_squared_error(y_test, y_pred)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(y_test, y_pred)
mape_val = mean_absolute_percentage_error(y_test, y_pred) * 100

print("\n" + "★"*50)
print(" 🏆 LightGBM 作为强基准模型的大考成绩单")
print("★"*50)
print(f"   - 均方误差 (MSE)      : {mse_val:.4f}")
print(f"   - 均方根误差 (RMSE)   : {rmse_val:.4f} 毫米")
print(f"   - 平均绝对误差 (MAE)  : {mae_val:.4f} 毫米")
print(f"   - 决定系数 (R²)       : {r2_val:.4f}")
print(f"   - 平均百分比误差(MAPE): {mape_val:.4f} %")
print("★"*50)

# ==========================================
# 4. 保存预测结果与可视化图表
# ==========================================
# 保存 CSV 成绩单
df_lgb_result = pd.DataFrame({
    '真实宽度_True': y_test,
    '预测宽度_LightGBM': y_pred
})
df_lgb_result.to_csv('result_LightGBM.csv', index=False, encoding='utf-8-sig')
print("\n💾 预测具体数值已保存为: result_LightGBM.csv")

# 画一张对角线散点对比图，为论文凑素材
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.6, color='mediumseagreen', edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='完美拟合线')
plt.title('LightGBM 宽展预测性能可视化', fontsize=15, fontweight='bold')
plt.xlabel('真实宽度 (mm)', fontsize=12)
plt.ylabel('LightGBM 预测宽度 (mm)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

plt.savefig('LightGBM_Prediction_Scatter.png', dpi=300)
print("📊 预测散点对比图已保存为: LightGBM_Prediction_Scatter.png")