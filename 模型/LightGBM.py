import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

#读取数据
file_path = r"F:\asus\Desktop\毕业设计数据\数据表格\最终数据.xlsx"
df = pd.read_excel(file_path)
target = '实测宽度-R2-5'
all_features = [col for col in df.columns if col != target]

# 提取动态特征列名
dynamic_cols = []
for i in range(1, 6):
    dynamic_cols.extend([
        f"R2-{i}压下量",
        f"R2轧制速度Pass{i}(H11_0)",
        f"平辊实际轧制力-R2-{i}"
    ])

# 提取静态特征列名
static_cols = [col for col in all_features if col not in dynamic_cols]

# 提取矩阵，与lstm写法同步
X_static = df[static_cols].values
X_dynamic_flat = df[dynamic_cols].values
y = df[target].values

# 将数据转为2D
X_2D = np.hstack((X_static, X_dynamic_flat))
print(f"数据准备完毕！LightGBM 接收 {X_2D.shape[1]} 个混合特征。")


# 7:2:1
total_samples = len(X_2D)
train_end = int(total_samples * 0.7)
val_end = int(total_samples * 0.9)

X_train, y_train = X_2D[:train_end], y[:train_end]
X_val, y_val = X_2D[train_end:val_end], y[train_end:val_end]
X_test, y_test = X_2D[val_end:], y[val_end:]

print(f"切分完毕：训练集 {len(X_train)} 条，验证集 {len(X_val)} 条，测试集 {len(X_test)} 条。")


# 4. 引入LightGBM

# 设置模型参数 (类似于 LSTM 的神经元数量等)
model_lgb = lgb.LGBMRegressor(
    n_estimators=500,      # 最多允许造 500 棵树
    learning_rate=0.05,    # 学习率
    max_depth=7,           # 树的深度
    random_state=42,       # 保证每次跑结果一样
    n_jobs=-1              # 火力全开，调用 CPU 所有核心
)

# 设立验证集监控（相当于 early_stopping），连续 20 棵树不进步就提前拔电源
callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=True)]

model_lgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='l1',
    callbacks=callbacks
)


# 测试集

y_pred = model_lgb.predict(X_test)

mae_val = mean_absolute_error(y_test, y_pred)
mse_val = mean_squared_error(y_test, y_pred)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(y_test, y_pred)
mape_val = mean_absolute_percentage_error(y_test, y_pred) * 100

print("\n" + "="*50)
print("="*50)
print(f"1. 平均绝对误差 (MAE)        : {mae_val:.4f} 毫米")
print(f"2. 均方误差 (MSE)            : {mse_val:.4f} ")
print(f"3. 均方根误差 (RMSE)         : {rmse_val:.4f} 毫米")
print(f"4. 决定系数 (R-squared)      : {r2_val:.4f}")
print(f"5. 平均百分比误差 (MAPE)     : {mape_val:.4f} %")
print("="*50)


df_lgb_result = pd.DataFrame({
    'y_test': y_test.flatten(),
    'y_pred': y_pred.flatten() # 如果你的变量名是 y_pred_lgb，这里就改成 y_pred_lgb.flatten()
})
df_lgb_result.to_csv('result_LightGBM.csv', index=False)
print("LightGBM 预测数据保存为: result_LightGBM.csv")