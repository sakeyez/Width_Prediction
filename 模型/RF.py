import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


file_path = r"F:\asus\Desktop\毕业设计数据\数据表格\最终数据.xlsx"
df = pd.read_excel(file_path)

# 提取初始板坯宽度
initial_width_col = '板坯宽度实测值(热态)'
target_col = '实测宽度-R2-5'


# y设置为宽展量 = 最终宽度 - 初始宽度
y_spread = df[target_col].values - df[initial_width_col].values

# 提取特征
all_features = [col for col in df.columns if col != target_col]
dynamic_cols = []
for i in range(1, 6):
    dynamic_cols.extend([f"R2-{i}压下量", f"R2轧制速度Pass{i}(H11_0)", f"平辊实际轧制力-R2-{i}"])

static_cols = [col for col in all_features if col not in dynamic_cols]

X_static = df[static_cols].values
X_dynamic_flat = df[dynamic_cols].values
X_2D = np.hstack((X_static, X_dynamic_flat))

# 7:2:1 切分
total_samples = len(X_2D)
train_end = int(total_samples * 0.7)
val_end = int(total_samples * 0.9)

X_train, y_train_spread = X_2D[:train_end], y_spread[:train_end]
X_test, y_test_spread = X_2D[val_end:], y_spread[val_end:]

# 保存测试集的初始宽度和最终绝对宽度，用于最后计算
initial_width_test = df[initial_width_col].values[val_end:]
y_test_absolute = df[target_col].values[val_end:]

# 训练
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)

rf_model.fit(X_train, y_train_spread)


# 预测
# 模型预测出这块钢板会变宽多少
y_pred_spread = rf_model.predict(X_test)

# 还原
y_pred_absolute = initial_width_test + y_pred_spread

# 还原计算
mae_val = mean_absolute_error(y_test_absolute, y_pred_absolute)
mse_val = mean_squared_error(y_test_absolute, y_pred_absolute)
r2_val = r2_score(y_test_absolute, y_pred_absolute)

print("\n" + "="*50)
print("随机森林")
print("="*50)
print(f"1. 平均绝对误差 (MAE)        : {mae_val:.4f} 毫米")
print(f"2. 均方误差 (MSE)            : {mse_val:.4f} ")
print(f"3. 决定系数 (R-squared)      : {r2_val:.4f}")
print("="*50)


df_rf_result = pd.DataFrame({
    'y_test': y_test_absolute.flatten(),
    'y_pred': y_pred_absolute.flatten()
})
df_rf_result.to_csv('result_RF.csv', index=False)
print("随机森林数据保存为: result_RF.csv")