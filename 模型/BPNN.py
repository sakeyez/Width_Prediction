import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. 读取并重组数据 (与 LightGBM 完全一致的 2D 拍平数据)
# ==========================================
file_path = r"F:\asus\Desktop\毕业设计数据\数据表格\最终数据.xlsx"
df = pd.read_excel(file_path)
target = '实测宽度-R2-5'
all_features = [col for col in df.columns if col != target]

dynamic_cols = []
for i in range(1, 6):
    dynamic_cols.extend([f"R2-{i}压下量", f"R2轧制速度Pass{i}(H11_0)", f"平辊实际轧制力-R2-{i}"])

static_cols = [col for col in all_features if col not in dynamic_cols]

X_static = df[static_cols].values
X_dynamic_flat = df[dynamic_cols].values
y = df[target].values

# 【核心：强行拍平为二维，剥夺时间维度】
X_2D = np.hstack((X_static, X_dynamic_flat))

# 7:2:1 切分
total = len(X_2D)
train_end, val_end = int(total * 0.7), int(total * 0.9)

X_train, y_train = X_2D[:train_end], y[:train_end]
X_val, y_val = X_2D[train_end:val_end], y[train_end:val_end]
X_test, y_test = X_2D[val_end:], y[val_end:]

# ==========================================
# 2. 搭建 BP 神经网络 (纯全连接层 MLP)
# ==========================================
print("\n🚀 正在启动 BP 神经网络 (MLP) 训练...")

bp_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, name='Output_Width')
])

bp_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 连续 15 轮验证集不下降就提前停止，防止过拟合
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

bp_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# ==========================================
# 3. 最终期末大考
# ==========================================
y_pred_bp = bp_model.predict(X_test).flatten()

mae_val = mean_absolute_error(y_test, y_pred_bp)
mse_val = mean_squared_error(y_test, y_pred_bp)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(y_test, y_pred_bp)
mape_val = mean_absolute_percentage_error(y_test, y_pred_bp) * 100


print("BP 神经网络")
print(f"1. 平均绝对误差 (MAE)        : {mae_val:.4f} 毫米")
print(f"2. 均方误差 (MSE)            : {mse_val:.4f} ")
print(f"3. 均方根误差 (RMSE)         : {rmse_val:.4f} 毫米")
print(f"4. 决定系数 (R-squared)      : {r2_val:.4f}")
print(f"5. 平均百分比误差 (MAPE)     : {mape_val:.4f} %")


df_bp_result = pd.DataFrame({
    'y_test': y_test.flatten(),
    'y_pred': y_pred_bp.flatten()
})
df_bp_result.to_csv('result_BPNN.csv', index=False)
print("✅ BP 神经网络预测成绩单已保存为: result_BPNN.csv")
