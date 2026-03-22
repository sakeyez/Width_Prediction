import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, Multiply, Lambda, Softmax
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pyswarm import pso

# 1. 基础配置与数据加载
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

file_path = r"F:\asus\Desktop\毕业设计数据\数据表格\最终数据.xlsx"
df = pd.read_excel(file_path)
target = '实测宽度-R2-5'

# 特征工程 (保持与原版一致)
dynamic_cols = []
for i in range(1, 6):
    dynamic_cols.extend([f"R2-{i}压下量", f"R2轧制速度Pass{i}(H11_0)", f"平辊实际轧制力-R2-{i}"])
static_cols = [col for col in df.columns if col != target and col not in dynamic_cols]

X_static = df[static_cols].values
X_dynamic_flat = df[dynamic_cols].values
time_steps, features_per_step = 5, 3
X_dynamic = X_dynamic_flat.reshape(-1, time_steps, features_per_step)
y = df[target].values

# 数据切分
train_end, val_end = int(len(y) * 0.7), int(len(y) * 0.9)
X_stat_train, X_dyn_train, y_train = X_static[:train_end], X_dynamic[:train_end], y[:train_end]
X_stat_val, X_dyn_val, y_val = X_static[train_end:val_end], X_dynamic[train_end:val_end], y[train_end:val_end]
X_stat_test, X_dyn_test, y_test = X_static[val_end:], X_dynamic[val_end:], y[val_end:]


# --- 2. 定义适应度函数 (PSO调用的核心) ---
def fitness_function(params):
    """
    params: [lstm_units, dense_units, learning_rate]
    由 PSO 自动传入
    """
    lstm_units = int(params[0])
    dense_units = int(params[1])
    lr = params[2]

    # 构建模型架构
    input_static = Input(shape=(X_stat_train.shape[1],))
    dense_s = Dense(32, activation='relu')(input_static)
    dense_s = Dense(16, activation='relu')(dense_s)

    input_dynamic = Input(shape=(time_steps, features_per_step))
    lstm_out = LSTM(lstm_units, return_sequences=True)(input_dynamic)

    # Attention
    att_scores = Dense(1, activation='tanh')(lstm_out)
    att_weights = Softmax(axis=1)(att_scores)
    context = Multiply()([lstm_out, att_weights])
    context = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context)

    merged = Concatenate()([dense_s, context])
    dense_f = Dense(dense_units, activation='relu')(merged)
    output = Dense(1)(dense_f)

    model = Model(inputs=[input_static, input_dynamic], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')

    # PSO阶段：快速训练评估
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        [X_stat_train, X_dyn_train], y_train,
        validation_data=([X_stat_val, X_dyn_val], y_val),
        epochs=20, batch_size=64, verbose=0, callbacks=[es]
    )

    val_mse = min(history.history['val_loss'])
    print(f"DEBUG >> 粒子尝试: LSTM={lstm_units}, Dense={dense_units}, LR={lr:.4f} -> MSE: {val_mse:.4f}")
    return val_mse


# --- 3. 执行 PSO 寻优 ---
# 参数范围: [LSTM神经元, 融合层神经元, 学习率]
lb = [32, 16, 0.0005]  # 下界
ub = [128, 64, 0.01]  # 上界

print("\n" + "=" * 50)
print("🚀 正在通过粒子群算法(PSO)寻找热连轧预测最优超参数...")
print("=" * 50)

# swarmsize: 粒子数量(建议10-20), maxiter: 迭代次数(建议5-10)
best_params, best_score = pso(fitness_function, lb, ub, swarmsize=10, maxiter=5)

final_lstm_units = int(best_params[0])
final_dense_units = int(best_params[1])
final_lr = best_params[2]

print("\n" + "★" * 50)
print(f"🏆 寻优完成！最优参数组合：")
print(f"   - LSTM 神经元: {final_lstm_units}")
print(f"   - 融合层神经元: {final_dense_units}")
print(f"   - 学习率: {final_lr:.5f}")
print("★" * 50 + "\n")

# --- 4. 使用最优参数进行最终完整训练 ---
print("正在使用最优参数进行 150 轮完整训练...")

# 构建最终模型
input_s = Input(shape=(X_stat_train.shape[1],))
d_s = Dense(32, activation='relu')(input_s)
d_s = Dense(16, activation='relu')(d_s)

input_d = Input(shape=(time_steps, features_per_step))
l_out = LSTM(final_lstm_units, return_sequences=True)(input_d)
a_w = Softmax(axis=1, name='final_attention')(Dense(1, activation='tanh')(l_out))
c_v = Lambda(lambda x: tf.reduce_sum(x, axis=1))(Multiply()([l_out, a_w]))

m = Concatenate()([d_s, c_v])
d_f = Dense(final_dense_units, activation='relu')(m)
out = Dense(1)(d_f)

final_model = Model(inputs=[input_s, input_d], outputs=out)
final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=final_lr), loss='mse', metrics=['mae'])

es_final = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
final_model.fit(
    [X_stat_train, X_dyn_train], y_train,
    validation_data=([X_stat_val, X_dyn_val], y_val),
    epochs=150, batch_size=32, callbacks=[es_final], verbose=1
)

# --- 5. 结果评估与保存 ---
y_pred = final_model.predict([X_stat_test, X_dyn_test]).flatten()
mae_val = mean_absolute_error(y_test, y_pred)
mse_val = mean_squared_error(y_test, y_pred)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("最终性能报告")
print("="*50)
print(f"1. 平均绝对误差 (MAE)    : {mae_val:.4f} 毫米")
print(f"2. 均方误差 (MSE)        : {mse_val:.4f}")
print(f"3. 均方根误差 (RMSE)     : {rmse_val:.4f} 毫米")
print(f"4. 决定系数 (R-squared)  : {r2_val:.4f}")
print("="*50)

# 保存结果供论文对比
df_results = pd.DataFrame({
    'Real_Width': y_test.flatten(),
    'PSO_LSTM_Pred': y_pred.flatten()
})
df_results.to_csv('result_PSO_LSTM.csv', index=False)
print("✅ 优化后的预测成绩单已保存为: result_PSO_LSTM.csv")