import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, Multiply, Lambda, Softmax
from tensorflow.keras.callbacks import EarlyStopping

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


#读取数据
file_path = r"F:\asus\Desktop\毕业设计数据\数据表格\最终数据.xlsx"
df = pd.read_excel(file_path)
target = '实测宽度-R2-5'
all_features = [col for col in df.columns if col != target]

# 选择动态特征
dynamic_cols = []
for i in range(1, 6):
    dynamic_cols.extend([
        f"R2-{i}压下量",
        f"R2轧制速度Pass{i}(H11_0)",
        f"平辊实际轧制力-R2-{i}"
    ])

# 定义静态特征
static_cols = [col for col in all_features if col not in dynamic_cols]



# 提取矩阵
X_static = df[static_cols].values
X_dynamic_flat = df[dynamic_cols].values
time_steps = 5  # 5个道次
features_per_step = 3 # 每道次3个参数

# 转为3维
X_dynamic = X_dynamic_flat.reshape(-1, time_steps, features_per_step)
print("\n" + "="*50)
print("="*50)
for i in range(time_steps):
    # 利用切片，把排好队的特征名按道次切开打印
    cols_in_step = dynamic_cols[i * features_per_step : (i + 1) * features_per_step]
    print(f"▶ 时间步 {i+1} (对应 R2-{i+1} 道次) 喂入的 {features_per_step} 个参数为：")
    for idx, col in enumerate(cols_in_step):
        print(f"  ├─ {idx+1}. {col}")
print("="*50 + "\n")
y = df[target].values

print(f"静态特征 {len(static_cols)} 个，动态特征每个道次 {features_per_step} 个。")

#切分
total_samples = len(X_static)
train_end = int(total_samples * 0.7)          # 70% 的分界线
val_end = int(total_samples * 0.9)            # 90% 的分界线

# 训练集
X_stat_train, X_dyn_train, y_train = X_static[:train_end], X_dynamic[:train_end], y[:train_end]
# 验证集
X_stat_val, X_dyn_val, y_val = X_static[train_end:val_end], X_dynamic[train_end:val_end], y[train_end:val_end]
# 测试集
X_stat_test, X_dyn_test, y_test = X_static[val_end:], X_dynamic[val_end:], y[val_end:]

print(f"训练集: {len(y_train)} 条, 验证集: {len(y_val)} 条, 测试集: {len(y_test)} 条")


# 静态特征处理，输出16个神经元
input_static = Input(shape=(X_stat_train.shape[1],), name='Static_Input')
dense_static = Dense(32, activation='relu')(input_static)
dense_static = Dense(16, activation='relu')(dense_static)

# 动态特征处理，拥有64个记忆细胞，可以逐步输出
input_dynamic = Input(shape=(time_steps, features_per_step), name='Dynamic_Input')
lstm_out = LSTM(64, return_sequences=True)(input_dynamic)

# 注意力机制打分
attention_scores = Dense(1, activation='tanh')(lstm_out)
# 换算百分比
attention_weights = Softmax(axis=1, name='attention_weights')(attention_scores)
# 应用系数
context_vector = Multiply()([lstm_out, attention_weights])
# 求和压缩
context_vector = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)

# 融合动静态
merged = Concatenate()([dense_static, context_vector])
dense_final = Dense(32, activation='relu')(merged)
output = Dense(1, name='Width_Output')(dense_final)

# 设定考核指数
model = Model(inputs=[input_static, input_dynamic], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])


# 训练模型
# 监控验证集
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

print("\n--- 开始训练模型 ---")
history = model.fit(
    [X_stat_train, X_dyn_train], y_train, #训练
    validation_data=([X_stat_val, X_dyn_val], y_val), #验证
    epochs=150,  #轮数
    batch_size=32, #每次训练样本数
    callbacks=[early_stopping], #检测
    verbose=1 #进度展示
)

# 测试
# flatten() 把矩阵拍平变成一维数组，方便和真实答案比对
y_pred = model.predict([X_stat_test, X_dyn_test]).flatten() #不加入宽度验证，直接预测


mae_val = mean_absolute_error(y_test, y_pred) #MAE
mse_val = mean_squared_error(y_test, y_pred) #MSE
rmse_val = np.sqrt(mse_val) #RMSE
r2_val = r2_score(y_test, y_pred) #R²

print(f"1. 平均绝对误差 (MAE)        : {mae_val:.4f} 毫米")
print(f"2. 均方误差 (MSE)            : {mse_val:.4f} ")
print(f"3. 均方根误差 (RMSE)         : {rmse_val:.4f} 毫米")
print(f"5. 决定系数 (R-squared)      : {r2_val:.4f}")



# 用最终训练好的模型提取所有测试集钢板的权重
final_attention_model = Model(inputs=model.input, outputs=model.get_layer('attention_weights').output)
# 一次性预测测试集里所有的钢板
all_test_weights = final_attention_model.predict([X_stat_test, X_dyn_test], verbose=0)

# 将形状 (样本数, 5, 1) 转换并计算每个道次的平均值
# axis=0 表示跨越所有的钢板样本求平均
avg_weights = np.mean(all_test_weights, axis=0).flatten()

print("注意力分配")
print(f"  ├─ R2-1 整体重要性: {avg_weights[0]*100:>5.2f}%")
print(f"  ├─ R2-2 整体重要性: {avg_weights[1]*100:>5.2f}%")
print(f"  ├─ R2-3 整体重要性: {avg_weights[2]*100:>5.2f}%")
print(f"  ├─ R2-4 整体重要性: {avg_weights[3]*100:>5.2f}%")
print(f"  ├─ R2-5 整体重要性: {avg_weights[4]*100:>5.2f}%")

df_lstm_result = pd.DataFrame({
    'y_test': y_test.flatten(),
    'y_pred': y_pred.flatten()
})
df_lstm_result.to_csv('result_LSTM.csv', index=False)
print("✅ LSTM 预测成绩单已保存为: result_LSTM.csv")

# ---------------------------------------------------------
# 💾 存档操作 2：保存 LSTM 的注意力权重矩阵 (用于画热力图)
# ---------------------------------------------------------
# 假设你之前提取的注意力权重变量叫 all_test_weights
test_weights_2d = all_test_weights.squeeze(-1) # 压平为 (样本数, 5)

df_attention = pd.DataFrame(test_weights_2d, columns=['R2-1', 'R2-2', 'R2-3', 'R2-4', 'R2-5'])
df_attention.to_csv('attention_LSTM.csv', index=False)
print("✅ LSTM 注意力指纹已保存为: attention_LSTM.csv")

