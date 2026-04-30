import numpy as np
import pandas as pd
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 读取真实粗轧数据并划分测试集
# ==========================================
raw_data_path = r"F:\asus\Desktop\毕业设计数据\2160粗轧数据.xlsx"
tg_dir = r"F:\asus\Desktop\毕业设计数据\TGDNN模型"

print("⏳ 正在加载真实粗轧数据集以进行最终微调...")
df = pd.read_excel(raw_data_path)
df.columns = df.columns.str.strip()

feature_cols = [
    '实测宽度-R1', '出口厚度-R1',
    'R2-1压下量', 'R2轧制速度Pass1(H11_0)', '道次出口温度-R2-1', 'R2工作辊辊径',
    'R2-2压下量', 'R2轧制速度Pass2(H11_0)', '道次出口温度-R2-2', 'R2工作辊辊径',
    'R2-3压下量', 'R2轧制速度Pass3(H11_0)', '道次出口温度-R2-3', 'R2工作辊辊径.1',
    'R2-4压下量', 'R2轧制速度Pass4(H11_0)', '道次出口温度-R2-4', 'R2工作辊辊径.1',
    'R2-5压下量', 'R2轧制速度Pass5(H11_0)', '道次出口温度-R2-4', 'R2工作辊辊径.2'
]

X_real = df[feature_cols].values
y_real = df['实测宽度-R2-5'].values

# 划分 80% 作为真实数据的训练集(用于微调)，20% 作为从未见过的最终大考测试集
# 这里的划分方式保证和前面基础模型评价标准一致
X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size=0.2, random_state=42)

# ==========================================
# 2. 真实特征的归一化处理
# ==========================================
# 提取我们在预训练阶段使用过的 Scaler，保证物理刻度的绝对统一
scaler_X = joblib.load(os.path.join(tg_dir, 'Scaler_X.pkl'))

X_train_scaled = scaler_X.transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# ==========================================
# 3. 唤醒预训练模型，进行参数迁移与微调 (TG-DNN)
# ==========================================
print("🧠 正在唤醒 PR-DNN (满分毕业的物理高材生)，注入真实产线数据进行参数微调...")

# 加载上一步保存的预训练模型
tg_dnn = joblib.load(os.path.join(tg_dir, 'PR_DNN_Model.pkl'))

# 核心步骤：再次调用 fit()。
# 因为我们在上一步设置了 warm_start=True，所以它不会清空之前的物理权重！
# 它会从之前 1.0000 的物理最佳状态出发，朝着真实数据的方向进行小幅度移动
tg_dnn.fit(X_train_scaled, y_train)

# ==========================================
# 4. 在最终的测试集上进行大考
# ==========================================
y_pred = tg_dnn.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n🏆 TG-DNN 模型在最终测试集 (Test) 上的惊艳大考成绩：")
print(f"   - MSE : {mse:.4f}")
print(f"   - MAE : {mae:.4f}")
print(f"   - R²  : {r2:.4f}")
print("\n✅ 恭喜！你已成功跑通了基于理论物理指导的知识迁移深度学习架构！")

joblib.dump(tg_dnn, os.path.join(tg_dir, 'Final_TGDNN_Model.pkl'))

# 画出预测对比散点图
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.6, color='dodgerblue', edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('TG-DNN: 理论指导神经网络测试集预测效果', fontsize=15, fontweight='bold')
plt.xlabel('真实宽度 (mm)', fontsize=12)
plt.ylabel('预测宽度 (mm)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plot_path = os.path.join(tg_dir, "TGDNN_Prediction_Scatter.png")
plt.savefig(plot_path, dpi=300)