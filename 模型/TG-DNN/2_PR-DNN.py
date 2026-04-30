import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 路径与数据准备
# ==========================================
raw_data_path = r"F:\asus\Desktop\毕业设计数据\2160粗轧数据.xlsx"
tg_dir = r"F:\asus\Desktop\毕业设计数据\TGDNN模型"

print("⏳ 正在加载基础数据以获取物理边界...")
df = pd.read_excel(raw_data_path)
df.columns = df.columns.str.strip()

# 提取特征列名 (共 22 个纯物理输入特征)
feature_cols = [
    '实测宽度-R1', '出口厚度-R1',
    'R2-1压下量', 'R2轧制速度Pass1(H11_0)', '道次出口温度-R2-1', 'R2工作辊辊径',
    'R2-2压下量', 'R2轧制速度Pass2(H11_0)', '道次出口温度-R2-2', 'R2工作辊辊径',  # 注意此处原表可能有重名列，以你的真实表头为准
    'R2-3压下量', 'R2轧制速度Pass3(H11_0)', '道次出口温度-R2-3', 'R2工作辊辊径.1',
    'R2-4压下量', 'R2轧制速度Pass4(H11_0)', '道次出口温度-R2-4', 'R2工作辊辊径.1',
    'R2-5压下量', 'R2轧制速度Pass5(H11_0)', '道次出口温度-R2-4', 'R2工作辊辊径.2'  # R2-5借用R2-4温度
]

# 获取这 22 个物理特征的真实上下限
X_real = df[feature_cols].values
X_min = X_real.min(axis=0)
X_max = X_real.max(axis=0)

# 加载刚才第一步跑出来的 9 个最优物理参数
best_coef = np.load(os.path.join(tg_dir, 'SSA_Tselikov_Coef.npy'))

# ==========================================
# 2. 凭空造物：生成 30,000 条高保真物理数据
# ==========================================
N_VIRTUAL = 30000
print(f"🏭 正在基于 Tselikov 物理机理公式，随机生成 {N_VIRTUAL} 条虚拟轧制数据...")

np.random.seed(42)
# 在真实的 min 和 max 之间，随机生成 30000 行 22 列的特征矩阵
X_virtual = np.random.uniform(X_min, X_max, (N_VIRTUAL, len(feature_cols)))


# 重新定义单道次公式 (用于生成数据)
def tselikov_single_pass(B, H, h, dh, R0, v, t, coef):
    a1, a2, a3, a4, k1, k2, eta1, eta2, eta3 = coef
    dh = np.clip(dh, 1e-3, None)
    H = np.clip(H, 1e-3, None)
    epsilon = (H - h) / H
    phi = k1 + k2 * epsilon
    mu = eta1 - eta2 * t - eta3 * v
    term = B / np.sqrt(R0 * dh)
    C = a1 * (term - a2) * np.exp(np.clip(a3 - term, -50, 50)) + a4
    return C * dh * np.sqrt(R0 / H) * phi * mu


# 批量计算 30000 条数据的虚拟出口宽度
y_virtual = np.zeros(N_VIRTUAL)

for i in range(N_VIRTUAL):
    row = X_virtual[i]
    B_in, H_in = row[0], row[1]

    # 逐道次累加
    h1 = H_in - row[2]
    db1 = tselikov_single_pass(B_in, H_in, h1, row[2], row[5], row[3], row[4], best_coef)
    B1 = B_in + db1

    h2 = h1 - row[6]
    db2 = tselikov_single_pass(B1, h1, h2, row[6], row[9], row[7], row[8], best_coef)
    B2 = B1 + db2

    h3 = h2 - row[10]
    db3 = tselikov_single_pass(B2, h2, h3, row[10], row[13], row[11], row[12], best_coef)
    B3 = B2 + db3

    h4 = h3 - row[14]
    db4 = tselikov_single_pass(B3, h3, h4, row[14], row[17], row[15], row[16], best_coef)
    B4 = B3 + db4

    h5 = h4 - row[18]
    db5 = tselikov_single_pass(B4, h4, h5, row[18], row[21], row[19], row[20], best_coef)

    y_virtual[i] = B4 + db5

print("✅ 虚拟物理数据集生成完毕！")

# ==========================================
# 3. 数据归一化 (MinMaxScaler)
# ==========================================
# 严格按照原论文 Eq(11) 使用 MinMaxScaler 将输入压缩到 0-1 之间
scaler_X = MinMaxScaler()
X_virtual_scaled = scaler_X.fit_transform(X_virtual)

# 保存 scaler 供最后一步使用
joblib.dump(scaler_X, os.path.join(tg_dir, 'Scaler_X.pkl'))

# ==========================================
# 4. 预训练 DNN (PR-DNN)
# ==========================================
print("🧠 正在使用 3 万条物理数据预训练深度神经网络 (这可能需要一两分钟)...")

# 构建结构为 25-10 的隐藏层 (加上输入输出刚好符合原论文的 22-25-10-1)
# 开启 warm_start=True 是参数迁移策略的核心！允许我们在下一步继续微调它
pr_dnn = MLPRegressor(
    hidden_layer_sizes=(25, 10),
    activation='relu',
    solver='adam',
    max_iter=500,
    learning_rate_init=0.005,
    warm_start=True,  # 关键参数：开启记忆功能，便于后续迁移学习
    random_state=42
)

pr_dnn.fit(X_virtual_scaled, y_virtual)

# 评估预训练模型在自身虚拟数据上的拟合度
score = pr_dnn.score(X_virtual_scaled, y_virtual)
print(f"🎉 预训练完成！PR-DNN 对物理公式的拟合优度 (R²) 达到了: {score:.4f}")

# 保存预训练模型
joblib.dump(pr_dnn, os.path.join(tg_dir, 'PR_DNN_Model.pkl'))
print(f"💾 预训练完成的 PR-DNN 已保存至: {tg_dir}\\PR_DNN_Model.pkl")