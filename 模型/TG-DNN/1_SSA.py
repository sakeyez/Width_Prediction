import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 读取包含 44 个参数的全局基础数据
# ==========================================
raw_data_path = r"F:\asus\Desktop\毕业设计数据\2160粗轧数据.xlsx"

tg_dir = r"F:\asus\Desktop\毕业设计数据\TGDNN模型"
if not os.path.exists(tg_dir):
    os.makedirs(tg_dir)

print("⏳ 正在加载原始粗轧数据集，这可能需要几秒钟...")
df = pd.read_excel(raw_data_path)

# 【核心防弹装甲】自动剥离所有列名首尾的隐形空格和回车符！
df.columns = df.columns.str.strip()

# 随机抽取 30% 真实数据用于理论模型寻优
n_samples = int(len(df) * 0.3)
df_opt = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
y_true = df_opt['实测宽度-R2-5'].values

print(f"📦 已加载 44 维全局数据集，抽取 {n_samples} 条真实数据准备寻优...")

# 提取各道次的物理向量 (将其转为 NumPy 数组以便极速运算)
B_in = df_opt['实测宽度-R1'].values  # 初始入口宽度
H_in = df_opt['出口厚度-R1'].values  # 初始入口厚度

# R2-1
dh1 = df_opt['R2-1压下量'].values
h1 = df_opt['出口厚度-R2-1'].values
v1 = df_opt['R2轧制速度Pass1(H11_0)'].values
t1 = df_opt['道次出口温度-R2-1'].values
R0_1 = df_opt['R2工作辊辊径'].values

# R2-2
dh2 = df_opt['R2-2压下量'].values
h2 = df_opt['出口厚度-R2-2'].values
v2 = df_opt['R2轧制速度Pass2(H11_0)'].values
t2 = df_opt['道次出口温度-R2-2'].values
R0_2 = df_opt['R2工作辊辊径'].values

# R2-3
dh3 = df_opt['R2-3压下量'].values
h3 = df_opt['出口厚度-R2-3'].values
v3 = df_opt['R2轧制速度Pass3(H11_0)'].values
t3 = df_opt['道次出口温度-R2-3'].values
R0_3 = df_opt['R2工作辊辊径.1'].values

# R2-4
dh4 = df_opt['R2-4压下量'].values
h4 = df_opt['出口厚度-R2-4'].values
v4 = df_opt['R2轧制速度Pass4(H11_0)'].values
t4 = df_opt['道次出口温度-R2-4'].values
R0_4 = df_opt['R2工作辊辊径.1'].values

# R2-5 (处理缺失项：出口厚度 = 上一道次厚度 - 本道次压下量)
dh5 = df_opt['R2-5压下量'].values
h5 = h4 - dh5
v5 = df_opt['R2轧制速度Pass5(H11_0)'].values
t5 = t4  # 借用上一道次温度
R0_5 = df_opt['R2工作辊辊径.2'].values


# ==========================================
# 2. 1:1 完美复刻 Tselikov 单道次理论宽展公式
# ==========================================
def tselikov_single_pass(B, H, h, dh, R0, v, t, coef):
    """
    单道次宽展计算：严格遵循论文 Eq(1) - Eq(4)
    coef 包含 9 个经验参数: a1, a2, a3, a4, k1, k2, eta1, eta2, eta3
    """
    a1, a2, a3, a4, k1, k2, eta1, eta2, eta3 = coef

    # 防止除零或负数引发物理报错
    dh = np.clip(dh, 1e-3, None)
    H = np.clip(H, 1e-3, None)

    # 压下率与形变系数
    epsilon = (H - h) / H
    phi = k1 + k2 * epsilon

    # 摩擦系数
    mu = eta1 - eta2 * t - eta3 * v

    # 接触弧与宽度比例系数 C
    term = B / np.sqrt(R0 * dh)
    C = a1 * (term - a2) * np.exp(np.clip(a3 - term, -50, 50)) + a4

    # 终极宽展计算公式
    delta_b = C * dh * np.sqrt(R0 / H) * phi * mu
    return delta_b


# 串联 5 个道次，击鼓传花累加宽展
def multi_pass_predict(coef):
    # Pass 1
    db1 = tselikov_single_pass(B_in, H_in, h1, dh1, R0_1, v1, t1, coef)
    B1_out = B_in + db1

    # Pass 2
    db2 = tselikov_single_pass(B1_out, h1, h2, dh2, R0_2, v2, t2, coef)
    B2_out = B1_out + db2

    # Pass 3
    db3 = tselikov_single_pass(B2_out, h2, h3, dh3, R0_3, v3, t3, coef)
    B3_out = B2_out + db3

    # Pass 4
    db4 = tselikov_single_pass(B3_out, h3, h4, dh4, R0_4, v4, t4, coef)
    B4_out = B3_out + db4

    # Pass 5
    db5 = tselikov_single_pass(B4_out, h4, h5, dh5, R0_5, v5, t5, coef)
    B5_final = B4_out + db5

    return B5_final


# ==========================================
# 3. 麻雀搜索算法 (SSA) 寻找 9 位核心参数
# ==========================================
# 严格导入论文 Table 1 中的寻优边界上下限
lb = np.array([1.005, 0.1125, 0.1125, 0.375, 0.1035, 0.246, 0.7875, 0.000375, 0.42])
ub = np.array([1.675, 0.1875, 0.1875, 0.625, 0.1725, 0.410, 1.3125, 0.000625, 0.70])

dim = 9
pop_size = 30
max_iter = 100


def ssa_theory_optimization():
    # 在指定物理边界内初始化种群
    X_pop = np.random.uniform(lb, ub, (pop_size, dim))

    def calc_fitness(pop):
        fits = np.zeros(pop_size)
        for i in range(pop_size):
            y_pred = multi_pass_predict(pop[i])
            fits[i] = np.mean(np.abs(y_pred - y_true))  # MAE
        return fits

    fitness = calc_fitness(X_pop)
    pBest = X_pop.copy()
    pBest_fit = fitness.copy()
    gBest = X_pop[np.argmin(fitness)].copy()
    gBest_fit = np.min(fitness)

    PD, SD = 0.2, 0.2
    p_num = int(pop_size * PD)
    s_num = int(pop_size * SD)

    loss_history = []

    print("\n🕊️ 启动麻雀搜索算法 (SSA) 深度优化 Tselikov 理论公式 9 大参数...")
    for t in range(max_iter):
        sort_idx = np.argsort(fitness)
        X_pop = X_pop[sort_idx]
        fitness = fitness[sort_idx]

        best_worst_idx = np.argmax(fitness)
        worst_pos = X_pop[best_worst_idx].copy()

        # 1. 发现者
        R2 = np.random.rand()
        for i in range(p_num):
            if R2 < 0.8:
                X_pop[i] = X_pop[i] * np.exp(-i / (np.random.rand() * max_iter + 1e-8))
            else:
                X_pop[i] = X_pop[i] + np.random.randn()

        # 2. 加入者
        for i in range(p_num, pop_size):
            if i > pop_size / 2:
                X_pop[i] = np.random.randn() * np.exp((worst_pos - X_pop[i]) / (i ** 2 + 1e-8))
            else:
                A = np.random.choice([-1, 1], size=dim)
                A_plus = A / dim
                X_pop[i] = X_pop[0] + np.abs(X_pop[i] - X_pop[0]) * A_plus

        # 3. 侦察者
        scout_idx = np.random.choice(pop_size, s_num, replace=False)
        for i in scout_idx:
            if fitness[i] > gBest_fit:
                X_pop[i] = gBest + np.random.randn() * np.abs(X_pop[i] - gBest)
            else:
                X_pop[i] = X_pop[i] + (np.random.choice([-1, 1]) * np.abs(X_pop[i] - worst_pos)) / (
                            fitness[i] - fitness[best_worst_idx] + 1e-8)

        # 强制约束：绝不允许任何参数越过论文规定的物理边界
        X_pop = np.clip(X_pop, lb, ub)

        fitness = calc_fitness(X_pop)
        for i in range(pop_size):
            if fitness[i] < pBest_fit[i]:
                pBest_fit[i] = fitness[i]
                pBest[i] = X_pop[i].copy()

        if np.min(pBest_fit) < gBest_fit:
            gBest_fit = np.min(pBest_fit)
            gBest = pBest[np.argmin(pBest_fit)].copy()

        loss_history.append(gBest_fit)

        if (t + 1) % 10 == 0:
            print(f"   ▶ 第 {t + 1:03d} 代 | 理论模型最佳 MAE: {gBest_fit:.4f} mm")

    return gBest, loss_history


# ==========================================
# 4. 执行寻优并保存结果
# ==========================================
best_coef, history = ssa_theory_optimization()

print(f"\n🎉 寻优完成！最终 Tselikov 理论公式预测 MAE 降至: {history[-1]:.4f} mm")
print("🌟 找出的 9 位最佳物理参数 (a1, a2, a3, a4, k1, k2, eta1, eta2, eta3) 分别为：")
print(np.round(best_coef, 6))

np.save(os.path.join(tg_dir, 'SSA_Tselikov_Coef.npy'), best_coef)
print(f"💾 物理核心参数已保存至: {tg_dir}\\SSA_Tselikov_Coef.npy")

plt.figure(figsize=(8, 5))
plt.plot(history, linewidth=2, color='darkred')
plt.title('SSA 优化 Tselikov 多道次理论模型收敛曲线', fontsize=15, fontweight='bold')
plt.xlabel('迭代次数', fontsize=12)
plt.ylabel('平均绝对误差 MAE', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plot_path = os.path.join(tg_dir, "SSA_Tselikov_Convergence.png")
plt.savefig(plot_path, dpi=300)