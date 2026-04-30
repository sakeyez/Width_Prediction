import pandas as pd
import numpy as np
import os
import pickle
import warnings

warnings.filterwarnings('ignore')

#加载
mgh_dir = r"F:\asus\Desktop\毕业设计数据\MGH模型"

data_path = os.path.join(mgh_dir, "MGH_Test_Expanded.xlsx")
model_path = os.path.join(mgh_dir, 'final_glr_model.pkl')

df_data = pd.read_excel(data_path)

with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

final_model = model_data['model']
selected_features = model_data['selected_features']

#找到关键参数
target_col = '出口宽度设定值-R2-5'  # 优化目标
real_col = '实测宽度-R2-5'  # 真实值
opt_cols = ['压下量-E2-1', '压下量-E2-3', '压下量-E2-5']  # 待优化的三个压下量


if target_col not in df_data.columns:
    raise ValueError(f"数据中未找到目标列 '{target_col}'。")


#PSO
def pso_optimize_row(base_features, target_width, model, selected_features, bounds, num_particles=30, max_iter=50):
    dim = len(opt_cols)

    # 初始化粒子位置 (在设定的上下限内随机生成)
    positions = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(num_particles, dim)
    )
    velocities = np.zeros((num_particles, dim))

    pbest_positions = positions.copy()
    pbest_scores = np.full(num_particles, np.inf)

    gbest_position = np.zeros(dim)
    gbest_score = np.inf

    w = 0.5  # 惯性权重：决定保留原有速度的程度，控制全局与局部搜索的平衡（通常0.4~0.9）
    c1 = 1.5  # 个体学习因子：粒子的自信度，决定向自身历史最优位置移动的步长权重
    c2 = 1.5  # 社会学习因子：粒子的从众度，决定向全局历史最优位置移动的步长权重

    def evaluate(pos):

        temp_features = base_features.copy()


        for i, col in enumerate(opt_cols):
            if col in temp_features.index:
                temp_features[col] = pos[i]
            sq_col = f"{col}_Square"
            if sq_col in temp_features.index:
                temp_features[sq_col] = pos[i] ** 2

        # 用模型进行预测
        X_input = temp_features[selected_features].values.reshape(1, -1)
        pred_width = model.predict(X_input)[0]

        # 适应度函数：预测宽度与设定宽度的绝对误差
        return abs(pred_width - target_width)

    # 迭代寻优
    for _ in range(max_iter):
        for i in range(num_particles):
            score = evaluate(positions[i])
            # 更新个体最优
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_positions[i] = positions[i].copy()
            # 更新全局最优
            if score < gbest_score:
                gbest_score = score
                gbest_position = positions[i].copy()

        # 更新速度和位置
        r1, r2 = np.random.rand(num_particles, dim), np.random.rand(num_particles, dim)
        velocities = (w * velocities +
                      c1 * r1 * (pbest_positions - positions) +
                      c2 * r2 * (gbest_position - positions))
        positions = positions + velocities

        # 越界处理
        for j in range(dim):
            positions[:, j] = np.clip(positions[:, j], bounds[j][0], bounds[j][1])

    # 寻优结束后，根据最优参数再测算一次最终的预测宽度
    best_features = base_features.copy()
    for i, col in enumerate(opt_cols):
        if col in best_features.index:
            best_features[col] = gbest_position[i]
        sq_col = f"{col}_Square"
        if sq_col in best_features.index:
            best_features[sq_col] = gbest_position[i] ** 2

    final_pred_width = model.predict(best_features[selected_features].values.reshape(1, -1))[0]

    return gbest_position, final_pred_width


#全局优化
results = []
print("🚀 开始对数据进行 PSO 寻优 (这可能需要几分钟)...")

# 设定优化的搜索空间
for index, row in df_data.iterrows():
    base_features = row.copy()
    target_width = row[target_col]
    real_width = row[real_col]

    # 获取优化前的模型预测宽度
    X_before = base_features[selected_features].values.reshape(1, -1)
    pred_before = final_model.predict(X_before)[0]


    bounds = []
    for col in opt_cols:
        val = row[col]
        # 设置容错，防止原值为0
        margin = abs(val * 0.2) if val != 0 else 0.1
        bounds.append((val - margin, val + margin))

    # 运行 PSO 优化
    best_params, pred_after = pso_optimize_row(
        base_features=base_features,
        target_width=target_width,
        model=final_model,
        selected_features=selected_features,
        bounds=bounds
    )

    # 计算评估指标 (依据方案设计)
    w_shi = abs(target_width - real_width)  # W实
    w_qian = abs(target_width - pred_before)  # W前
    w_hou = abs(target_width - pred_after)  # W后

    results.append({
        '原始压下量R1-1': row[opt_cols[0]],
        '原始压下量R1-3': row[opt_cols[1]],
        '原始压下量R1-5': row[opt_cols[2]],
        '优化后压下量R1-1': best_params[0],
        '优化后压下量R1-3': best_params[1],
        '优化后压下量R1-5': best_params[2],
        '出口宽度设定值': target_width,
        '实测宽度': real_width,
        '优化前预测宽度': pred_before,
        '优化后预测宽度': pred_after,
        'W实': w_shi,
        'W前': w_qian,
        'W后': w_hou,
        '相对真实改善(W实-W后)': w_shi - w_hou,
        '相对模型改善(W前-W后)': w_qian - w_hou
    })

    if (index + 1) % 50 == 0:
        print(f"   已完成 {index + 1} / {len(df_data)} 块钢的优化...")


# 保存
df_results = pd.DataFrame(results)

# 打印整体评价报告

print("-" * 50)
mean_real_improve = df_results['相对真实改善(W实-W后)'].mean()
mean_model_improve = df_results['相对模型改善(W前-W后)'].mean()

print(f"平均 (W实 - W后) : {mean_real_improve:.4f} mm")
if mean_real_improve > 0:
    print(" 1：优化后的预测宽度比真实轧制的误差更小，具有显著的工业指导价值")
else:
    print(" 1：发生负优化。")

print(f"\n平均 (W前 - W后) : {mean_model_improve:.4f} mm")
if mean_model_improve > 0:
    print(" 2：在同一套评价体系下，PSO 成功降低了设定值与模型预测值的虚拟误差")
else:
    print(" 2：模型虚拟误差未降低。")
print("-" * 50)

# 保存最终的优化对比表
results_path = os.path.join(mgh_dir, 'PSO_Optimization_Results.xlsx')
df_results.to_excel(results_path, index=False)
print(f"详细的压下量优化对比清单已保存至: {results_path}")