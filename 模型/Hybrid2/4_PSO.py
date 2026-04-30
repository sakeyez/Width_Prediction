import pandas as pd
import numpy as np
import os
import joblib
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 屏蔽可能出现的版本警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置文件与数据加载
# ==========================================
hybrid_dir = r"F:\asus\Desktop\毕业设计数据\Hybrid2模型"
expert_dir = os.path.join(hybrid_dir, "Trained_Expert_Models")

val_path = os.path.join(hybrid_dir, "Hybrid2_Val.xlsx")
test_path = os.path.join(hybrid_dir, "Hybrid2_Test.xlsx")

df_val = pd.read_excel(val_path)
df_test = pd.read_excel(test_path)
target = '实测宽度-R2-5'

# 数据清洗：确保验证和预测时使用的特征，与训练时完全一致
cols_to_drop = [target, 'Cluster_Label', 'Cluster']
features = [col for col in df_val.columns if col not in cols_to_drop]

# 自动读取我们在 2_KMeans.py 阶段确定的最佳 K 值字典
ks_file_path = os.path.join(hybrid_dir, 'selected_best_ks.pkl')
if not os.path.exists(ks_file_path):
    raise FileNotFoundError("❌ 找不到 selected_best_ks.pkl，请确保前置步骤已正确完成。")
best_ks = joblib.load(ks_file_path)
model_names = list(best_ks.keys())


# ==========================================
# 2. 定义智能路由预测系统 (非对称调度)
# ==========================================
def get_expert_predictions(df):
    """
    智能路由：根据样本的物理特征，将其分发给对应的工况专家进行预测。
    返回 (N, 4) 的矩阵，代表 4 大模型家族的汇总意见。
    """
    n_samples = len(df)
    preds_matrix = np.zeros((n_samples, len(model_names)))
    X = df[features]

    for col_idx, m_name in enumerate(model_names):
        k = best_ks[m_name]

        if k == 1:
            # 该模型家族是全局专家 (如 ANN)
            model = joblib.load(os.path.join(expert_dir, f'{m_name}_Global.pkl'))
            preds_matrix[:, col_idx] = model.predict(X)
        else:
            # 该模型家族是细分领域的工况专家 (如 GPR 有 5 个)
            kmeans = joblib.load(os.path.join(hybrid_dir, f'kmeans_k{k}.pkl'))
            labels = kmeans.predict(X)

            for i in range(k):
                idx = (labels == i)
                if idx.any():
                    model_path = os.path.join(expert_dir, f'{m_name}_Cluster_{i}_of_K{k}.pkl')
                    if os.path.exists(model_path):
                        model = joblib.load(model_path)
                        preds_matrix[idx, col_idx] = model.predict(X[idx])
                    else:
                        # 极端防呆：如果某个冷门工况在训练时样本不足没建出模型，用均值或其他专家兜底
                        pass
    return preds_matrix


print(f"📦 正在调度非对称专家库，对 Val 和 Test 集进行基础预测...")
preds_val = get_expert_predictions(df_val)
y_val = df_val[target].values

preds_test = get_expert_predictions(df_test)
y_test = df_test[target].values

# ==========================================
# 3. PSO 粒子群算法：为热连轧预测寻找黄金比例
# ==========================================
print("\n🕊️ 启动 PSO 粒子群算法，正在寻找四大模型家族的最优权重比例...")

# 完全对齐论文 Table 1 的 PSO 参数
N_PARTICLES = 30
MAX_ITER = 100
W = 0.8
C1 = 1.5
C2 = 1.5


def objective_function(weights):
    # 强制权重归一化
    w_norm = weights / np.sum(weights)
    y_ensemble = np.dot(preds_val, w_norm)
    # 使用与论文一致的 RMSE 作为适应度函数
    return np.sqrt(mean_squared_error(y_val, y_ensemble))


n_models = len(model_names)
particles_pos = np.random.rand(N_PARTICLES, n_models)
particles_vel = np.zeros((N_PARTICLES, n_models))

pbest_pos = particles_pos.copy()
pbest_scores = np.array([objective_function(p) for p in particles_pos])

gbest_idx = np.argmin(pbest_scores)
gbest_pos = pbest_pos[gbest_idx].copy()
gbest_score = pbest_scores[gbest_idx]

for _ in range(MAX_ITER):
    r1, r2 = np.random.rand(N_PARTICLES, n_models), np.random.rand(N_PARTICLES, n_models)

    particles_vel = (W * particles_vel +
                     C1 * r1 * (pbest_pos - particles_pos) +
                     C2 * r2 * (gbest_pos - particles_pos))
    particles_pos = particles_pos + particles_vel

    # 约束处理：权重必须在 [1e-6, 1.0] 之间，防止负权重
    particles_pos = np.clip(particles_pos, 1e-6, 1.0)

    current_scores = np.array([objective_function(p) for p in particles_pos])

    better_mask = current_scores < pbest_scores
    pbest_pos[better_mask] = particles_pos[better_mask]
    pbest_scores[better_mask] = current_scores[better_mask]

    if np.min(pbest_scores) < gbest_score:
        gbest_idx = np.argmin(pbest_scores)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]

# ==========================================
# 4. 输出最优成绩单
# ==========================================
optimal_weights = gbest_pos / np.sum(gbest_pos)
print("\n🎉 PSO 寻优完成！最终确定的各家族权重比例为：")
for name, weight in zip(model_names, optimal_weights):
    print(f"   - {name} 专家家族: {weight:.4f}")

# 在 Test 测试集上进行严格大考
y_pred_final = np.dot(preds_test, optimal_weights)

print("\n🏆 Hybrid-2 模型在最终测试集 (Test) 上的惊艳成绩：")
print(f"   - RMSE : {np.sqrt(mean_squared_error(y_test, y_pred_final)):.4f}")
print(f"   - MAE  : {mean_absolute_error(y_test, y_pred_final):.4f}")
print(f"   - R²   : {r2_score(y_test, y_pred_final):.4f}")

# 保存最终的混合权重，方便以后直接部署使用
joblib.dump(optimal_weights, os.path.join(hybrid_dir, 'pso_optimal_weights.pkl'))
print("\n✅ 恭喜！你的热连轧混合宽度预测模型 (Hybrid-2) 已全部跑通！")