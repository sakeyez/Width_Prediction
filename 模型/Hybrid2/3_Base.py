import pandas as pd
import numpy as np
import os
import joblib
import warnings
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 隐藏收敛警告，保持控制台整洁
warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置文件与数据读取
# ==========================================
hybrid_dir = r"F:\asus\Desktop\毕业设计数据\Hybrid2模型"
train_path = os.path.join(hybrid_dir, "Hybrid2_Train.xlsx")

df_train = pd.read_excel(train_path)
target = '实测宽度-R2-5'

# 【安全锁】：强制清洗，确保只保留你选出的 18 个物理特征
cols_to_drop = [target, 'Cluster_Label', 'Cluster']
features = [col for col in df_train.columns if col not in cols_to_drop]

X_train_raw = df_train[features]
y_train = df_train[target].values

ks_file_path = os.path.join(hybrid_dir, 'selected_best_ks.pkl')
if not os.path.exists(ks_file_path):
    raise FileNotFoundError("❌ 找不到 selected_best_ks.pkl，请确保你已经完整运行了 2_KMeans.py！")

best_ks = joblib.load(ks_file_path)
print(f"📥 成功读取你在上一阶段选定的最优 K 值组合: {best_ks}")

models_dir = os.path.join(hybrid_dir, "Trained_Expert_Models")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)


# ==========================================
# 2. 专家模型配置库 (加入标准化 Pipeline)
# ==========================================
def get_model_pipeline(m_type):
    if m_type == 'SVM':
        return make_pipeline(StandardScaler(), SVR(kernel='rbf', C=10.0, gamma='scale'))

    if m_type == 'GPR':
        kernel = C(1.0) * RBF(1.0)
        return make_pipeline(StandardScaler(),
                             GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True,
                                                      random_state=42))

    if m_type == 'ANN':
        return make_pipeline(StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=3000, learning_rate_init=0.01,
                                          early_stopping=True, random_state=42))

    if m_type == 'RF':
        # RF 对量纲不敏感
        return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)


# ==========================================
# 3. 循环构建 10 个专家模型
# ==========================================
print(f"🚀 开始构建非对称专家库，当前使用物理特征数: {X_train_raw.shape[1]}")

for m_name, k in best_ks.items():
    print(f"\n--- 正在处理 {m_name} 模型家族 (K={k}) ---")

    if k == 1:
        # 训练全局专家
        model = get_model_pipeline(m_name)
        model.fit(X_train_raw, y_train)
        joblib.dump(model, os.path.join(models_dir, f'{m_name}_Global.pkl'))
        print(f"  ✅ {m_name} 全局模型已保存。")

    else:
        # 训练分簇专家
        km_path = os.path.join(hybrid_dir, f'kmeans_k{k}.pkl')
        if not os.path.exists(km_path):
            print(f"  ❌ 错误：找不到 {km_path}，请确认第二步生成成功。")
            continue

        kmeans = joblib.load(km_path)
        # 用对应的 K-Means 对训练集进行工况划分
        labels = kmeans.predict(X_train_raw)

        for i in range(k):
            idx = (labels == i)
            if idx.sum() > 5:  # 防止因某个极端冷门工况导致样本过少报错
                model = get_model_pipeline(m_name)
                model.fit(X_train_raw[idx], y_train[idx])
                save_name = f'{m_name}_Cluster_{i}_of_K{k}.pkl'
                joblib.dump(model, os.path.join(models_dir, save_name))
                print(f"  ✅ {m_name} 专家簇 {i} 已保存 (训练样本数: {idx.sum()})。")

print(f"\n🎉 专家库构建完毕！文件已安全打包至: {models_dir}")