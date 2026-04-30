import pandas as pd
import numpy as np
import os
import joblib
from sklearn.cluster import KMeans
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==========================================
# 1. 配置路径与数据加载
# ==========================================
hybrid_dir = r"F:\asus\Desktop\毕业设计数据\Hybrid2模型"
train_path = os.path.join(hybrid_dir, "Hybrid2_Train.xlsx")
val_path = os.path.join(hybrid_dir, "Hybrid2_Val.xlsx")

target = '实测宽度-R2-5'
df_train = pd.read_excel(train_path)
df_val = pd.read_excel(val_path)
# 确保只删除目标列，并且如果存在旧的标签列也一并删除
cols_to_drop = [target, 'Cluster_Label']
X_train = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns])
X_val = df_val.drop(columns=[c for c in cols_to_drop if c in df_val.columns])


print(f"当前参与计算的特征数量: {X_train.shape[1]}")

y_train = df_train[target].values

y_val = df_val[target].values


def get_regressor(m_type):
    if m_type == 'SVM': return SVR(kernel='rbf', C=10)
    if m_type == 'GPR': return GaussianProcessRegressor(normalize_y=True, random_state=42)
    if m_type == 'ANN': return MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=2000, early_stopping=True,
                                            random_state=42)
    if m_type == 'RF':  return RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)


# ==========================================
# 2. 评估阶段：展示多指标数据
# ==========================================
model_list = ['SVM', 'GPR', 'ANN', 'RF']
print("🚀 开始多指标评估 (K=1~6)...")

for m_type in model_list:
    print(f"\n📊 --- {m_type} 模型性能评估 ---")
    print(f"{'K值':<5} | {'R²':<8} | {'MAE':<8} | {'MSE':<8}")
    print("-" * 40)

    for k in range(1, 7):
        if k == 1:
            reg = get_regressor(m_type)
            reg.fit(X_train, y_train)
            preds = reg.predict(X_val)
        else:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            tr_labels = km.fit_predict(X_train)
            v_labels = km.predict(X_val)
            preds = np.zeros(len(y_val))
            for i in range(k):
                idx_tr, idx_v = (tr_labels == i), (v_labels == i)
                if idx_tr.sum() > 5:
                    reg = get_regressor(m_type)
                    reg.fit(X_train[idx_tr], y_train[idx_tr])
                    if idx_v.sum() > 0:
                        preds[idx_v] = reg.predict(X_val[idx_v])

        r2 = r2_score(y_val, preds)
        mae = mean_absolute_error(y_val, preds)
        mse = mean_squared_error(y_val, preds)
        print(f"{k:<5} | {r2:<8.4f} | {mae:<8.4f} | {mse:<8.4f}")

# ==========================================
# 3. 人工干预：输入确定的 K 值
# ==========================================
print("\n" + "=" * 50)
print("请根据上方指标，输入你为每个模型选定的最优 K 值：")
input_ks = {}
for m in model_list:
    val = input(f"请输入 {m} 的 K 值: ")
    input_ks[m] = int(val)

# ==========================================
# 4. 生成阶段：保存所需的 pkl 文件
# ==========================================
print("\n💾 正在根据输入生成聚类模型文件...")
needed_ks = set(input_ks.values())
for k_val in needed_ks:
    if k_val > 1:
        km_final = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        km_final.fit(X_train)
        save_path = os.path.join(hybrid_dir, f'kmeans_k{k_val}.pkl')
        joblib.dump(km_final, save_path)
        print(f"✅ 已生成并保存: kmeans_k{k_val}.pkl")
    else:
        print(f"ℹ️  K=1 不需要生成聚类文件。")

# 将选择记录保存，方便查阅
joblib.dump(input_ks, os.path.join(hybrid_dir, 'selected_best_ks.pkl'))
print("\n🎉 第二阶段修改完成！请记录好你选择的 K 值，然后运行 3_Base.py。")