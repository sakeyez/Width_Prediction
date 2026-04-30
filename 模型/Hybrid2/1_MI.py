import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 配置文件路径
base_dir = r"F:\asus\Desktop\毕业设计数据\数据表格"
train_path = os.path.join(base_dir, "Train_Data.xlsx")
val_path = os.path.join(base_dir, "Val_Data.xlsx")
test_path = os.path.join(base_dir, "Test_Data.xlsx")

hybrid_dir = r"F:\asus\Desktop\毕业设计数据\Hybrid2模型"
if not os.path.exists(hybrid_dir):
    os.makedirs(hybrid_dir)

# 加载原始数据
df_train = pd.read_excel(train_path)
df_val = pd.read_excel(val_path)
target = '实测宽度-R2-5'

# 【重要】清理数据：剔除目标列和任何可能存在的旧标签列
cols_to_drop = [target, 'Cluster_Label', 'Cluster']
features = [col for col in df_train.columns if col not in cols_to_drop]

X_train_raw = df_train[features]
y_train = df_train[target]
X_val_raw = df_val[features]
y_val = df_val[target]

print(f"✅ 数据清理完成。参与评估的基础特征总数: {len(features)}")

# ==========================================
# 2. 计算互信息 (MI) 得分
# ==========================================
print(f"🚀 正在计算特征与目标宽度的互信息(MI)得分...")
mi_scores = mutual_info_regression(X_train_raw, y_train, random_state=42)
mic_df = pd.DataFrame({'Feature': features, 'MI_Score': mi_scores})
mic_df = mic_df.sort_values(by='MI_Score', ascending=False).reset_index(drop=True)

# ==========================================
# 3. 维度评估阶段 (6-42)
# ==========================================
print("\n🔍 开始特征维度扫描 (6-42)...")
print(f"{'维度':<6} | {'R²':<8} | {'MAE':<8} | {'MSE':<8}")
print("-" * 45)

dims = range(6, len(features) + 1)
history = []

for n in dims:
    selected = mic_df['Feature'][:n].tolist()
    # 使用 Pipeline 加入标准化，确保评估精度真实
    model = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    )
    model.fit(X_train_raw[selected], y_train)
    preds = model.predict(X_val_raw[selected])

    r2 = r2_score(y_val, preds)
    mae = mean_absolute_error(y_val, preds)
    mse = mean_squared_error(y_val, preds)

    history.append({'n': n, 'r2': r2, 'mae': mae, 'mse': mse})
    print(f"{n:<6} | {r2:<8.4f} | {mae:<8.4f} | {mse:<8.4f}")

# ==========================================
# 4. 人工决策与保存
# ==========================================
print("\n" + "=" * 50)
best_n_input = input("请根据上方指标，输入你选定的特征数量 N: ")
N = int(best_n_input)

selected_features = mic_df['Feature'][:N].tolist()
columns_to_keep = selected_features + [target]

print(f"\n💾 正在保存 Hybrid-2 专属数据集 (维度: {N})...")
# 保存时确保不带旧的标签
df_train[columns_to_keep].to_excel(os.path.join(hybrid_dir, "Hybrid2_Train.xlsx"), index=False)
df_val[columns_to_keep].to_excel(os.path.join(hybrid_dir, "Hybrid2_Val.xlsx"), index=False)
pd.read_excel(test_path)[columns_to_keep].to_excel(os.path.join(hybrid_dir, "Hybrid2_Test.xlsx"), index=False)

print(f"🎉 第一阶段修改完成！已剔除干扰列并根据你的选择生成了 {N} 维数据集。")