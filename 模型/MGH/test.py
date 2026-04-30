import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 1. 考场准备：加载路径
# ==========================================
mgh_dir = r"F:\asus\Desktop\毕业设计数据\MGH模型"
test_path = os.path.join(mgh_dir, "MGH_Test_Expanded.xlsx")
model_path = os.path.join(mgh_dir, 'final_glr_model.pkl')

print("📥 正在布置考场...")

# ==========================================
# 2. 宣读规则：加载“试卷”和“考生（模型）”
# ==========================================
df_test = pd.read_excel(test_path)

with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

final_model = model_data['model']
selected_features = model_data['selected_features']

# ==========================================
# 3. 严格监考：把试卷的“题目(X)”和“答案(y)”无情分离
# ==========================================
target = '实测宽度-R2-5'
# 标准答案 y_test_true 被我们扣下，绝对不给模型看！
y_test_true = df_test[target].values
# 模型只能拿到它在 3 号脚本里挑中的那 33 列题干
X_test = df_test[selected_features].values

# ==========================================
# 4. 闭卷考试（模型极其无助地盲猜）
# ==========================================
print("🧠 考生（GLR模型）正在进行闭卷考试，请稍候...")
# 注意：predict 里面只有 X_test，没有 y！
y_test_pred = final_model.predict(X_test)

# ==========================================
# 5. 阅卷老师登场，无情打分
# ==========================================
mse = mean_squared_error(y_test_true, y_test_pred)
mae = mean_absolute_error(y_test_true, y_test_pred)
r2 = r2_score(y_test_true, y_test_pred)

print("\n🎉 期末大考（Test集）成绩单出炉！")
print("-" * 40)
print(f"   - 测试集 MSE: {mse:.4f}")
print(f"   - 测试集 MAE: {mae:.4f} mm")
print(f"   - 测试集 R² : {r2:.4f}")
print("-" * 40)

# ==========================================
# 6. 保存考试明细（用来放进毕业论文画图）
# ==========================================
df_results = pd.DataFrame({
    '真实宽度(标准答案)': y_test_true,
    '预测宽度(考生答卷)': y_test_pred,
    '绝对误差(mm)': np.abs(y_test_true - y_test_pred)
})
results_path = os.path.join(mgh_dir, 'GLR_Test_Final_Predictions.xlsx')
df_results.to_excel(results_path, index=False)

print(f"📊 详细的盲测对比表已保存至: {results_path}")
print("🏁 传统机器学习基准测试（Baseline）圆满收官！")