import pandas as pd
import os

# ==========================================
# 1. 设置路径 (请根据你的实际文件夹结构修改)
# ==========================================
mgh_dir = r"F:\asus\Desktop\毕业设计数据\MGH模型"
test_expanded_path = os.path.join(mgh_dir, "MGH_Test_Expanded.xlsx")

# 导师发给你的第一个文件（包含目标靶心和完整原始数据）
target_file_path = r"F:\asus\Desktop\数据\宽度离线数据导出.xlsx"

# ==========================================
# 2. 读取数据：我们坚决使用带有真实物理意义的 Expanded 数据
# ==========================================
print("📥 正在读取物理状态的 Test 测试集...")
df_test = pd.read_excel(test_expanded_path)
df_raw = pd.read_excel(target_file_path)

# ==========================================
# 3. 双重指纹匹配：找回丢失的靶心 (目标宽度)
# ==========================================
# 我们使用实测宽度和 E2-1压下量 作为联合主键，确保匹配绝对精准
match_keys = ['实测宽度-R2-5', '压下量-E2-1']
target_col_name = '出口宽度设定值-R2-5' # 也就是中间坯宽度设定值

print("🔍 正在进行数据指纹交叉比对...")

# 提取：钥匙 + 我们急需的靶心
df_target_subset = df_raw[match_keys + [target_col_name]].copy()

# 为防止导师原始数据中有完全重复的行，先去重
df_target_subset = df_target_subset.drop_duplicates(subset=match_keys)

# 执行左连接拼接
df_ready_for_pso = pd.merge(df_test, df_target_subset, on=match_keys, how='left')

# ==========================================
# 4. 严苛的工程体检
# ==========================================
missing_count = df_ready_for_pso[target_col_name].isnull().sum()

if missing_count > 0:
    print(f"\n⚠️ 警告：有 {missing_count} 条数据未能成功匹配！")
    print("原因通常是浮点数精度截断（例如 1250.1 != 1250.100001）。")
    print("可以尝试将两张表的实测宽度用 round(2) 保留两位小数后再匹配。")
else:
    print("\n✅ 完美！所有测试集带钢均已成功找回其控制靶心。")
    print("数据状态：全物理量级、无归一化干扰，随时可供 PSO 调遣。")

# ==========================================
# 5. 保存优化专用的最终底座
# ==========================================
save_path = os.path.join(mgh_dir, "Optimization_Test_Base.xlsx")
df_ready_for_pso.to_excel(save_path, index=False)
print(f"💾 优化专用测试底座已保存至: {save_path}")