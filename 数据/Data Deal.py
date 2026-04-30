import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. 读取原始数据与定义目标
# ==========================================
file_path = r"F:\asus\Desktop\毕业设计数据\数据表格\2160粗轧数据.xlsx"
df = pd.read_excel(file_path)
print("原有", len(df), "行。")

# 明确定义实测目标和设定值基准
target = '实测宽度-R2-5'
setting_target = '出口宽度设定值-R2-5'

# ==========================================
# 2. 特征筛选与白名单保护
# ==========================================
domain_irrelevant_features = [
    # 无关化学元素
    'Al', 'As', 'B', 'Ca', 'Cr', 'Cu', 'Mo', 'N',
    'Nb', 'Ni', 'O', 'P', 'Pb', 'Sn', 'Ti', 'V', 'W', 'Zr', 'Fe',
    # 序号列
    'Unnamed: 0',
    # 设备累计损耗参数
    'R2上工作辊轧制公里', 'R2上工作辊轧制公里.1', 'R2上工作辊轧制公里.2',
    # 中间宽度
    '实测宽度-R1',
    '实测宽度-R2-1',
    'AD-平辊出口宽度R2-2',
    '实测宽度-R2-3',
    'AD-平辊出口宽度R2-4'
]

# 【关键白名单】：保护后续PSO优化必须要调参的立辊压下量
optimization_vars = ['压下量-E2-1', '压下量-E2-3', '压下量-E2-5']

# 剔除狗骨效应及其他无关立辊参数
for col in df.columns:
    if any(keyword in col for keyword in ['立辊', 'E1', 'E2', 'E3', 'E4', '狗骨']):
        # 既不在无关特征列表里，又不是我们要保护的优化变量，才予以剔除
        if col not in domain_irrelevant_features and col not in optimization_vars:
            domain_irrelevant_features.append(col)

df = df.drop(columns=domain_irrelevant_features, errors='ignore')
print(f"初步剔除剩余列数：{len(df.columns)}")


# ==========================================
# 3. 处理缺失值与0值
# ==========================================
physical_keywords = ['温度', '厚度', '宽度', '速度', '辊径', '压下量', '轧制力', '力矩']
physical_columns = [col for col in df.columns if any(keyword in col for keyword in physical_keywords)]

for col in physical_columns:
    df[col] = df[col].replace(0, np.nan)

df_clean = df.dropna()
print("清理完缺失值和 0 值后，还剩下", len(df_clean), "行。")


# ==========================================
# 4. 处理离散值 (3倍 IQR)
# ==========================================
# 【修改点】：将实测宽度和目标设定值双双排除，不参与离散值截断
features_to_check = [col for col in df_clean.columns if col not in [target, setting_target]]

for feature in features_to_check:
    Q1 = df_clean[feature].quantile(0.25)
    Q3 = df_clean[feature].quantile(0.75)
    IQR = Q3 - Q1

    multiplier = 3.0
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    df_clean = df_clean[(df_clean[feature] >= lower_bound) & (df_clean[feature] <= upper_bound)]

print("处理离散值后", len(df_clean), "行。")


clean_path = r"F:\asus\Desktop\毕业设计数据\数据表格\2160粗轧数据_清洗完整版.xlsx"
df_clean.to_excel(clean_path, index=False)
print("\n✅ 中间文件已输出：清洗后完整数据集！")


# ==========================================
# 5. 全局 MinMaxScaler 归一化
# ==========================================
scaler = MinMaxScaler()
# 【修改点】：特征列表中排除 target 和 setting_target
features = [col for col in df_clean.columns if col not in [target, setting_target]]

df_scaled_features = pd.DataFrame(scaler.fit_transform(df_clean[features]), columns=features)

# 【修改点】：将两个目标列原封不动地拼接到归一化后的数据末尾
df_scaled_features[target] = df_clean[target].values
df_scaled_features[setting_target] = df_clean[setting_target].values


# ==========================================
# 6. 严格按 7:2:1 划分数据集
# ==========================================
train_val_data, test_data = train_test_split(df_scaled_features, test_size=0.1, random_state=42)

val_ratio_in_remaining = 2.0 / 9.0
train_data, val_data = train_test_split(train_val_data, test_size=val_ratio_in_remaining, random_state=42)

print(f"\n📊 数据集 7:2:1 划分完毕：")
print(f"   - 训练集 (Train): {len(train_data)} 行")
print(f"   - 验证集 (Val)  : {len(val_data)} 行")
print(f"   - 测试集 (Test) : {len(test_data)} 行")


# ==========================================
# 7. 导出至通用数据池
# ==========================================
train_path = r"F:\asus\Desktop\毕业设计数据\数据表格\Train_Data.xlsx"
val_path = r"F:\asus\Desktop\毕业设计数据\数据表格\Val_Data.xlsx"
test_path = r"F:\asus\Desktop\毕业设计数据\数据表格\Test_Data.xlsx"

train_data.to_excel(train_path, index=False)
val_data.to_excel(val_path, index=False)
test_data.to_excel(test_path, index=False)

print("\n✅ 全局通用数据清洗与切分完成，已输出到数据表格文件夹！")