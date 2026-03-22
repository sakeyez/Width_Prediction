import pandas as pd
import numpy as np

# 读取原始数据
file_path = r"F:\asus\Desktop\毕业设计数据\数据表格\2160粗轧数据.xlsx"
df = pd.read_excel(file_path)
print("原有", len(df), "行。")


# 根据专业知识剔除无关参数
domain_irrelevant_features = [
    # 无关化学元素
    'Al', 'As', 'B', 'Ca', 'Cr', 'Cu', 'Mo', 'N',
    'Nb', 'Ni', 'O', 'P', 'Pb', 'S', 'Sn', 'Ti', 'V', 'W', 'Zr', 'Fe',
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

df = df.drop(columns=domain_irrelevant_features, errors='ignore')
print(f"初步剔除剩余列数：：{len(df.columns)}")


# 处理缺失值
target = '实测宽度-R2-5'

# 下列参数若为0必是缺失值
physical_keywords = ['温度', '厚度', '宽度', '速度', '辊径']
physical_columns = [col for col in df.columns if any(keyword in col for keyword in physical_keywords)]
# 替换成真正的缺失值 NaN
for col in physical_columns:
    df[col] = df[col].replace(0, np.nan)
# 现在执行整行删除
df_clean = df.dropna()
print("清理完缺失值和 0 值后，还剩下", len(df_clean), "行。")


# 处理离散值 (3倍)
features_to_check = [col for col in df_clean.columns if col != target]

for feature in features_to_check:
    Q1 = df_clean[feature].quantile(0.25)
    Q3 = df_clean[feature].quantile(0.75)
    IQR = Q3 - Q1


    multiplier = 3.0
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    df_clean = df_clean[(df_clean[feature] >= lower_bound) & (df_clean[feature] <= upper_bound)]

print("处理离散值后", len(df_clean), "行。")


# 保存清洗后的数据
save_path = r"F:\asus\Desktop\毕业设计数据\数据表格\初步处理版数据.xlsx"
df_clean.to_excel(save_path, index=False)
print("数据已经保存到：", save_path)