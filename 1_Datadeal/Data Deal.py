import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_output_dir = os.path.join(project_root, "data", "datadeal")
image_output_dir = os.path.join(project_root, "image", "datadeal")
os.makedirs(data_output_dir, exist_ok=True)
os.makedirs(image_output_dir, exist_ok=True)


def scale_split(dataframe, feature_columns, scaler, target, setting_target):
    scaled_features = pd.DataFrame(
        scaler.transform(dataframe[feature_columns]),
        columns=feature_columns,
        index=dataframe.index,
    )
    scaled_features[target] = dataframe[target].values
    scaled_features[setting_target] = dataframe[setting_target].values
    return scaled_features


# ==========================================
# 1. 读取原始数据与定义目标
# ==========================================
file_path = r"F:\asus\Desktop\毕业设计数据\数据表格\2160粗轧数据.xlsx"
df = pd.read_excel(file_path)
print("原有", len(df), "行。")

target = "实测宽度-R2-5"
setting_target = "出口宽度设定值-R2-5"


# ==========================================
# 2. 特征筛选与白名单保护
# ==========================================
domain_irrelevant_features = [
    "Al",
    "As",
    "B",
    "Ca",
    "Cr",
    "Cu",
    "Mo",
    "N",
    "Nb",
    "Ni",
    "O",
    "P",
    "Pb",
    "Sn",
    "Ti",
    "V",
    "W",
    "Zr",
    "Fe",
    "Unnamed: 0",
    "R2上工作辊轧制公里",
    "R2上工作辊轧制公里.1",
    "R2上工作辊轧制公里.2",
    "实测宽度-R1",
    "实测宽度-R2-1",
    "AD-平辊出口宽度R2-2",
    "实测宽度-R2-3",
    "AD-平辊出口宽度R2-4",
]

optimization_vars = ["压下量E2-1", "压下量E2-3", "压下量E2-5"]

for col in df.columns:
    if any(keyword in col for keyword in ["立辊", "E1", "E2", "E3", "E4", "狗骨"]):
        if col not in domain_irrelevant_features and col not in optimization_vars:
            domain_irrelevant_features.append(col)

df = df.drop(columns=domain_irrelevant_features, errors="ignore")
print(f"初步剔除剩余列数：{len(df.columns)}")


# ==========================================
# 3. 处理缺失值与 0 值
# ==========================================
physical_keywords = ["温度", "厚度", "宽度", "速度", "辊径", "压下量", "轧制力", "力矩"]
physical_columns = [col for col in df.columns if any(keyword in col for keyword in physical_keywords)]

for col in physical_columns:
    df[col] = df[col].replace(0, np.nan)

df_clean = df.dropna()
print("清理完缺失值和 0 值后，还剩下", len(df_clean), "行。")


# ==========================================
# 4. 处理离散值（3倍IQR）
# ==========================================
features_to_check = [col for col in df_clean.columns if col not in [target, setting_target]]

for feature in features_to_check:
    q1 = df_clean[feature].quantile(0.25)
    q3 = df_clean[feature].quantile(0.75)
    iqr = q3 - q1

    multiplier = 3.0
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    df_clean = df_clean[(df_clean[feature] >= lower_bound) & (df_clean[feature] <= upper_bound)]

print("处理离散值后", len(df_clean), "行。")

clean_path = os.path.join(data_output_dir, "2160粗轧数据_清洗完整版.xlsx")
df_clean.to_excel(clean_path, index=False)
print("\n中间文件已输出：清洗后完整数据集！")


# ==========================================
# 5. 严格按 7:2:1 划分数据集
# ==========================================
train_val_data, test_data = train_test_split(df_clean, test_size=0.1, random_state=42)
val_ratio_in_remaining = 2.0 / 9.0
train_data, val_data = train_test_split(train_val_data, test_size=val_ratio_in_remaining, random_state=42)


# ==========================================
# 6. 用训练集最大最小值做 MinMaxScaler
# ==========================================
features = [col for col in df_clean.columns if col not in [target, setting_target]]
scaler = MinMaxScaler()
scaler.fit(train_data[features])

train_data = scale_split(train_data, features, scaler, target, setting_target)
val_data = scale_split(val_data, features, scaler, target, setting_target)
test_data = scale_split(test_data, features, scaler, target, setting_target)

print("\n数据集 7:2:1 划分完毕：")
print(f"   - 训练集(Train): {len(train_data)} 行")
print(f"   - 验证集(Val)  : {len(val_data)} 行")
print(f"   - 测试集(Test) : {len(test_data)} 行")


# ==========================================
# 7. 导出至通用数据池
# ==========================================
train_path = os.path.join(data_output_dir, "Train_Data.xlsx")
val_path = os.path.join(data_output_dir, "Val_Data.xlsx")
test_path = os.path.join(data_output_dir, "Test_Data.xlsx")

train_data.to_excel(train_path, index=False)
val_data.to_excel(val_path, index=False)
test_data.to_excel(test_path, index=False)

print("\n全局通用数据清洗与切分完成，已输出到数据表格文件夹！")
