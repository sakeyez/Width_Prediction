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

no_setting_source_path = os.path.join(project_root, "originaldata", "2160粗轧无设定值数据.xlsx")
with_setting_source_path = os.path.join(project_root, "originaldata", "带设定值粗轧数据.xlsx")
complete_data_path = os.path.join(project_root, "originaldata", "完整版数据.xlsx")


def scale_split(dataframe, feature_columns, scaler, target, setting_target):
    scaled_features = pd.DataFrame(
        scaler.transform(dataframe[feature_columns]),
        columns=feature_columns,
        index=dataframe.index,
    )
    scaled_features[target] = dataframe[target].values
    scaled_features[setting_target] = dataframe[setting_target].values
    return scaled_features


def save_excel_with_fallback(dataframe, preferred_path):
    try:
        dataframe.to_excel(preferred_path, index=False)
        return preferred_path
    except PermissionError:
        fallback_path = preferred_path.replace(".xlsx", "_new.xlsx")
        dataframe.to_excel(fallback_path, index=False)
        print(f"检测到 {preferred_path} 正在被占用，已改存为: {fallback_path}")
        return fallback_path


def build_complete_dataset(target, setting_target):
    if not os.path.exists(no_setting_source_path):
        raise FileNotFoundError(f"找不到无设定值源文件: {no_setting_source_path}")
    if not os.path.exists(with_setting_source_path):
        raise FileNotFoundError(f"找不到带设定值源文件: {with_setting_source_path}")

    no_setting_df = pd.read_excel(no_setting_source_path)
    with_setting_df = pd.read_excel(with_setting_source_path)

    preferred_match_columns = ["实测宽度-R2-1", "实测宽度-R2-3", "实测宽度-R2-5"]
    match_columns = [
        column
        for column in preferred_match_columns
        if column in no_setting_df.columns and column in with_setting_df.columns
    ]
    missing_match_columns = [column for column in preferred_match_columns if column not in match_columns]
    if missing_match_columns:
        print(f"以下匹配列在源文件中不存在，将自动跳过: {missing_match_columns}")
    if not match_columns:
        raise KeyError(f"找不到可用于匹配设定值的列，候选列: {preferred_match_columns}")
    if setting_target not in with_setting_df.columns:
        raise KeyError(f"带设定值源文件缺少设定值列: {setting_target}")

    mapping_df = (
        with_setting_df.groupby(match_columns, dropna=False)[setting_target]
        .agg(lambda values: tuple(pd.unique(pd.Series(values).dropna())))
        .reset_index(name="候选设定值")
    )
    mapping_df["候选数量"] = mapping_df["候选设定值"].apply(len)
    unique_mapping_df = (
        mapping_df[mapping_df["候选数量"] == 1][match_columns + ["候选设定值"]]
        .copy()
        .rename(columns={"候选设定值": setting_target})
    )
    unique_mapping_df[setting_target] = unique_mapping_df[setting_target].apply(lambda values: values[0])

    complete_df = no_setting_df.drop(columns=[setting_target], errors="ignore").merge(
        unique_mapping_df,
        on=match_columns,
        how="left",
    )
    if len(complete_df) != len(no_setting_df):
        raise ValueError("设定值回填后行数发生变化，已中止。")

    matched_count = int(complete_df[setting_target].notna().sum())
    unmatched_count = int(complete_df[setting_target].isna().sum())
    ambiguous_count = int((mapping_df["候选数量"] > 1).sum())

    saved_complete_path = save_excel_with_fallback(complete_df, complete_data_path)
    print("\n完整版数据已生成：")
    print(f"   - 使用底表: {no_setting_source_path}")
    print(f"   - 使用设定值源表: {with_setting_source_path}")
    print(f"   - 实际匹配列: {match_columns}")
    print(f"   - 唯一可映射键数: {len(unique_mapping_df)}")
    print(f"   - 多设定值冲突键数: {ambiguous_count}")
    print(f"   - 成功回填设定值样本数: {matched_count}")
    print(f"   - 设定值留空样本数: {unmatched_count}")
    print(f"   - 完整版文件: {saved_complete_path}")
    return complete_df


# ==========================================
# 1. 读取原始数据与定义目标
# ==========================================
target = "实测宽度-R2-5"
setting_target = "出口宽度设定值-R2-5"

df = build_complete_dataset(target=target, setting_target=setting_target)
print("原有", len(df), "行。")

# 只保留真正想优化的立辊压下量。
protected_edger_reduction_features = [
    "压下量-E2-1",
    "压下量-E2-3",
    "压下量-E2-5",
]


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

# 删除所有立辊相关变量，但对白名单中的立辊压下量放行。
# 这里不再删除“侧压”相关字段，因为 SSP 与立辊是不同步骤。
for col in df.columns:
    is_edger_related = any(keyword in col for keyword in ["立辊", "E1", "E2", "E3", "E4", "狗骨"])
    if is_edger_related and col not in protected_edger_reduction_features:
        domain_irrelevant_features.append(col)

domain_irrelevant_features = list(dict.fromkeys(domain_irrelevant_features))
df = df.drop(columns=domain_irrelevant_features, errors="ignore")

missing_protected_features = [col for col in protected_edger_reduction_features if col not in df.columns]
if missing_protected_features:
    raise KeyError(f"原始数据中缺少需要保留的立辊压下量列: {missing_protected_features}")

print(f"初步剔除后剩余列数：{len(df.columns)}")


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
# 4. 处理离群值（3 倍 IQR）
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

print("处理离群值后", len(df_clean), "行。")

clean_path = os.path.join(data_output_dir, "2160粗轧数据_清洗完整版.xlsx")
df_clean.to_excel(clean_path, index=False)
print("\n中间文件已输出：清洗后完整数据集。")


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

print("\n全局通用数据清洗与切分完成，已输出到数据表格文件夹。")
