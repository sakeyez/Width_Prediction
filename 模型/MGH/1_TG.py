import pandas as pd
import numpy as np
import os

# 读取文件
base_dir = r"F:\asus\Desktop\毕业设计数据\数据表格"
train_path = os.path.join(base_dir, "Train_Data.xlsx")
val_path = os.path.join(base_dir, "Val_Data.xlsx")
test_path = os.path.join(base_dir, "Test_Data.xlsx")

# 保存路径
mgh_dir = r"F:\asus\Desktop\毕业设计数据\MGH模型"
if not os.path.exists(mgh_dir):
    os.makedirs(mgh_dir)

target = '实测宽度-R2-5'
# 【修改点1】：新增定义设定值列名
setting_target = '出口宽度设定值-R2-5'


# 构建特征池
def expand_mgh_features(df, target_col):
    # 【修改点2】：为了准确统计扩容前特征数，减去实测值和设定值两列
    print(f"  扩容前特征数量：{len(df.columns) - 2}")
    df_expanded = df.copy()

    # 【修改点3】：把实测宽度和设定值都排除掉，不让它们参与平方扩容
    base_features = [col for col in df.columns if col not in [target_col, setting_target]]

    # 平方项 (捕获非线性关系)
    for col in base_features:
        df_expanded[f"{col}_Square"] = df_expanded[col] ** 2

    # 道次差分项 与道次绝对差分项
    sequence_groups = {
        '压下量': ['R1压下量Pass1(H11_0)', 'R2-1压下量', 'R2-2压下量', 'R2-3压下量', 'R2-4压下量', 'R2-5压下量'],
        '轧制力': ['平辊实际轧制力-R1', '平辊实际轧制力-R2-1', '平辊实际轧制力-R2-2', '平辊实际轧制力-R2-3',
                   '平辊实际轧制力-R2-4', '平辊实际轧制力-R2-5'],
        '出口厚度': ['出口厚度-R1', '出口厚度-R2-1', '出口厚度-R2-2', '出口厚度-R2-3', '出口厚度-R2-4'],
        '出口温度': ['道次出口温度-R1', '道次出口温度-R2-1', '道次出口温度-R2-2', '道次出口温度-R2-3',
                     '道次出口温度-R2-4']
    }

    for group_name, cols in sequence_groups.items():
        # 过滤出当前数据集中确实存在的列，防止因全局清洗被误删而报错
        valid_cols = [c for c in cols if c in df.columns]

        # 计算相邻道次的变化量
        for i in range(1, len(valid_cols)):
            prev_col = valid_cols[i - 1]
            curr_col = valid_cols[i]

            # 1. 差分项 (当前道次 - 上一道次，体现衰减或增加的具体值)
            diff_col_name = f"{group_name}_Diff_{i}"
            df_expanded[diff_col_name] = df_expanded[curr_col] - df_expanded[prev_col]

            # 2. 绝对差分项 (波动的绝对幅度)
            abs_diff_col_name = f"{group_name}_AbsDiff_{i}"
            df_expanded[abs_diff_col_name] = df_expanded[diff_col_name].abs()

    # 【修改点4】：为了准确统计扩容后特征数，减去两列目标列
    print(f"  扩容后特征数量：{len(df_expanded.columns) - 2}")

    # 【修改点5】：重新排列列顺序，把所有特征放在前面，把 设定值 和 实测值 移到最后两列
    cols = [c for c in df_expanded.columns if c not in [target_col, setting_target]] + [setting_target, target_col]

    return df_expanded[cols]


# 批量处理并保存
df_train = pd.read_excel(train_path)
df_val = pd.read_excel(val_path)
df_test = pd.read_excel(test_path)

print("\n处理 Train 集:")
df_train_mgh = expand_mgh_features(df_train, target)

print("\n处理 Val 集:")
df_val_mgh = expand_mgh_features(df_val, target)

print("\n处理 Test 集:")
df_test_mgh = expand_mgh_features(df_test, target)

# 保存至 MGH 模型专属文件夹
df_train_mgh.to_excel(os.path.join(mgh_dir, "MGH_Train_Expanded.xlsx"), index=False)
df_val_mgh.to_excel(os.path.join(mgh_dir, "MGH_Val_Expanded.xlsx"), index=False)
df_test_mgh.to_excel(os.path.join(mgh_dir, "MGH_Test_Expanded.xlsx"), index=False)

print(f"\n特征池扩容完毕！已将海量特征保存至：{mgh_dir}")