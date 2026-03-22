import pandas as pd
from sklearn.preprocessing import StandardScaler

# 读取数据
file_path = r"F:\asus\Desktop\毕业设计数据\数据表格\初步处理版数据.xlsx"
df_clean = pd.read_excel(file_path)


target = '实测宽度-R2-5'

# 把实测宽度单独提取出来
features = [col for col in df_clean.columns if col != target and col != 'Unnamed: 0']

# 引入工具箱
scaler = StandardScaler()

# 进行标准化转换
scaled_features = scaler.fit_transform(df_clean[features])

# 组装回数据框
df_scaled = pd.DataFrame(scaled_features, columns=features)

# 把y写入
df_scaled[target] = df_clean[target].values


save_path = r"F:\asus\Desktop\毕业设计数据\数据表格\标准化后数据.xlsx"
df_scaled.to_excel(save_path, index=False)

print("已保存至：", save_path)