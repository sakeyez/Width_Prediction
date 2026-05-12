import pandas as pd
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_output_dir = os.path.join(project_root, "data", "datadeal")
image_output_dir = os.path.join(project_root, "image", "datadeal")
os.makedirs(data_output_dir, exist_ok=True)
os.makedirs(image_output_dir, exist_ok=True)

print("=== 开始生成数据清洗前后对比表 ===")

# ==========================================
# 1. 请在这里修改为你的实际文件路径！
# ==========================================

# 假设你有两个文件：一个是清洗前的原始数据，一个是清洗后的数据
# 如果你的数据是 Excel 格式，请把 read_csv 改成 read_excel
RAW_DATA_PATH = r"F:\asus\Desktop\毕业设计数据\数据表格\2160粗轧数据.xlsx" # 替换为你的原始数据文件名
CLEANED_DATA_PATH = r"F:\asus\Desktop\毕业设计数据\数据表格\2160粗轧数据_清洗完整版.xlsx"  # 替换为你清洗后的文件名

# ==========================================
# 2. 请在这里填入你想要在表格中展示的特征名
# ==========================================
# 建议从 52 个特征中挑选 8-10 个最具代表性的核心参数
TARGET_FEATURES = [
    # --- 1. 预测目标 ---
    '实测宽度-R2-5',             # 必须保留，作为参照点

    # --- 2. 初始几何尺寸 (绝对主导因素) ---
    '板坯宽度实测值(热态)',      # 决定宽度的基数
    '板坯厚度热态',              # 决定总压下量（压下量越大，宽展越明显）
    '侧压机压下量',              # 粗轧阶段控制宽度的最主要手段

    # --- 3. 材料化学成分 (决定基础变形抗力) ---
    'C',                       # 碳当量对屈服强度的影响最大
    'Mn',                      # 锰元素影响材料的硬化指数

    # --- 4. 关键温度节点 (决定热态塑性) ---
    '出炉实测温度',              # 初始热状态
    '除鳞机后实测温度',          # 进入轧机前的实际表面温度
    '道次出口温度-R2-4',         # 末道次前的高温状态，直接影响最后一次宽展

    # --- 5. 早期轧制状态 (R1) ---
    'R1压下量Pass1(H11_0)',    # R1的变形量
    '平辊实际轧制力-R1',         # 反映早期的材料硬度与变形状况

    # --- 6. 晚期轧制状态 (R2 后半段，最关键) ---
    'R2-3压下量',
    '平辊实际轧制力-R2-3',
    'R2-4压下量',              # 倒数第二道次压下量
    '平辊实际轧制力-R2-4',       # 倒数第二道次力能参数
    'R2-5压下量',              # 末道次压下量（引发最终宽展的直接动力）
    '平辊实际轧制力-R2-5',       # 末道次轧制力（与接触弧长、宽展量高度非线性相关）

    # --- 7. 动力学与设备状态 ---
    'R1轧制速度Pass1(H11_0)'
    'R2轧制速度Pass5(H11_0)',  # 轧制速度影响应变速率和摩擦系数，进而影响宽展
    'R2工作辊辊径.2',            # 辊径大小直接改变接触弧长 (摩擦区)，对宽展有几何影响

]


def generate_three_line_table():
    # 检查文件是否存在
    if not os.path.exists(RAW_DATA_PATH) or not os.path.exists(CLEANED_DATA_PATH):
        print("⚠️ 找不到数据文件！请确保文件路径填写正确，且文件和本代码在同一文件夹下。")
        return

    # 读取数据
    print("正在读取数据...")
    df_raw = pd.read_excel(RAW_DATA_PATH)
    df_cleaned = pd.read_excel(CLEANED_DATA_PATH)

    # 提取需要的统计量 (最小值, 平均值, 标准差, 最大值)
    def get_stats(df, features):
        # 过滤出存在于数据集中的特征，防止列名写错报错
        valid_features = [f for f in features if f in df.columns]
        stats = df[valid_features].describe().T
        return stats[['min', 'mean', 'std', 'max']]

    stats_before = get_stats(df_raw, TARGET_FEATURES)
    stats_after = get_stats(df_cleaned, TARGET_FEATURES)

    # 重命名列名，加上(前)/(后)的后缀以区分
    stats_before.columns = ['最小值(前)', '平均值(前)', '标准差(前)', '最大值(前)']
    stats_after.columns = ['最小值(后)', '平均值(后)', '标准差(后)', '最大值(后)']

    # 按照 三线表 的展示逻辑进行横向拼接交替排列
    print("正在计算统计指标...")
    final_table = pd.concat([
        stats_before[['最小值(前)']], stats_after[['最小值(后)']],
        stats_before[['平均值(前)']], stats_after[['平均值(后)']],
        stats_before[['标准差(前)']], stats_after[['标准差(后)']],
        stats_before[['最大值(前)']], stats_after[['最大值(后)']]
    ], axis=1)

    # 保留 3 位小数，符合学术规范
    final_table = final_table.round(3)

    # 导出为 Excel 文件
    output_filename = os.path.join(data_output_dir, '数据清洗前后对比表_输出.xlsx')
    final_table.to_excel(output_filename, index_label='变量名称')

    print(f"✅ 表格生成成功！已保存为: {output_filename}")
    print("你可以直接打开该 Excel 文件，将其复制到 Word 中套用三线表格式。")


if __name__ == '__main__':
    generate_three_line_table()
