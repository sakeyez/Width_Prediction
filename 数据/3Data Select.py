import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
file_path = r"F:\asus\Desktop\毕业设计数据\数据表格\初步处理版数据.xlsx"
df = pd.read_excel(file_path)

target = '实测宽度-R2-5'

#旧代码，第一步中已经删除，不影响运行先保留
domain_irrelevant_features = [
    'Al', 'As', 'B', 'Ca', 'Cr', 'Cu', 'Mo', 'N',
    'Nb', 'Ni', 'O', 'P', 'Pb', 'S', 'Sn', 'Ti', 'V', 'W', 'Zr', 'Fe',
    'Unnamed: 0',
    'R2上工作辊轧制公里', 'R2上工作辊轧制公里.1', 'R2上工作辊轧制公里.2'
]
df = df.drop(columns=domain_irrelevant_features, errors='ignore')


dynamic_cols = []
static_cols = []

# 1. 直接精准定义你想要的 15 个动态特征名
target_dynamic_features = []
for i in range(1, 6):
    target_dynamic_features.extend([
        f"R2-{i}压下量",
        f"R2轧制速度Pass{i}(H11_0)",
        f"平辊实际轧制力-R2-{i}"
    ])

# 2. 检查这些特征是否在读取的 df 中（防止由于空格或微小命名差异报错）
dynamic_cols = [col for col in target_dynamic_features if col in df.columns]

# 3. 剩下的作为静态参数候选池（排除掉 target 和已经选走的动态参数）
potential_static_cols = [
    col for col in df.columns
    if col != target and col not in target_dynamic_features and col not in domain_irrelevant_features
]

# 4. 从静态池中选出 Top 15
print(f"静态候选池共有 {len(potential_static_cols)} 个，开始选拔 Top 15...")
X_static_pool = df[potential_static_cols]
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_static_pool, df[target])

importance_df = pd.DataFrame({'静态特征': potential_static_cols, '重要性': rf_model.feature_importances_})
importance_df = importance_df.sort_values(by='重要性', ascending=False)
top_15_static_cols = importance_df.head(15)['静态特征'].tolist()

# 5. 合并：15个静态 + 15个动态 + 1个标签 = 31列
final_column_order = top_15_static_cols + dynamic_cols + [target]
df_final_lstm = df[final_column_order]

print(f"\n✅ 最终列数验证: {len(df_final_lstm.columns)} (包含 15静 + 15动 + 1目标)")


# 写一个提取道次数字的小工具
def get_pass_num(col_name):
    if '-1' in col_name or 'Pass1' in col_name: return 1
    if '-2' in col_name or 'Pass2' in col_name: return 2
    if '-3' in col_name or 'Pass3' in col_name: return 3
    if '-4' in col_name or 'Pass4' in col_name: return 4
    if '-5' in col_name or 'Pass5' in col_name: return 5
    return 99  # 防御性编程，如果找不到数字就扔到最后

# 利用小工具对动态参数进行多重排序（先按道次排，同道次的按拼音/字母排）
dynamic_cols.sort(key=lambda x: (get_pass_num(x), x))

print(f"\n--- 保留并排好序的动态时序参数共 {len(dynamic_cols)} 个 ---")
# 打印前几个给你看看排序效果
print("排序展示 (前5个):", dynamic_cols[:5])


final_column_order = top_15_static_cols + dynamic_cols + [target]

# 按照新顺序从原始大表里抽取数据
df_final_lstm = df[final_column_order]



plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(15), x='重要性', y='静态特征', hue='静态特征', palette='mako', legend=False)
plt.title('送入全连接层的 Top 15 静态核心特征', fontsize=14)
plt.tight_layout()
plt.savefig(r"F:\宽度预测\静态特征Top15排名图.png", dpi=300)

potential_static_cols = [col for col in df.columns if col != target and col not in dynamic_cols]

if len(potential_static_cols) == 0:
    print("❌ 错误：未找到有效的静态特征列，请检查 dynamic_cols 是否包含所有特征。")
else:
    print(f"📊 正在从 {len(potential_static_cols)} 个静态特征中筛选 Top 15 并生成权重图...")

    # 2. 训练随机森林计算重要性
    X_static_temp = df[potential_static_cols]
    y_temp = df[target]

    # n_jobs=-1 开启全核计算加速
    rf_selector = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector.fit(X_static_temp, y_temp)

    # 3. 构造重要性数据框并取前 15 名
    importance_df = pd.DataFrame({
        'Feature': potential_static_cols,
        'Importance': rf_selector.feature_importances_ * 100  # 转化为百分比
    }).sort_values(by='Importance', ascending=False)

    top_15_static_cols = importance_df.head(15)['Feature'].tolist()

    # 4. 绘制【静态参数权重比例图】
    plt.figure(figsize=(12, 8))
    # 取前15名进行绘图
    plot_data = importance_df.head(15)

    # 绘制横向柱状图
    ax = sns.barplot(data=plot_data, x='Importance', y='Feature', palette='viridis', hue='Feature', legend=False)

    # 在柱条末尾添加百分比数值标签
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f'{width:.2f}%',
                    (width + 0.2, p.get_y() + p.get_height() / 2),
                    va='center', fontsize=11, color='black')

    plt.title('Top 15 静态核心特征对宽度预测的贡献权重 (%)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('贡献权重 (Importance %)', fontsize=12)
    plt.ylabel('静态工艺参数', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 保存权重比例图
    weight_fig_path = r"F:\宽度预测\静态特征权重比例图.png"
    plt.savefig(weight_fig_path, dpi=300)
    print(f"✅ 权重比例图已保存至: {weight_fig_path}")

    # 5. 按照 15静 + 15动 + 1目标 的顺序重新整理列并保存 Excel
    final_column_order = top_15_static_cols + dynamic_cols + [target]
    df_final = df[final_column_order]

    save_path = r"F:\asus\Desktop\毕业设计数据\数据表格\最终数据.xlsx"
    try:
        df_final.to_excel(save_path, index=False)
        print(f"🚀 最终数据表(31列)已成功保存至: {save_path}")
    except PermissionError:
        print(f"❌ 权限错误：请关闭正在打开的 Excel 文件 '{save_path}' 后重试。")