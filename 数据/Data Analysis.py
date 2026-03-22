import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# 设置中文字体和负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取 Excel 数据
file_path = r"F:\asus\Desktop\毕业设计数据\数据表格\2160粗轧数据.xlsx"
df = pd.read_excel(file_path)

# 确定预测目标
target = '实测宽度-R2-5'

# 获取参数
features = [col for col in df.columns if col != target and col != 'Unnamed: 0']

# 抽样，约20%
df_sampled = df.sample(frac=0.2, random_state=42)

# 计算画板的布局
n_cols = 5
n_rows = (len(features) + n_cols - 1) // n_cols

# 准备画布（高度会根据行数自动拉长）
fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 5 * n_rows))
axes = axes.flatten()

# 循环画出所有参数的图
for i, feature in enumerate(features):
    sns.scatterplot(data=df_sampled, x=feature, y=target, ax=axes[i], color='blue', s=15)
    axes[i].set_title(f"{feature} 与预测宽度的关系", fontsize=14)
    axes[i].set_xlabel(feature, fontsize=12)
    axes[i].set_ylabel(target, fontsize=12)

# 删除空白画板
for j in range(len(features), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()

# 保存图片
save_path = r"/所有参数散点图_实心精简版.png"
plt.savefig(save_path, dpi=300)  # dpi=300 保证图片放大依然清晰

print(f"散点分布图已生成：{save_path}")


file_path = r"F:\asus\Desktop\毕业设计数据\数据表格\2160粗轧数据.xlsx"
df = pd.read_excel(file_path)

# 确定我们在图表中要展示的四个关键列名
# 请确保你的 Excel 文件包含这些列，否则会报错
x_col = '板坯宽度实测值(热态)'    # X轴：初始宽度
y_col = 'R2-5压下量'             # Y轴：最后一道次压下量
z_col = '平辊实际轧制力-R2-5'    # Z轴：最后一道次轧制力
c_col = '实测宽度-R2-5'          # 颜色深浅：你的终极预测目标！

# 抽样（约20%）
df_sampled = df.sample(frac=0.2, random_state=42)

#创建画布
fig = plt.figure(figsize=(10, 8)) # 设置图片比例
ax = fig.add_subplot(111, projection='3d') # 指定为 3D 投影

# 获取颜色映射的数据
c_data = df_sampled[c_col]

# 绘制散点图
# s=20 设置点的大小
# alpha=0.6 设置透明度，防止点重叠完全遮挡
sc = ax.scatter(df_sampled[x_col],
                df_sampled[y_col],
                df_sampled[z_col],
                c=c_data,
                cmap='jet',
                s=20,
                alpha=0.6,
                edgecolors='none') # 移除点的白色边缘，使画面更整洁


# 设置坐标轴标签
ax.set_xlabel(f'{x_col} (mm)', fontsize=12)
ax.set_ylabel(f'{y_col} (%)', fontsize=12)
ax.set_zlabel(f'{z_col} (°C)', fontsize=12)
# ax.set_title(f'轧制参数与{c_col}的 3D 分布', fontsize=14, fontweight='bold', pad=20)

# 添加颜色条
cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label(f'{c_col} (mm)', fontsize=12)
cbar.set_ticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]) # 可以根据需要调整刻度

# 调整视角
ax.view_init(elev=30, azim=120)

# 自动调整排版
plt.tight_layout()


save_path = r"/轧制数据3D分布图_静止版.png"
# dpi=300 保证图片保存后非常清晰，适合贴入Word
plt.savefig(save_path, dpi=300)

print(f"\n🏆3D 散点图已生成并保存：{save_path}")
# 运行后图片会弹出来（可选项，如果想看一眼）
plt.show()
