import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# 1. 设置中文字体，防止中文显示为方块
# Windows 系统通常使用 'SimHei' (黑体)
# Mac 系统建议改为 'Arial Unicode MS' 或 'PingFang SC'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 2. 生成模拟数据 (为了尽量还原图片中的分布形态)
np.random.seed(42) # 设置随机种子保证每次生成一样的数据
# 中间温度：中位数大概在650，有下方密集的异常值和一个上方异常值
data_mid = np.random.normal(loc=650, scale=30, size=100)
data_mid = np.append(data_mid, [800]) # 上方异常值
data_mid = np.append(data_mid, np.random.uniform(510, 550, 15)) # 下方密集异常值

# 卷取温度：数据分布较广，中位数偏下，无明显异常值
data_coil = np.random.normal(loc=520, scale=60, size=100)

data = [data_mid, data_coil]

# 3. 创建画布
fig, ax = plt.subplots(figsize=(7, 6))

# 4. 绘制箱线图
# patch_artist=True 允许为箱体填充颜色
# showmeans=True 允许显示均值
bplot = ax.boxplot(data,
                   patch_artist=True,
                   showmeans=True,
                   widths=0.4, # 箱体宽度
                   # 设置均值（红星）样式
                   meanprops={'marker': '*', 'markerfacecolor': 'red', 'markeredgecolor': 'red', 'markersize': 8},
                   # 设置异常值（黑点）样式
                   flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markersize': 3, 'linestyle': 'none'},
                   # 设置中位线（黑色实线）样式
                   medianprops={'color': 'black', 'linewidth': 1})

# 5. 设置箱体颜色
colors = ['#F58220', '#1F6A8B'] # 近似原图的橙色和深蓝色
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('black') # 箱体边框颜色

# 6. 设置坐标轴标签和刻度
ax.set_xticks([1, 2])
ax.set_xticklabels(['中间温度', '卷取温度'], fontsize=12)
ax.set_ylabel('温度(℃)', fontsize=12)

# 设置Y轴范围和刻度，与原图保持一致 (350 到 850)
ax.set_ylim(350, 850)
ax.set_yticks(np.arange(350, 900, 50))

# 7. 手动构建自定义图例
# 原图的图例比较复杂，包含箱体色块、线条和散点，需要用 mpatches 和 mlines 组合构建
legend_box = mpatches.Patch(facecolor='#F58220', edgecolor='black', label='25%~75%')
# 用类似误差棒的形状表示 1.5IQR
legend_whisker = mlines.Line2D([], [], color='black', marker='|', markersize=12, markeredgewidth=1.5, label='1.5IQR内的范围', linestyle='-')
legend_median = mlines.Line2D([], [], color='black', label='中位线')
legend_mean = mlines.Line2D([], [], color='white', marker='*', markerfacecolor='red', markeredgecolor='red', markersize=10, label='均值')
legend_flier = mlines.Line2D([], [], color='white', marker='o', markerfacecolor='black', markersize=4, label='异常值')

# 添加图例到右上角，取消图例边框 (frameon=False)
ax.legend(handles=[legend_box, legend_whisker, legend_median, legend_mean, legend_flier],
          loc='upper right', frameon=False, fontsize=10)

# 8. 添加底部图片标题
# y=-0.1 代表将其放置在X轴下方
plt.figtext(0.5, 0.02, '图 2.3 中间温度、卷取温度箱线图', ha='center', fontsize=14)

# 调整布局以防止底部标题被遮挡
plt.subplots_adjust(bottom=0.15)

# 9. 显示图片
plt.show()