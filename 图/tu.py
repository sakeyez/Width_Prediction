import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

# 配置中文字体，确保中文和减号正常显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布，稍微加宽以容纳 6 个层
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_aspect('equal')
ax.axis('off')

# ---------------- 核心参数设置 ----------------
# 6个网络层的 X 坐标 (拉开间距)
x_coords = [0, 3.5, 7.0, 10.5, 14.0, 17.5]

# 每一层节点的 Y 坐标 (中间留出空隙给省略号)
y_L0 = [3.0, 1.5, 0.0, -1.5, -3.0]  # 输入层 (5个节点代表特征)
y_L1 = [3.5, 2.0, -2.0, -3.5]  # RBM层 128节点 (上下各2个，中间留空)
y_L2 = [3.5, 2.0, -2.0, -3.5]  # Scaler层 (形状同上)
y_L3 = [3.5, 2.0, -2.0, -3.5]  # MLP隐藏层1 64节点
y_L4 = [2.5, 1.0, -1.0, -2.5]  # MLP隐藏层2 32节点
y_L5 = [0.0]  # 输出层 1节点

layers = [y_L0, y_L1, y_L2, y_L3, y_L4, y_L5]
node_radius = 0.45  # 节点圆的半径

# ---------------- 1. 绘制层间连接线 (保持绿色箭头风格) ----------------
arrow_color = '#A0C49D'  # 稍微调淡一点连线颜色，避免画面太乱
for i in range(len(layers) - 1):
    x0 = x_coords[i]
    x1 = x_coords[i + 1]

    # 针对输出层，将所有的线汇聚到中心节点
    for y0 in layers[i]:
        for y1 in layers[i + 1]:
            # 计算连接线起点和终点的偏移量，刚好触碰圆的边缘
            dx = x1 - x0
            dy = y1 - y0
            dist = math.hypot(dx, dy)
            ox = (dx / dist) * node_radius
            oy = (dy / dist) * node_radius

            # 使用原来的高颜值箭头样式
            arrow = patches.FancyArrowPatch(
                (x0 + ox, y0 + oy), (x1 - ox, y1 - oy),
                color=arrow_color,
                arrowstyle='-|>,head_length=5,head_width=3',
                lw=1.0, alpha=0.7,
                zorder=1
            )
            ax.add_patch(arrow)

# ---------------- 2. 绘制神经元节点 ----------------
for i, x in enumerate(x_coords):
    for y in layers[i]:
        # 为特殊的节点设置不同的颜色
        if i == 2:  # StandardScaler层填充橘黄色
            face_color = '#FFF3E0'
            edge_color = '#E65100'
        elif i == 5:  # 输出层填充淡绿色
            face_color = '#E8F5E9'
            edge_color = '#2E7D32'
        else:
            face_color = 'white'
            edge_color = 'black'

        circle = patches.Circle((x, y), node_radius, facecolor=face_color,
                                edgecolor=edge_color, lw=1.5, zorder=3)
        ax.add_patch(circle)

# ---------------- 3. 绘制省略号 (表示大量神经元) ----------------
dot_y = [0.5, 0.0, -0.5]  # 省略号的垂直坐标
# 在 RBM、Scaler、隐藏层1、隐藏层2 画省略号
for idx in [1, 2, 3, 4]:
    for y in dot_y:
        ax.plot(x_coords[idx], y, 'ko', markersize=3, zorder=3, color='#555555')


# ---------------- 4. 绘制功能包围框 (复用你的精美虚线框) ----------------
def draw_bounding_box(x_min, y_min, width, height, color, label, label_y, dashes):
    """绘制带圆角的虚线框和顶部标签"""
    box = patches.FancyBboxPatch(
        (x_min, y_min), width, height,
        boxstyle="round,pad=0.4,rounding_size=0.6",
        edgecolor=color, facecolor='none', lw=2.0,
        linestyle=(0, dashes), zorder=0
    )
    ax.add_patch(box)
    ax.text(x_min + width / 2, label_y, label, fontsize=16, fontweight='bold',
            ha='center', va='bottom', color=color)


# 框1：无监督 RBM 特征提取区 (包含输入层和RBM层)
draw_bounding_box(-1.2, -4.5, 5.9, 9.0, '#1E3163', '无监督特征提取 (RBM)', 5.0, dashes=(6, 4))

# 框2：有监督 MLP 回归预测区 (包含隐藏层1、2和输出层)
draw_bounding_box(9.3, -4.5, 9.4, 9.0, '#5F9E52', '有监督回归预测 (MLP)', 5.0, dashes=(6, 4))

# ---------------- 5. 添加文字标签 ----------------
# 底部层名称标签 (增加具体的神经元数量说明)
labels = [
    "输入层\n(Features)",
    "RBM 特征层\n(128 Nodes)",
    "StandardScaler\n(方差重塑)",
    "MLP 隐层 1\n(64 Nodes)",
    "MLP 隐层 2\n(32 Nodes)",
    "输出层\n(1 Node)"
]
for i, x in enumerate(x_coords):
    # 对于特殊层换个颜色标注
    text_color = '#E65100' if i == 2 else 'black'
    ax.text(x, -5.2, labels[i], fontsize=13, ha='center', va='top',
            fontweight='bold', color=text_color)

# ---------------- 6. 调整视图并保存为高清矢量图 ----------------
ax.set_xlim(-2.5, 20)
ax.set_ylim(-7.5, 6.5)

plt.tight_layout()

# 保存为 SVG 和 PNG，论文首选插入 SVG
plt.savefig("My_Hybrid_DBN.svg", format="svg", bbox_inches='tight')
plt.savefig("My_Hybrid_DBN.png", dpi=600, bbox_inches='tight')

plt.show()