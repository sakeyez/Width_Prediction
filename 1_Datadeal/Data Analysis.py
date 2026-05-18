import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import warnings

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_output_dir = os.path.join(project_root, "data", "datadeal")
image_output_dir = os.path.join(project_root, "image", "datadeal")
source_data_path = os.path.join(project_root, "originaldata", "2160粗轧无设定值数据.xlsx")
os.makedirs(data_output_dir, exist_ok=True)
os.makedirs(image_output_dir, exist_ok=True)

# 忽略一些不必要的 seaborn 警告
warnings.filterwarnings("ignore")


def setup_matplotlib():
    """设置字体和全局绘图参数"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    # 提高全局图表的分辨率和清晰度
    plt.rcParams['figure.dpi'] = 100


def load_and_preprocess_data(file_path, sample_size=1000, random_seed=42):
    """读取数据并进行预处理和抽样"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")

    print("正在读取数据...")
    df = pd.read_excel(file_path)

    # 抽样（如果数据量极大，建议在此之前先清除异常值/空值）
    sample_size = min(sample_size, len(df))
    df_sampled = df.sample(n=sample_size, random_state=random_seed).copy()

    # 如果有不参与绘图的无用列，在这里剔除
    if 'Unnamed: 0' in df_sampled.columns:
        df_sampled.drop(columns=['Unnamed: 0'], inplace=True)

    print(f"数据读取完成，抽样后数据量: {df_sampled.shape[0]} 行")
    return df_sampled


def plot_2d_scatter_grid(df, target, save_dir):
    """绘制适合论文排版的多列二维散点图矩阵"""
    print("正在生成 2D 散点图阵列...")

    feature_groups = [
        ("宽度 / mm", ["板坯宽度实测值(热态)", "实测宽度-R1", "实测宽度-R2-1", "实测宽度-R2-3"]),
        ("温度 / °C", ["出炉实测温度", "道次出口温度-R1", "道次出口温度-R2-1", "道次出口温度-R2-3"]),
        ("压下量 / mm", ["侧压机压下量", "R1压下量Pass1(H11_0)", "R2-1压下量", "R2-3压下量"]),
        ("轧制力 / kN", ["平辊实际轧制力-R1", "平辊实际轧制力-R2-1", "平辊实际轧制力-R2-3", "平辊实际轧制力-R2-5"]),
        ("厚度 / mm", ["板坯厚度热态", "出口厚度-R1", "出口厚度-R2-1", "出口厚度-R2-3"]),
    ]

    selected_features = [feature for _, group in feature_groups for feature in group]
    missing_features = [feature for feature in selected_features if feature not in df.columns]
    if missing_features:
        raise KeyError(f"以下特征在数据中不存在: {missing_features}")

    n_rows = max(len(features) for _, features in feature_groups)
    n_cols = len(feature_groups)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 12), sharey=True)

    for col_idx, (unit_label, features) in enumerate(feature_groups):
        for row_idx, feature in enumerate(features):
            ax = axes[row_idx, col_idx]
            sns.scatterplot(
                data=df, x=feature, y=target, ax=ax,
                color='#1f77b4', s=12, alpha=0.35, edgecolor=None
            )
            ax.text(
                0.03, 0.95, feature,
                transform=ax.transAxes, ha='left', va='top', fontsize=13.8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.75)
            )
            ax.grid(True, linestyle='--', alpha=0.35, linewidth=0.6)
            ax.tick_params(axis='both', which='both', direction='in', length=3.5, width=0.8, labelsize=12.6)

            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color('#666666')

            if row_idx < n_rows - 1:
                ax.set_xlabel("")
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel(unit_label, fontsize=16.8)

            if col_idx > 0:
                ax.set_ylabel("")
                ax.tick_params(axis='y', labelleft=False)
            else:
                ax.set_ylabel("")

    fig.text(0.02, 0.5, f"{target} / mm", va='center', rotation=90, fontsize=18)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.08, top=0.995, wspace=0.0, hspace=0.0)

    save_path = os.path.join(save_dir, f"论文版二维散点图矩阵_{n_rows}x{n_cols}.png")
    svg_path = os.path.splitext(save_path)[0] + ".svg"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"2D 散点分布图已保存至：{save_path}")
    print(f"2D 散点分布图 SVG 已保存至：{svg_path}")


def plot_3d_scatter(df, x_col, y_col, z_col, c_col, save_dir):
    """绘制 3D 核心参数散点图"""
    print("正在生成 3D 散点图...")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    c_data = df[c_col]

    # 优化点：弃用 jet 颜色条（视觉不均匀），改用 viridis（数值越大越黄，越小越紫）
    sc = ax.scatter(
        df[x_col], df[y_col], df[z_col],
        c=c_data, cmap='viridis', s=25, alpha=0.7, edgecolors='none'
    )

    # 设置坐标轴标签（注意单位的修正）
    ax.set_xlabel(f'\n{x_col}\n(mm)', fontsize=14, linespacing=1.5)
    ax.set_ylabel(f'\n{y_col}\n(mm/%)', fontsize=14, linespacing=1.5)  # 压下量单位通常是mm或%
    ax.set_zlabel(f'\n{z_col}\n(kN)', fontsize=14, linespacing=1.5)  # 轧制力单位通常是kN或t，而不是°C

    ax.set_title(f'核心轧制参数与 [{c_col}] 的 3D 映射', fontsize=18, pad=15)

    # 优化颜色条：移除硬编码的 ticks，让 matplotlib 自动根据数据分布计算刻度
    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label(f'{c_col}', fontsize=14)

    # 调整视角 (仰角, 方位角)
    ax.view_init(elev=25, azim=135)

    # 调整面板背景颜色，让图看起来更现代
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.tight_layout()

    save_path = os.path.join(save_dir, "轧制数据3D分布图_静止版.png")
    svg_path = os.path.splitext(save_path)[0] + ".svg"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')

    print(f"3D 散点图已保存至：{save_path}")
    print(f"3D 散点图 SVG 已保存至：{svg_path}")
    plt.close()


def main():
    # 1. 基础配置
    setup_matplotlib()

    file_path = source_data_path
    target_col = '实测宽度-R2-5'

    # 创建保存图片的目录
    save_dir = image_output_dir
    os.makedirs(save_dir, exist_ok=True)

    # 2. 读取并预处理数据 (仅执行一次)
    try:
        df = load_and_preprocess_data(file_path, sample_size=1000)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 3. 绘制 2D 全量散点图
    plot_2d_scatter_grid(df, target_col, save_dir)

    # 4. 绘制 3D 核心参数散点图
    x_col = '板坯宽度实测值(热态)'
    y_col = 'R2-5压下量'
    z_col = '平辊实际轧制力-R2-5'
    plot_3d_scatter(df, x_col, y_col, z_col, target_col, save_dir)


if __name__ == "__main__":
    main()
