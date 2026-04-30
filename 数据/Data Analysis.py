import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import warnings

# 忽略一些不必要的 seaborn 警告
warnings.filterwarnings("ignore")


def setup_matplotlib():
    """设置字体和全局绘图参数"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    # 提高全局图表的分辨率和清晰度
    plt.rcParams['figure.dpi'] = 100


def load_and_preprocess_data(file_path, sample_frac=0.2, random_seed=42):
    """读取数据并进行预处理和抽样"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")

    print("正在读取数据...")
    df = pd.read_excel(file_path)

    # 抽样（如果数据量极大，建议在此之前先清除异常值/空值）
    df_sampled = df.sample(frac=sample_frac, random_state=random_seed).copy()

    # 如果有不参与绘图的无用列，在这里剔除
    if 'Unnamed: 0' in df_sampled.columns:
        df_sampled.drop(columns=['Unnamed: 0'], inplace=True)

    print(f"数据读取完成，抽样后数据量: {df_sampled.shape[0]} 行")
    return df_sampled


def plot_2d_scatter_grid(df, target, save_dir):
    """绘制所有特征与目标变量的 2D 散点分布图"""
    print("正在生成 2D 散点图阵列...")
    features = [col for col in df.columns if col != target]

    n_cols = 5
    n_rows = (len(features) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 4.5 * n_rows))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        # 优化点：加入 alpha 透明度，去除边缘色(edgecolor)，能有效解决点太密集导致“乱”的问题
        sns.scatterplot(
            data=df, x=feature, y=target, ax=axes[i],
            color='#1f77b4', s=15, alpha=0.3, edgecolor=None
        )
        axes[i].set_title(f"{feature} vs 预测宽度", fontsize=12)
        axes[i].set_xlabel(feature, fontsize=10)
        axes[i].set_ylabel(target, fontsize=10)
        axes[i].grid(True, linestyle='--', alpha=0.5)  # 添加淡色网格线辅助查看

    # 删除多余的空白子图
    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=2.0)

    save_path = os.path.join(save_dir, "所有参数散点图_实心精简版.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 2D 散点分布图已保存至：{save_path}")


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
    ax.set_xlabel(f'\n{x_col}\n(mm)', fontsize=11, linespacing=1.5)
    ax.set_ylabel(f'\n{y_col}\n(mm/%)', fontsize=11, linespacing=1.5)  # 压下量单位通常是mm或%
    ax.set_zlabel(f'\n{z_col}\n(kN)', fontsize=11, linespacing=1.5)  # 轧制力单位通常是kN或t，而不是°C

    ax.set_title(f'核心轧制参数与 [{c_col}] 的 3D 映射', fontsize=15, pad=15)

    # 优化颜色条：移除硬编码的 ticks，让 matplotlib 自动根据数据分布计算刻度
    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label(f'{c_col}', fontsize=12)

    # 调整视角 (仰角, 方位角)
    ax.view_init(elev=25, azim=135)

    # 调整面板背景颜色，让图看起来更现代
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.tight_layout()

    save_path = os.path.join(save_dir, "轧制数据3D分布图_静止版.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    print(f"✅ 3D 散点图已保存至：{save_path}")
    plt.show()


def main():
    # 1. 基础配置
    setup_matplotlib()

    file_path = r"F:\asus\Desktop\毕业设计数据\2160粗轧数据.xlsx"
    target_col = '实测宽度-R2-5'

    # 创建保存图片的目录
    save_dir = r"F:\asus\Desktop\毕业设计数据"
    os.makedirs(save_dir, exist_ok=True)

    # 2. 读取并预处理数据 (仅执行一次)
    try:
        df = load_and_preprocess_data(file_path, sample_frac=0.2)
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