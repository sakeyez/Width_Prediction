import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


plt.style.use('seaborn-v0_8-whitegrid')



def draw_parity_plot(ax, df_path, model_name, color):

    try:
        df = pd.read_csv(df_path)
    except FileNotFoundError:
        print(f"❌ Error: File {df_path} not found.")
        ax.set_title(f"{model_name} (Data Missing)", color='red')
        return

    y_true = df['y_test'].values
    y_pred = df['y_pred'].values

    # 计算指标
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # 绘制散点
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='w', linewidth=0.3, s=30, c=color)

    # 画完美预测对角线
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label='Target Line ($y=x$)')

    # 细节美化 (全英文标签)
    ax.set_title(f'{model_name}', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Measured Final Width (mm)', fontsize=12)
    ax.set_ylabel('Predicted Final Width (mm)', fontsize=12)

    # 1:1 比例
    ax.set_aspect('equal', adjustable='box')

    # 成绩印章
    textstr = '\n'.join((
        r'$R^2=%.4f$' % (r2,),
        r'$MAE=%.2f$ mm' % (mae,),
        r'$MSE=%.1f$' % (mse,)))

    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    # 只有第一个图加图例
    if 'Proposed' in model_name:
        ax.legend(loc='lower right', fontsize=11)


# ==========================================
# 2. 主程序：构建 2×2 的四子图擂台
# ==========================================
print("\n🔥 Starting plot generation...")

fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=90)
ax_list = axs.flatten()

# 纯英文模型名称，高端大气
model_infos = [
    ('result_LSTM.csv', 'LSTM+Attention', '#1f77b4'),
    ('result_RF.csv', 'Random Forest', '#ff7f0e'),
    ('result_BPNN.csv', 'BPNN', '#2ca02c'),
    ('result_LightGBM.csv', 'LightGBM', '#d62728')
]

for i, info in enumerate(model_infos):
    draw_parity_plot(ax_list[i], info[0], info[1], info[2])

plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=2.0)
plt.show()

print("\n🏆 Plot generation complete!")