import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import BernoulliRBM, MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RANDOM_SEED = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_output_dir = os.path.join(project_root, "data", "DBN")
image_output_dir = os.path.join(project_root, "image", "DBN")
base_dir = os.path.join(project_root, "data", "datadeal")
os.makedirs(data_output_dir, exist_ok=True)
os.makedirs(image_output_dir, exist_ok=True)

train_path = os.path.join(base_dir, "Train_Data.xlsx")
val_path = os.path.join(base_dir, "Val_Data.xlsx")
test_path = os.path.join(base_dir, "Test_Data.xlsx")

target = "实测宽度-R2-5"
setting_target = "出口宽度设定值-R2-5"
rbm_hidden_layers = (72,)
mlp_hidden_sizes = ()
RUN_OPTION = 1
# 1: 完整训练并导出结果
# 2: 只画图，不重新训练

RBM_LEARNING_RATE = 0.001
RBM_N_ITER = 18
SUPERVISED_MAX_EPOCHS = 75
SUPERVISED_LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10
MIN_IMPROVEMENT = 1e-4
MLP_ALPHA = 5e-4


def evaluate_dataset(name, x_scaled, y_true, model, scaler_y):
    y_pred_scaled = model.predict(x_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    metrics = {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
    }

    print(f"\n{name}指标:")
    print(f"R2 : {metrics['R2']:.4f}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"MSE: {metrics['MSE']:.4f}")
    return y_pred, metrics


def pad_array(values, target_length):
    padded = np.full(target_length, np.nan, dtype=float)
    padded[: len(values)] = values
    return padded


def build_excel_result(y_train, y_val, y_test, train_pred, val_pred, test_pred):
    max_length = max(len(train_pred), len(val_pred), len(test_pred))
    return pd.DataFrame(
        {
            "验证集预测宽度": pad_array(val_pred, max_length),
            "验证集实测宽度": pad_array(y_val, max_length),
            "训练集预测宽度": pad_array(train_pred, max_length),
            "训练集实测宽度": pad_array(y_train, max_length),
            "测试集预测宽度": pad_array(test_pred, max_length),
            "测试集实测宽度": pad_array(y_test, max_length),
        }
    )


def save_excel_with_fallback(dataframe, preferred_path):
    try:
        dataframe.to_excel(preferred_path, index=False)
        return preferred_path
    except PermissionError:
        fallback_path = preferred_path.replace(".xlsx", "_new.xlsx")
        dataframe.to_excel(fallback_path, index=False)
        print(f"\n检测到 {preferred_path} 正在被占用，已改存为: {fallback_path}")
        return fallback_path


def save_figure_with_fallback(figure, preferred_path, dpi=300):
    try:
        figure.savefig(preferred_path, dpi=dpi, bbox_inches="tight")
        return preferred_path
    except PermissionError:
        fallback_path = preferred_path.replace(".png", "_new.png")
        figure.savefig(fallback_path, dpi=dpi, bbox_inches="tight")
        print(f"\n检测到 {preferred_path} 正在被占用，图像已改存为: {fallback_path}")
        return fallback_path
    finally:
        plt.close(figure)


class JointFineTunedDBNRegressor:
    def __init__(self, rbm_hidden_layers, mlp_hidden_sizes, random_state=42):
        self.rbm_hidden_layers = tuple(rbm_hidden_layers)
        self.mlp_hidden_sizes = tuple(mlp_hidden_sizes)
        self.random_state = random_state
        self.rbms = []
        self.regressor = None
        self.best_epoch_ = 0
        self.best_val_rmse_ = np.inf

    @property
    def full_hidden_layers(self):
        return self.rbm_hidden_layers + self.mlp_hidden_sizes

    def _build_rbms(self):
        self.rbms = []
        for index, hidden_size in enumerate(self.rbm_hidden_layers):
            rbm = BernoulliRBM(
                n_components=hidden_size,
                learning_rate=RBM_LEARNING_RATE,
                n_iter=RBM_N_ITER,
                random_state=self.random_state + index,
                verbose=False,
            )
            self.rbms.append(rbm)

    def _pretrain_rbms(self, x_train):
        self._build_rbms()
        transformed = x_train
        for layer_index, rbm in enumerate(self.rbms, start=1):
            print(f"开始预训练第 {layer_index} 个 RBM...")
            rbm.fit(transformed)
            transformed = rbm.transform(transformed)
            print(f"第 {layer_index} 个 RBM 训练完成，隐藏特征维度: {transformed.shape[1]}")

    def _build_finetune_network(self, x_train, y_train):
        self.regressor = MLPRegressor(
            hidden_layer_sizes=self.full_hidden_layers,
            activation="logistic",
            solver="adam",
            learning_rate_init=SUPERVISED_LEARNING_RATE,
            max_iter=1,
            warm_start=True,
            shuffle=True,
            random_state=self.random_state,
            batch_size=min(32, len(x_train)),
            alpha=MLP_ALPHA,
        )

        # Run one light fit so sklearn creates all internal arrays, then
        # replace the first hidden layers with RBM pretrained weights.
        self.regressor.fit(x_train, y_train)

        for layer_index, rbm in enumerate(self.rbms):
            self.regressor.coefs_[layer_index] = rbm.components_.T.copy()
            self.regressor.intercepts_[layer_index] = rbm.intercept_hidden_.copy()

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        self._pretrain_rbms(x_train)
        self._build_finetune_network(x_train, y_train)

        best_weights = [coef.copy() for coef in self.regressor.coefs_]
        best_intercepts = [intercept.copy() for intercept in self.regressor.intercepts_]
        stale_epochs = 0

        print("\n开始整网联合微调...")
        for epoch in range(1, SUPERVISED_MAX_EPOCHS + 1):
            self.regressor.partial_fit(x_train, y_train)

            if x_val is not None and y_val is not None:
                val_pred = self.regressor.predict(x_val)
                val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
            else:
                train_pred = self.regressor.predict(x_train)
                val_rmse = float(np.sqrt(mean_squared_error(y_train, train_pred)))

            print(f"联合微调轮次 {epoch:03d} | 当前RMSE: {val_rmse:.6f}")

            if val_rmse + MIN_IMPROVEMENT < self.best_val_rmse_:
                self.best_val_rmse_ = val_rmse
                self.best_epoch_ = epoch
                best_weights = [coef.copy() for coef in self.regressor.coefs_]
                best_intercepts = [intercept.copy() for intercept in self.regressor.intercepts_]
                stale_epochs = 0
            else:
                stale_epochs += 1

            if stale_epochs >= EARLY_STOPPING_PATIENCE:
                print(f"验证集性能连续 {EARLY_STOPPING_PATIENCE} 轮未提升，提前停止联合微调。")
                break

        self.regressor.coefs_ = best_weights
        self.regressor.intercepts_ = best_intercepts
        print(f"联合微调完成，最佳轮次: {self.best_epoch_}，最佳RMSE: {self.best_val_rmse_:.6f}")
        return self

    def predict(self, x_input):
        return self.regressor.predict(x_input)


def compute_layer_positions(x_coord, node_count, y_min=0.18, y_max=0.82):
    if node_count == 1:
        return [(x_coord, 0.5)]
    y_values = np.linspace(y_min, y_max, node_count)
    return [(x_coord, float(y_value)) for y_value in y_values]


def draw_layer_nodes(axis, positions, radius, edge_color="#333333", face_color="white", linewidth=1.2):
    for x_coord, y_coord in positions:
        axis.add_patch(
            Circle(
                (x_coord, y_coord),
                radius=radius,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=linewidth,
                zorder=3,
            )
        )


def draw_dense_connections(axis, left_positions, right_positions, color, alpha=0.28, linewidth=0.8):
    for left_x, left_y in left_positions:
        for right_x, right_y in right_positions:
            axis.plot(
                [left_x, right_x],
                [left_y, right_y],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                zorder=1,
            )


def add_architecture_group(axis, xy, width, height, edge_color, title):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=1.3,
        edgecolor=edge_color,
        facecolor="none",
        linestyle=(0, (5, 3)),
        zorder=0,
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height + 0.03,
        title,
        ha="center",
        va="bottom",
        fontsize=12,
        color=edge_color,
        fontweight="bold",
    )


def plot_dbn_architecture(feature_count, output_path):
    figure, axis = plt.subplots(figsize=(14, 6), dpi=140)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    add_architecture_group(axis, (0.06, 0.09), 0.27, 0.82, "#3c5a99", "RBM 逐层预训练")
    add_architecture_group(axis, (0.38, 0.09), 0.54, 0.82, "#5aaf5a", "整网联合微调")

    layer_specs = [("输入层\nFeatures", 0.12, 5, "#333333", "white")]
    rbm_x_positions = np.linspace(0.28, 0.46, max(len(rbm_hidden_layers), 1))
    mlp_x_positions = np.linspace(0.66, 0.74, max(len(mlp_hidden_sizes), 1))

    for index, hidden_size in enumerate(rbm_hidden_layers, start=1):
        layer_specs.append((f"RBM 层 {index}\n{hidden_size}", float(rbm_x_positions[index - 1]), 5, "#3c5a99", "white"))

    for index, hidden_size in enumerate(mlp_hidden_sizes, start=1):
        layer_specs.append((f"监督隐层 {index}\n{hidden_size}", float(mlp_x_positions[index - 1]), 4, "#5aaf5a", "#f1fff1"))

    layer_specs.append(("回归输出\n1 Node", 0.86, 1, "#5aaf5a", "#dff3df"))

    layer_positions = []
    for _, x_coord, node_count, edge_color, face_color in layer_specs:
        positions = compute_layer_positions(x_coord, node_count)
        layer_positions.append(positions)
        draw_layer_nodes(axis, positions, radius=0.018, edge_color=edge_color, face_color=face_color, linewidth=1.4 if node_count == 1 else 1.2)

    for left_positions, right_positions in zip(layer_positions[:-1], layer_positions[1:]):
        draw_dense_connections(axis, left_positions, right_positions, color="#8cbf8c")

    axis.text(0.20, 0.5, "预训练权重", ha="center", va="center", fontsize=11, color="#3c5a99")
    axis.text(0.56, 0.5, "反向传播联合更新", ha="center", va="center", fontsize=11, color="#5aaf5a")

    bottom_y = 0.04
    for label, x_coord, _, _, _ in layer_specs:
        axis.text(x_coord, bottom_y, label, ha="center", va="center", fontsize=10, fontweight="bold")

    axis.text(0.12, 0.94, f"输入特征数: {feature_count}", ha="left", va="center", fontsize=10, color="#555555")
    axis.text(0.88, 0.94, "输出: 宽度预测值", ha="right", va="center", fontsize=10, color="#555555")
    return save_figure_with_fallback(figure, output_path)


def draw_flow_box(axis, center, text, width=0.28, height=0.08, face_color="white", edge_color="#2f2f2f"):
    x_coord = center[0] - width / 2
    y_coord = center[1] - height / 2
    patch = FancyBboxPatch(
        (x_coord, y_coord),
        width,
        height,
        boxstyle="round,pad=0.01,rounding_size=0.012",
        linewidth=1.4,
        edgecolor=edge_color,
        facecolor=face_color,
        zorder=2,
    )
    axis.add_patch(patch)
    axis.text(center[0], center[1], text, ha="center", va="center", fontsize=11, zorder=3)


def draw_flow_terminal(axis, center, text, width=0.20, height=0.07):
    x_coord = center[0] - width / 2
    y_coord = center[1] - height / 2
    patch = FancyBboxPatch(
        (x_coord, y_coord),
        width,
        height,
        boxstyle="round,pad=0.01,rounding_size=0.05",
        linewidth=1.5,
        edgecolor="#2f2f2f",
        facecolor="#f8fbff",
        zorder=2,
    )
    axis.add_patch(patch)
    axis.text(center[0], center[1], text, ha="center", va="center", fontsize=11, zorder=3)


def draw_flow_diamond(axis, center, text, width=0.22, height=0.12):
    x_coord, y_coord = center
    points = [
        (x_coord, y_coord + height / 2),
        (x_coord + width / 2, y_coord),
        (x_coord, y_coord - height / 2),
        (x_coord - width / 2, y_coord),
    ]
    patch = Polygon(points, closed=True, edgecolor="#2f2f2f", facecolor="white", linewidth=1.4, zorder=2)
    axis.add_patch(patch)
    axis.text(center[0], center[1], text, ha="center", va="center", fontsize=10.5, zorder=3)


def draw_arrow(axis, start, end, text=None, text_offset=(0.0, 0.0)):
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.4, color="#3a3a3a", shrinkA=0, shrinkB=0),
        zorder=1,
    )
    if text is not None:
        axis.text(
            (start[0] + end[0]) / 2 + text_offset[0],
            (start[1] + end[1]) / 2 + text_offset[1],
            text,
            fontsize=10,
            ha="center",
            va="center",
            color="#444444",
        )


def plot_dbn_flowchart(feature_count, output_path):
    figure, axis = plt.subplots(figsize=(10, 14), dpi=160)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(0.5, 0.965, "DBN.py 代码流程图", ha="center", va="center", fontsize=16, fontweight="bold")
    axis.text(
        0.5,
        0.935,
        f"特征数: {feature_count}    RBM层: {rbm_hidden_layers}    输出层: {mlp_hidden_sizes}",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#555555",
    )

    draw_flow_terminal(axis, (0.5, 0.885), "开始")
    draw_flow_box(axis, (0.5, 0.81), "读取训练集、验证集、测试集", width=0.34, face_color="#f6fbff")
    draw_flow_box(axis, (0.5, 0.73), "提取特征列与目标列", width=0.28, face_color="#f6fbff")
    draw_flow_box(axis, (0.5, 0.645), "输入特征做 MinMaxScaler\n范围 [0.1, 0.9]", width=0.32, height=0.09, face_color="#fff8ef")
    draw_flow_box(axis, (0.5, 0.555), "目标值做 StandardScaler", width=0.28, face_color="#fff8ef")
    draw_flow_box(axis, (0.5, 0.465), "逐层训练 RBM\n得到预训练权重", width=0.30, height=0.09, face_color="#f4fff5")
    draw_flow_box(axis, (0.5, 0.37), "将 RBM 权重注入整网 MLP\n补齐监督隐层与输出层", width=0.34, height=0.09, face_color="#f4fff5")
    draw_flow_box(axis, (0.5, 0.275), "整网反向传播联合微调\n基于验证集早停", width=0.30, height=0.09, face_color="#fefbf3")
    draw_flow_box(axis, (0.5, 0.18), "在训练/验证/测试集上预测\n反标准化并导出指标与图像", width=0.34, height=0.09, face_color="#fefbf3")
    draw_flow_diamond(axis, (0.5, 0.085), "RUN_OPTION\n是否为 2\n只画图模式？", width=0.22, height=0.12)

    draw_flow_box(axis, (0.20, 0.085), "读取已有预测结果 CSV", width=0.24, face_color="#f6fbff")
    draw_flow_box(axis, (0.20, 0.175), "只生成散点图\n和流程图", width=0.22, height=0.085, face_color="#f6fbff")
    draw_flow_terminal(axis, (0.20, 0.265), "结束")
    draw_flow_terminal(axis, (0.82, 0.085), "结束")

    draw_arrow(axis, (0.5, 0.85), (0.5, 0.845))
    draw_arrow(axis, (0.5, 0.77), (0.5, 0.765))
    draw_arrow(axis, (0.5, 0.685), (0.5, 0.68))
    draw_arrow(axis, (0.5, 0.595), (0.5, 0.59))
    draw_arrow(axis, (0.5, 0.505), (0.5, 0.5))
    draw_arrow(axis, (0.5, 0.41), (0.5, 0.405))
    draw_arrow(axis, (0.5, 0.315), (0.5, 0.32))
    draw_arrow(axis, (0.5, 0.225), (0.5, 0.145))
    draw_arrow(axis, (0.61, 0.085), (0.72, 0.085), text="否", text_offset=(0.0, 0.025))

    axis.plot([0.39, 0.30], [0.085, 0.085], color="#3a3a3a", lw=1.4, zorder=1)
    axis.plot([0.30, 0.30], [0.085, 0.175], color="#3a3a3a", lw=1.4, zorder=1)
    axis.annotate("", xy=(0.31, 0.175), xytext=(0.30, 0.175), arrowprops=dict(arrowstyle="->", lw=1.4, color="#3a3a3a"))
    axis.text(0.345, 0.103, "是", fontsize=10, ha="center", va="bottom", color="#444444")
    draw_arrow(axis, (0.20, 0.127), (0.20, 0.133))
    draw_arrow(axis, (0.20, 0.217), (0.20, 0.23))

    return save_figure_with_fallback(figure, output_path)


def plot_prediction_scatter(y_test, test_pred, output_path):
    figure, axis = plt.subplots(figsize=(8, 8))
    axis.scatter(y_test, test_pred, alpha=0.6, color="mediumpurple", edgecolor="k")
    axis.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        lw=2,
        label="理想拟合线",
    )
    axis.set_title("联合微调 DBN 模型宽度预测性能可视化", fontsize=15, fontweight="bold")
    axis.set_xlabel("真实宽度 (mm)", fontsize=12)
    axis.set_ylabel("DBN 预测宽度 (mm)", fontsize=12)
    axis.legend()
    axis.grid(True, linestyle="--", alpha=0.4)
    figure.tight_layout()
    return save_figure_with_fallback(figure, output_path)


def load_datasets():
    df_train = pd.read_excel(train_path)
    df_val = pd.read_excel(val_path)
    df_test = pd.read_excel(test_path)

    df_train.columns = df_train.columns.str.strip()
    df_val.columns = df_val.columns.str.strip()
    df_test.columns = df_test.columns.str.strip()
    return df_train, df_val, df_test


def build_model():
    return JointFineTunedDBNRegressor(
        rbm_hidden_layers=rbm_hidden_layers,
        mlp_hidden_sizes=mlp_hidden_sizes,
        random_state=RANDOM_SEED,
    )


def run_train_and_export():
    print("正在加载全局标准化数据集...")
    df_train, df_val, df_test = load_datasets()

    features = [col for col in df_train.columns if col not in [target, setting_target]]
    X_train, y_train = df_train[features].values, df_train[target].values
    X_val, y_val = df_val[features].values, df_val[target].values
    X_test, y_test = df_test[features].values, df_test[target].values

    print("正在进行数据归一化与标准化...")
    scaler_X = MinMaxScaler(feature_range=(0.1, 0.9))
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

    print("\n开始训练联合微调 DBN 模型...")
    dbn_model = build_model()
    dbn_model.fit(X_train_scaled, y_train_scaled, x_val=X_val_scaled, y_val=y_val_scaled)

    print("\n训练完成，开始在训练集、验证集、测试集上进行预测...")
    train_pred, _ = evaluate_dataset("训练集", X_train_scaled, y_train, dbn_model, scaler_y)
    val_pred, _ = evaluate_dataset("验证集", X_val_scaled, y_val, dbn_model, scaler_y)
    test_pred, _ = evaluate_dataset("测试集", X_test_scaled, y_test, dbn_model, scaler_y)

    result_excel = build_excel_result(
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        train_pred=train_pred,
        val_pred=val_pred,
        test_pred=test_pred,
    )
    excel_result_path = save_excel_with_fallback(
        result_excel,
        os.path.join(data_output_dir, "DBN_预测结果汇总.xlsx"),
    )

    csv_result_path = os.path.join(data_output_dir, "result_DBN_Pure.csv")
    pd.DataFrame({"真实宽度_True": y_test, "预测宽度_DBN": test_pred}).to_csv(
        csv_result_path,
        index=False,
        encoding="utf-8-sig",
    )

    image_result_path = plot_prediction_scatter(
        y_test,
        test_pred,
        os.path.join(image_output_dir, "DBN_Pure_Prediction_Scatter.png"),
    )
    architecture_image_path = plot_dbn_flowchart(
        feature_count=X_train.shape[1],
        output_path=os.path.join(image_output_dir, "DBN_Code_Flowchart.png"),
    )

    print("\n结果文件已保存:")
    print(f"1. {excel_result_path}")
    print(f"2. {csv_result_path}")
    print(f"3. {image_result_path}")
    print(f"4. {architecture_image_path}")


def run_plot_only():
    print("只画图模式：不重新训练 DBN。")
    df_train, _, _ = load_datasets()
    features = [col for col in df_train.columns if col not in [target, setting_target]]

    csv_result_path = os.path.join(data_output_dir, "result_DBN_Pure.csv")
    if not os.path.exists(csv_result_path):
        raise FileNotFoundError(f"未找到已有预测结果文件: {csv_result_path}")

    result_df = pd.read_csv(csv_result_path)
    y_test = result_df["真实宽度_True"].to_numpy(dtype=float)
    test_pred = result_df["预测宽度_DBN"].to_numpy(dtype=float)

    image_result_path = plot_prediction_scatter(
        y_test,
        test_pred,
        os.path.join(image_output_dir, "DBN_Pure_Prediction_Scatter.png"),
    )
    architecture_image_path = plot_dbn_architecture(
        feature_count=len(features),
        output_path=os.path.join(image_output_dir, "DBN_Code_Flowchart.png"),
    )

    print("\n图像文件已保存:")
    print(f"1. {image_result_path}")
    print(f"2. {architecture_image_path}")


if __name__ == "__main__":
    if RUN_OPTION == 1:
        run_train_and_export()
    elif RUN_OPTION == 2:
        run_plot_only()
    else:
        raise ValueError(f"不支持的 RUN_OPTION: {RUN_OPTION}")
