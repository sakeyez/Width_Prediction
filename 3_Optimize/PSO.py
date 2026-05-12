import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "MGH")

TEST_DATA_PATH = os.path.join(DATA_OUTPUT_DIR, "MGH_Test_Expanded.xlsx")
MODEL_BUNDLE_PATH = os.path.join(DATA_OUTPUT_DIR, "final_glr_model.pkl")
RESULT_EXCEL_PATH = os.path.join(DATA_OUTPUT_DIR, "PSO_Optimization_Results.xlsx")
RESULT_SUMMARY_PATH = os.path.join(DATA_OUTPUT_DIR, "PSO_Optimization_Summary.json")

NUM_PARTICLES = 30
MAX_ITERATIONS = 50
SEARCH_MARGIN_RATIO = 0.2
ZERO_FALLBACK_MARGIN = 0.1
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def load_inputs():
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"缺少测试集文件: {TEST_DATA_PATH}")
    if not os.path.exists(MODEL_BUNDLE_PATH):
        raise FileNotFoundError(f"缺少新版 MGH 模型文件: {MODEL_BUNDLE_PATH}")

    test_df = pd.read_excel(TEST_DATA_PATH)
    with open(MODEL_BUNDLE_PATH, "rb") as file_obj:
        bundle = pickle.load(file_obj)

    required_keys = [
        "model",
        "selected_features",
        "target",
        "setting_target",
        "optimization_columns",
        "sequence_groups",
    ]
    missing_keys = [key for key in required_keys if key not in bundle]
    if missing_keys:
        raise KeyError(f"模型文件缺少必要字段: {missing_keys}")

    return test_df, bundle


def update_engineered_features(row, changed_columns, sequence_groups):
    for column in changed_columns:
        square_column = f"{column}_Square"
        if square_column in row.index:
            row[square_column] = row[column] ** 2

    changed_set = set(changed_columns)
    for group_name, group_columns in sequence_groups.items():
        valid_columns = [column for column in group_columns if column in row.index]
        if len(valid_columns) < 2 or not changed_set.intersection(valid_columns):
            continue

        for index in range(1, len(valid_columns)):
            prev_column = valid_columns[index - 1]
            curr_column = valid_columns[index]
            diff_name = f"{group_name}_Diff_{index}"
            abs_diff_name = f"{group_name}_AbsDiff_{index}"

            diff_value = row[curr_column] - row[prev_column]
            if diff_name in row.index:
                row[diff_name] = diff_value
            if abs_diff_name in row.index:
                row[abs_diff_name] = abs(diff_value)

    return row


def apply_candidate_values(base_row, optimization_columns, candidate_values, sequence_groups):
    updated_row = base_row.copy()
    for index, column in enumerate(optimization_columns):
        updated_row[column] = candidate_values[index]

    return update_engineered_features(updated_row, optimization_columns, sequence_groups)


def build_bounds(row, optimization_columns):
    bounds = []
    for column in optimization_columns:
        if column not in row.index:
            raise KeyError(f"测试数据中缺少待优化列: {column}")
        value = float(row[column])
        margin = abs(value) * SEARCH_MARGIN_RATIO if value != 0 else ZERO_FALLBACK_MARGIN
        bounds.append((value - margin, value + margin))
    return bounds


def predict_width(model, row, selected_features):
    x_input = row[selected_features].to_numpy(dtype=float).reshape(1, -1)
    return float(model.predict(x_input)[0])


def pso_optimize_row(base_row, target_width, model, selected_features, optimization_columns, sequence_groups, bounds):
    dim = len(optimization_columns)

    positions = np.random.uniform(
        low=[bound[0] for bound in bounds],
        high=[bound[1] for bound in bounds],
        size=(NUM_PARTICLES, dim),
    )
    velocities = np.zeros((NUM_PARTICLES, dim), dtype=float)

    pbest_positions = positions.copy()
    pbest_scores = np.full(NUM_PARTICLES, np.inf, dtype=float)
    gbest_position = positions[0].copy()
    gbest_score = float("inf")

    inertia_weight = 0.5
    individual_factor = 1.5
    social_factor = 1.5

    def evaluate(candidate_values):
        candidate_row = apply_candidate_values(
            base_row=base_row,
            optimization_columns=optimization_columns,
            candidate_values=candidate_values,
            sequence_groups=sequence_groups,
        )
        predicted_width = predict_width(model, candidate_row, selected_features)
        return abs(predicted_width - target_width), predicted_width

    for _ in range(MAX_ITERATIONS):
        for particle_index in range(NUM_PARTICLES):
            score, _ = evaluate(positions[particle_index])
            if score < pbest_scores[particle_index]:
                pbest_scores[particle_index] = score
                pbest_positions[particle_index] = positions[particle_index].copy()
            if score < gbest_score:
                gbest_score = score
                gbest_position = positions[particle_index].copy()

        r1 = np.random.rand(NUM_PARTICLES, dim)
        r2 = np.random.rand(NUM_PARTICLES, dim)
        velocities = (
            inertia_weight * velocities
            + individual_factor * r1 * (pbest_positions - positions)
            + social_factor * r2 * (gbest_position - positions)
        )
        positions = positions + velocities

        for dim_index in range(dim):
            positions[:, dim_index] = np.clip(
                positions[:, dim_index],
                bounds[dim_index][0],
                bounds[dim_index][1],
            )

    optimized_row = apply_candidate_values(
        base_row=base_row,
        optimization_columns=optimization_columns,
        candidate_values=gbest_position,
        sequence_groups=sequence_groups,
    )
    optimized_prediction = predict_width(model, optimized_row, selected_features)
    return gbest_position, optimized_prediction


def build_result_record(row_index, original_row, optimized_values, target_width, real_width, pred_before, pred_after, optimization_columns):
    record = {
        "row_index": int(row_index),
        "target_width": float(target_width),
        "real_width": float(real_width),
        "pred_before": float(pred_before),
        "pred_after": float(pred_after),
        "real_error_before": float(abs(target_width - real_width)),
        "model_error_before": float(abs(target_width - pred_before)),
        "model_error_after": float(abs(target_width - pred_after)),
        "improvement_vs_real": float(abs(target_width - real_width) - abs(target_width - pred_after)),
        "improvement_vs_model": float(abs(target_width - pred_before) - abs(target_width - pred_after)),
    }

    for index, column in enumerate(optimization_columns):
        record[f"original_{column}"] = float(original_row[column])
        record[f"optimized_{column}"] = float(optimized_values[index])

    return record


def main():
    test_df, bundle = load_inputs()

    model = bundle["model"]
    selected_features = bundle["selected_features"]
    target_column = bundle["target"]
    setting_target_column = bundle["setting_target"]
    optimization_columns = bundle["optimization_columns"]
    sequence_groups = bundle["sequence_groups"]

    required_columns = [target_column, setting_target_column, *optimization_columns]
    missing_columns = [column for column in required_columns if column not in test_df.columns]
    if missing_columns:
        raise KeyError(f"测试集缺少必要列: {missing_columns}")

    print("开始执行新版 MGH 的 PSO 优化...")
    print(f"模型文件: {MODEL_BUNDLE_PATH}")
    print(f"待优化列: {optimization_columns}")

    results = []
    for row_index, row in test_df.iterrows():
        base_row = row.copy()
        target_width = float(base_row[setting_target_column])
        real_width = float(base_row[target_column])
        pred_before = predict_width(model, base_row, selected_features)
        bounds = build_bounds(base_row, optimization_columns)

        best_values, pred_after = pso_optimize_row(
            base_row=base_row,
            target_width=target_width,
            model=model,
            selected_features=selected_features,
            optimization_columns=optimization_columns,
            sequence_groups=sequence_groups,
            bounds=bounds,
        )

        results.append(
            build_result_record(
                row_index=row_index,
                original_row=base_row,
                optimized_values=best_values,
                target_width=target_width,
                real_width=real_width,
                pred_before=pred_before,
                pred_after=pred_after,
                optimization_columns=optimization_columns,
            )
        )

        if (row_index + 1) % 50 == 0:
            print(f"已完成 {row_index + 1} / {len(test_df)} 条样本优化")

    results_df = pd.DataFrame(results)
    results_df.to_excel(RESULT_EXCEL_PATH, index=False)

    summary = {
        "num_rows": int(len(results_df)),
        "mean_real_error_before": float(results_df["real_error_before"].mean()),
        "mean_model_error_before": float(results_df["model_error_before"].mean()),
        "mean_model_error_after": float(results_df["model_error_after"].mean()),
        "mean_improvement_vs_real": float(results_df["improvement_vs_real"].mean()),
        "mean_improvement_vs_model": float(results_df["improvement_vs_model"].mean()),
        "optimization_columns": optimization_columns,
    }
    with open(RESULT_SUMMARY_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print("-" * 50)
    print(f"平均真实误差基线: {summary['mean_real_error_before']:.4f} mm")
    print(f"平均模型优化前误差: {summary['mean_model_error_before']:.4f} mm")
    print(f"平均模型优化后误差: {summary['mean_model_error_after']:.4f} mm")
    print(f"相对真实值平均改善: {summary['mean_improvement_vs_real']:.4f} mm")
    print(f"相对模型预测平均改善: {summary['mean_improvement_vs_model']:.4f} mm")
    print("-" * 50)
    print(f"优化明细已保存到: {RESULT_EXCEL_PATH}")
    print(f"优化摘要已保存到: {RESULT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
