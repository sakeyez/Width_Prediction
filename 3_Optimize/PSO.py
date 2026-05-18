import json
import os
import pickle
import sys
import __main__
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MGH_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "MGH")
DATA_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "PSO")
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

TEST_DATA_PATH = os.path.join(MGH_DATA_DIR, "MGH_Test_Expanded.xlsx")
MODEL_BUNDLE_PATH = os.path.join(MGH_DATA_DIR, "final_glr_model.pkl")
RESULT_EXCEL_PATH = os.path.join(DATA_OUTPUT_DIR, "PSO_Optimization_Results.xlsx")
RESULT_SUMMARY_PATH = os.path.join(DATA_OUTPUT_DIR, "PSO_Optimization_Summary.json")
TABLE_41_PATH = os.path.join(DATA_OUTPUT_DIR, "4.1.xlsx")
TABLE_42_PATH = os.path.join(DATA_OUTPUT_DIR, "4.2.xlsx")
TABLE_41_TOP20_PATH = os.path.join(DATA_OUTPUT_DIR, "4.1_top20.xlsx")
TABLE_42_TOP20_PATH = os.path.join(DATA_OUTPUT_DIR, "4.2_top20.xlsx")
SENSITIVITY_DETAIL_PATH = os.path.join(DATA_OUTPUT_DIR, "PSO_Sensitivity_Detail.xlsx")
SENSITIVITY_SUMMARY_PATH = os.path.join(DATA_OUTPUT_DIR, "PSO_Sensitivity_Summary.xlsx")

OPTIMIZE_SAMPLE_COUNT = 100
TOP_SAMPLE_COUNT = 20
DEFAULT_MENU_OPTION = "1"

ZERO_FALLBACK_MARGIN = 0.05
RANDOM_SEED = 42

INERTIA_WEIGHT = 0.50
INDIVIDUAL_FACTOR = 1.35
SOCIAL_FACTOR = 1.45

ERROR_WEIGHT = 1.0
UNDER_ERROR_THRESHOLD = 1.0
HIGH_ERROR_THRESHOLD = 20.0

# 误差 > 20 mm: 维持更激进的搜索
AGGRESSIVE_NUM_PARTICLES = 36
AGGRESSIVE_MAX_ITERATIONS = 60
AGGRESSIVE_SEARCH_MARGIN_RATIO = 0.55
AGGRESSIVE_MAX_ADJUSTMENT_MM = 8.0
AGGRESSIVE_ADJUSTMENT_WEIGHT = 0.12
AGGRESSIVE_UNDER_ERROR_WEIGHT = 27.0

# 误差 <= 20 mm: 用更保守的小范围搜索，避免大量被解到 0
CONSERVATIVE_NUM_PARTICLES = 36
CONSERVATIVE_MAX_ITERATIONS = 60
CONSERVATIVE_SEARCH_MARGIN_RATIO = 0.55
CONSERVATIVE_MAX_ADJUSTMENT_MM = 8.0
CONSERVATIVE_ADJUSTMENT_WEIGHT = 0.12
CONSERVATIVE_UNDER_ERROR_WEIGHT = 27.0

COL_SAMPLE_INDEX = "样本序号"
COL_TARGET = "设定宽度"
COL_REAL = "实测宽度"
COL_PRED_BEFORE = "优化前预测宽度"
COL_PRED_AFTER = "优化后预测宽度"
COL_REAL_ERR = "实测值与设定值绝对偏差"
COL_MODEL_ERR_BEFORE = "优化前模型与设定值绝对偏差"
COL_MODEL_ERR_AFTER = "优化后模型与设定值绝对偏差"
COL_PRED_REAL_ERR_BEFORE = "优化前预测值与实测值绝对偏差"
COL_PRED_REAL_ERR_AFTER = "优化后预测值与实测值绝对偏差"
COL_IMPROVE_MODEL = "相对模型改善量"
COL_OBJECTIVE = "目标函数值"
COL_MEAN_ADJ = "平均绝对调整量"
COL_MAX_ADJ = "最大绝对调整量"
COL_SIGNED_BEFORE = "优化前宽度偏差量"
COL_SIGNED_AFTER = "优化后宽度偏差量"

np.random.seed(RANDOM_SEED)


@dataclass
class PredictOnlyLinearModel:
    coef_: np.ndarray
    intercept_: float

    def predict(self, x_input):
        x_array = np.asarray(x_input, dtype=float)
        return x_array @ self.coef_ + self.intercept_


def load_inputs():
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"未找到测试集文件: {TEST_DATA_PATH}")
    if not os.path.exists(MODEL_BUNDLE_PATH):
        raise FileNotFoundError(f"未找到模型文件: {MODEL_BUNDLE_PATH}")

    test_df = pd.read_excel(TEST_DATA_PATH)
    __main__.PredictOnlyLinearModel = PredictOnlyLinearModel
    with open(MODEL_BUNDLE_PATH, "rb") as file_obj:
        bundle = pickle.load(file_obj)
    return test_df, bundle


def build_scaler_maps(bundle):
    payload = bundle.get("datadeal_scaler_payload")
    if not payload:
        raise KeyError("模型包中缺少 datadeal_scaler_payload，无法进行反归一化优化。")

    feature_names = payload["feature_names"]
    scale_values = payload["scale"]
    min_values = payload["min"]
    data_min_values = payload["data_min"]
    scale_map = {feature_names[index]: float(scale_values[index]) for index in range(len(feature_names))}
    min_map = {feature_names[index]: float(min_values[index]) for index in range(len(feature_names))}
    data_min_map = {feature_names[index]: float(data_min_values[index]) for index in range(len(feature_names))}
    return scale_map, min_map, data_min_map


def inverse_scale_feature(scaled_value, column_name, scale_map, min_map, data_min_map):
    scale_value = scale_map.get(column_name)
    min_value = min_map.get(column_name)
    if scale_value is None or min_value is None:
        raise KeyError(f"scaler 中缺少列 {column_name} 的缩放信息。")
    if abs(scale_value) < 1e-12:
        return float(data_min_map.get(column_name, 0.0))
    return float((scaled_value - min_value) / scale_value)


def forward_scale_feature(raw_value, column_name, scale_map, min_map):
    scale_value = scale_map.get(column_name)
    min_value = min_map.get(column_name)
    if scale_value is None or min_value is None:
        raise KeyError(f"scaler 中缺少列 {column_name} 的缩放信息。")
    return float(raw_value * scale_value + min_value)


def get_search_regime(baseline_error):
    if baseline_error > HIGH_ERROR_THRESHOLD:
        return {
            "name": "aggressive",
            "num_particles": AGGRESSIVE_NUM_PARTICLES,
            "max_iterations": AGGRESSIVE_MAX_ITERATIONS,
            "search_margin_ratio": AGGRESSIVE_SEARCH_MARGIN_RATIO,
            "max_adjustment_mm": AGGRESSIVE_MAX_ADJUSTMENT_MM,
            "adjustment_weight": AGGRESSIVE_ADJUSTMENT_WEIGHT,
            "under_error_weight": AGGRESSIVE_UNDER_ERROR_WEIGHT,
        }
    return {
        "name": "conservative",
        "num_particles": CONSERVATIVE_NUM_PARTICLES,
        "max_iterations": CONSERVATIVE_MAX_ITERATIONS,
        "search_margin_ratio": CONSERVATIVE_SEARCH_MARGIN_RATIO,
        "max_adjustment_mm": CONSERVATIVE_MAX_ADJUSTMENT_MM,
        "adjustment_weight": CONSERVATIVE_ADJUSTMENT_WEIGHT,
        "under_error_weight": CONSERVATIVE_UNDER_ERROR_WEIGHT,
    }


def resolve_optimization_columns(bundle, dataframe):
    optimization_columns = list(bundle["optimization_columns"])
    side_pressure_column = "侧压机压下量"
    if side_pressure_column in dataframe.columns and side_pressure_column not in optimization_columns:
        optimization_columns = [side_pressure_column] + optimization_columns
    return optimization_columns


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


def apply_candidate_values(base_row, optimization_columns, candidate_values, sequence_groups, scale_map, min_map):
    updated_row = base_row.copy()
    for index, column in enumerate(optimization_columns):
        updated_row[column] = forward_scale_feature(candidate_values[index], column, scale_map, min_map)
    return update_engineered_features(updated_row, optimization_columns, sequence_groups)


def build_bounds(row, optimization_columns, scale_map, min_map, data_min_map, regime):
    bounds = []
    for column in optimization_columns:
        raw_value = inverse_scale_feature(float(row[column]), column, scale_map, min_map, data_min_map)
        base_margin = abs(raw_value) * regime["search_margin_ratio"] if raw_value != 0 else ZERO_FALLBACK_MARGIN
        margin = min(max(base_margin, ZERO_FALLBACK_MARGIN), regime["max_adjustment_mm"])
        bounds.append((raw_value - margin, raw_value + margin))
    return bounds


def predict_width(model, row, selected_features):
    x_input = row[selected_features].to_numpy(dtype=float).reshape(1, -1)
    return float(model.predict(x_input)[0])


def compute_adjustment_penalty(candidate_values, original_values, max_adjustment_mm):
    absolute_delta = np.abs(np.asarray(candidate_values, dtype=float) - np.asarray(original_values, dtype=float))
    normalized_delta = np.clip(absolute_delta / max(max_adjustment_mm, 1e-8), 0.0, 1.0)
    return float(np.mean(normalized_delta ** 2))


def compute_under_error_penalty(width_error):
    if width_error >= UNDER_ERROR_THRESHOLD:
        return 0.0
    shortage = UNDER_ERROR_THRESHOLD - width_error
    normalized_shortage = shortage / UNDER_ERROR_THRESHOLD
    return float(normalized_shortage ** 2)


def evaluate_candidate(
    candidate_values,
    base_row,
    target_width,
    model,
    selected_features,
    optimization_columns,
    sequence_groups,
    original_values,
    scale_map,
    min_map,
    regime,
):
    candidate_row = apply_candidate_values(
        base_row=base_row,
        optimization_columns=optimization_columns,
        candidate_values=candidate_values,
        sequence_groups=sequence_groups,
        scale_map=scale_map,
        min_map=min_map,
    )
    predicted_width = predict_width(model, candidate_row, selected_features)
    width_error = abs(predicted_width - target_width)
    adjustment_penalty = compute_adjustment_penalty(
        candidate_values=candidate_values,
        original_values=original_values,
        max_adjustment_mm=regime["max_adjustment_mm"],
    )
    under_error_penalty = compute_under_error_penalty(width_error)
    objective_score = (
        ERROR_WEIGHT * width_error
        + regime["adjustment_weight"] * adjustment_penalty
        + regime["under_error_weight"] * under_error_penalty
    )
    return float(objective_score), {
        "predicted_width": float(predicted_width),
        "width_error": float(width_error),
        "adjustment_penalty": float(adjustment_penalty),
        "under_error_penalty": float(under_error_penalty),
        "regime_name": regime["name"],
    }


def pso_optimize_row(
    base_row,
    target_width,
    model,
    selected_features,
    optimization_columns,
    sequence_groups,
    scale_map,
    min_map,
    data_min_map,
):
    dim = len(optimization_columns)
    original_values = np.array(
        [
            inverse_scale_feature(float(base_row[column]), column, scale_map, min_map, data_min_map)
            for column in optimization_columns
        ],
        dtype=float,
    )

    baseline_predicted_width = predict_width(model, base_row, selected_features)
    baseline_error = abs(baseline_predicted_width - target_width)
    regime = get_search_regime(baseline_error)
    bounds = build_bounds(base_row, optimization_columns, scale_map, min_map, data_min_map, regime)
    lower_bounds = np.array([bound[0] for bound in bounds], dtype=float)
    upper_bounds = np.array([bound[1] for bound in bounds], dtype=float)

    positions = np.random.uniform(
        low=lower_bounds,
        high=upper_bounds,
        size=(regime["num_particles"], dim),
    )
    positions[0] = original_values.copy()
    velocities = np.zeros((regime["num_particles"], dim), dtype=float)

    baseline_score, baseline_metrics = evaluate_candidate(
        candidate_values=original_values,
        base_row=base_row,
        target_width=target_width,
        model=model,
        selected_features=selected_features,
        optimization_columns=optimization_columns,
        sequence_groups=sequence_groups,
        original_values=original_values,
        scale_map=scale_map,
        min_map=min_map,
        regime=regime,
    )
    best_position = original_values.copy()
    best_score = baseline_score
    best_metrics = baseline_metrics

    pbest_positions = positions.copy()
    pbest_scores = np.full(regime["num_particles"], np.inf, dtype=float)

    for _ in range(regime["max_iterations"]):
        for particle_index in range(regime["num_particles"]):
            score, metrics = evaluate_candidate(
                candidate_values=positions[particle_index],
                base_row=base_row,
                target_width=target_width,
                model=model,
                selected_features=selected_features,
                optimization_columns=optimization_columns,
                sequence_groups=sequence_groups,
                original_values=original_values,
                scale_map=scale_map,
                min_map=min_map,
                regime=regime,
            )
            if score < pbest_scores[particle_index]:
                pbest_scores[particle_index] = score
                pbest_positions[particle_index] = positions[particle_index].copy()
            if score < best_score:
                best_score = score
                best_position = positions[particle_index].copy()
                best_metrics = metrics

        r1 = np.random.rand(regime["num_particles"], dim)
        r2 = np.random.rand(regime["num_particles"], dim)
        velocities = (
            INERTIA_WEIGHT * velocities
            + INDIVIDUAL_FACTOR * r1 * (pbest_positions - positions)
            + SOCIAL_FACTOR * r2 * (best_position - positions)
        )
        positions = np.clip(positions + velocities, lower_bounds, upper_bounds)

    return best_position, best_metrics, best_score, regime


def select_rows_for_optimization(test_df, setting_target_column, model, selected_features, sample_count):
    usable_df = test_df[test_df[setting_target_column].notna()].copy()
    usable_df[COL_PRED_BEFORE] = usable_df.apply(lambda row: predict_width(model, row, selected_features), axis=1)
    usable_df[COL_MODEL_ERR_BEFORE] = (usable_df[COL_PRED_BEFORE] - usable_df[setting_target_column]).abs()
    usable_df = usable_df.sort_values(by=[COL_MODEL_ERR_BEFORE], ascending=False).reset_index()
    return usable_df.head(sample_count).copy()


def build_result_record(
    row_index,
    original_row,
    optimized_values,
    target_width,
    real_width,
    pred_before,
    optimization_metrics,
    objective_score,
    optimization_columns,
    regime,
):
    pred_after = optimization_metrics["predicted_width"]
    original_values = np.array(
        [float(original_row[f"原始值_{column}"]) for column in optimization_columns],
        dtype=float,
    )
    optimized_values = np.array(optimized_values, dtype=float)
    abs_delta = np.abs(optimized_values - original_values)

    record = {
        COL_SAMPLE_INDEX: int(row_index),
        COL_TARGET: float(target_width),
        COL_REAL: float(real_width),
        COL_PRED_BEFORE: float(pred_before),
        COL_PRED_AFTER: float(pred_after),
        COL_REAL_ERR: float(abs(target_width - real_width)),
        COL_MODEL_ERR_BEFORE: float(abs(target_width - pred_before)),
        COL_MODEL_ERR_AFTER: float(abs(target_width - pred_after)),
        COL_PRED_REAL_ERR_BEFORE: float(abs(pred_before - real_width)),
        COL_PRED_REAL_ERR_AFTER: float(abs(pred_after - real_width)),
        COL_IMPROVE_MODEL: float(abs(target_width - pred_before) - abs(target_width - pred_after)),
        COL_OBJECTIVE: float(objective_score),
        COL_MEAN_ADJ: float(abs_delta.mean()),
        COL_MAX_ADJ: float(abs_delta.max()),
        COL_SIGNED_BEFORE: float(pred_before - target_width),
        COL_SIGNED_AFTER: float(pred_after - target_width),
        "搜索档位": regime["name"],
        "档位粒子数": int(regime["num_particles"]),
        "档位迭代数": int(regime["max_iterations"]),
        "档位搜索比例": float(regime["search_margin_ratio"]),
        "档位最大调整量": float(regime["max_adjustment_mm"]),
    }
    for index, column in enumerate(optimization_columns):
        record[f"原始_{column}"] = float(original_values[index])
        record[f"优化后_{column}"] = float(optimized_values[index])
        record[f"调整量_{column}"] = float(optimized_values[index] - original_values[index])
    return record


def write_excel_with_fallback(dataframe, output_path):
    base_path, ext = os.path.splitext(output_path)
    candidate_path = output_path
    suffix_index = 0
    while True:
        try:
            dataframe.to_excel(candidate_path, index=False)
            return candidate_path
        except PermissionError:
            suffix_index += 1
            candidate_path = f"{base_path}_new_{suffix_index}{ext}"


def format_transition_text(before_value, after_value, decimals=4):
    return f"{before_value:.{decimals}f}->{after_value:.{decimals}f}"


def resolve_optimization_result_columns(results_df):
    preferred_suffixes = ["侧压机压下量", "压下量-E2-1", "压下量-E2-3", "压下量-E2-5"]
    before_columns = [f"原始_{suffix}" for suffix in preferred_suffixes if f"原始_{suffix}" in results_df.columns]
    if len(before_columns) < 3:
        before_columns = [column for column in results_df.columns if column.startswith("原始_")]
    return before_columns


def build_compact_tables(results_df, renumber=False, include_prediction_cols=False):
    sample_numbers = results_df[COL_SAMPLE_INDEX].astype(int) + 1
    if renumber:
        sample_numbers = pd.Series(np.arange(1, len(results_df) + 1), index=results_df.index)

    before_columns = resolve_optimization_result_columns(results_df)
    display_columns = before_columns[:3]
    if len(display_columns) < 3:
        raise KeyError("结果文件中的优化变量列少于 3 列，无法导出紧凑表。")

    after_columns = [column.replace("原始_", "优化后_", 1) for column in display_columns]

    table_41 = pd.DataFrame(
        {
            "样本编号": sample_numbers,
            "立辊压下量E1": results_df[display_columns[0]],
            "立辊压下量E2": results_df[display_columns[1]],
            "立辊压下量E3": results_df[display_columns[2]],
            "预测宽度": results_df[COL_PRED_AFTER],
            "实测宽度": results_df[COL_REAL],
        }
    )

    table_42 = pd.DataFrame(
        {
            "样本编号": sample_numbers,
            "立辊压下量E1": [
                format_transition_text(before_value, after_value)
                for before_value, after_value in zip(results_df[display_columns[0]], results_df[after_columns[0]])
            ],
            "立辊压下量E2": [
                format_transition_text(before_value, after_value)
                for before_value, after_value in zip(results_df[display_columns[1]], results_df[after_columns[1]])
            ],
            "立辊压下量E3": [
                format_transition_text(before_value, after_value)
                for before_value, after_value in zip(results_df[display_columns[2]], results_df[after_columns[2]])
            ],
            "宽度偏差量（优化前）": results_df[COL_SIGNED_BEFORE],
            "宽度偏差量（优化后）": results_df[COL_SIGNED_AFTER],
        }
    )
    if include_prediction_cols:
        table_42["设定值"] = results_df[COL_TARGET]
        table_42["优化前预测值"] = results_df[COL_PRED_BEFORE]
        table_42["优化后预测值"] = results_df[COL_PRED_AFTER]
    return table_41, table_42


def export_compact_tables(results_df):
    table_41, table_42 = build_compact_tables(results_df, renumber=False)
    table_41_path = write_excel_with_fallback(table_41, TABLE_41_PATH)
    table_42_path = write_excel_with_fallback(table_42, TABLE_42_PATH)
    return table_41_path, table_42_path


def resolve_existing_results_path():
    candidate_paths = [
        RESULT_EXCEL_PATH,
        os.path.join(DATA_OUTPUT_DIR, "PSO_Optimization_Results_new_1.xlsx"),
        os.path.join(MGH_DATA_DIR, "PSO_Optimization_Results_200Samples.xlsx"),
        os.path.join(MGH_DATA_DIR, "PSO_Optimization_Results_200Samples_new.xlsx"),
        os.path.join(MGH_DATA_DIR, "PSO_Optimization_Results.xlsx"),
    ]
    existing_paths = [path for path in candidate_paths if os.path.exists(path)]
    if not existing_paths:
        raise FileNotFoundError("未找到已有的 PSO 结果文件，无法直接处理选样。")
    return max(existing_paths, key=os.path.getmtime)


def export_top20_compact_tables_from_existing_results():
    result_path = resolve_existing_results_path()
    results_df = pd.read_excel(result_path)
    filtered_df = results_df[
        (results_df[COL_MODEL_ERR_AFTER] >= 0.5) & (results_df[COL_MODEL_ERR_AFTER] <= 10.0)
    ].copy()
    top20_df = (
        filtered_df.sort_values(
            by=[COL_TARGET, COL_MODEL_ERR_AFTER, COL_SAMPLE_INDEX],
            ascending=[True, True, True],
        )
        .head(TOP_SAMPLE_COUNT)
        .copy()
        .reset_index(drop=True)
    )

    table_41_top20, table_42_top20 = build_compact_tables(
        top20_df,
        renumber=True,
        include_prediction_cols=True,
    )
    numeric_columns_41 = table_41_top20.select_dtypes(include=[np.number]).columns
    numeric_columns_42 = table_42_top20.select_dtypes(include=[np.number]).columns
    table_41_top20[numeric_columns_41] = table_41_top20[numeric_columns_41].round(3)
    table_42_top20[numeric_columns_42] = table_42_top20[numeric_columns_42].round(3)
    for column in ["立辊压下量E1", "立辊压下量E2", "立辊压下量E3"]:
        if column in table_42_top20.columns:
            table_42_top20[column] = table_42_top20[column].apply(
                lambda text: format_transition_text(
                    float(text.split("->")[0]),
                    float(text.split("->")[1]),
                    decimals=3,
                )
            )
    table_41_path = write_excel_with_fallback(table_41_top20, TABLE_41_TOP20_PATH)
    table_42_path = write_excel_with_fallback(table_42_top20, TABLE_42_TOP20_PATH)

    print("-" * 60)
    print(f"已从已有优化结果中选出误差位于 0.5-10 mm 且按设定值排序的前 {TOP_SAMPLE_COUNT} 条样本。")
    print(f"取样依据: 先筛 {COL_MODEL_ERR_AFTER} ∈ [0.5, 10]，再按 {COL_TARGET} 从小到大")
    print(f"原始结果文件: {result_path}")
    print(f"Top20 4.1 表已保存到: {table_41_path}")
    print(f"Top20 4.2 表已保存到: {table_42_path}")


def run_sensitivity_test():
    test_df, bundle = load_inputs()
    model = bundle["model"]
    selected_features = bundle["selected_features"]
    target_column = bundle["target"]
    setting_target_column = bundle["setting_target"]
    optimization_columns = resolve_optimization_columns(bundle, test_df)
    sequence_groups = bundle["sequence_groups"]
    scale_map, min_map, data_min_map = build_scaler_maps(bundle)

    selected_test_df = select_rows_for_optimization(
        test_df=test_df,
        setting_target_column=setting_target_column,
        model=model,
        selected_features=selected_features,
        sample_count=OPTIMIZE_SAMPLE_COUNT,
    )

    print(f"开始敏感性测试，样本数: {len(selected_test_df)}")
    detail_records = []
    summary_records = []

    for _, row in selected_test_df.iterrows():
        base_row = row.copy()
        row_index = int(base_row["index"])
        target_width = float(base_row[setting_target_column])
        real_width = float(base_row[target_column])
        baseline_pred = predict_width(model, base_row, selected_features)
        baseline_error = abs(baseline_pred - target_width)
        regime = get_search_regime(baseline_error)
        bounds = build_bounds(base_row, optimization_columns, scale_map, min_map, data_min_map, regime)
        original_values = np.array(
            [
                inverse_scale_feature(float(base_row[column]), column, scale_map, min_map, data_min_map)
                for column in optimization_columns
            ],
            dtype=float,
        )

        sample_summary = {
            "样本序号": row_index,
            "设定宽度": target_width,
            "实测宽度": real_width,
            "基线预测宽度": baseline_pred,
            "基线绝对误差": baseline_error,
            "搜索档位": regime["name"],
        }

        for column_index, column_name in enumerate(optimization_columns):
            lower_bound, upper_bound = bounds[column_index]
            scan_values = np.linspace(lower_bound, upper_bound, 21)
            best_error = float("inf")
            best_value = float(original_values[column_index])
            predictions = []

            for candidate_value in scan_values:
                candidate_values = original_values.copy()
                candidate_values[column_index] = float(candidate_value)
                candidate_row = apply_candidate_values(
                    base_row=base_row,
                    optimization_columns=optimization_columns,
                    candidate_values=candidate_values,
                    sequence_groups=sequence_groups,
                    scale_map=scale_map,
                    min_map=min_map,
                )
                predicted_width = predict_width(model, candidate_row, selected_features)
                current_error = abs(predicted_width - target_width)
                predictions.append(predicted_width)

                if current_error < best_error:
                    best_error = current_error
                    best_value = float(candidate_value)

            prediction_range = float(max(predictions) - min(predictions))
            improvement = float(baseline_error - best_error)
            detail_records.append(
                {
                    "样本序号": row_index,
                    "搜索档位": regime["name"],
                    "变量": column_name,
                    "原始值": float(original_values[column_index]),
                    "下界": float(lower_bound),
                    "上界": float(upper_bound),
                    "最优值": best_value,
                    "基线绝对误差": baseline_error,
                    "单变量最优绝对误差": float(best_error),
                    "误差改善量": improvement,
                    "预测宽度扫描范围": prediction_range,
                }
            )

            sample_summary[f"{column_name}_最优值"] = best_value
            sample_summary[f"{column_name}_单变量最优绝对误差"] = float(best_error)
            sample_summary[f"{column_name}_误差改善量"] = improvement
            sample_summary[f"{column_name}_预测宽度扫描范围"] = prediction_range

        summary_records.append(sample_summary)

    detail_df = pd.DataFrame(detail_records)
    summary_df = pd.DataFrame(summary_records)
    detail_path = write_excel_with_fallback(detail_df, SENSITIVITY_DETAIL_PATH)
    summary_path = write_excel_with_fallback(summary_df, SENSITIVITY_SUMMARY_PATH)

    variable_summary = (
        detail_df.groupby("变量")[["误差改善量", "预测宽度扫描范围", "单变量最优绝对误差"]]
        .agg(["mean", "max", "min"])
        .round(4)
    )

    print("-" * 60)
    print("立辊变量敏感性测试完成。")
    print(variable_summary)
    print(f"明细文件已保存到: {detail_path}")
    print(f"汇总文件已保存到: {summary_path}")


def run_pso_optimization():
    test_df, bundle = load_inputs()
    model = bundle["model"]
    selected_features = bundle["selected_features"]
    target_column = bundle["target"]
    setting_target_column = bundle["setting_target"]
    optimization_columns = resolve_optimization_columns(bundle, test_df)
    sequence_groups = bundle["sequence_groups"]
    scale_map, min_map, data_min_map = build_scaler_maps(bundle)

    selected_test_df = select_rows_for_optimization(
        test_df=test_df,
        setting_target_column=setting_target_column,
        model=model,
        selected_features=selected_features,
        sample_count=OPTIMIZE_SAMPLE_COUNT,
    )

    print(f"设定值非空样本数: {int(test_df[setting_target_column].notna().sum())}")
    print(f"本次参与优化样本数: {len(selected_test_df)}")

    results = []
    aggressive_count = 0
    conservative_count = 0

    for local_index, row in selected_test_df.iterrows():
        base_row = row.copy()
        row_index = int(base_row["index"])
        target_width = float(base_row[setting_target_column])
        real_width = float(base_row[target_column])
        pred_before = float(base_row[COL_PRED_BEFORE])

        for column in optimization_columns:
            base_row[f"原始值_{column}"] = inverse_scale_feature(
                float(base_row[column]),
                column,
                scale_map,
                min_map,
                data_min_map,
            )

        best_values, optimization_metrics, objective_score, regime = pso_optimize_row(
            base_row=base_row,
            target_width=target_width,
            model=model,
            selected_features=selected_features,
            optimization_columns=optimization_columns,
            sequence_groups=sequence_groups,
            scale_map=scale_map,
            min_map=min_map,
            data_min_map=data_min_map,
        )

        if regime["name"] == "aggressive":
            aggressive_count += 1
        else:
            conservative_count += 1

        results.append(
            build_result_record(
                row_index=row_index,
                original_row=base_row,
                optimized_values=best_values,
                target_width=target_width,
                real_width=real_width,
                pred_before=pred_before,
                optimization_metrics=optimization_metrics,
                objective_score=objective_score,
                optimization_columns=optimization_columns,
                regime=regime,
            )
        )

        if (local_index + 1) % 10 == 0:
            print(f"已完成 {local_index + 1} / {len(selected_test_df)} 条样本")

    results_df = pd.DataFrame(results)
    result_path = write_excel_with_fallback(results_df, RESULT_EXCEL_PATH)
    table_41_path, table_42_path = export_compact_tables(results_df)

    summary = {
        "设定值非空样本数": int(test_df[setting_target_column].notna().sum()),
        "参与优化样本数": int(len(results_df)),
        "激进档样本数": int(aggressive_count),
        "保守档样本数": int(conservative_count),
        "优化前实测与设定平均绝对偏差": float(results_df[COL_REAL_ERR].mean()),
        "优化前模型与设定平均绝对偏差": float(results_df[COL_MODEL_ERR_BEFORE].mean()),
        "优化后模型与设定平均绝对偏差": float(results_df[COL_MODEL_ERR_AFTER].mean()),
        "优化前预测与实测平均绝对偏差": float(results_df[COL_PRED_REAL_ERR_BEFORE].mean()),
        "优化后预测与实测平均绝对偏差": float(results_df[COL_PRED_REAL_ERR_AFTER].mean()),
        "相对模型平均改善量": float(results_df[COL_IMPROVE_MODEL].mean()),
        "平均绝对调整量": float(results_df[COL_MEAN_ADJ].mean()),
        "最大绝对调整量均值": float(results_df[COL_MAX_ADJ].mean()),
        "优化后误差低于1mm样本数": int((results_df[COL_MODEL_ERR_AFTER] < 1.0).sum()),
        "优化后误差位于1到20mm样本数": int(((results_df[COL_MODEL_ERR_AFTER] >= 1.0) & (results_df[COL_MODEL_ERR_AFTER] <= 20.0)).sum()),
    }
    with open(RESULT_SUMMARY_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print("-" * 60)
    print(f"优化前实测与设定平均绝对偏差: {summary['优化前实测与设定平均绝对偏差']:.4f} mm")
    print(f"优化前模型与设定平均绝对偏差: {summary['优化前模型与设定平均绝对偏差']:.4f} mm")
    print(f"优化后模型与设定平均绝对偏差: {summary['优化后模型与设定平均绝对偏差']:.4f} mm")
    print(f"相对模型平均改善量: {summary['相对模型平均改善量']:.4f} mm")
    print(f"平均绝对调整量: {summary['平均绝对调整量']:.4f} mm")
    print("-" * 60)
    print(f"优化结果已保存到: {result_path}")
    print(f"优化摘要已保存到: {RESULT_SUMMARY_PATH}")
    print(f"4.1.xlsx 已保存到: {table_41_path}")
    print(f"4.2.xlsx 已保存到: {table_42_path}")


def prompt_menu_option():
    print("请选择 PSO 运行模式：")
    print("1. 完整流程")
    print("2. 提取部分数据")

    if not sys.stdin or not sys.stdin.isatty():
        print(f"未检测到交互式终端，自动使用默认选项 {DEFAULT_MENU_OPTION}。")
        return DEFAULT_MENU_OPTION

    while True:
        choice = input("请输入编号 (1-2): ").strip()
        if choice in {"1", "2"}:
            return choice
        print("输入无效，请输入 1 或 2。")


def main():
    menu_option = prompt_menu_option()
    if menu_option == "1":
        run_pso_optimization()
    elif menu_option == "2":
        export_top20_compact_tables_from_existing_results()
    else:
        raise ValueError("菜单选项只能是 1、2。")


if __name__ == "__main__":
    main()
