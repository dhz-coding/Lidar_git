"""Quality-driven adaptive registration demo for the ship-hold point clouds.

Inputs can be individual PCD files or directories containing PCD files.
The demo produces:

* a merged, registered before-sand cloud
* a merged, registered after-sand cloud
* the after-sand cloud transformed into the before-sand frame
* a coloured before/after overlay for quick visual inspection

The first version deliberately keeps the adaptive controller transparent.
Every parameter choice and every retry is written to JSON so it can be
calibrated against the user's real scans later.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import yaml
from scipy.spatial import cKDTree

from quality_assessment import (
    AdaptiveSettings,
    QualityThresholds,
    assess_cloud,
    choose_parameters,
    serializable_settings,
    settings_from_mapping,
)
from registration_curvature_roi import (
    DEFAULT_PARAMETERS,
    CurvatureROIRegistrationError,
    register_curvature_roi,
)


DEMO_VERSION = "0.3.0"


def transform_difference(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    delta = np.linalg.inv(expected) @ actual
    angle = math.degrees(math.acos(float(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))))
    return {
        "translation_error": float(np.linalg.norm(delta[:3, 3])),
        "rotation_error_deg": angle,
    }


def rotation_angles_deg(rotation: np.ndarray) -> dict[str, float]:
    """Return ZYX yaw, pitch and roll angles for a rigid rotation matrix."""
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    pitch = math.atan2(
        -float(rotation[2, 0]),
        math.hypot(float(rotation[0, 0]), float(rotation[1, 0])),
    )
    if math.hypot(float(rotation[0, 0]), float(rotation[1, 0])) > 1e-8:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = math.atan2(-float(rotation[0, 1]), float(rotation[1, 1]))
        yaw = 0.0
    return {
        "yaw_deg": math.degrees(yaw),
        "pitch_deg": math.degrees(pitch),
        "roll_deg": math.degrees(roll),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="装沙前后点云质量评估、参数自适应初值和配准 demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--before", "--before_dir", default="", help="装沙前 PCD 文件或目录，留空则使用 config.yaml")
    parser.add_argument("--after", "--after_dir", default="", help="装沙后 PCD 文件或目录，留空则使用 config.yaml")
    parser.add_argument("--output_dir", default="", help="输出根目录，留空则使用 config.yaml")
    parser.add_argument("--config", default="config.yaml", help="demo 配置文件")
    parser.add_argument("--run_name", default="", help="可选输出目录名，默认使用时间戳")
    parser.add_argument("--accept_review", action="store_true", help="允许达到 review 门槛的结果合并")
    parser.add_argument("--ascii", action="store_true", help="以 ASCII PCD 保存，文件会明显变大")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists() and not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"配置文件必须是 YAML 字典: {config_path}")
    return loaded


def json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法序列化类型: {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)


def collect_pcd_paths(value: str, label: str) -> list[Path]:
    path = Path(value)
    if path.is_file():
        if path.suffix.lower() != ".pcd":
            raise ValueError(f"{label} 文件必须是 .pcd: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"{label} 文件或目录不存在: {path}")
    files = sorted((item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".pcd"), key=lambda item: item.name.lower())
    if not files:
        raise FileNotFoundError(f"{label} 目录中没有 .pcd 文件: {path}")
    return files


def load_cloud(path: Path, label: str) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))
    if len(cloud.points) == 0:
        raise ValueError(f"{label} 为空或读取失败: {path}")
    points = np.asarray(cloud.points)
    finite = np.isfinite(points).all(axis=1)
    if not np.all(finite):
        cloud = cloud.select_by_index(np.flatnonzero(finite).tolist())
    if len(cloud.points) == 0:
        raise ValueError(f"{label} 清理无效点后为空: {path}")
    return cloud


def clone_cloud(cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    return o3d.geometry.PointCloud(cloud)


def deterministic_sample_cloud(cloud: o3d.geometry.PointCloud, max_points: int, seed: int = 20261112):
    if len(cloud.points) <= max_points:
        return clone_cloud(cloud)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(cloud.points), size=max_points, replace=False)
    return cloud.select_by_index(np.sort(indices).tolist())


def validate_transform(matrix: np.ndarray) -> bool:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return False
    rotation = matrix[:3, :3]
    return bool(
        np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4)
    )


def _direction_metrics(source_points: np.ndarray, target_points: np.ndarray, threshold: float):
    tree = cKDTree(target_points)
    distances, indices = tree.query(source_points, k=1, workers=1)
    matched = np.isfinite(distances) & (distances <= threshold)
    count = int(np.count_nonzero(matched))
    global_fitness = float(count / max(len(source_points), 1))
    return {
        "fitness": global_fitness,
        "global_fitness": global_fitness,
        "inlier_rmse": float(np.sqrt(np.mean(distances[matched] ** 2))) if count else None,
        "correspondence_count": count,
        "point_count": int(len(source_points)),
        "overlap_candidate_count": int(len(source_points)),
        "overlap_candidate_fraction": 1.0,
        "p90_distance": float(np.percentile(distances[matched], 90)) if count else None,
        "_matched_mask": matched,
        "_eligible_mask": np.ones(len(source_points), dtype=bool),
        "_distances": distances,
        "_nearest_indices": indices,
    }


def _expanded_bbox_mask(points: np.ndarray, reference_points: np.ndarray, margin: float) -> np.ndarray:
    """Estimate which points can belong to the observable overlap region."""
    if len(points) == 0 or len(reference_points) == 0:
        return np.zeros(len(points), dtype=bool)
    minimum = np.min(reference_points, axis=0) - margin
    maximum = np.max(reference_points, axis=0) + margin
    return np.all((points >= minimum) & (points <= maximum), axis=1)


def _direction_metrics_in_overlap(
    source_points: np.ndarray,
    target_points: np.ndarray,
    threshold: float,
    eligible_mask: np.ndarray,
):
    metrics = _direction_metrics(source_points, target_points, threshold)
    eligible_mask = np.asarray(eligible_mask, dtype=bool)
    if eligible_mask.shape != (len(source_points),):
        raise ValueError("overlap eligibility mask has an invalid shape")
    eligible = eligible_mask & np.isfinite(metrics["_nearest_indices"])
    matched = metrics["_matched_mask"] & eligible
    candidate_count = int(np.count_nonzero(eligible))
    match_count = int(np.count_nonzero(matched))
    metrics.update({
        "fitness": float(match_count / max(candidate_count, 1)),
        "overlap_candidate_count": candidate_count,
        "overlap_candidate_fraction": float(candidate_count / max(len(source_points), 1)),
        "inlier_rmse": float(np.sqrt(np.mean(metrics["_distances"][matched] ** 2))) if match_count else None,
        "_matched_mask": matched,
        "_eligible_mask": eligible,
    })
    return metrics


def evaluate_transform(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    reference_spacing: float,
    settings: AdaptiveSettings,
    validation_voxel: float | None = None,
) -> dict[str, Any]:
    """Fixed bidirectional validation used for every adaptive attempt."""
    if not validate_transform(transformation):
        return {"state": "rejected", "passed": False, "failure_reasons": ["invalid_transform"]}

    angles = rotation_angles_deg(transformation[:3, :3])
    angle_failures = []
    if abs(angles["pitch_deg"]) > settings.max_pitch_deg:
        angle_failures.append("pitch_limit_exceeded")
    if abs(angles["roll_deg"]) > settings.max_roll_deg:
        angle_failures.append("roll_limit_exceeded")
    if angle_failures:
        return {
            "state": "rejected",
            "passed": False,
            "failure_reasons": angle_failures,
            "rotation_angles_deg": angles,
            "max_pitch_deg": settings.max_pitch_deg,
            "max_roll_deg": settings.max_roll_deg,
        }

    voxel = float(validation_voxel or max(reference_spacing * 4.0, 1e-3))
    threshold = max(reference_spacing * settings.validation_distance_factor, voxel * 1.5)
    source_snapshot = source.voxel_down_sample(voxel)
    target_snapshot = target.voxel_down_sample(voxel)
    source_snapshot = deterministic_sample_cloud(source_snapshot, settings.validation_sample_points, 20261113)
    target_snapshot = deterministic_sample_cloud(target_snapshot, settings.validation_sample_points, 20261114)
    source_points = np.asarray(source_snapshot.points, dtype=np.float64)
    target_points = np.asarray(target_snapshot.points, dtype=np.float64)
    if len(source_points) < 3 or len(target_points) < 3:
        return {"state": "rejected", "passed": False, "failure_reasons": ["validation_cloud_too_small"]}

    rotation = transformation[:3, :3]
    translation = transformation[:3, 3]
    transformed_source = source_points @ rotation.T + translation
    inverse = np.linalg.inv(transformation)
    transformed_target = target_points @ inverse[:3, :3].T + inverse[:3, 3]
    overlap_gate = max(
        threshold * settings.validation_overlap_gate_factor,
        voxel * 2.0,
    )
    forward_overlap = _expanded_bbox_mask(transformed_source, target_points, overlap_gate)
    reverse_overlap = _expanded_bbox_mask(transformed_target, source_points, overlap_gate)
    forward = _direction_metrics_in_overlap(
        transformed_source, target_points, threshold, forward_overlap
    )
    reverse = _direction_metrics_in_overlap(
        transformed_target, source_points, threshold, reverse_overlap
    )
    fitness = min(forward["fitness"], reverse["fitness"])
    global_fitness = min(forward["global_fitness"], reverse["global_fitness"])
    rmses = [metric["inlier_rmse"] for metric in (forward, reverse) if metric["inlier_rmse"] is not None]
    rmse = max(rmses) if rmses else math.inf
    max_rmse = settings.validation_rmse_factor * reference_spacing
    review_rmse = settings.review_rmse_factor * reference_spacing
    enough = (
        min(forward["correspondence_count"], reverse["correspondence_count"]) >= 30
        and min(
            forward["overlap_candidate_count"],
            reverse["overlap_candidate_count"],
        ) >= settings.min_overlap_candidate_points
    )

    # Estimate how much of the stable, approximately vertical structure agrees.
    # This is a proxy for hull walls and rails, while avoiding a dependence on
    # the changing sand surface.
    stable_wall_overlap = None
    normal_consistency = None
    try:
        source_snapshot.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=max(3.0 * voxel, 1e-3), max_nn=50)
        )
        target_snapshot.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=max(3.0 * voxel, 1e-3), max_nn=50)
        )
        source_normals = np.asarray(source_snapshot.normals)
        target_normals = np.asarray(target_snapshot.normals)
        source_stable = np.abs(source_normals[:, 2]) <= 0.60
        target_stable = np.abs(target_normals[:, 2]) <= 0.60
        source_matched = forward["_matched_mask"] & source_stable
        target_matched = reverse["_matched_mask"] & target_stable
        source_stable_overlap = source_stable & forward["_eligible_mask"]
        target_stable_overlap = target_stable & reverse["_eligible_mask"]
        forward_stable = float(
            np.count_nonzero(source_matched) / max(np.count_nonzero(source_stable_overlap), 1)
        )
        reverse_stable = float(
            np.count_nonzero(target_matched) / max(np.count_nonzero(target_stable_overlap), 1)
        )
        stable_wall_overlap = min(forward_stable, reverse_stable)

        rotated_source_normals = source_normals @ rotation.T
        source_indices = forward["_nearest_indices"][source_matched]
        source_normal_values = rotated_source_normals[source_matched]
        target_normal_values = target_normals[source_indices]
        products = np.linalg.norm(source_normal_values, axis=1) * np.linalg.norm(target_normal_values, axis=1)
        valid_normals = products > 1e-12
        if np.any(valid_normals):
            cosines = np.abs(np.sum(source_normal_values[valid_normals] * target_normal_values[valid_normals], axis=1)) / products[valid_normals]
            normal_consistency = float(np.mean(cosines >= math.cos(math.radians(35.0))))
    except Exception:
        stable_wall_overlap = None
        normal_consistency = None

    if enough and fitness >= settings.min_validation_fitness and rmse <= max_rmse:
        state = "passed"
        reasons: list[str] = []
    elif enough and fitness >= settings.review_validation_fitness and rmse <= review_rmse:
        state = "review"
        reasons = ["borderline_geometry"]
    else:
        state = "rejected"
        reasons = ["fixed_bidirectional_geometry_failed"]

    public_forward = {key: value for key, value in forward.items() if not key.startswith("_")}
    public_reverse = {key: value for key, value in reverse.items() if not key.startswith("_")}
    return {
        "state": state,
        "passed": state == "passed",
        "failure_reasons": reasons,
        "validation_voxel": voxel,
        "validation_distance": threshold,
        "validation_overlap_gate": overlap_gate,
        "min_overlap_candidate_points": settings.min_overlap_candidate_points,
        "rotation_angles_deg": angles,
        "max_pitch_deg": settings.max_pitch_deg,
        "max_roll_deg": settings.max_roll_deg,
        "max_rmse": max_rmse,
        "review_max_rmse": review_rmse,
        "fitness": float(fitness),
        "global_fitness": float(global_fitness),
        "inlier_rmse": float(rmse) if np.isfinite(rmse) else None,
        "forward": public_forward,
        "reverse": public_reverse,
        "stable_wall_overlap": stable_wall_overlap,
        "normal_consistency": normal_consistency,
        "pair_quality": {
            "symmetric_overlap": float(fitness),
            "global_symmetric_fitness": float(global_fitness),
            "forward_overlap_candidates": forward["overlap_candidate_count"],
            "reverse_overlap_candidates": reverse["overlap_candidate_count"],
            "stable_wall_overlap": stable_wall_overlap,
            "normal_consistency": normal_consistency,
            "distance_p90": float(max(
                forward.get("p90_distance") or math.inf,
                reverse.get("p90_distance") or math.inf,
            )),
        },
    }


def solver_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in parameters.items() if key in DEFAULT_PARAMETERS}


def make_retry_parameters(base: dict[str, Any], attempt_index: int, previous_report: dict[str, Any] | None):
    """Adjust one failure cause at a time and keep every change bounded."""
    params = solver_parameters(copy.deepcopy(base))
    params["source_curvature_percentile"] = float(base["source_curvature_percentile"])
    params["target_curvature_percentile"] = float(base["target_curvature_percentile"])
    report = previous_report or {}
    source_roi = report.get("source", {}).get("roi_points", 0)
    target_roi = report.get("target", {}).get("roi_points", 0)
    ransac = report.get("ransac", {}) or {}
    icp = report.get("icp", {}) or {}
    previous_validation = report.get("validation", {}) or {}
    previous_pair_quality = previous_validation.get("pair_quality", {}) or {}

    if attempt_index == 1:
        # First rescue keeps the same scale and gives FPFH more usable features.
        params["source_curvature_percentile"] = max(40.0, params["source_curvature_percentile"] - 10.0)
        params["target_curvature_percentile"] = max(40.0, params["target_curvature_percentile"] - 10.0)
        params["feature_radius_ratio"] = min(12.0, params["feature_radius_ratio"] * 1.25)
        params["ransac_distance_ratio"] = min(8.0, params["ransac_distance_ratio"] * 1.25)
        if min(source_roi or 10**9, target_roi or 10**9) < params["min_roi_points"]:
            params["disable_cluster_filter"] = True
    elif attempt_index == 2:
        # Second rescue changes scale for sparse or noisy clouds.
        params["voxel_size"] = min(0.35, params["voxel_size"] * 1.30)
        params["curvature_radius_ratio"] = min(7.0, params["curvature_radius_ratio"] * 1.20)
        params["feature_radius_ratio"] = min(14.0, params["feature_radius_ratio"] * 1.15)
        params["ransac_distance_ratio"] = min(10.0, params["ransac_distance_ratio"] * 1.20)
        params["icp_distance_ratio"] = min(5.0, params["icp_distance_ratio"] * 1.25)
        params["disable_cluster_filter"] = True
    elif attempt_index == 3:
        # Final rescue uses a finer scale and a less strict geometric checker.
        params["voxel_size"] = max(0.025, params["voxel_size"] * 0.75)
        params["source_curvature_percentile"] = max(35.0, params["source_curvature_percentile"] - 8.0)
        params["target_curvature_percentile"] = max(35.0, params["target_curvature_percentile"] - 8.0)
        params["ransac_n"] = 3
        params["edge_length_threshold"] = 0.85
        params["ransac_max_iteration"] = min(4_000_000, int(params["ransac_max_iteration"] * 1.5))
        params["ransac_distance_ratio"] = min(12.0, params["ransac_distance_ratio"] * 1.15)
        params["icp_distance_ratio"] = min(6.0, params["icp_distance_ratio"] * 1.15)
        params["disable_cluster_filter"] = True
    elif attempt_index == 4:
        # Multi-start candidate: keep the dense ROI but remove mutual filtering.
        # This is useful when two scans see different portions of the same hull.
        params["mutual_filter"] = False
        params["edge_length_threshold"] = 0.82
        params["ransac_distance_ratio"] = min(10.0, params["ransac_distance_ratio"] * 1.10)
        params["icp_distance_ratio"] = min(5.0, params["icp_distance_ratio"] * 1.10)
        params["disable_cluster_filter"] = True
    else:
        # A second multi-start candidate with more features and RANSAC n=3.
        params["mutual_filter"] = False
        params["ransac_n"] = 3
        params["edge_length_threshold"] = 0.78
        params["source_curvature_percentile"] = max(35.0, params["source_curvature_percentile"] - 12.0)
        params["target_curvature_percentile"] = max(35.0, params["target_curvature_percentile"] - 12.0)
        params["feature_radius_ratio"] = min(14.0, params["feature_radius_ratio"] * 1.25)
        params["ransac_distance_ratio"] = min(12.0, params["ransac_distance_ratio"] * 1.20)
        params["disable_cluster_filter"] = True

    # If a previous attempt reached RANSAC but ICP was weak, keep the same
    # feature scale and only widen the ICP capture range.
    if icp and float(icp.get("fitness", 1.0)) < 0.15:
        params["icp_distance_ratio"] = min(6.0, params["icp_distance_ratio"] * 1.20)
    if ransac and int(ransac.get("correspondence_count", 999999)) < params["min_roi_points"]:
        params["ransac_distance_ratio"] = min(12.0, params["ransac_distance_ratio"] * 1.15)
    if previous_pair_quality:
        if float(previous_pair_quality.get("symmetric_overlap", 0.0)) < 0.25:
            params["voxel_size"] = min(0.35, params["voxel_size"] * 1.15)
            params["feature_radius_ratio"] = min(14.0, params["feature_radius_ratio"] * 1.20)
        if float(previous_pair_quality.get("stable_wall_overlap") or 0.0) < 0.20:
            params["source_curvature_percentile"] = max(35.0, params["source_curvature_percentile"] - 8.0)
            params["target_curvature_percentile"] = max(35.0, params["target_curvature_percentile"] - 8.0)

    return params


def adaptive_register_pair(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    base_parameters: dict[str, Any],
    settings: AdaptiveSettings,
    pair_name: str,
    accept_review: bool = False,
    strict_consistency: bool = False,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Run all bounded retries and select the strongest validated transform."""
    started = time.perf_counter()
    spacing = float(base_parameters.get("reference_spacing", 0.01))
    attempts: list[dict[str, Any]] = []
    previous_report: dict[str, Any] | None = None
    selected = None
    selected_validation = None
    candidate_rows: list[dict[str, Any]] = []

    for index in range(max(1, settings.max_attempts)):
        params = make_retry_parameters(base_parameters, index, previous_report)
        attempt_started = time.perf_counter()
        row: dict[str, Any] = {
            "attempt_index": index + 1,
            "pair_name": pair_name,
            "parameters": params,
        }
        print(
            f"[{pair_name}] 尝试 {index + 1}/{settings.max_attempts} "
            f"质量档位={base_parameters.get('quality_level')} "
            f"voxel={params['voxel_size']:.5g} "
            f"RANSAC距离={params['ransac_distance_ratio'] * params['voxel_size']:.5g}",
            flush=True,
        )
        try:
            transformation, coarse, delta, solver_report = register_curvature_roi(
                source,
                target,
                params,
            )
            validation = evaluate_transform(
                source,
                target,
                transformation,
                spacing,
                settings,
                validation_voxel=max(spacing * 4.0, params["voxel_size"]),
            )
            row.update({
                "solver_state": "completed",
                "solver_report": solver_report,
                "transformation": transformation.tolist(),
                "coarse_transformation": coarse.tolist(),
                "icp_delta_transformation": delta.tolist(),
                "validation": validation,
                "state": validation["state"],
            })
            previous_report = {**solver_report, "validation": validation}
            eligible = validation["state"] == "passed" or (
                accept_review and validation["state"] == "review"
            )
            if eligible:
                candidate_rows.append(row)
        except CurvatureROIRegistrationError as error:
            previous_report = error.report
            row.update({
                "solver_state": "exception",
                "exception": repr(error),
                "solver_report": error.report,
                "state": "rejected",
            })
        except Exception as error:  # keep the demo moving and preserve the failure evidence
            row.update({
                "solver_state": "exception",
                "exception": repr(error),
                "state": "rejected",
            })

        row["selected"] = False
        row["elapsed_seconds"] = time.perf_counter() - attempt_started
        attempts.append(row)
        print(f"[{pair_name}] 本次状态={row.get('state', 'rejected')}", flush=True)

    # Do not stop at the first threshold pass. Symmetric ship structures can
    # produce several plausible transforms, so compare every retry result.
    translation_consensus_limit = max(0.20, 8.0 * spacing)
    rotation_consensus_limit = 5.0
    for candidate in candidate_rows:
        candidate_transform = np.asarray(candidate["transformation"], dtype=float)
        neighbours = []
        for other in candidate_rows:
            other_transform = np.asarray(other["transformation"], dtype=float)
            difference = transform_difference(candidate_transform, other_transform)
            if (
                difference["translation_error"] <= translation_consensus_limit
                and difference["rotation_error_deg"] <= rotation_consensus_limit
            ):
                neighbours.append(other["attempt_index"])
        candidate["transform_consensus"] = {
            "neighbour_attempts": neighbours,
            "consensus_count": len(neighbours),
            "translation_limit": translation_consensus_limit,
            "rotation_limit_deg": rotation_consensus_limit,
        }

    candidate_rows.sort(
        key=lambda item: (
            0 if item["validation"]["state"] == "passed" else 1,
            -int(item.get("transform_consensus", {}).get("consensus_count", 0)),
            -float(item["validation"].get("stable_wall_overlap") or 0.0),
            -float(item["validation"].get("normal_consistency") or 0.0),
            -float(item["validation"].get("fitness", 0.0)),
            float(item["validation"].get("inlier_rmse") or math.inf),
        )
    )
    consistency_checks = []
    if strict_consistency and candidate_rows:
        for candidate in candidate_rows[:2]:
            reverse_report = {"state": "not_run"}
            try:
                reverse_parameters = solver_parameters(candidate["parameters"])
                reverse, _, _, reverse_report = register_curvature_roi(
                    target, source, reverse_parameters
                )
                forward = np.asarray(candidate["transformation"], dtype=float)
                cycle = reverse @ forward
                rotation_error = math.degrees(
                    math.acos(float(np.clip((np.trace(cycle[:3, :3]) - 1) / 2, -1, 1)))
                )
                translation_error = float(np.linalg.norm(cycle[:3, 3]))
                check = {
                    "attempt_index": candidate["attempt_index"],
                    "translation_error": translation_error,
                    "rotation_error_deg": rotation_error,
                    "passed": translation_error <= 0.25 and rotation_error <= 5.0,
                    "reverse_report": reverse_report,
                }
            except Exception as error:
                check = {
                    "attempt_index": candidate["attempt_index"],
                    "passed": False,
                    "exception": repr(error),
                    "reverse_report": reverse_report,
                }
            candidate["reverse_consistency"] = check
            consistency_checks.append(check)
            if check["passed"]:
                selected = np.asarray(candidate["transformation"], dtype=float)
                selected_validation = candidate["validation"]
                candidate["selected"] = True
                break
    elif candidate_rows:
        candidate_rows[0]["selected"] = True
        selected = np.asarray(candidate_rows[0]["transformation"], dtype=float)
        selected_validation = candidate_rows[0]["validation"]

    attempt_validation_rows = [
        row.get("validation")
        for row in attempts
        if row.get("validation") and row["validation"].get("rotation_angles_deg")
    ]
    attempt_validation_rows.sort(
        key=lambda item: (
            0 if item.get("state") == "passed" else 1,
            0 if item.get("state") == "review" else 1,
            -float(item.get("fitness", 0.0)),
            float(item.get("inlier_rmse") or math.inf),
        )
    )
    best_attempt_validation = attempt_validation_rows[0] if attempt_validation_rows else None

    report = {
        "pair_name": pair_name,
        "success": selected is not None,
        "selected_validation": selected_validation,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "candidate_count": len(candidate_rows),
        "consistency_checks": consistency_checks,
        "strict_consistency": strict_consistency,
        "elapsed_seconds": time.perf_counter() - started,
        "base_parameters": base_parameters,
        "selected_pair_quality": (
            selected_validation.get("pair_quality") if selected_validation else None
        ),
        "selected_rotation_angles_deg": (
            selected_validation.get("rotation_angles_deg") if selected_validation else None
        ),
        "best_attempt_rotation_angles_deg": (
            best_attempt_validation.get("rotation_angles_deg")
            if best_attempt_validation else None
        ),
    }
    return selected, report


def transform_cloud(cloud: o3d.geometry.PointCloud, transformation: np.ndarray):
    result = clone_cloud(cloud)
    result.transform(transformation)
    return result


def refine_with_prior(source, target, prior, voxel_size):
    """Use the corresponding before-scan pose as a seed for after-scan ICP."""
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    if len(source_down.points) < 30 or len(target_down.points) < 30:
        return None
    radius = max(3.0 * voxel_size, 1e-3)
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50))
    result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        3.0 * voxel_size,
        np.asarray(prior, dtype=float),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
    )
    return result.transformation


def write_cloud(path: Path, cloud: o3d.geometry.PointCloud, ascii_output: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=ascii_output, compressed=False):
        raise IOError(f"点云保存失败: {path}")


def register_group(
    paths: list[Path],
    label: str,
    output_dir: Path,
    quality_thresholds: QualityThresholds,
    adaptive_settings: AdaptiveSettings,
    accept_review: bool,
    ascii_output: bool,
    prior_transforms: list[list[float] | None] | None = None,
) -> tuple[o3d.geometry.PointCloud, list[dict[str, Any]], dict[str, Any]]:
    """Register all scans in one time phase into the first scan's frame."""
    clouds = [load_cloud(path, f"{label} {path.name}") for path in paths]
    qualities = [
        assess_cloud(cloud, f"{label}/{path.name}", quality_thresholds)
        for path, cloud in zip(paths, clouds)
    ]
    for quality in qualities:
        print(
            f"[{label} 质量] {quality['name']}："
            f"点数={quality.get('point_count', 0)}，"
            f"s50={quality.get('nearest_neighbor_p50', 0.0):.6g}，"
            f"密度比={quality.get('density_uniformity_ratio', 0.0):.3f}，"
            f"粗糙度P90={quality.get('roughness_p90', 0.0):.6g}，"
            f"等级={quality.get('quality_level', 'D')}",
            flush=True,
        )
    if any(item.get("quality_level") == "D" for item in qualities):
        print(f"[{label}] 存在 D 级输入，仍会尝试配准并在报告中保留风险标记", flush=True)

    registered_dir = output_dir / f"{label}_registered_scans"
    registered_dir.mkdir(parents=True, exist_ok=True)
    reference = clone_cloud(clouds[0])
    merged = clone_cloud(reference)
    write_cloud(registered_dir / f"scan_01_{paths[0].stem}_registered.pcd", reference, ascii_output)
    transforms: list[list[float] | None] = [np.eye(4).tolist()]
    pair_reports: list[dict[str, Any]] = []

    for index in range(1, len(clouds)):
        base = choose_parameters(qualities[index], qualities[0], adaptive_settings)
        transformation = None
        report = None
        prior = None
        if prior_transforms and index < len(prior_transforms):
            prior = prior_transforms[index]
        if label == "after" and prior is not None:
            prior_refined = refine_with_prior(
                clouds[index], merged, prior, base["voxel_size"]
            )
            if prior_refined is not None:
                prior_validation = evaluate_transform(
                    clouds[index], merged, prior_refined,
                    float(base["reference_spacing"]), adaptive_settings,
                    validation_voxel=base["voxel_size"],
                )
                print(
                    f"[{label} scan {index + 1}] 使用装沙前相对位姿先验，"
                    f"状态={prior_validation['state']}，"
                    f"fitness={prior_validation.get('fitness', 0.0):.3f}",
                    flush=True,
                )
                if prior_validation["state"] == "passed" or (
                    accept_review and prior_validation["state"] == "review"
                ):
                    transformation = prior_refined
                    report = {
                        "pair_name": f"{label} scan {index + 1} -> registered global",
                        "success": True,
                        "selected_validation": prior_validation,
                        "selected_by": "before_pose_prior_icp",
                        "attempt_count": 0,
                        "attempts": [],
                        "base_parameters": base,
                    }
        if transformation is None:
            transformation, report = adaptive_register_pair(
                clouds[index],
                merged,
                base,
                adaptive_settings,
                f"{label} scan {index + 1} -> registered global",
                accept_review=accept_review if label != "before" else False,
                strict_consistency=(label == "before"),
            )
        report["source_path"] = str(paths[index])
        report["target"] = "merged_registered_cloud"
        pair_reports.append(report)
        if transformation is None:
            transforms.append(None)
            print(f"[{label}] scan {index + 1} 未通过验证，跳过合并", flush=True)
            continue

        registered = transform_cloud(clouds[index], transformation)
        merged += registered
        transforms.append(transformation.tolist())
        write_cloud(registered_dir / f"scan_{index + 1:02d}_{paths[index].stem}_registered.pcd", registered, ascii_output)
        np.savetxt(registered_dir / f"scan_{index + 1:02d}_transform.txt", transformation, fmt="%.10f")

    report = {
        "label": label,
        "reference": str(paths[0]),
        "scan_count": len(paths),
        "registered_count": int(sum(item is not None for item in transforms)),
        "quality_reports": qualities,
        "transforms_to_reference": transforms,
        "pair_reports": pair_reports,
        "selected_rotation_angles_deg": (
            pair_reports[-1].get("selected_rotation_angles_deg") if pair_reports else None
        ),
        "best_attempt_rotation_angles_deg": (
            pair_reports[-1].get("best_attempt_rotation_angles_deg") if pair_reports else None
        ),
        "status": "complete" if all(item is not None for item in transforms) else "partial",
    }
    write_cloud(output_dir / f"{label}_registered.pcd", merged, ascii_output)
    write_json(output_dir / f"{label}_registration_report.json", report)
    return merged, qualities, report


def make_coloured_overlay(before: o3d.geometry.PointCloud, after_aligned: o3d.geometry.PointCloud):
    before_coloured = clone_cloud(before)
    after_coloured = clone_cloud(after_aligned)
    before_coloured.paint_uniform_color([0.10, 0.45, 1.00])
    after_coloured.paint_uniform_color([1.00, 0.12, 0.05])
    return before_coloured + after_coloured


def print_rotation_summary(label: str, report: dict[str, Any]) -> None:
    angles = report.get("selected_rotation_angles_deg") or report.get("best_attempt_rotation_angles_deg")
    if angles:
        print(
            f"[{label}] yaw={angles.get('yaw_deg', 0.0):.2f}°，"
            f"pitch={angles.get('pitch_deg', 0.0):.2f}°，"
            f"roll={angles.get('roll_deg', 0.0):.2f}°",
            flush=True,
        )
    else:
        print(f"[{label}] 没有可记录的旋转角度", flush=True)


def run(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    quality_thresholds, adaptive_settings = settings_from_mapping(config)
    accept_review = bool(args.accept_review or config.get("accept_review", True))
    before_value = args.before or config.get("before_path", "")
    after_value = args.after or config.get("after_path", "")
    output_value = args.output_dir or config.get("output_dir", "output")
    if not before_value or not after_value:
        raise ValueError("请先在 config.yaml 中填写 before_path 和 after_path")
    before_paths = collect_pcd_paths(before_value, "装沙前")
    after_paths = collect_pcd_paths(after_value, "装沙后")

    output_root = Path(output_value)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {output_dir.resolve()}")
    print(f"装沙前输入: {len(before_paths)} 个 PCD")
    print(f"装沙后输入: {len(after_paths)} 个 PCD")

    before_merged, before_qualities, before_report = register_group(
        before_paths,
        "before",
        output_dir,
        quality_thresholds,
        adaptive_settings,
        accept_review,
        args.ascii,
    )
    after_merged, after_qualities, after_report = register_group(
        after_paths,
        "after",
        output_dir,
        quality_thresholds,
        adaptive_settings,
        accept_review,
        args.ascii,
        prior_transforms=(
            before_report.get("transforms_to_reference")
            if bool(config.get("use_before_pose_prior", False))
            else None
        ),
    )
    print_rotation_summary("前配准", before_report)
    print_rotation_summary("后配准", after_report)

    before_quality = assess_cloud(before_merged, "before_registered", quality_thresholds)
    after_quality = assess_cloud(after_merged, "after_registered", quality_thresholds)
    pair_parameters = choose_parameters(after_quality, before_quality, adaptive_settings)
    after_aligned, before_after_registration_report = adaptive_register_pair(
        after_merged,
        before_merged,
        pair_parameters,
        adaptive_settings,
        "after -> before",
        accept_review=accept_review,
    )
    print_rotation_summary("前后配准", before_after_registration_report)
    if after_aligned is None:
        write_json(output_dir / "before_after_registration_report.json", before_after_registration_report)
        write_json(output_dir / "adaptive_parameters.json", {
            "before": before_report.get("pair_reports", []),
            "after": after_report.get("pair_reports", []),
            "before_after": pair_parameters,
        })
        raise RuntimeError(
            "装沙后到装沙前的整体配准没有通过固定双向验证。"
            "请查看 before_after_registration_report.json 中的每轮参数和失败原因。"
        )

    after_aligned_cloud = transform_cloud(after_merged, after_aligned)
    write_cloud(output_dir / "after_aligned_to_before.pcd", after_aligned_cloud, args.ascii)
    overlay = make_coloured_overlay(before_merged, after_aligned_cloud)
    write_cloud(output_dir / "before_after_aligned_coloured.pcd", overlay, args.ascii)

    quality_report = {
        "demo_version": DEMO_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "before": [str(path) for path in before_paths],
            "after": [str(path) for path in after_paths],
        },
        "settings": {
            "quality_thresholds": serializable_settings(quality_thresholds),
            "adaptive_settings": serializable_settings(adaptive_settings),
            "accept_review": accept_review,
        },
        "before_inputs_quality": before_report["quality_reports"],
        "after_inputs_quality": after_report["quality_reports"],
        "registered_cloud_quality": {
            "before": before_quality,
            "after": after_quality,
        },
        "before_registration": before_report,
        "after_registration": after_report,
        "before_after_initial_parameters": pair_parameters,
        "before_after_registration": before_after_registration_report,
        "registration_rotation_angles_deg": {
            "before": before_report.get("selected_rotation_angles_deg")
            or before_report.get("best_attempt_rotation_angles_deg"),
            "after": after_report.get("selected_rotation_angles_deg")
            or after_report.get("best_attempt_rotation_angles_deg"),
            "before_after": before_after_registration_report.get("selected_rotation_angles_deg")
            or before_after_registration_report.get("best_attempt_rotation_angles_deg"),
        },
        "outputs": {
            "before_registered": str(output_dir / "before_registered.pcd"),
            "after_registered": str(output_dir / "after_registered.pcd"),
            "after_aligned_to_before": str(output_dir / "after_aligned_to_before.pcd"),
            "before_after_aligned_coloured": str(output_dir / "before_after_aligned_coloured.pcd"),
        },
    }
    write_json(output_dir / "quality_and_adaptive_report.json", quality_report)
    write_json(output_dir / "before_after_registration_report.json", before_after_registration_report)
    write_json(output_dir / "adaptive_parameters.json", {
        "before": before_report.get("pair_reports", []),
        "after": after_report.get("pair_reports", []),
        "before_after": pair_parameters,
    })

    print("\n完成。主要输出文件:")
    for name in (
        "before_registered.pcd",
        "after_registered.pcd",
        "after_aligned_to_before.pcd",
        "before_after_aligned_coloured.pcd",
        "quality_and_adaptive_report.json",
    ):
        print(f"  {output_dir / name}")
    return output_dir


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        print(f"运行失败: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

