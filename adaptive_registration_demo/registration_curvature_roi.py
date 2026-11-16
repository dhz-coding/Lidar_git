"""Curvature ROI registration migrated from registration_compare_simplified.py.

Both RANSAC and point-to-point ICP operate on the downsampled curvature ROI.
The returned source-to-target transform can be applied to the original cloud.
"""

import time
from types import SimpleNamespace

import numpy as np
import open3d as o3d


DEFAULT_PARAMETERS = {
    "voxel_size": 0.1,
    "normal_radius_ratio": 3.0,
    "feature_radius_ratio": 5.0,
    "ransac_distance_ratio": 2.0,
    "ransac_n": 3,
    "ransac_max_iteration": 4000000,
    "ransac_confidence": 0.99,
    "edge_length_threshold": 0.9,
    "icp_distance_ratio": 2.0,
    "icp_max_iteration": 500,
    "curvature_radius_ratio": 4.0,
    "source_curvature_percentile": 60.0,
    "target_curvature_percentile": 60.0,
    "mid_curvature_percentile": 35.0,
    "high_curvature_fraction": 0.4,
    "mid_curvature_fraction": 0.3,
    "uniform_fraction": 0.3,
    "max_roi_points": 12000,
    "min_roi_points": 100,
    "cluster_eps_ratio": 4.0,
    "min_cluster_points": 8,
    "max_clusters": 5,
    "disable_cluster_filter": False,
    "enable_normal_axis_filter": False,
    "normal_axis": [0.0, 0.0, 1.0],
    "normal_axis_angle": 35.0,
    "enable_post_normal_filter": False,
    "post_normal_angle": 35.0,
    "max_downsample_points": 40000,
    "mutual_filter": True,
    "icp_use_full_cloud": True,
}


class CurvatureROIRegistrationError(ValueError):
    """Keep partial metrics when ROI extraction or registration fails."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def curvature_roi_parameters(parameters=None):
    parameters = {} if parameters is None else dict(parameters)
    unknown = set(parameters) - set(DEFAULT_PARAMETERS)
    if unknown:
        raise ValueError(f"Unknown curvature_roi parameters: {sorted(unknown)}")
    result = {**DEFAULT_PARAMETERS, **parameters}
    positive = (
        "voxel_size", "normal_radius_ratio", "feature_radius_ratio",
        "ransac_distance_ratio", "icp_distance_ratio", "curvature_radius_ratio",
        "cluster_eps_ratio",
        "ransac_max_iteration", "icp_max_iteration", "min_roi_points",
        "min_cluster_points", "max_clusters", "ransac_n",
        "max_downsample_points", "max_roi_points",
    )
    for key in positive:
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"curvature_roi.{key} must be finite and positive")
    for key in (
        "ransac_n", "ransac_max_iteration", "icp_max_iteration",
        "min_roi_points", "min_cluster_points", "max_clusters",
        "max_downsample_points",
    ):
        if int(result[key]) != result[key]:
            raise ValueError(f"curvature_roi.{key} must be an integer")
        result[key] = int(result[key])
    if result["ransac_n"] < 3:
        raise ValueError("curvature_roi.ransac_n must be at least 3")
    for key in ("source_curvature_percentile", "target_curvature_percentile"):
        if not 0 <= result[key] < 100:
            raise ValueError(f"curvature_roi.{key} must be in [0, 100)")
    if not 0 <= result["mid_curvature_percentile"] < 100:
        raise ValueError("curvature_roi.mid_curvature_percentile must be in [0, 100)")
    for key in ("high_curvature_fraction", "mid_curvature_fraction", "uniform_fraction"):
        if not np.isfinite(result[key]) or result[key] < 0:
            raise ValueError(f"curvature_roi.{key} must be finite and non-negative")
    if result["high_curvature_fraction"] + result["mid_curvature_fraction"] + result["uniform_fraction"] <= 0:
        raise ValueError("curvature_roi sampling fractions must have a positive sum")
    for key in ("ransac_confidence", "edge_length_threshold"):
        if not 0 < result[key] <= 1:
            raise ValueError(f"curvature_roi.{key} must be in (0, 1]")
    for key in ("normal_axis_angle", "post_normal_angle"):
        if not 0 <= result[key] <= 90:
            raise ValueError(f"curvature_roi.{key} must be in [0, 90]")
    if result["enable_normal_axis_filter"]:
        axis = np.asarray(result["normal_axis"], dtype=float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
            raise ValueError("curvature_roi.normal_axis must be a finite nonzero 3-vector")
    return result


def valid_transform(matrix):
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return False
    rotation = matrix[:3, :3]
    return bool(
        np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4)
    )


def compute_curvatures(cloud, radius, max_nn=50, min_nn=6):
    """Local PCA surface variation, with the same neighborhood as the experiment."""
    points = np.asarray(cloud.points)
    curvatures = np.zeros(len(points), dtype=np.float64)
    tree = o3d.geometry.KDTreeFlann(cloud)
    for point_index, point in enumerate(points):
        _, indices, _ = tree.search_hybrid_vector_3d(point, radius, max_nn)
        if len(indices) < min_nn:
            continue
        neighbors = points[indices]
        centered = neighbors - neighbors.mean(axis=0)
        covariance = centered.T @ centered / (len(indices) - 1)
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalue_sum = np.sum(eigenvalues)
        if eigenvalue_sum > 1e-12:
            curvatures[point_index] = max(0.0, eigenvalues[0]) / eigenvalue_sum
    return curvatures


def estimate_normals(cloud, radius):
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50))
    cloud.normalize_normals()


def normal_axis_mask(normals, axis, max_angle_deg):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    cosines = np.zeros(len(normals))
    cosines[valid] = np.abs(normals[valid] @ axis) / lengths[valid]
    return valid & (cosines >= np.cos(np.deg2rad(max_angle_deg)))


def _sample_indices(indices, count, rng):
    indices = np.asarray(indices, dtype=np.int64)
    if count <= 0 or len(indices) == 0:
        return np.empty(0, dtype=np.int64)
    if len(indices) <= count:
        return indices.copy()
    return np.sort(rng.choice(indices, size=count, replace=False))


def extract_roi(cloud, percentile, args):
    """Build a mixed ROI instead of keeping only the highest-curvature tail.

    High-curvature points improve feature distinctiveness, while mid-curvature
    and spatially uniform points keep the hypothesis tied to larger structures.
    """
    down = cloud.voxel_down_sample(args.voxel_size)
    if len(down.points) > args.max_downsample_points:
        # Keep preprocessing bounded and deterministic for very dense scans.
        rng = np.random.default_rng(20261112)
        indices = rng.choice(
            len(down.points), size=args.max_downsample_points, replace=False
        )
        down = down.select_by_index(np.sort(indices).tolist())
    curvatures = compute_curvatures(down, args.curvature_radius_ratio * args.voxel_size)
    valid_indices = np.flatnonzero(np.isfinite(curvatures) & (curvatures > 1e-12))
    if len(valid_indices) < max(args.min_roi_points, args.ransac_n):
        raise ValueError("Insufficient positive-curvature points; no full-cloud fallback")
    valid_cloud = down.select_by_index(valid_indices.tolist())
    valid_curvatures = curvatures[valid_indices]
    high_threshold = np.percentile(valid_curvatures, percentile)
    mid_threshold = np.percentile(
        valid_curvatures,
        min(float(args.mid_curvature_percentile), max(float(percentile) - 1e-6, 0.0)),
    )
    high_local = np.flatnonzero(valid_curvatures >= high_threshold)
    mid_local = np.flatnonzero(
        (valid_curvatures >= mid_threshold) & (valid_curvatures < high_threshold)
    )
    all_local = np.arange(len(valid_indices), dtype=np.int64)
    rng = np.random.default_rng(20261112)
    fractions = np.asarray([
        args.high_curvature_fraction,
        args.mid_curvature_fraction,
        args.uniform_fraction,
    ], dtype=float)
    fractions /= np.sum(fractions)
    roi_limit = max(int(args.max_roi_points), int(args.min_roi_points), int(args.ransac_n))
    quotas = np.floor(fractions * roi_limit).astype(int)
    quotas[-1] += roi_limit - int(np.sum(quotas))

    chosen_high = _sample_indices(high_local, quotas[0], rng)
    chosen_mid = _sample_indices(mid_local, quotas[1], rng)
    chosen = np.unique(np.concatenate([chosen_high, chosen_mid]))
    remaining = np.setdiff1d(all_local, chosen, assume_unique=False)
    chosen_uniform = _sample_indices(remaining, quotas[2], rng)
    selected_local = np.unique(np.concatenate([chosen, chosen_uniform]))
    if len(selected_local) < roi_limit:
        fill_pool = np.setdiff1d(all_local, selected_local, assume_unique=False)
        fill = _sample_indices(fill_pool, roi_limit - len(selected_local), rng)
        selected_local = np.unique(np.concatenate([selected_local, fill]))

    roi = valid_cloud.select_by_index(selected_local.tolist())
    counts = {
        "down_points": len(down.points),
        "curvature_points": len(valid_indices),
        "high_curvature_points": len(chosen_high),
        "mid_curvature_points": len(chosen_mid),
        "uniform_points": len(chosen_uniform),
        "roi_points_before_filter": len(roi.points),
    }
    if args.enable_normal_axis_filter:
        estimate_normals(roi, args.normal_radius_ratio * args.voxel_size)
        mask = normal_axis_mask(np.asarray(roi.normals), args.normal_axis, args.normal_axis_angle)
        roi = roi.select_by_index(np.flatnonzero(mask).tolist())
    if not args.disable_cluster_filter and len(roi.points):
        labels = np.asarray(roi.cluster_dbscan(
            eps=args.cluster_eps_ratio * args.voxel_size,
            min_points=args.min_cluster_points,
            print_progress=False,
        ))
        unique_labels, sizes = np.unique(labels[labels >= 0], return_counts=True)
        keep_labels = unique_labels[np.argsort(sizes)[::-1][:args.max_clusters]]
        roi = roi.select_by_index(np.flatnonzero(np.isin(labels, keep_labels)).tolist())
    counts["roi_points"] = len(roi.points)
    if len(roi.points) < max(args.min_roi_points, args.ransac_n):
        raise ValueError(f"ROI has only {len(roi.points)} points; no full-cloud fallback")
    return roi, counts


def compute_fpfh(cloud, args):
    # The ROI was already downsampled before computing curvature.
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=args.normal_radius_ratio * args.voxel_size, max_nn=30,
    ))
    return o3d.pipelines.registration.compute_fpfh_feature(
        cloud, o3d.geometry.KDTreeSearchParamHybrid(
            radius=args.feature_radius_ratio * args.voxel_size, max_nn=100,
        ),
    )


def filter_by_post_normal_consistency(source, target, transformation, threshold, max_angle_deg):
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, transformation)
    correspondences = np.asarray(evaluation.correspondence_set)
    if len(correspondences) < 3:
        raise ValueError("Insufficient correspondences for post-normal filtering")
    source_normals = np.asarray(source.normals)[correspondences[:, 0]] @ transformation[:3, :3].T
    target_normals = np.asarray(target.normals)[correspondences[:, 1]]
    lengths = np.linalg.norm(source_normals, axis=1) * np.linalg.norm(target_normals, axis=1)
    valid = lengths > 1e-12
    cosines = np.zeros(len(lengths))
    cosines[valid] = np.abs(np.sum(source_normals[valid] * target_normals[valid], axis=1)) / lengths[valid]
    keep = valid & (cosines >= np.cos(np.deg2rad(max_angle_deg)))
    if np.count_nonzero(keep) < 3:
        raise ValueError("Insufficient points after post-normal filtering")
    return (
        source.select_by_index(correspondences[keep, 0].tolist()),
        target.select_by_index(np.unique(correspondences[keep, 1]).tolist()),
    )


def result_metrics(result, elapsed=0.0):
    return {
        "elapsed": float(elapsed),
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "correspondence_count": len(result.correspondence_set),
    }


def register_curvature_roi(source, target, parameters=None):
    """Return (total, RANSAC, ICP_delta, report), without changing either input."""
    parameters = curvature_roi_parameters(parameters)
    args = SimpleNamespace(**parameters)
    started = time.perf_counter()
    report = {"method": "curvature_roi", "parameters": parameters, "timing": {}}
    registration = o3d.pipelines.registration
    try:
        for name, cloud in (("source", source), ("target", target)):
            if len(cloud.points) < args.ransac_n or not np.isfinite(np.asarray(cloud.points)).all():
                raise ValueError(f"{name} must contain enough finite points")
        print(
            f"[ROI 参数] 体素={args.voxel_size:g}，曲率半径={args.curvature_radius_ratio * args.voxel_size:g}，"
            f"源/目标百分位=P{args.source_curvature_percentile:g}/P{args.target_curvature_percentile:g}，"
            f"聚类筛选={'关闭' if args.disable_cluster_filter else '开启'}",
            flush=True,
        )
        preparation_started = time.perf_counter()
        roi_clouds = []
        for name, label, cloud, percentile in (
            ("source", "源点云", source, args.source_curvature_percentile),
            ("target", "目标点云", target, args.target_curvature_percentile),
        ):
            selected, counts = extract_roi(cloud, percentile, args)
            roi_clouds.append(selected)
            report[name] = counts
            keep_ratio = counts["roi_points"] / counts["down_points"]
            print(
                f"[曲率 ROI] {label}：下采样 {counts['down_points']} -> ROI {counts['roi_points']} 点"
                f"（保留 {keep_ratio:.1%}）",
                flush=True,
            )
        source_roi, target_roi = roi_clouds
        report["timing"]["roi_seconds"] = time.perf_counter() - preparation_started

        ransac_threshold = args.ransac_distance_ratio * args.voxel_size
        print(f"[ROI 阶段] 开始 FPFH + RANSAC 粗配准，距离阈值={ransac_threshold:g}", flush=True)
        coarse_started = time.perf_counter()
        source_fpfh = compute_fpfh(source_roi, args)
        target_fpfh = compute_fpfh(target_roi, args)
        coarse = registration.registration_ransac_based_on_feature_matching(
            source_roi, target_roi, source_fpfh, target_fpfh,
            mutual_filter=args.mutual_filter,
            max_correspondence_distance=ransac_threshold,
            estimation_method=registration.TransformationEstimationPointToPoint(False),
            ransac_n=args.ransac_n,
            checkers=[
                registration.CorrespondenceCheckerBasedOnEdgeLength(args.edge_length_threshold),
                registration.CorrespondenceCheckerBasedOnDistance(ransac_threshold),
            ],
            criteria=registration.RANSACConvergenceCriteria(
                max_iteration=args.ransac_max_iteration, confidence=args.ransac_confidence,
            ),
        )
        report["ransac"] = result_metrics(coarse, time.perf_counter() - coarse_started)
        if not valid_transform(coarse.transformation) or len(coarse.correspondence_set) < 3:
            raise ValueError("RANSAC did not produce a valid coarse registration")

        post_started = time.perf_counter()
        source_icp, target_icp = source_roi, target_roi
        if args.enable_post_normal_filter:
            source_icp, target_icp = filter_by_post_normal_consistency(
                source_roi, target_roi, coarse.transformation, ransac_threshold, args.post_normal_angle,
            )
        report["timing"]["post_filter_seconds"] = time.perf_counter() - post_started
        if parameters.get("icp_use_full_cloud", True):
            source_icp = source.voxel_down_sample(args.voxel_size)
            target_icp = target.voxel_down_sample(args.voxel_size)
            if len(source_icp.points) > args.max_downsample_points:
                source_icp = source_icp.select_by_index(
                    np.sort(np.random.default_rng(20261112).choice(
                        len(source_icp.points), size=args.max_downsample_points, replace=False
                    )).tolist()
                )
            if len(target_icp.points) > args.max_downsample_points:
                target_icp = target_icp.select_by_index(
                    np.sort(np.random.default_rng(20261113).choice(
                        len(target_icp.points), size=args.max_downsample_points, replace=False
                    )).tolist()
                )
        report["source"]["icp_points"] = len(source_icp.points)
        report["target"]["icp_points"] = len(target_icp.points)
        source_aligned = o3d.geometry.PointCloud(source_icp)
        source_aligned.transform(coarse.transformation)
        print(
            f"[ROI 阶段] 开始 ROI ICP 精配准，源/目标={len(source_icp.points)}/{len(target_icp.points)} 点，"
            f"距离阈值={args.icp_distance_ratio * args.voxel_size:g}",
            flush=True,
        )
        icp_started = time.perf_counter()
        icp = registration.registration_icp(
            source_aligned, target_icp, args.icp_distance_ratio * args.voxel_size, np.eye(4),
            registration.TransformationEstimationPointToPoint(),
            registration.ICPConvergenceCriteria(max_iteration=args.icp_max_iteration),
        )
        report["icp"] = result_metrics(icp, time.perf_counter() - icp_started)
        transformation = icp.transformation @ coarse.transformation
        if not valid_transform(transformation) or len(icp.correspondence_set) < 3:
            raise ValueError("ICP did not produce a valid fine registration")

        report["timing"]["total_seconds"] = time.perf_counter() - started
        return transformation, coarse.transformation.copy(), icp.transformation.copy(), report
    except Exception as error:
        report["timing"]["total_seconds"] = time.perf_counter() - started
        raise CurvatureROIRegistrationError(str(error), report) from error
