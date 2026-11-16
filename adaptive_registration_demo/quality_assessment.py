"""Point-cloud quality assessment and adaptive registration profiles.

The measurements are intentionally scale-aware.  Registration parameters are
derived from the observed nearest-neighbour spacing instead of assuming that
every scanner and every scan has the same density.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class QualityThresholds:
    """Initial rules for the demo quality grades.

    These values are deliberately kept in the configuration file so they can
    be calibrated with the user's own scanner data.
    """

    min_points: int = 1000
    min_finite_ratio: float = 0.995
    max_duplicate_ratio: float = 0.03
    grade_a_density_ratio: float = 5.0
    grade_b_density_ratio: float = 9.0
    grade_a_roughness_p90: float = 0.08
    grade_b_roughness_p90: float = 0.18


@dataclass
class AdaptiveSettings:
    """Knobs that convert quality measurements into solver parameters."""

    voxel_factor: float = 8.0
    min_voxel_size: float = 0.04
    max_voxel_size: float = 0.25
    normal_radius_ratio: float = 3.0
    feature_radius_ratio: float = 6.0
    curvature_radius_ratio: float = 4.0
    ransac_distance_ratio: float = 3.5
    icp_distance_ratio: float = 2.0
    ransac_n: int = 4
    ransac_max_iteration: int = 1_000_000
    ransac_confidence: float = 0.999
    edge_length_threshold: float = 0.90
    icp_max_iteration: int = 500
    max_downsample_points: int = 40_000
    min_roi_points: int = 300
    min_cluster_points: int = 8
    max_clusters: int = 8
    enable_cluster_filter: bool = True
    max_attempts: int = 6
    validation_distance_factor: float = 3.0
    validation_rmse_factor: float = 3.5
    min_validation_fitness: float = 0.35
    review_validation_fitness: float = 0.20
    review_rmse_factor: float = 5.0
    validation_sample_points: int = 60_000


def _finite_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("点云坐标必须是 N×3 数组")
    return points[np.isfinite(points).all(axis=1)]


def _sample_points(points: np.ndarray, max_points: int, seed: int = 20261112) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[np.sort(indices)]


def _local_roughness(points: np.ndarray, neighbours: int = 16) -> np.ndarray:
    """Estimate local surface variation with a vectorized neighbourhood PCA."""
    if len(points) < 4:
        return np.empty(0, dtype=np.float64)
    k = min(max(4, neighbours), len(points))
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k, workers=1)
    neighbourhoods = points[indices]
    centred = neighbourhoods - neighbourhoods.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centred, centred) / max(k - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    sums = eigenvalues.sum(axis=1)
    result = np.zeros(len(points), dtype=np.float64)
    valid = sums > 1e-12
    result[valid] = np.maximum(eigenvalues[valid, 0], 0.0) / sums[valid]
    return result


def _duplicate_ratio(points: np.ndarray, spacing: float) -> float:
    if len(points) < 2 or spacing <= 0:
        return 0.0
    # A small quantization cell detects exact and near-exact repeated returns
    # without an O(N²) pairwise comparison.
    cell = max(spacing * 0.20, 1e-9)
    quantized = np.floor(points / cell).astype(np.int64)
    unique = np.unique(quantized, axis=0)
    return float(1.0 - len(unique) / len(points))


def assess_points(
    points: np.ndarray,
    name: str,
    thresholds: QualityThresholds | None = None,
    sample_points: int = 60_000,
) -> dict[str, Any]:
    """Assess one point cloud without modifying the source cloud."""
    thresholds = thresholds or QualityThresholds()
    raw = np.asarray(points)
    total_points = int(len(raw))
    finite_mask = np.isfinite(raw).all(axis=1) if raw.ndim == 2 and raw.shape[1] == 3 else np.zeros(total_points, dtype=bool)
    finite_ratio = float(finite_mask.mean()) if total_points else 0.0
    clean = _finite_points(raw) if raw.ndim == 2 and raw.shape[1] == 3 else np.empty((0, 3))

    if len(clean) < 2:
        return {
            "name": name,
            "point_count": total_points,
            "finite_point_count": int(len(clean)),
            "finite_ratio": finite_ratio,
            "quality_level": "D",
            "quality_status": "invalid_or_empty",
            "quality_scores": {},
        }

    sampled = _sample_points(clean, sample_points)
    minimum = sampled.min(axis=0)
    maximum = sampled.max(axis=0)
    extent = maximum - minimum
    diagonal = float(np.linalg.norm(extent))
    tree = cKDTree(sampled)
    distances, _ = tree.query(sampled, k=2, workers=1)
    nearest = distances[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    if len(nearest) == 0:
        spacing_p50 = spacing_p90 = density_ratio = 0.0
    else:
        spacing_p50 = float(np.percentile(nearest, 50))
        spacing_p90 = float(np.percentile(nearest, 90))
        density_ratio = float(spacing_p90 / max(spacing_p50, 1e-12))

    duplicate_ratio = _duplicate_ratio(sampled, spacing_p50)
    roughness = _local_roughness(sampled)
    roughness_p50 = float(np.percentile(roughness, 50)) if len(roughness) else 1.0
    roughness_p90 = float(np.percentile(roughness, 90)) if len(roughness) else 1.0
    roughness_valid_ratio = float(np.count_nonzero(roughness > 0) / max(len(roughness), 1))

    integrity_score = min(1.0, finite_ratio / max(thresholds.min_finite_ratio, 1e-9))
    duplicate_score = max(0.0, 1.0 - duplicate_ratio / max(thresholds.max_duplicate_ratio, 1e-9))
    density_score = min(1.0, thresholds.grade_b_density_ratio / max(density_ratio, 1e-9))
    noise_score = min(1.0, thresholds.grade_b_roughness_p90 / max(roughness_p90, 1e-9))
    feature_score = min(1.0, roughness_valid_ratio / 0.50)

    if (
        len(clean) < thresholds.min_points
        or finite_ratio < thresholds.min_finite_ratio * 0.98
        or spacing_p50 <= 0
    ):
        level = "D"
        status = "insufficient_input"
    elif (
        finite_ratio >= thresholds.min_finite_ratio
        and duplicate_ratio <= thresholds.max_duplicate_ratio
        and density_ratio <= thresholds.grade_a_density_ratio
        and roughness_p90 <= thresholds.grade_a_roughness_p90
    ):
        level = "A"
        status = "good"
    elif (
        finite_ratio >= thresholds.min_finite_ratio * 0.99
        and density_ratio <= thresholds.grade_b_density_ratio
        and roughness_p90 <= thresholds.grade_b_roughness_p90
    ):
        level = "B"
        status = "usable"
    else:
        level = "C"
        status = "degraded_but_attemptable"

    return {
        "name": name,
        "point_count": total_points,
        "finite_point_count": int(len(clean)),
        "sampled_point_count": int(len(sampled)),
        "finite_ratio": finite_ratio,
        "duplicate_ratio": duplicate_ratio,
        "bbox_min": minimum.tolist(),
        "bbox_max": maximum.tolist(),
        "bbox_extent": extent.tolist(),
        "bbox_diagonal": diagonal,
        "nearest_neighbor_p50": spacing_p50,
        "nearest_neighbor_p90": spacing_p90,
        "density_uniformity_ratio": density_ratio,
        "roughness_p50": roughness_p50,
        "roughness_p90": roughness_p90,
        "roughness_valid_ratio": roughness_valid_ratio,
        "quality_scores": {
            "integrity": float(integrity_score),
            "duplicate_free": float(duplicate_score),
            "density": float(density_score),
            "noise": float(noise_score),
            "feature_support": float(feature_score),
        },
        "quality_level": level,
        "quality_status": status,
    }


def assess_cloud(cloud, name: str, thresholds: QualityThresholds | None = None, sample_points: int = 60_000):
    """Open3D adapter for :func:`assess_points`."""
    return assess_points(np.asarray(cloud.points), name, thresholds, sample_points)


def _worst_level(left: str, right: str) -> str:
    order = {"A": 0, "B": 1, "C": 2, "D": 3}
    return left if order.get(left, 3) >= order.get(right, 3) else right


def choose_parameters(
    source_quality: dict[str, Any],
    target_quality: dict[str, Any],
    settings: AdaptiveSettings | None = None,
) -> dict[str, Any]:
    """Create initial curvature-ROI registration parameters from two reports."""
    settings = settings or AdaptiveSettings()
    source_spacing = float(source_quality.get("nearest_neighbor_p50", 0.0))
    target_spacing = float(target_quality.get("nearest_neighbor_p50", 0.0))
    spacing = max(source_spacing, target_spacing, 1e-4)
    tier = _worst_level(
        str(source_quality.get("quality_level", "D")),
        str(target_quality.get("quality_level", "D")),
    )

    tier_factor = {"A": 0.85, "B": 1.0, "C": 1.25, "D": 1.50}.get(tier, 1.50)
    voxel = float(np.clip(
        settings.voxel_factor * spacing * tier_factor,
        settings.min_voxel_size,
        settings.max_voxel_size,
    ))

    # A sparse or noisy source needs more candidate features.  The before and
    # after clouds use separate percentiles because sand can hide structure.
    source_percentile = {"A": 78.0, "B": 72.0, "C": 62.0, "D": 55.0}.get(tier, 55.0)
    target_percentile = {"A": 85.0, "B": 78.0, "C": 68.0, "D": 60.0}.get(tier, 60.0)
    if source_quality.get("quality_level") in {"C", "D"}:
        source_percentile -= 5.0
    if target_quality.get("quality_level") in {"C", "D"}:
        target_percentile -= 5.0

    feature_radius_ratio = settings.feature_radius_ratio * (1.15 if tier in {"C", "D"} else 1.0)
    ransac_distance_ratio = settings.ransac_distance_ratio * (1.20 if tier in {"C", "D"} else 1.0)
    icp_distance_ratio = settings.icp_distance_ratio * (1.15 if tier in {"C", "D"} else 1.0)

    return {
        "voxel_size": voxel,
        "normal_radius_ratio": settings.normal_radius_ratio,
        "feature_radius_ratio": feature_radius_ratio,
        "ransac_distance_ratio": ransac_distance_ratio,
        "ransac_n": settings.ransac_n,
        "ransac_max_iteration": settings.ransac_max_iteration,
        "ransac_confidence": settings.ransac_confidence,
        "edge_length_threshold": settings.edge_length_threshold,
        "icp_distance_ratio": icp_distance_ratio,
        "icp_max_iteration": settings.icp_max_iteration,
        "curvature_radius_ratio": settings.curvature_radius_ratio,
        "source_curvature_percentile": float(np.clip(source_percentile, 35.0, 92.0)),
        "target_curvature_percentile": float(np.clip(target_percentile, 35.0, 95.0)),
        "min_roi_points": settings.min_roi_points,
        "cluster_eps_ratio": 4.0 if tier in {"A", "B"} else 5.0,
        "min_cluster_points": settings.min_cluster_points,
        "max_clusters": settings.max_clusters,
        "disable_cluster_filter": not settings.enable_cluster_filter or tier == "D",
        "enable_normal_axis_filter": False,
        "normal_axis": [0.0, 0.0, 1.0],
        "normal_axis_angle": 35.0,
        "enable_post_normal_filter": False,
        "post_normal_angle": 35.0,
        "max_downsample_points": settings.max_downsample_points,
        "mutual_filter": True,
        "quality_level": tier,
        "reference_spacing": spacing,
    }


def settings_from_mapping(mapping: dict[str, Any] | None) -> tuple[QualityThresholds, AdaptiveSettings]:
    mapping = mapping or {}
    quality = QualityThresholds(**{k: v for k, v in mapping.get("quality", {}).items() if k in QualityThresholds.__annotations__})
    adaptive = AdaptiveSettings(**{k: v for k, v in mapping.get("adaptive", {}).items() if k in AdaptiveSettings.__annotations__})
    return quality, adaptive


def serializable_settings(settings: QualityThresholds | AdaptiveSettings) -> dict[str, Any]:
    return asdict(settings)

