import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    trimesh = None
logger = logging.getLogger(__name__)
@dataclass
class QualityMetrics:
    mesh_validity_score: float = 0.0
    mesh_manifold_score: float = 0.0
    mesh_watertight_score: float = 0.0
    mesh_smoothness_score: float = 0.0
    edge_quality_score: float = 0.0
    face_quality_score: float = 0.0
    vertex_count: int = 0
    face_count: int = 0
    edge_count: int = 0
    volume: float = 0.0
    surface_area: float = 0.0
    bounding_box_volume: float = 0.0
    compactness: float = 0.0
    uv_coverage: float = 0.0
    uv_distortion: float = 0.0
    uv_overlaps: int = 0
    uv_islands: int = 0
    texture_resolution: Optional[Tuple[int, int]] = None
    texture_sharpness: float = 0.0
    texture_color_variance: float = 0.0
    has_albedo: bool = False
    has_normal: bool = False
    has_roughness: bool = False
    has_metallic: bool = False
    polycount_score: float = 0.0
    draw_call_estimate: int = 0
    material_count: int = 0
    bone_count: int = 0
    overall_mesh_quality: float = 0.0
    overall_texture_quality: float = 0.0
    game_ready_score: float = 0.0
    prompt_adherence_score: float = 0.0
    def calculate_overall_scores(self) -> None:
        mesh_scores = [
            self.mesh_validity_score,
            self.mesh_manifold_score,
            self.mesh_watertight_score,
            self.mesh_smoothness_score,
            self.edge_quality_score,
            self.face_quality_score,
        ]
        self.overall_mesh_quality = np.mean([s for s in mesh_scores if s > 0])
        texture_scores = []
        if self.texture_resolution:
            texture_scores.append(self.texture_sharpness)
            texture_scores.append(100 - min(self.uv_distortion * 10, 100))
            if self.uv_coverage > 0:
                texture_scores.append(self.uv_coverage)
        if texture_scores:
            self.overall_texture_quality = np.mean(texture_scores)
        game_factors = []
        if 10000 <= self.face_count <= 50000:
            game_factors.append(100)
        elif 5000 <= self.face_count <= 100000:
            game_factors.append(70)
        else:
            game_factors.append(30)
        if self.uv_overlaps == 0 and self.uv_coverage > 50:
            game_factors.append(90)
        elif self.uv_overlaps < 5 and self.uv_coverage > 30:
            game_factors.append(60)
        else:
            game_factors.append(20)
        if self.material_count <= 2:
            game_factors.append(100)
        elif self.material_count <= 5:
            game_factors.append(70)
        else:
            game_factors.append(40)
        self.game_ready_score = np.mean(game_factors) if game_factors else 0.0
class QualityAssessor:
    def __init__(self):
        self.last_metrics: Optional[QualityMetrics] = None
    def assess_mesh_file(self, file_path: Path) -> QualityMetrics:
        metrics = QualityMetrics()
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return metrics
        if not TRIMESH_AVAILABLE:
            logger.warning("Trimesh not available, returning basic metrics")
            return self._get_basic_metrics(file_path)
        try:
            mesh = trimesh.load(file_path, force='mesh')
            if isinstance(mesh, trimesh.Scene):
                meshes = list(mesh.geometry.values())
                if meshes:
                    mesh = trimesh.util.concatenate(meshes)
                else:
                    logger.warning("No geometry found in scene")
                    return metrics
            metrics.vertex_count = len(mesh.vertices)
            metrics.face_count = len(mesh.faces)
            metrics.edge_count = len(mesh.edges)
            if mesh.is_watertight:
                metrics.volume = float(mesh.volume)
                metrics.mesh_watertight_score = 100.0
            else:
                metrics.mesh_watertight_score = 0.0
            metrics.surface_area = float(mesh.area)
            bounds = mesh.bounds
            if bounds is not None and len(bounds) == 2:
                bbox_size = bounds[1] - bounds[0]
                metrics.bounding_box_volume = float(np.prod(bbox_size))
                if metrics.bounding_box_volume > 0 and metrics.volume > 0:
                    metrics.compactness = metrics.volume / metrics.bounding_box_volume
            metrics.mesh_validity_score = 100.0 if mesh.is_valid else 0.0
            metrics.mesh_manifold_score = 100.0 if mesh.is_winding_consistent else 50.0
            edge_lengths = mesh.edges_unique_length
            if len(edge_lengths) > 0:
                edge_std = np.std(edge_lengths)
                edge_mean = np.mean(edge_lengths)
                if edge_mean > 0:
                    edge_cv = edge_std / edge_mean
                    metrics.edge_quality_score = max(0, 100 - edge_cv * 50)
            if hasattr(mesh, 'face_angles'):
                face_angles = mesh.face_angles
                good_angles = np.logical_and(face_angles > np.pi/6, face_angles < np.pi/2)
                metrics.face_quality_score = (np.sum(good_angles) / face_angles.size) * 100
            if hasattr(mesh, 'face_normals'):
                normals = mesh.face_normals
                smoothness = self._calculate_smoothness(mesh, normals)
                metrics.mesh_smoothness_score = smoothness
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
                uv = mesh.visual.uv
                if uv is not None and len(uv) > 0:
                    metrics.uv_coverage = self._calculate_uv_coverage(uv)
                    metrics.uv_distortion = self._calculate_uv_distortion(mesh, uv)
                    metrics.uv_overlaps = self._count_uv_overlaps(uv)
                    metrics.uv_islands = self._count_uv_islands(uv)
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
                metrics.material_count = 1
            metrics.calculate_overall_scores()
        except Exception as e:
            logger.error(f"Error assessing mesh: {e}")
        self.last_metrics = metrics
        return metrics
    def _get_basic_metrics(self, file_path: Path) -> QualityMetrics:
        metrics = QualityMetrics()
        file_size_mb = file_path.stat().st_size / (1024**2)
        if 0.5 <= file_size_mb <= 10:
            metrics.overall_mesh_quality = 70
        elif 0.1 <= file_size_mb <= 50:
            metrics.overall_mesh_quality = 50
        else:
            metrics.overall_mesh_quality = 30
        return metrics
    def _calculate_smoothness(self, mesh: Any, normals: np.ndarray) -> float:
        try:
            face_adjacency = mesh.face_adjacency
            if len(face_adjacency) == 0:
                return 50.0
            angles = []
            for face_pair in face_adjacency:
                n1 = normals[face_pair[0]]
                n2 = normals[face_pair[1]]
                dot_product = np.clip(np.dot(n1, n2), -1, 1)
                angle = np.arccos(dot_product)
                angles.append(angle)
            mean_angle = np.mean(angles)
            smoothness = max(0, 100 - (mean_angle * 180 / np.pi) * 2)
            return float(smoothness)
        except Exception as e:
            logger.debug(f"Could not calculate smoothness: {e}")
            return 50.0
    def _calculate_uv_coverage(self, uv: np.ndarray) -> float:
        try:
            if len(uv) == 0:
                return 0.0
            grid_size = 100
            grid = np.zeros((grid_size, grid_size), dtype=bool)
            uv_scaled = np.clip(uv * grid_size, 0, grid_size - 1).astype(int)
            for coord in uv_scaled:
                grid[coord[1], coord[0]] = True
            coverage = (np.sum(grid) / grid.size) * 100
            return float(coverage)
        except Exception as e:
            logger.debug(f"Could not calculate UV coverage: {e}")
            return 0.0
    def _calculate_uv_distortion(self, mesh: Any, uv: np.ndarray) -> float:
        try:
            edges_3d = mesh.edges_unique
            if len(edges_3d) == 0 or len(uv) < len(mesh.vertices):
                return 0.0
            distortions = []
            for edge in edges_3d[:100]:
                v1_3d = mesh.vertices[edge[0]]
                v2_3d = mesh.vertices[edge[1]]
                len_3d = np.linalg.norm(v2_3d - v1_3d)
                if edge[0] < len(uv) and edge[1] < len(uv):
                    v1_uv = uv[edge[0]]
                    v2_uv = uv[edge[1]]
                    len_uv = np.linalg.norm(v2_uv - v1_uv)
                    if len_3d > 0 and len_uv > 0:
                        ratio = len_uv / len_3d
                        distortion = abs(1 - ratio)
                        distortions.append(distortion)
            if distortions:
                return float(np.mean(distortions))
            return 0.0
        except Exception as e:
            logger.debug(f"Could not calculate UV distortion: {e}")
            return 0.0
    def _count_uv_overlaps(self, uv: np.ndarray) -> int:
        unique_uv = np.unique(uv, axis=0)
        overlaps = len(uv) - len(unique_uv)
        return max(0, overlaps // 10)
    def _count_uv_islands(self, uv: np.ndarray) -> int:
        if len(uv) == 0:
            return 0
        grid_size = 10
        uv_grid = (uv * grid_size).astype(int)
        unique_cells = np.unique(uv_grid, axis=0)
        return max(1, len(unique_cells) // 10)
    def compare_quality(
        self,
        metrics1: QualityMetrics,
        metrics2: QualityMetrics,
    ) -> Dict[str, float]:
        comparison = {}
        comparison["mesh_quality_diff"] = metrics1.overall_mesh_quality - metrics2.overall_mesh_quality
        comparison["texture_quality_diff"] = metrics1.overall_texture_quality - metrics2.overall_texture_quality
        comparison["game_ready_diff"] = metrics1.game_ready_score - metrics2.game_ready_score
        if metrics2.face_count > 0:
            comparison["polycount_ratio"] = metrics1.face_count / metrics2.face_count
        if metrics2.surface_area > 0:
            comparison["surface_area_ratio"] = metrics1.surface_area / metrics2.surface_area
        score1 = (metrics1.overall_mesh_quality + metrics1.overall_texture_quality + metrics1.game_ready_score) / 3
        score2 = (metrics2.overall_mesh_quality + metrics2.overall_texture_quality + metrics2.game_ready_score) / 3
        comparison["winner"] = 1 if score1 >= score2 else 2
        comparison["score_difference"] = score1 - score2
        return comparison