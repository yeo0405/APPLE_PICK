"""Small geometry primitives used by RACE-6D without compiled extensions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert the Zhou et al. 6D representation to rotation matrices.

    This follows PyTorch3D's row-vector convention and is intentionally kept
    API-compatible with ``pytorch3d.transforms.rotation_6d_to_matrix``.
    """
    if d6.shape[-1] != 6:
        raise ValueError(f"Expected a final dimension of 6, got {d6.shape}")
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to the non-unique 6D representation."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3), got {matrix.shape}")
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    if axis == "X":
        values = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        values = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        values = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError(f"Unknown axis {axis!r}")
    return torch.stack(values, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """Convert Euler angles to matrices using a three-letter convention."""
    if euler_angles.shape[-1] != 3:
        raise ValueError(f"Expected a final dimension of 3, got {euler_angles.shape}")
    if len(convention) != 3 or any(axis not in "XYZ" for axis in convention):
        raise ValueError(f"Invalid Euler convention {convention!r}")
    matrices = [
        _axis_angle_rotation(axis, angle)
        for axis, angle in zip(convention, euler_angles.unbind(-1))
    ]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def sample_farthest_points(
    points: torch.Tensor,
    K: int,
    max_candidates: int = 10000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic farthest-point sampling for small CAD initialization jobs.

    Dense clouds are first reduced to evenly spaced candidates.  This bounds
    initialization cost while retaining broad surface coverage; RACE-6D calls
    this once per CAD model and requests 1,500 points.
    """
    if points.ndim != 3:
        raise ValueError(f"Expected [batch, points, dimensions], got {points.shape}")
    batch_size, point_count, dimensions = points.shape
    if point_count == 0 or K <= 0:
        return (
            points.new_empty((batch_size, 0, dimensions)),
            torch.empty((batch_size, 0), dtype=torch.long, device=points.device),
        )

    candidate_indices = torch.arange(point_count, device=points.device)
    if point_count > max_candidates:
        candidate_indices = torch.linspace(
            0, point_count - 1, max_candidates, device=points.device
        ).long()
        candidates = points[:, candidate_indices]
    else:
        candidates = points

    candidate_count = candidates.shape[1]
    sample_count = min(int(K), candidate_count)
    selected_local = torch.empty(
        (batch_size, sample_count), dtype=torch.long, device=points.device
    )
    distances = torch.full(
        (batch_size, candidate_count), float("inf"), device=points.device
    )
    centroid = candidates.mean(dim=1, keepdim=True)
    farthest = ((candidates - centroid) ** 2).sum(-1).argmax(-1)
    batch_indices = torch.arange(batch_size, device=points.device)

    for index in range(sample_count):
        selected_local[:, index] = farthest
        selected_points = candidates[batch_indices, farthest].unsqueeze(1)
        squared_distance = ((candidates - selected_points) ** 2).sum(-1)
        distances = torch.minimum(distances, squared_distance)
        farthest = distances.argmax(-1)

    sampled = candidates.gather(
        1, selected_local.unsqueeze(-1).expand(-1, -1, dimensions)
    )
    selected_original = candidate_indices[selected_local]
    return sampled, selected_original
