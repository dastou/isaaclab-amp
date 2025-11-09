# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utilitaires mathématiques pour Isaac Lab - Adapté d'Isaac Gym.

IMPORTANT: Convention de quaternions Isaac Lab
- Isaac Gym utilisait: [x, y, z, w]
- Isaac Lab utilise: [w, x, y, z]

Tous les quaternions dans ce fichier suivent la convention Isaac Lab [w, x, y, z].
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple


def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    """Convertit un tableau numpy ou une valeur en tenseur torch."""
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    """Normalise un tenseur."""
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def normalize_angle(x):
    """Normalise un angle dans [-pi, pi]."""
    return torch.atan2(torch.sin(x), torch.cos(x))


def copysign(a, b):
    """Applique le signe de b à la valeur absolue de a.

    Args:
        a: Scalaire (float) ou Tensor
        b: Tensor

    Returns:
        Tensor avec la magnitude de a et le signe de b
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, device=b.device, dtype=torch.float)
    if a.dim() == 0:  # Scalaire
        a = a.repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


# ==================== Fonctions de Quaternions ====================
# Convention Isaac Lab: quaternions au format [w, x, y, z]
# ================================================================


@torch.jit.script
def quat_mul(a, b):
    """Multiplie deux quaternions.

    Args:
        a: Premier quaternion [w, x, y, z], shape (..., 4)
        b: Second quaternion [w, x, y, z], shape (..., 4)

    Returns:
        Produit des quaternions [w, x, y, z], shape (..., 4)
    """
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    # Convention Isaac Lab: [w, x, y, z]
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    # Formule de multiplication de quaternions
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    quat = torch.stack([w, x, y, z], dim=-1).view(shape)
    return quat


@torch.jit.script
def quat_conjugate(a):
    """Calcule le conjugué d'un quaternion.

    Args:
        a: Quaternion [w, x, y, z], shape (..., 4)

    Returns:
        Conjugué [w, -x, -y, -z], shape (..., 4)
    """
    shape = a.shape
    a = a.reshape(-1, 4)
    # Conjugué: [w, -x, -y, -z]
    return torch.cat((a[:, :1], -a[:, 1:]), dim=-1).view(shape)


@torch.jit.script
def quat_unit(a):
    """Normalise un quaternion.

    Args:
        a: Quaternion [w, x, y, z], shape (..., 4)

    Returns:
        Quaternion normalisé, shape (..., 4)
    """
    return normalize(a)


@torch.jit.script
def quat_apply(a, b):
    """Applique une rotation par quaternion à un vecteur.

    Args:
        a: Quaternion [w, x, y, z], shape (..., 4)
        b: Vecteur 3D, shape (..., 3)

    Returns:
        Vecteur tourné, shape (..., 3)
    """
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)

    # Extraire les composantes [w, x, y, z]
    xyz = a[:, 1:]  # [x, y, z]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, :1] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_rotate(q, v):
    """Applique une rotation par quaternion à un vecteur.

    Args:
        q: Quaternion [w, x, y, z], shape (N, 4)
        v: Vecteur 3D, shape (N, 3)

    Returns:
        Vecteur tourné, shape (N, 3)
    """
    shape = q.shape
    # Convention Isaac Lab: [w, x, y, z]
    q_w = q[:, 0]
    q_vec = q[:, 1:]  # [x, y, z]

    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0

    return a + b + c


@torch.jit.script
def get_basis_vector(q, v):
    """Obtient un vecteur de base après rotation.

    Alias pour quat_rotate pour compatibilité.

    Args:
        q: Quaternion [w, x, y, z], shape (N, 4)
        v: Vecteur 3D, shape (N, 3)

    Returns:
        Vecteur tourné, shape (N, 3)
    """
    return quat_rotate(q, v)


@torch.jit.script
def quat_rotate_inverse(q, v):
    """Applique la rotation inverse par quaternion à un vecteur.

    Args:
        q: Quaternion [w, x, y, z], shape (N, 4)
        v: Vecteur 3D, shape (N, 3)

    Returns:
        Vecteur tourné, shape (N, 3)
    """
    shape = q.shape
    # Convention Isaac Lab: [w, x, y, z]
    q_w = q[:, 0]
    q_vec = q[:, 1:]  # [x, y, z]

    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0

    return a - b + c


@torch.jit.script
def quat_from_angle_axis(angle, axis):
    """Crée un quaternion depuis une représentation angle-axe.

    Args:
        angle: Angle de rotation (radians), shape (N,)
        axis: Axe de rotation (normalisé), shape (N, 3)

    Returns:
        Quaternion [w, x, y, z], shape (N, 4)
    """
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    # Convention Isaac Lab: [w, x, y, z]
    return quat_unit(torch.cat([w, xyz], dim=-1))


@torch.jit.script
def quat_to_angle_axis(q):
    """Convertit un quaternion en représentation angle-axe.

    Args:
        q: Quaternion [w, x, y, z], shape (..., 4)

    Returns:
        Tuple (angle, axis) où:
        - angle: Angle de rotation (radians), shape (...)
        - axis: Axe de rotation (normalisé), shape (..., 3)
    """
    min_theta = 1e-5
    # Convention Isaac Lab: [w, x, y, z]
    qw, qx, qy, qz = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw+3] / sin_theta_expand  # [x, y, z]

    mask = sin_theta > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)

    return angle, axis


@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw):
    """Crée un quaternion depuis les angles d'Euler (XYZ).

    Args:
        roll: Rotation autour de l'axe X (radians), shape (N,)
        pitch: Rotation autour de l'axe Y (radians), shape (N,)
        yaw: Rotation autour de l'axe Z (radians), shape (N,)

    Returns:
        Quaternion [w, x, y, z], shape (N, 4)
    """
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    # Convention Isaac Lab: [w, x, y, z]
    return torch.stack([qw, qx, qy, qz], dim=-1)


@torch.jit.script
def get_euler_xyz(q):
    """Extrait les angles d'Euler (XYZ) depuis un quaternion.

    Args:
        q: Quaternion [w, x, y, z], shape (N, 4)

    Returns:
        Tuple (roll, pitch, yaw) en radians, chaque tensor de shape (N,)
    """
    # Convention Isaac Lab: [w, x, y, z]
    qw, qx, qy, qz = 0, 1, 2, 3

    # Roulis (rotation axe x)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Tangage (rotation axe y)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    # Utiliser torch.sign au lieu de copysign pour compatibilité JIT
    pitch = torch.where(torch.abs(sinp) >= 1, (np.pi / 2.0) * torch.sign(sinp), torch.asin(sinp))

    # Lacet (rotation axe z)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)


@torch.jit.script
def quat_diff_rad(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Calcule la différence en radians entre deux quaternions.

    Args:
        a: Premier quaternion [w, x, y, z], shape (N, 4)
        b: Second quaternion [w, x, y, z], shape (N, 4)

    Returns:
        Différence en radians, shape (N,)
    """
    b_conj = quat_conjugate(b)
    mul = quat_mul(a, b_conj)
    # Convention Isaac Lab: partie vectorielle est [1:4]
    return 2.0 * torch.asin(
        torch.clamp(torch.norm(mul[:, 1:4], p=2, dim=-1), max=1.0)
    )


@torch.jit.script
def quat_to_tan_norm(q):
    """Représente une rotation en utilisant les vecteurs tangent et normal.

    Args:
        q: Quaternion [w, x, y, z], shape (..., 4)

    Returns:
        Vecteur tangent et normal concaténés, shape (..., 6)
    """
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = quat_rotate(q, ref_tan)

    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = quat_rotate(q, ref_norm)

    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan


@torch.jit.script
def angle_axis_to_exp_map(angle, axis):
    """Calcule la carte exponentielle depuis angle-axe.

    Args:
        angle: Angle de rotation (radians), shape (...)
        axis: Axe de rotation (normalisé), shape (..., 3)

    Returns:
        Carte exponentielle, shape (..., 3)
    """
    angle_expand = angle.unsqueeze(-1)
    exp_map = angle_expand * axis
    return exp_map


@torch.jit.script
def quat_to_exp_map(q):
    """Calcule la carte exponentielle depuis un quaternion.

    Args:
        q: Quaternion normalisé [w, x, y, z], shape (..., 4)

    Returns:
        Carte exponentielle, shape (..., 3)
    """
    angle, axis = quat_to_angle_axis(q)
    exp_map = angle_axis_to_exp_map(angle, axis)
    return exp_map


@torch.jit.script
def exp_map_to_angle_axis(exp_map):
    """Extrait angle-axe depuis une carte exponentielle.

    Args:
        exp_map: Carte exponentielle, shape (..., 3)

    Returns:
        Tuple (angle, axis)
    """
    min_theta = 1e-5

    angle = torch.norm(exp_map, dim=-1)
    angle_exp = torch.unsqueeze(angle, dim=-1)
    axis = exp_map / angle_exp
    angle = normalize_angle(angle)

    default_axis = torch.zeros_like(exp_map)
    default_axis[..., -1] = 1

    mask = angle > min_theta
    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)

    return angle, axis


@torch.jit.script
def exp_map_to_quat(exp_map):
    """Convertit une carte exponentielle en quaternion.

    Args:
        exp_map: Carte exponentielle, shape (..., 3)

    Returns:
        Quaternion [w, x, y, z], shape (..., 4)
    """
    angle, axis = exp_map_to_angle_axis(exp_map)
    q = quat_from_angle_axis(angle, axis)
    return q


@torch.jit.script
def slerp(q0, q1, t):
    """Interpolation sphérique linéaire entre deux quaternions.

    Args:
        q0: Premier quaternion [w, x, y, z], shape (..., 4)
        q1: Second quaternion [w, x, y, z], shape (..., 4)
        t: Paramètre d'interpolation [0, 1], shape (..., 1)

    Returns:
        Quaternion interpolé [w, x, y, z], shape (..., 4)
    """
    # Convention Isaac Lab: [w, x, y, z]
    qw, qx, qy, qz = 0, 1, 2, 3

    cos_half_theta = q0[..., qw] * q1[..., qw] \
                   + q0[..., qx] * q1[..., qx] \
                   + q0[..., qy] * q1[..., qy] \
                   + q0[..., qz] * q1[..., qz]

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratioB = torch.sin(t * half_theta) / sin_half_theta

    new_q_w = ratioA * q0[..., qw:qw+1] + ratioB * q1[..., qw:qw+1]
    new_q_x = ratioA * q0[..., qx:qx+1] + ratioB * q1[..., qx:qx+1]
    new_q_y = ratioA * q0[..., qy:qy+1] + ratioB * q1[..., qy:qy+1]
    new_q_z = ratioA * q0[..., qz:qz+1] + ratioB * q1[..., qz:qz+1]

    cat_dim = len(new_q_w.shape) - 1
    new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=cat_dim)

    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q


@torch.jit.script
def quat_axis(q, axis: int = 0):
    """Obtient un vecteur de base après rotation par quaternion.

    Args:
        q: Quaternion [w, x, y, z], shape (N, 4)
        axis: Index de l'axe (0=x, 1=y, 2=z)

    Returns:
        Vecteur de base tourné, shape (N, 3)
    """
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


# ==================== Fonctions de Transformation ====================


@torch.jit.script
def tf_inverse(q, t):
    """Inverse une transformation (rotation + translation).

    Args:
        q: Quaternion de rotation [w, x, y, z], shape (N, 4)
        t: Translation, shape (N, 3)

    Returns:
        Tuple (q_inv, t_inv) de la transformation inverse
    """
    q_inv = quat_conjugate(q)
    return q_inv, -quat_apply(q_inv, t)


@torch.jit.script
def tf_apply(q, t, v):
    """Applique une transformation à un vecteur.

    Args:
        q: Quaternion de rotation [w, x, y, z], shape (N, 4)
        t: Translation, shape (N, 3)
        v: Vecteur à transformer, shape (N, 3)

    Returns:
        Vecteur transformé, shape (N, 3)
    """
    return quat_apply(q, v) + t


@torch.jit.script
def tf_vector(q, v):
    """Applique uniquement la rotation (pas de translation).

    Args:
        q: Quaternion de rotation [w, x, y, z], shape (N, 4)
        v: Vecteur à tourner, shape (N, 3)

    Returns:
        Vecteur tourné, shape (N, 3)
    """
    return quat_apply(q, v)


@torch.jit.script
def tf_combine(q1, t1, q2, t2):
    """Combine deux transformations.

    Args:
        q1, t1: Première transformation
        q2, t2: Seconde transformation

    Returns:
        Tuple (q_combined, t_combined)
    """
    return quat_mul(q1, q2), quat_apply(q1, t2) + t1


@torch.jit.script
def local_to_world_space(pos_offset_local: torch.Tensor, pose_global: torch.Tensor):
    """Convertit un point du repère local au repère global.

    Args:
        pos_offset_local: Point dans le repère local, shape (N, 3)
        pose_global: La pose spatiale [pos, quat], shape (N, 7)
                    où quat est en format [w, x, y, z]

    Returns:
        Position dans le repère global, shape (N, 3)
    """
    quat_pos_local = torch.cat(
        [torch.zeros(pos_offset_local.shape[0], 1, dtype=torch.float32, device=pos_offset_local.device),
         pos_offset_local],
        dim=-1
    )
    # Convention Isaac Lab: quaternion dans pose_global[3:7] est [w, x, y, z]
    quat_global = pose_global[:, 3:7]
    quat_global_conj = quat_conjugate(quat_global)
    pos_offset_global = quat_mul(quat_global, quat_mul(quat_pos_local, quat_global_conj))[:, 1:4]

    result_pos_global = pos_offset_global + pose_global[:, 0:3]

    return result_pos_global


# ==================== Fonctions de Cap (Heading) ====================


@torch.jit.script
def calc_heading(q):
    """Calcule la direction du cap depuis un quaternion.

    Args:
        q: Quaternion normalisé [w, x, y, z], shape (..., 4)

    Returns:
        Cap (angle sur plan xy), shape (...)
    """
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = quat_rotate(q, ref_dir)

    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading


@torch.jit.script
def calc_heading_quat(q):
    """Calcule la rotation du cap depuis un quaternion.

    Args:
        q: Quaternion normalisé [w, x, y, z], shape (..., 4)

    Returns:
        Quaternion de cap [w, x, y, z], shape (..., 4)
    """
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(heading, axis)
    return heading_q


@torch.jit.script
def calc_heading_quat_inv(q):
    """Calcule l'inverse de la rotation du cap depuis un quaternion.

    Args:
        q: Quaternion normalisé [w, x, y, z], shape (..., 4)

    Returns:
        Quaternion de cap inverse [w, x, y, z], shape (..., 4)
    """
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(-heading, axis)
    return heading_q


@torch.jit.script
def compute_heading_and_up(torso_rotation, inv_start_rot, to_target, vec0, vec1, up_idx):
    """Calcule le cap et la direction vers le haut.

    Returns:
        Tuple (torso_quat, up_proj, heading_proj, up_vec, heading_vec)
    """
    num_envs = torso_rotation.shape[0]
    target_dirs = normalize(to_target)

    torso_quat = quat_mul(torso_rotation, inv_start_rot)
    up_vec = get_basis_vector(torso_quat, vec1).view(num_envs, 3)
    heading_vec = get_basis_vector(torso_quat, vec0).view(num_envs, 3)
    up_proj = up_vec[:, up_idx]
    heading_proj = torch.bmm(heading_vec.view(num_envs, 1, 3), target_dirs.view(num_envs, 3, 1)).view(num_envs)

    return torso_quat, up_proj, heading_proj, up_vec, heading_vec


@torch.jit.script
def compute_rot(torso_quat, velocity, ang_velocity, targets, torso_positions):
    """Calcule les informations de rotation pour les observations."""
    vel_loc = quat_rotate_inverse(torso_quat, velocity)
    angvel_loc = quat_rotate_inverse(torso_quat, ang_velocity)

    roll, pitch, yaw = get_euler_xyz(torso_quat)

    walk_target_angle = torch.atan2(targets[:, 2] - torso_positions[:, 2],
                                    targets[:, 0] - torso_positions[:, 0])
    angle_to_target = walk_target_angle - yaw

    return vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target


@torch.jit.script
def get_basis_vector(q, v):
    """Obtient un vecteur de base après rotation."""
    return quat_rotate(q, v)


# ==================== Fonctions Utilitaires ====================


@torch.jit.script
def euler_xyz_to_exp_map(roll, pitch, yaw):
    """Convertit angles d'Euler en carte exponentielle."""
    q = quat_from_euler_xyz(roll, pitch, yaw)
    exp_map = quat_to_exp_map(q)
    return exp_map


# ==================== Conversion Matrice/Quaternion ====================


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convertit les quaternions en matrices de rotation.

    Args:
        quaternions: Quaternions [w, x, y, z], shape (..., 4)

    Returns:
        Matrices de rotation, shape (..., 3, 3)
    """
    # Convention Isaac Lab: [w, x, y, z]
    w, x, y, z = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    mat = torch.stack(
        (
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ),
        -1,
    )
    return mat.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Retourne torch.sqrt(torch.max(0, x))."""
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convertit les matrices de rotation en quaternions.

    Args:
        matrix: Matrices de rotation, shape (..., 3, 3)

    Returns:
        Quaternions [w, x, y, z], shape (..., 4)
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # Convention Isaac Lab: retourne [w, x, y, z]
    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


# ==================== Normalisation et Scaling ====================


def torch_rand_float(lower, upper, shape, device):
    """Génère des flottants aléatoires uniformes.

    Args:
        lower: Borne inférieure
        upper: Borne supérieure
        shape: Forme du tenseur (tuple d'entiers)
        device: Device ('cpu' ou 'cuda')

    Returns:
        Tenseur aléatoire de forme shape

    Note: Non-JIT car PyTorch JIT ne supporte pas Tuple[int, ...] avec décompression.
    """
    return (upper - lower) * torch.rand(*shape, device=device) + lower


def torch_random_dir_2(shape, device):
    """Génère des directions 2D aléatoires.

    Args:
        shape: Forme du tenseur (tuple d'entiers)
        device: Device ('cpu' ou 'cuda')

    Returns:
        Tenseur de directions 2D normalisées

    Note: Non-JIT car appelle torch_rand_float (non-JIT).
    """
    angle = torch_rand_float(-np.pi, np.pi, shape, device).squeeze(-1)
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    """Clamp un tenseur."""
    return torch.max(torch.min(t, max_t), min_t)


@torch.jit.script
def scale(x, lower, upper):
    """Scale depuis [-1, 1] vers [lower, upper]."""
    return (0.5 * (x + 1.0) * (upper - lower) + lower)


@torch.jit.script
def unscale(x, lower, upper):
    """Unscale depuis [lower, upper] vers [-1, 1]."""
    return (2.0 * x - upper - lower) / (upper - lower)


def unscale_np(x, lower, upper):
    """Version numpy de unscale."""
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def scale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Normalise un tenseur dans [-1, 1]."""
    offset = (lower + upper) * 0.5
    return 2 * (x - offset) / (upper - lower)


@torch.jit.script
def unscale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Dénormalise un tenseur de [-1, 1] vers [lower, upper]."""
    offset = (lower + upper) * 0.5
    return x * (upper - lower) * 0.5 + offset


@torch.jit.script
def saturate(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Limite un tenseur entre [lower, upper]."""
    return torch.max(torch.min(x, upper), lower)


# ==================== Fonctions de Pose ====================


def normalise_quat_in_pose(pose):
    """Normalise la partie quaternion d'une pose.

    Args:
        pose: Pose [pos(3), quat(4)], shape (N, 7)
            où quat est en format [w, x, y, z]

    Returns:
        Pose avec quaternion normalisé, shape (N, 7)
    """
    pos = pose[:, 0:3]
    quat = pose[:, 3:7]  # [w, x, y, z]
    quat /= torch.norm(quat, dim=-1, p=2).reshape(-1, 1)
    return torch.cat([pos, quat], dim=-1)


def get_axis_params(value, axis_idx, x_value=0., dtype=float, n_dims=3):
    """Construit les arguments pour Vec en fonction de l'index de l'axe."""
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.
    params = np.where(zs == 1., value, zs)
    params[0] = x_value
    return list(params.astype(dtype))


# Alias pour compatibilité
my_quat_rotate = quat_rotate

# EOF
