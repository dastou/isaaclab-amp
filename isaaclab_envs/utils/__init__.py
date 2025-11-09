# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilitaires pour Isaac Lab."""

# Import conditionnel pour éviter erreur quand Isaac Lab n'est pas installé
try:
    # Motion Library
    from .motion_lib import MotionLib, convert_quat_xyzw_to_wxyz, convert_quat_wxyz_to_xyzw

    # RL-Games Wrapper
    from .rl_games_wrapper import RLGamesVecEnv, create_rlgames_env

    _ISAAC_LAB_AVAILABLE = True
except ImportError:
    # Isaac Lab non installé, imports Isaac Lab non disponibles
    _ISAAC_LAB_AVAILABLE = False
    MotionLib = None
    RLGamesVecEnv = None
    create_rlgames_env = None
    convert_quat_xyzw_to_wxyz = None
    convert_quat_wxyz_to_xyzw = None

# Utilitaires mathématiques (quaternions au format Isaac Lab [w, x, y, z])
from .math import (
    # Quaternions
    quat_mul,
    quat_conjugate,
    quat_unit,
    quat_apply,
    quat_rotate,
    quat_rotate_inverse,
    quat_from_angle_axis,
    quat_to_angle_axis,
    quat_from_euler_xyz,
    get_euler_xyz,
    quat_diff_rad,
    quat_to_tan_norm,
    quat_to_exp_map,
    quat_axis,
    # Transformations
    tf_inverse,
    tf_apply,
    tf_vector,
    tf_combine,
    local_to_world_space,
    # Cap (heading)
    calc_heading,
    calc_heading_quat,
    calc_heading_quat_inv,
    compute_heading_and_up,
    compute_rot,
    # Utilitaires
    normalize,
    normalize_angle,
    to_torch,
    # Conversions
    quaternion_to_matrix,
    matrix_to_quaternion,
    # Scaling
    scale,
    unscale,
    scale_transform,
    unscale_transform,
    saturate,
    # Pose
    normalise_quat_in_pose,
)

__all__ = [
    "MotionLib",
    "convert_quat_xyzw_to_wxyz",
    "convert_quat_wxyz_to_xyzw",
    "RLGamesVecEnv",
    "create_rlgames_env",
    "quat_mul",
    "quat_conjugate",
    "quat_unit",
    "quat_apply",
    "quat_rotate",
    "quat_rotate_inverse",
    "quat_from_angle_axis",
    "quat_to_angle_axis",
    "quat_from_euler_xyz",
    "get_euler_xyz",
    "quat_diff_rad",
    "quat_to_tan_norm",
    "quat_to_exp_map",
    "quat_axis",
    "tf_inverse",
    "tf_apply",
    "tf_vector",
    "tf_combine",
    "local_to_world_space",
    "calc_heading",
    "calc_heading_quat",
    "calc_heading_quat_inv",
    "compute_heading_and_up",
    "compute_rot",
    "normalize",
    "normalize_angle",
    "to_torch",
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "scale",
    "unscale",
    "scale_transform",
    "unscale_transform",
    "saturate",
    "normalise_quat_in_pose",
]
