# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fonctions de calcul des observations pour HumanoidAMP.

IMPORTANT: Convention de quaternions Isaac Lab
- Tous les quaternions sont au format [w, x, y, z]
- root_states[:, 3:7] contient la rotation racine en WXYZ
"""

import torch
from typing import TYPE_CHECKING

from isaaclab_envs.utils.math import (
    calc_heading_quat_inv,
    quat_mul,
    quat_to_tan_norm,
    quat_rotate,
    exp_map_to_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv


# Constantes pour les observations
# NOTE: Ces valeurs devront être mises à jour si DOF_BODY_IDS change
DOF_OFFSETS = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]


@torch.jit.script
def dof_to_obs(pose: torch.Tensor) -> torch.Tensor:
    """Convertit les positions DOF en observations.

    Pour les articulations sphériques (3 DOF), utilise une représentation
    tangent-normal (6D) depuis la carte exponentielle.
    Pour les articulations à charnière (1 DOF), utilise l'angle directement.

    Args:
        pose: Positions DOF, shape (N, num_dofs)

    Returns:
        Observations DOF, shape (N, 52)
        - 52 = somme des tailles d'observations pour chaque articulation
    """
    dof_obs_size = 52
    dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # Articulation sphérique (3 DOF) → représentation 6D
        if dof_size == 3:
            # Convertir carte exponentielle → quaternion [w, x, y, z]
            joint_pose_q = exp_map_to_quat(joint_pose)
            # Convertir quaternion → tangent-normal (6D)
            joint_dof_obs = quat_to_tan_norm(joint_pose_q)
            joint_dof_obs_size = 6
        else:
            # Articulation à charnière (1 DOF) → angle
            joint_dof_obs = joint_pose
            joint_dof_obs_size = 1

        dof_obs[:, dof_obs_offset:(dof_obs_offset + joint_dof_obs_size)] = joint_dof_obs
        dof_obs_offset += joint_dof_obs_size

    return dof_obs


@torch.jit.script
def compute_humanoid_observations(
    root_states: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    key_body_pos: torch.Tensor,
    local_root_obs: bool
) -> torch.Tensor:
    """Calcule les observations du humanoid.

    Observations (105 dimensions):
    - 1: hauteur racine
    - 12: rotation racine (représentation tangent-normal 6D × 2)
    - 3: vélocité racine (locale)
    - 3: vélocité angulaire racine (locale)
    - 52: positions DOF (représentation 6D pour sphériques, angle pour charnières)
    - 28: vélocités DOF
    - 12: positions des corps clés (4 corps × 3 coords, locales)

    Args:
        root_states: États racine [pos(3), quat_w(4), vel(3), ang_vel(3)], shape (N, 13)
                     IMPORTANT: quat au format [w, x, y, z]
        dof_pos: Positions DOF, shape (N, num_dofs)
        dof_vel: Vélocités DOF, shape (N, num_dofs)
        key_body_pos: Positions des corps clés, shape (N, num_key_bodies, 3)
        local_root_obs: Si True, observations racine dans le référentiel local

    Returns:
        Observations, shape (N, 105)
    """
    # Extraire les composantes de root_states
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]  # Format [w, x, y, z] ✅
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    # Hauteur racine (z)
    root_h = root_pos[:, 2:3]

    # Quaternion de direction (heading) inverse
    heading_rot = calc_heading_quat_inv(root_rot)  # [w, x, y, z] ✅

    # Rotation racine (locale ou globale)
    if local_root_obs:
        root_rot_obs = quat_mul(heading_rot, root_rot)  # [w, x, y, z] ✅
    else:
        root_rot_obs = root_rot

    # Convertir quaternion → représentation tangent-normal (12D)
    root_rot_obs = quat_to_tan_norm(root_rot_obs)  # (N, 12)

    # Vélocités dans le référentiel local
    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    # Positions des corps clés dans le référentiel local
    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    # Rotation des positions clés dans le référentiel local
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
        local_key_body_pos.shape[2]
    )
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2]
    )
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0],
        local_key_body_pos.shape[1] * local_key_body_pos.shape[2]
    )

    # Observations DOF (52D)
    dof_obs = dof_to_obs(dof_pos)

    # Concaténer toutes les observations
    obs = torch.cat((
        root_h,              # 1
        root_rot_obs,        # 12
        local_root_vel,      # 3
        local_root_ang_vel,  # 3
        dof_obs,             # 52
        dof_vel,             # 28
        flat_local_key_pos   # 12
    ), dim=-1)

    return obs


@torch.jit.script
def build_amp_observations(
    root_states: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    key_body_pos: torch.Tensor,
    local_root_obs: bool
) -> torch.Tensor:
    """Construit les observations AMP (alias pour compute_humanoid_observations).

    Cette fonction est utilisée par la motion library pour créer des observations
    de démonstration depuis les mouvements de référence.

    Args:
        root_states: États racine, shape (N, 13), quaternion [w, x, y, z]
        dof_pos: Positions DOF, shape (N, num_dofs)
        dof_vel: Vélocités DOF, shape (N, num_dofs)
        key_body_pos: Positions des corps clés, shape (N, num_key_bodies, 3)
        local_root_obs: Observations locales ou globales

    Returns:
        Observations AMP, shape (N, 105)
    """
    return compute_humanoid_observations(
        root_states, dof_pos, dof_vel, key_body_pos, local_root_obs
    )


# Fonction non-JIT pour usage avec DirectRLEnv
def compute_observations(env: "DirectRLEnv") -> torch.Tensor:
    """Calcule les observations pour l'environnement DirectRLEnv.

    Args:
        env: Instance de l'environnement

    Returns:
        Observations, shape (num_envs, 105)
    """
    # Accéder aux données du robot
    robot = env.scene.articulations["robot"]

    root_states = robot.data.root_state_w  # (N, 13), quat [w, x, y, z] ✅
    dof_pos = robot.data.joint_pos  # (N, num_dofs)
    dof_vel = robot.data.joint_vel  # (N, num_dofs)

    # Positions des corps clés
    key_body_pos = robot.data.body_pos_w[:, env._key_body_ids, :]

    # Calcul des observations
    obs = compute_humanoid_observations(
        root_states,
        dof_pos,
        dof_vel,
        key_body_pos,
        env.cfg.local_root_obs
    )

    return obs
