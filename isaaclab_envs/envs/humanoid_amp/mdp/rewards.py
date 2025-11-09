# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fonctions de calcul des récompenses pour HumanoidAMP.

Pour AMP pur (task_reward_w = 0.0), les récompenses viennent uniquement
du discriminateur. Ce fichier fournit des récompenses de base pour
l'environnement Humanoid standard.
"""

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import DirectRLEnv


@torch.jit.script
def compute_humanoid_reward(obs_buf: torch.Tensor) -> torch.Tensor:
    """Calcule la récompense de base pour le humanoid.

    Pour AMP pur, cette récompense n'est pas utilisée (task_reward_w = 0.0).
    Le discriminateur AMP fournit la récompense basée sur la similarité
    avec les mouvements de référence.

    Args:
        obs_buf: Buffer d'observations, shape (N, num_obs)

    Returns:
        Récompenses, shape (N,)
    """
    # Récompense constante de 1.0 (non utilisée pour AMP pur)
    reward = torch.ones_like(obs_buf[:, 0])
    return reward


# Récompenses optionnelles pour Humanoid standard (non-AMP)

@torch.jit.script
def reward_alive(env_ids: torch.Tensor, num_envs: int, device: str) -> torch.Tensor:
    """Récompense pour rester en vie.

    Args:
        env_ids: IDs des environnements actifs
        num_envs: Nombre total d'environnements
        device: Device PyTorch

    Returns:
        Récompenses, shape (num_envs,)
    """
    reward = torch.zeros(num_envs, device=device)
    reward[env_ids] = 1.0
    return reward


@torch.jit.script
def reward_forward_velocity(
    root_vel: torch.Tensor,
    target_vel: float = 1.0,
    sigma: float = 0.25
) -> torch.Tensor:
    """Récompense pour vitesse vers l'avant.

    Args:
        root_vel: Vélocités racine, shape (N, 3)
        target_vel: Vitesse cible
        sigma: Écart-type pour la gaussienne

    Returns:
        Récompenses, shape (N,)
    """
    # Vitesse en x (avant)
    vel_x = root_vel[:, 0]

    # Récompense gaussienne centrée sur target_vel
    reward = torch.exp(-((vel_x - target_vel) ** 2) / (2 * sigma ** 2))

    return reward


@torch.jit.script
def reward_upright(root_quat: torch.Tensor) -> torch.Tensor:
    """Récompense pour rester droit (orientation verticale).

    Args:
        root_quat: Quaternions racine [w, x, y, z], shape (N, 4)

    Returns:
        Récompenses, shape (N,)
    """
    # Composante z du vecteur "up" après rotation
    # Pour quaternion [w, x, y, z], le vecteur z devient:
    # [2(xz + wy), 2(yz - wx), 1 - 2(x^2 + y^2)]
    w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
    up_z = 1.0 - 2.0 * (x * x + y * y)

    # Récompense proportionnelle à l'alignement vertical
    reward = torch.clamp(up_z, min=0.0)

    return reward


@torch.jit.script
def reward_energy(joint_torques: torch.Tensor, scale: float = 0.01) -> torch.Tensor:
    """Pénalité pour la consommation d'énergie.

    Args:
        joint_torques: Torques des articulations, shape (N, num_dofs)
        scale: Facteur d'échelle pour la pénalité

    Returns:
        Pénalités (négatives), shape (N,)
    """
    # Somme des carrés des torques
    energy = torch.sum(joint_torques ** 2, dim=-1)

    # Pénalité (négative)
    penalty = -scale * energy

    return penalty


@torch.jit.script
def reward_action_smoothness(
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    scale: float = 0.1
) -> torch.Tensor:
    """Récompense pour la fluidité des actions.

    Args:
        actions: Actions actuelles, shape (N, num_actions)
        prev_actions: Actions précédentes, shape (N, num_actions)
        scale: Facteur d'échelle

    Returns:
        Récompenses, shape (N,)
    """
    # Différence entre actions consécutives
    action_diff = torch.sum((actions - prev_actions) ** 2, dim=-1)

    # Pénalité pour variations brusques
    penalty = -scale * action_diff

    return penalty


# Fonction principale pour DirectRLEnv
def compute_rewards(env: "DirectRLEnv") -> torch.Tensor:
    """Calcule les récompenses pour l'environnement DirectRLEnv.

    Pour AMP pur:
    - task_reward_w = 0.0 (pas de récompense de tâche)
    - disc_reward_w = 1.0 (récompense du discriminateur uniquement)

    Le discriminateur AMP compare les observations avec les mouvements
    de référence et fournit une récompense basée sur la similarité.

    Args:
        env: Instance de l'environnement

    Returns:
        Récompenses, shape (num_envs,)
    """
    # Pour AMP pur, retourner zéro (le discriminateur ajoutera sa récompense)
    if hasattr(env.cfg, 'task_reward_w') and env.cfg.task_reward_w == 0.0:
        return torch.zeros(env.num_envs, device=env.device)

    # Pour Humanoid standard, utiliser la récompense de base
    reward = compute_humanoid_reward(env.obs_buf)

    return reward


# Fonction combinée pour récompenses pondérées
def compute_weighted_rewards(
    env: "DirectRLEnv",
    weights: dict[str, float]
) -> torch.Tensor:
    """Calcule une récompense pondérée combinant plusieurs termes.

    Args:
        env: Instance de l'environnement
        weights: Dictionnaire {nom_recompense: poids}

    Returns:
        Récompenses totales, shape (num_envs,)
    """
    total_reward = torch.zeros(env.num_envs, device=env.device)

    robot = env.scene.articulations["robot"]
    root_states = robot.data.root_state_w

    # Récompense pour rester en vie
    if 'alive' in weights:
        active_envs = torch.arange(env.num_envs, device=env.device)
        total_reward += weights['alive'] * reward_alive(
            active_envs, env.num_envs, env.device
        )

    # Récompense pour vitesse vers l'avant
    if 'forward_vel' in weights:
        total_reward += weights['forward_vel'] * reward_forward_velocity(
            root_states[:, 7:10]  # vélocité
        )

    # Récompense pour rester droit
    if 'upright' in weights:
        total_reward += weights['upright'] * reward_upright(
            root_states[:, 3:7]  # quaternion [w, x, y, z]
        )

    # Pénalité énergie
    if 'energy' in weights and hasattr(robot.data, 'joint_torques'):
        total_reward += weights['energy'] * reward_energy(
            robot.data.joint_torques,
            scale=abs(weights['energy'])
        )

    return total_reward
