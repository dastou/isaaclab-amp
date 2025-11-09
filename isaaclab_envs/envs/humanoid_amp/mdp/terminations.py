# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fonctions de termination pour HumanoidAMP.

Les terminations déterminent quand un épisode doit se terminer:
- Termination normale: fin de l'épisode (max_episode_length)
- Early termination: chute détectée (hauteur, contact non-pied)
"""

import torch
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from omni.isaac.lab.envs import DirectRLEnv


@torch.jit.script
def compute_humanoid_reset(
    reset_buf: torch.Tensor,
    progress_buf: torch.Tensor,
    contact_buf: torch.Tensor,
    contact_body_ids: torch.Tensor,
    rigid_body_pos: torch.Tensor,
    max_episode_length: float,
    enable_early_termination: bool,
    termination_height: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calcule les conditions de reset pour le humanoid.

    Args:
        reset_buf: Buffer de reset, shape (N,)
        progress_buf: Buffer de progrès (nb de steps), shape (N,)
        contact_buf: Forces de contact, shape (N, num_bodies, 3)
        contact_body_ids: IDs des corps de contact autorisés (pieds), shape (K,)
        rigid_body_pos: Positions des corps rigides, shape (N, num_bodies, 3)
        max_episode_length: Longueur maximale d'un épisode
        enable_early_termination: Si True, terminer sur chute
        termination_height: Hauteur minimale avant termination

    Returns:
        Tuple de (reset, terminated):
        - reset: Environments à reset (timeout OU chute), shape (N,)
        - terminated: Environments terminés par chute, shape (N,)
    """
    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        # Masquer les contacts autorisés (pieds)
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0

        # Détecter contact sur corps non-pied
        fall_contact = torch.any(masked_contact_buf > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        # Détecter hauteur trop basse
        body_height = rigid_body_pos[..., 2]  # Coordonnée z
        fall_height = body_height < termination_height

        # Ne pas considérer les pieds pour la hauteur
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        # Chute = contact non-pied ET hauteur basse
        has_fallen = torch.logical_and(fall_contact, fall_height)

        # La première étape peut avoir des forces de contact résiduelles
        # Ne vérifier qu'après quelques steps
        has_fallen *= (progress_buf > 1)

        # Marquer comme terminé
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    # Reset sur timeout OU termination
    reset = torch.where(
        progress_buf >= max_episode_length - 1,
        torch.ones_like(reset_buf),
        terminated
    )

    return reset, terminated


@torch.jit.script
def check_termination_height(
    body_pos: torch.Tensor,
    min_height: float,
    exclude_body_ids: torch.Tensor
) -> torch.Tensor:
    """Vérifie si des corps sont en dessous de la hauteur minimale.

    Args:
        body_pos: Positions des corps, shape (N, num_bodies, 3)
        min_height: Hauteur minimale autorisée
        exclude_body_ids: IDs des corps à exclure (pieds)

    Returns:
        Masque de termination, shape (N,)
    """
    # Hauteurs (z)
    body_height = body_pos[..., 2]

    # Détecter hauteur trop basse
    too_low = body_height < min_height

    # Exclure certains corps (pieds)
    too_low[:, exclude_body_ids] = False

    # Au moins un corps trop bas
    terminated = torch.any(too_low, dim=-1)

    return terminated


@torch.jit.script
def check_illegal_contact(
    contact_forces: torch.Tensor,
    allowed_body_ids: torch.Tensor,
    threshold: float = 0.1
) -> torch.Tensor:
    """Vérifie s'il y a des contacts non autorisés.

    Args:
        contact_forces: Forces de contact, shape (N, num_bodies, 3)
        allowed_body_ids: IDs des corps autorisés à toucher le sol (pieds)
        threshold: Seuil de force pour détecter contact

    Returns:
        Masque de termination, shape (N,)
    """
    # Masquer les contacts autorisés
    masked_contacts = contact_forces.clone()
    masked_contacts[:, allowed_body_ids, :] = 0

    # Détecter contact non autorisé
    has_contact = torch.any(masked_contacts > threshold, dim=-1)
    has_illegal_contact = torch.any(has_contact, dim=-1)

    return has_illegal_contact


@torch.jit.script
def check_timeout(
    progress_buf: torch.Tensor,
    max_length: int
) -> torch.Tensor:
    """Vérifie si l'épisode a atteint la durée maximale.

    Args:
        progress_buf: Nombre de steps écoulés, shape (N,)
        max_length: Durée maximale de l'épisode

    Returns:
        Masque de timeout, shape (N,)
    """
    timeout = progress_buf >= max_length - 1
    return timeout


# Fonction principale pour DirectRLEnv
def compute_terminations(env: "DirectRLEnv") -> Tuple[torch.Tensor, torch.Tensor]:
    """Calcule les terminations pour l'environnement DirectRLEnv.

    Isaac Lab distingue deux types de done:
    - terminated: Termination due à une condition (chute)
    - truncated: Termination due au timeout (max_episode_length)

    Args:
        env: Instance de l'environnement

    Returns:
        Tuple de (terminated, truncated), chaque tensor de shape (num_envs,)
    """
    robot = env.scene.articulations["robot"]

    # Accéder aux données
    body_pos = robot.data.body_pos_w  # (N, num_bodies, 3)
    contact_forces = env.scene.contact_forces["robot"]  # (N, num_bodies, 3)

    # Timeout
    truncated = check_timeout(env.episode_length_buf, env.max_episode_length)

    # Early termination activée
    if env.cfg.enable_early_termination:
        # Termination par hauteur
        height_term = check_termination_height(
            body_pos,
            env.cfg.termination_height,
            env._contact_body_ids
        )

        # Termination par contact illégal
        contact_term = check_illegal_contact(
            contact_forces,
            env._contact_body_ids,
            threshold=0.1
        )

        # Combiner (chute = hauteur ET contact)
        has_fallen = torch.logical_and(height_term, contact_term)

        # Ignorer première step (forces résiduelles)
        has_fallen = torch.logical_and(has_fallen, env.episode_length_buf > 1)

        terminated = has_fallen
    else:
        # Pas de early termination
        terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    return terminated, truncated


# Fonction alternative utilisant la fonction JIT originale
def compute_terminations_jit(env: "DirectRLEnv") -> Tuple[torch.Tensor, torch.Tensor]:
    """Version utilisant la fonction JIT compute_humanoid_reset.

    Args:
        env: Instance de l'environnement

    Returns:
        Tuple de (terminated, truncated)
    """
    robot = env.scene.articulations["robot"]

    # Accéder aux données
    body_pos = robot.data.body_pos_w
    contact_forces = env.scene.contact_forces["robot"]

    # Appeler fonction JIT
    reset_buf = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    reset, terminated = compute_humanoid_reset(
        reset_buf,
        env.episode_length_buf,
        contact_forces,
        env._contact_body_ids,
        body_pos,
        float(env.max_episode_length),
        env.cfg.enable_early_termination,
        env.cfg.termination_height
    )

    # Convertir en bool
    terminated = terminated.bool()

    # Timeout (truncated)
    truncated = torch.logical_and(
        reset.bool(),
        torch.logical_not(terminated)
    )

    return terminated, truncated
