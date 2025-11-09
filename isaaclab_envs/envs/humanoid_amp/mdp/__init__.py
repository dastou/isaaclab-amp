# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Définitions MDP (Markov Decision Process) pour HumanoidAMP.

Ce module contient toutes les fonctions nécessaires pour définir le processus
de décision de Markov de l'environnement HumanoidAMP:
- Observations: Calcul de l'état observable (105 dimensions)
- Récompenses: Calcul des récompenses (pour AMP: discriminateur uniquement)
- Terminations: Conditions de fin d'épisode (chute, timeout)
"""

# Observations
from .observations import (
    dof_to_obs,
    compute_humanoid_observations,
    build_amp_observations,
    compute_observations,
)

# Récompenses
from .rewards import (
    compute_humanoid_reward,
    reward_alive,
    reward_forward_velocity,
    reward_upright,
    reward_energy,
    reward_action_smoothness,
    compute_rewards,
    compute_weighted_rewards,
)

# Terminations
from .terminations import (
    compute_humanoid_reset,
    check_termination_height,
    check_illegal_contact,
    check_timeout,
    compute_terminations,
    compute_terminations_jit,
)

__all__ = [
    # Observations
    "dof_to_obs",
    "compute_humanoid_observations",
    "build_amp_observations",
    "compute_observations",
    # Récompenses
    "compute_humanoid_reward",
    "reward_alive",
    "reward_forward_velocity",
    "reward_upright",
    "reward_energy",
    "reward_action_smoothness",
    "compute_rewards",
    "compute_weighted_rewards",
    # Terminations
    "compute_humanoid_reset",
    "check_termination_height",
    "check_illegal_contact",
    "check_timeout",
    "compute_terminations",
    "compute_terminations_jit",
]
