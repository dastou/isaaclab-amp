# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Algorithmes d'apprentissage pour Isaac Lab.

Ce module contient les algorithmes RL utilisés pour entraîner les agents:
- AMP (Adversarial Motion Priors): Apprentissage depuis mouvements de référence
- Réseaux de neurones (Actor-Critic + Discriminateur)
- Agents RL (CommonAgent, AMPAgent)
- Replay buffers pour stockage d'expériences
"""

from .amp_network import ModelAMPContinuous, AMPBuilder
from .amp_datasets import AMPDataset
from .amp_agent import AMPAgent
from .common_agent import CommonAgent
from .replay_buffer import ReplayBuffer

__all__ = [
    "ModelAMPContinuous",
    "AMPBuilder",
    "AMPDataset",
    "AMPAgent",
    "CommonAgent",
    "ReplayBuffer",
]
