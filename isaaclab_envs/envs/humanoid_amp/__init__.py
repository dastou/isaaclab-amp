# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environnement HumanoidAMP pour Isaac Lab.

Hiérarchie des classes:
- HumanoidAMPBaseEnv: Classe de base avec observations, terminations, reset
- HumanoidEnv: Humanoid standard (hérite de Base, pas d'AMP)
- HumanoidAMPEnv: Humanoid avec AMP (hérite de Base, ajoute discriminateur)
"""

from .humanoid_amp_base_env import HumanoidAMPBaseEnv
from .humanoid_amp_env import HumanoidAMPEnv

__all__ = [
    "HumanoidAMPBaseEnv",
    "HumanoidAMPEnv",
]
