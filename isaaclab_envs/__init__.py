# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package contenant les environnements Isaac Lab pour AMP (Adversarial Motion Priors)."""

import os

# Configuration du répertoire des données
ISAACLAB_ENVS_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
ISAACLAB_ENVS_METADATA = {
    "version": "1.0.0",
    "description": "Environnements Isaac Lab pour AMP Humanoid",
}
