# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration de la scène pour l'environnement HumanoidAMP."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

from isaaclab_envs.assets.humanoid_cfg import HUMANOID_CFG

##
# Scene configuration
##


@configclass
class HumanoidSceneCfg(InteractiveSceneCfg):
    """Configuration de la scène pour l'environnement Humanoid AMP."""

    # Sol plan
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )

    # Humanoid articulé
    robot: ArticulationCfg = HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Lumière
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
