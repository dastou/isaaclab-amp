# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration pour l'asset humanoid AMP."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

##
# Configuration
##

HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"assets/mjcf/amp_humanoid.xml",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness={
                ".*_waist.*": 20.0,
                ".*_upper_arm.*": 10.0,
                ".*_lower_arm.*": 2.0,
                "pelvis": 10.0,
                ".*_thigh:0": 10.0,
                ".*_thigh:1": 20.0,
                ".*_thigh:2": 10.0,
                ".*_shin": 5.0,
                ".*_foot.*": 2.0,
            },
            damping={
                ".*_waist.*": 5.0,
                ".*_upper_arm.*": 5.0,
                ".*_lower_arm.*": 1.0,
                "pelvis": 5.0,
                ".*_thigh:0": 5.0,
                ".*_thigh:1": 5.0,
                ".*_thigh:2": 5.0,
                ".*_shin": 0.1,
                ".*_foot.*": 1.0,
            },
        ),
    },
)
"""Configuration pour l'humanoid AMP basée sur le fichier MJCF.

Cette configuration définit un asset humanoid articulé avec:
- Capteurs de contact activés
- Propriétés de corps rigides (pas de gravité désactivée)
- Propriétés d'articulation (4 itérations de solveur de position)
- Actuateurs implicites avec raideur et amortissement par articulation
"""
