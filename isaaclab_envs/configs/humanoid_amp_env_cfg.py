# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration de l'environnement pour HumanoidAMP."""

import math

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

from .scene_cfg import HumanoidSceneCfg

##
# Pre-defined configs
##
from isaaclab_assets.humanoid import HUMANOID_CFG  # isort: skip


##
# Configuration de l'environnement
##


@configclass
class HumanoidAMPEnvCfg(DirectRLEnvCfg):
    """Configuration de l'environnement Humanoid AMP."""

    # Paramètres de l'environnement de base
    episode_length_s = 5.0  # 300 steps @ 60Hz
    decimation = 2  # Control à 30Hz (60Hz / 2)
    num_actions = 21  # Nombre d'articulations du humanoid
    num_observations = 210  # Observations AMP (2 steps * 105)
    num_states = 0

    # Paramètres de simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,  # 60Hz
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Scène
    scene: HumanoidSceneCfg = HumanoidSceneCfg(num_envs=4096, env_spacing=5.0)

    # Paramètres AMP
    num_amp_obs_steps = 2  # Nombre de timesteps pour les observations AMP
    local_root_obs = False  # Si les observations sont dans le référentiel local
    contact_bodies = ["right_foot", "left_foot"]  # Corps pour détection de contact
    termination_height = 0.5  # Hauteur minimale avant termination
    enable_early_termination = True  # Active la termination anticipée

    # Fichier de mouvement pour AMP
    motion_file = "amp_humanoid_run.npy"
    # Autres options:
    # motion_file = "amp_humanoid_walk.npy"
    # motion_file = "amp_humanoid_dance.npy"
    # motion_file = "amp_humanoid_hop.npy"  # Nécessite HumanoidAMPPPOLowGP
    # motion_file = "amp_humanoid_backflip.npy"  # Nécessite HumanoidAMPPPOLowGP

    # Initialisation
    state_init = "Random"  # "Default" ou "Random"
    hybrid_init_prob = 0.5  # Probabilité d'initialisation hybride (random + motion)

    # Contrôle PD
    pd_control = True
    power_scale = 1.0

    # Paramètres de caméra
    camera_follow = True  # La caméra suit l'humanoïde
    enable_debug_vis = False  # Visualisation de debug

    # MDP settings
    # Les observations, récompenses et terminations seront définies dans des fichiers séparés
    # dans le dossier mdp/
    observations = None  # À définir dans mdp/observations.py
    actions = None  # À définir dans mdp/actions.py
    rewards = None  # À définir dans mdp/rewards.py
    terminations = None  # À définir dans mdp/terminations.py
    events = None  # À définir dans mdp/events.py (domain randomization)


@configclass
class HumanoidAMPEnvCfg_PLAY(HumanoidAMPEnvCfg):
    """Configuration pour le mode play/test (inférence)."""

    def __post_init__(self):
        # post init de la classe parent
        super().__post_init__()
        # Ajuster les paramètres pour le mode play
        self.scene.num_envs = 64
        self.episode_length_s = 10.0  # Episodes plus longs pour visualisation
        self.enable_debug_vis = True
        self.camera_follow = True
