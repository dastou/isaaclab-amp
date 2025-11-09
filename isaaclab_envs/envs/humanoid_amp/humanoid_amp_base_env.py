# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environnement de base pour HumanoidAMP utilisant Isaac Lab.

Ce fichier remplace humanoid_amp_base.py d'Isaac Gym.
Changements majeurs:
- Hérite de DirectRLEnv au lieu de VecTask
- Utilise ArticulationView au lieu de gym.acquire_*
- Quaternions au format [w, x, y, z] (Isaac Lab) au lieu de [x, y, z, w]
- Cloner API pour la création d'environnements
"""

import torch
from typing import Dict, Tuple

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.assets import Articulation

from isaaclab_envs.utils.motion_lib import MotionLib
from .mdp import compute_observations, compute_rewards, compute_terminations

# Constantes du modèle humanoid
# NOTE: Ces valeurs sont identiques à Isaac Gym pour ce modèle spécifique
# (voir DOF_ANALYSIS.md pour l'explication détaillée)
DOF_BODY_IDS = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
DOF_OFFSETS = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]

# Dimensions
NUM_OBS = 105  # 1 + 12 + 3 + 3 + 52 + 28 + 12
NUM_ACTIONS = 28

# Corps clés (extrémités)
KEY_BODY_NAMES = ["right_hand", "left_hand", "right_foot", "left_foot"]
KEY_BODY_IDS = [5, 8, 11, 14]

# Corps de contact autorisés (pieds)
CONTACT_BODY_NAMES = ["right_foot", "left_foot"]
CONTACT_BODY_IDS = [11, 14]


class HumanoidAMPBaseEnv(DirectRLEnv):
    """Environnement de base pour HumanoidAMP.

    Fournit les fonctionnalités de base pour:
    - Observations (105D)
    - Terminations (chute, timeout)
    - Reset des environnements
    - Contrôle PD

    Cette classe est étendue par:
    - HumanoidEnv: Humanoid standard sans AMP
    - HumanoidAMPEnv: Humanoid avec AMP (discriminateur)
    """

    cfg: "HumanoidAMPEnvCfg"  # Type hint pour la configuration

    def __init__(self, cfg, render_mode=None, **kwargs):
        """Initialise l'environnement HumanoidAMP.

        Args:
            cfg: Configuration de l'environnement (HumanoidAMPEnvCfg)
            render_mode: Mode de rendu (pour compatibilité Gym)
            **kwargs: Arguments supplémentaires
        """
        # Stocker la configuration
        self.cfg = cfg

        # Initialiser les IDs des corps
        self._key_body_ids = torch.tensor(KEY_BODY_IDS, dtype=torch.long)
        self._contact_body_ids = torch.tensor(CONTACT_BODY_IDS, dtype=torch.long)

        # Initialiser la bibliothèque de mouvements (si utilisée)
        self._motion_lib = None
        if hasattr(cfg, 'motion_file') and cfg.motion_file is not None:
            self._load_motion_lib()

        # Appeler le constructeur parent (DirectRLEnv)
        # Cela va appeler _setup_scene() automatiquement
        super().__init__(cfg, render_mode, **kwargs)

        # Buffers supplémentaires
        self._init_amp_buffers()

    def _load_motion_lib(self):
        """Charge la bibliothèque de mouvements de référence."""
        motion_file = self.cfg.motion_file

        # Créer MotionLib
        self._motion_lib = MotionLib(
            motion_file=motion_file,
            num_dofs=NUM_ACTIONS,
            key_body_ids=self._key_body_ids.cpu().numpy(),
            device=self.device
        )

        print(f"[HumanoidAMP] Motion library chargée: {self._motion_lib.num_motions()} mouvements")

    def _setup_scene(self):
        """Configure la scène avec le robot et le sol.

        Cette méthode est appelée automatiquement par DirectRLEnv.__init__().
        Elle remplace create_sim() et _create_envs() d'Isaac Gym.
        """
        # Récupérer le robot depuis la config de scène
        self.robot = Articulation(self.cfg.scene.robot)

        # Ajouter à la scène
        self.scene.articulations["robot"] = self.robot

        # Ajouter le sol
        self.scene.add_ground_plane(**self.cfg.scene.ground)

        # Cloner les environnements (automatique dans Isaac Lab)
        self.scene.clone_environments(copy_from_source=False)

        # Filtrer les collisions self-collision (optionnel)
        # Isaac Lab gère cela automatiquement via PhysX

        print(f"[HumanoidAMP] Scène créée avec {self.num_envs} environnements")

    def _init_amp_buffers(self):
        """Initialise les buffers spécifiques à AMP.

        Ces buffers sont utilisés pour:
        - Stocker les observations AMP
        - Gérer le sampling de mouvements
        """
        # Buffer pour les observations AMP (utilisé par le discriminateur)
        # Shape: (num_envs, num_obs)
        self.amp_obs_buf = torch.zeros(
            self.num_envs, NUM_OBS,
            device=self.device, dtype=torch.float
        )

        # IDs des mouvements actuels (si motion_lib utilisée)
        if self._motion_lib is not None:
            self._motion_ids = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.long
            )
            self._motion_times = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.float
            )

    def _pre_physics_step(self, actions: torch.Tensor):
        """Appelée avant chaque pas de simulation.

        Applique les actions au robot (contrôle PD ou forces).

        Args:
            actions: Actions normalisées [-1, 1], shape (num_envs, num_actions)
        """
        # Clipper les actions
        actions = torch.clamp(actions, -1.0, 1.0)

        # Appliquer les actions
        if self.cfg.use_pd_control:
            # Contrôle PD: actions → positions cibles
            self._apply_pd_control(actions)
        else:
            # Contrôle direct: actions → forces/torques
            self._apply_force_control(actions)

    def _apply_pd_control(self, actions: torch.Tensor):
        """Applique le contrôle PD aux articulations.

        Args:
            actions: Actions normalisées [-1, 1], shape (num_envs, num_actions)
        """
        # Dénormaliser les actions vers les positions cibles
        # Les limites viennent de ArticulationCfg
        joint_pos_target = self.robot.data.default_joint_pos + actions * self.cfg.action_scale

        # Appliquer les cibles (Isaac Lab fait le contrôle PD automatiquement)
        self.robot.set_joint_position_target(joint_pos_target)

    def _apply_force_control(self, actions: torch.Tensor):
        """Applique le contrôle en force/torque direct.

        Args:
            actions: Actions normalisées [-1, 1], shape (num_envs, num_actions)
        """
        # Dénormaliser les actions vers les forces/torques
        forces = actions * self.cfg.power_scale

        # Appliquer les forces
        self.robot.set_joint_effort_target(forces)

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Calcule les observations pour la policy.

        Returns:
            Dictionnaire {"policy": obs}, obs de shape (num_envs, num_obs)
        """
        # Utiliser la fonction MDP
        obs = compute_observations(self)

        # Isaac Lab attend un dictionnaire
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Calcule les récompenses.

        Pour AMP pur (task_reward_w=0.0), retourne zéro.
        Le discriminateur ajoutera sa récompense ensuite.

        Returns:
            Récompenses, shape (num_envs,)
        """
        return compute_rewards(self)

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcule les conditions de fin d'épisode.

        Returns:
            Tuple de (terminated, truncated):
            - terminated: Fin due à une condition (chute), shape (num_envs,)
            - truncated: Fin due au timeout, shape (num_envs,)
        """
        return compute_terminations(self)

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset les environnements spécifiés.

        Args:
            env_ids: IDs des environnements à reset, shape (num_reset,)
        """
        num_reset = len(env_ids)

        # Reset depuis la motion library si disponible
        if self._motion_lib is not None and self.cfg.reset_from_motion:
            self._reset_from_motion(env_ids)
        else:
            self._reset_default(env_ids)

        # Reset des buffers
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_default(self, env_ids: torch.Tensor):
        """Reset par défaut (position debout).

        Args:
            env_ids: IDs des environnements à reset
        """
        num_reset = len(env_ids)

        # Positions DOF par défaut (debout)
        dof_pos = self.robot.data.default_joint_pos[env_ids].clone()
        dof_vel = torch.zeros((num_reset, NUM_ACTIONS), device=self.device)

        # Position racine par défaut
        root_states = self.robot.data.default_root_state[env_ids].clone()

        # Petite randomisation (optionnel)
        if self.cfg.randomize_reset:
            dof_pos += torch.randn_like(dof_pos) * 0.1
            root_states[:, :3] += torch.randn((num_reset, 3), device=self.device) * 0.1

        # Écrire les états
        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(root_states, env_ids=env_ids)

    def _reset_from_motion(self, env_ids: torch.Tensor):
        """Reset depuis un mouvement de référence.

        Args:
            env_ids: IDs des environnements à reset
        """
        num_reset = len(env_ids)

        # Échantillonner des mouvements aléatoires
        motion_ids, motion_times = self._motion_lib.sample_motions(num_reset)

        # Obtenir les états depuis la motion library
        (root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos) = \
            self._motion_lib.get_motion_state(motion_ids, motion_times)

        # Construire root_states [pos(3), quat(4), vel(3), ang_vel(3)]
        # NOTE: root_rot est déjà en [w, x, y, z] (conversion dans MotionLib)
        root_states = torch.cat([
            root_pos,      # (num_reset, 3)
            root_rot,      # (num_reset, 4) [w, x, y, z]
            root_vel,      # (num_reset, 3)
            root_ang_vel   # (num_reset, 3)
        ], dim=-1)  # (num_reset, 13)

        # Écrire les états
        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(root_states, env_ids=env_ids)

        # Sauvegarder les IDs/temps de mouvement
        self._motion_ids[env_ids] = motion_ids
        self._motion_times[env_ids] = motion_times

    # --- Méthodes utilitaires pour AMP ---

    def fetch_amp_obs_demo(self, num_samples: int) -> torch.Tensor:
        """Récupère des observations AMP depuis les mouvements de référence.

        Utilisé pour entraîner le discriminateur AMP.

        Args:
            num_samples: Nombre d'échantillons à générer

        Returns:
            Observations AMP, shape (num_samples, num_obs)
        """
        if self._motion_lib is None:
            raise RuntimeError("Motion library non chargée")

        # Échantillonner des états de mouvement aléatoires
        motion_ids, motion_times = self._motion_lib.sample_motions(num_samples)

        # Obtenir les états
        (root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos) = \
            self._motion_lib.get_motion_state(motion_ids, motion_times)

        # Construire root_states
        root_states = torch.cat([
            root_pos, root_rot, root_vel, root_ang_vel
        ], dim=-1)

        # Calculer les observations
        from .mdp.observations import build_amp_observations

        amp_obs = build_amp_observations(
            root_states,
            dof_pos,
            dof_vel,
            key_pos,
            self.cfg.local_root_obs
        )

        return amp_obs

    def get_num_amp_obs(self) -> int:
        """Retourne le nombre de dimensions des observations AMP."""
        return NUM_OBS


# Fonctions utilitaires pour la compatibilité

def get_dof_body_ids():
    """Retourne les IDs des corps avec des DOF."""
    return DOF_BODY_IDS

def get_dof_offsets():
    """Retourne les offsets des DOF."""
    return DOF_OFFSETS

def get_key_body_names():
    """Retourne les noms des corps clés."""
    return KEY_BODY_NAMES
