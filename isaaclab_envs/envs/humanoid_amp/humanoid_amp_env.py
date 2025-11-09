# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environnement HumanoidAMP avec discriminateur.

Ce fichier remplace humanoid_amp.py d'Isaac Gym.
Il hérite de HumanoidAMPBaseEnv et ajoute:
- Support pour le discriminateur AMP
- Échantillonnage de démonstrations depuis la motion library
- Observations AMP avec historique temporel
"""

import torch
from enum import Enum
from typing import Tuple

from .humanoid_amp_base_env import HumanoidAMPBaseEnv, NUM_OBS
from .mdp.observations import build_amp_observations


class StateInit(Enum):
    """Modes d'initialisation d'état pour le reset."""
    Default = 0  # Reset à la pose par défaut
    Start = 1    # Reset au début d'un mouvement de référence
    Random = 2   # Reset à un temps aléatoire dans un mouvement
    Hybrid = 3   # Mélange de Default et Random


class HumanoidAMPEnv(HumanoidAMPBaseEnv):
    """Environnement Humanoid avec AMP (Adversarial Motion Priors).

    Ajoute à HumanoidAMPBaseEnv:
    - Gestion de l'historique d'observations AMP (pour le discriminateur)
    - Échantillonnage de démonstrations depuis la motion library
    - Modes de reset différents (default, start, random, hybrid)

    Le discriminateur AMP est entraîné dans l'agent (amp_continuous.py),
    pas dans l'environnement.
    """

    cfg: "HumanoidAMPEnvCfg"

    def __init__(self, cfg, render_mode=None, **kwargs):
        """Initialise l'environnement HumanoidAMP.

        Args:
            cfg: Configuration de l'environnement (HumanoidAMPEnvCfg)
            render_mode: Mode de rendu (pour compatibilité Gym)
            **kwargs: Arguments supplémentaires
        """
        # Initialiser l'enum pour le mode d'initialisation
        self._state_init = StateInit[cfg.state_init]
        self._hybrid_init_prob = cfg.hybrid_init_prob
        self._num_amp_obs_steps = cfg.num_amp_obs_steps

        assert self._num_amp_obs_steps >= 2, "num_amp_obs_steps doit être >= 2"

        # Listes pour traquer les resets
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        # Appeler le constructeur parent
        super().__init__(cfg, render_mode, **kwargs)

        # Buffers pour observations AMP avec historique temporel
        # Shape: (num_envs, num_amp_obs_steps, NUM_OBS)
        self._amp_obs_buf = torch.zeros(
            (self.num_envs, self._num_amp_obs_steps, NUM_OBS),
            device=self.device,
            dtype=torch.float,
        )
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]  # Observations actuelles (t)
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]  # Historique (t-1, t-2, ...)

        # Buffer pour observations de démonstration (lazy init)
        self._amp_obs_demo_buf = None

        # Calculer le nombre total d'observations AMP
        # Utilisé par le discriminateur
        self.num_amp_obs = self._num_amp_obs_steps * NUM_OBS

        print(f"[HumanoidAMP] AMP observations: {self.num_amp_obs} ({self._num_amp_obs_steps} steps × {NUM_OBS} dims)")

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset les environnements spécifiés.

        Étend la méthode parent pour supporter différents modes d'initialisation.

        Args:
            env_ids: IDs des environnements à reset
        """
        # Reset selon le mode d'initialisation
        if self._state_init == StateInit.Default:
            self._reset_default_only(env_ids)
        elif self._state_init in [StateInit.Start, StateInit.Random]:
            self._reset_ref_state_init(env_ids)
        elif self._state_init == StateInit.Hybrid:
            self._reset_hybrid_state_init(env_ids)
        else:
            raise ValueError(f"Mode d'initialisation non supporté: {self._state_init}")

        # Reset des buffers
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        # Initialiser les observations AMP
        self._init_amp_obs(env_ids)

    def _reset_default_only(self, env_ids: torch.Tensor):
        """Reset à la pose par défaut uniquement.

        Args:
            env_ids: IDs des environnements à reset
        """
        # Utiliser la méthode parent
        super()._reset_default(env_ids)
        self._reset_default_env_ids = env_ids.cpu().tolist()

    def _reset_ref_state_init(self, env_ids: torch.Tensor):
        """Reset à un état de mouvement de référence.

        Args:
            env_ids: IDs des environnements à reset
        """
        if self._motion_lib is None:
            print("[WARNING] Motion library non chargée, reset par défaut")
            self._reset_default_only(env_ids)
            return

        # Utiliser la méthode parent qui gère déjà le sampling
        if self._state_init == StateInit.Start:
            # Reset au début du mouvement (t=0)
            # TODO: Modifier motion_lib.sample_motions pour accepter time_min/max
            super()._reset_from_motion(env_ids)
        else:
            # Random: temps aléatoire
            super()._reset_from_motion(env_ids)

        self._reset_ref_env_ids = env_ids.cpu().tolist()

    def _reset_hybrid_state_init(self, env_ids: torch.Tensor):
        """Reset hybride: mélange de default et référence.

        Args:
            env_ids: IDs des environnements à reset
        """
        num_reset = len(env_ids)

        # Probabilité de reset depuis référence
        ref_probs = torch.full((num_reset,), self._hybrid_init_prob, device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs).bool()

        # Reset avec référence
        ref_reset_ids = env_ids[ref_init_mask]
        if len(ref_reset_ids) > 0:
            self._reset_ref_state_init(ref_reset_ids)

        # Reset par défaut
        default_reset_ids = env_ids[~ref_init_mask]
        if len(default_reset_ids) > 0:
            self._reset_default_only(default_reset_ids)

    def _init_amp_obs(self, env_ids: torch.Tensor):
        """Initialise les observations AMP après un reset.

        Remplit l'historique avec l'observation actuelle répétée.

        Args:
            env_ids: IDs des environnements réinitialisés
        """
        # Calculer observation actuelle
        self._compute_amp_observations(env_ids)

        # Initialiser l'historique en répétant l'observation actuelle
        # Cela évite d'avoir des valeurs nulles dans l'historique
        for i in range(self._num_amp_obs_steps - 1):
            self._hist_amp_obs_buf[env_ids, i] = self._curr_amp_obs_buf[env_ids]

    def _compute_amp_observations(self, env_ids: torch.Tensor = None):
        """Calcule les observations AMP.

        Ces observations sont utilisées par le discriminateur pour
        distinguer mouvements réels vs générés.

        Args:
            env_ids: IDs des environnements (None = tous)
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Accéder aux données du robot
        robot = self.scene.articulations["robot"]

        # Construire root_states [pos(3), quat(4), vel(3), ang_vel(3)]
        root_states = robot.data.root_state_w[env_ids]  # (N, 13)

        # Positions et vélocités DOF
        dof_pos = robot.data.joint_pos[env_ids]  # (N, num_dofs)
        dof_vel = robot.data.joint_vel[env_ids]  # (N, num_dofs)

        # Positions des corps clés
        key_body_pos = robot.data.body_pos_w[env_ids][:, self._key_body_ids, :]  # (N, 4, 3)

        # Calculer observations AMP
        amp_obs = build_amp_observations(
            root_states,
            dof_pos,
            dof_vel,
            key_body_pos,
            self.cfg.local_root_obs
        )

        # Sauvegarder
        self._curr_amp_obs_buf[env_ids] = amp_obs

    def _update_amp_obs_after_step(self):
        """Met à jour l'historique des observations AMP après un step.

        Appelée automatiquement après chaque pas de simulation.
        """
        # Décaler l'historique: [t-1, t-2, ...] <- [t, t-1, ...]
        self._hist_amp_obs_buf[:, 1:] = self._hist_amp_obs_buf[:, :-1].clone()
        self._hist_amp_obs_buf[:, 0] = self._curr_amp_obs_buf.clone()

        # Calculer nouvelles observations actuelles
        self._compute_amp_observations()

    def step(self, actions: torch.Tensor) -> Tuple:
        """Exécute un pas de simulation.

        Étend la méthode parent pour mettre à jour l'historique AMP.

        Args:
            actions: Actions de la policy, shape (num_envs, num_actions)

        Returns:
            Tuple (obs, rewards, terminated, truncated, info)
        """
        # Appeler step parent
        result = super().step(actions)

        # Mettre à jour historique AMP
        self._update_amp_obs_after_step()

        return result

    def get_num_amp_obs(self) -> int:
        """Retourne le nombre d'observations AMP.

        Utilisé par l'agent pour dimensionner le discriminateur.

        Returns:
            Nombre total d'observations AMP (steps × dims)
        """
        return self.num_amp_obs

    def fetch_amp_obs_demo(self, num_samples: int) -> torch.Tensor:
        """Récupère des observations de démonstration depuis la motion library.

        Utilisé pour entraîner le discriminateur AMP.

        Args:
            num_samples: Nombre d'échantillons à générer

        Returns:
            Observations AMP de démonstration, shape (num_samples, num_amp_obs)
        """
        if self._motion_lib is None:
            raise RuntimeError("Motion library non chargée")

        # Créer buffer si nécessaire (lazy init)
        if self._amp_obs_demo_buf is None or self._amp_obs_demo_buf.shape[0] != num_samples:
            self._build_amp_obs_demo_buf(num_samples)

        # Échantillonner des séquences temporelles
        # Pour chaque échantillon, on veut num_amp_obs_steps observations consécutives
        dt = self.cfg.sim.dt * self.cfg.decimation  # Timestep effectif

        for i in range(num_samples):
            # Échantillonner un mouvement et un temps de départ
            motion_ids, motion_times = self._motion_lib.sample_motions(1)
            motion_id = motion_ids[0]
            start_time = motion_times[0]

            # Obtenir num_amp_obs_steps frames consécutives
            for step in range(self._num_amp_obs_steps):
                # Temps pour ce step
                t = start_time + step * dt

                # Obtenir l'état du mouvement
                (root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos) = \
                    self._motion_lib.get_motion_state(
                        torch.tensor([motion_id], device=self.device),
                        torch.tensor([t], device=self.device)
                    )

                # Construire root_states
                root_states = torch.cat([
                    root_pos, root_rot, root_vel, root_ang_vel
                ], dim=-1)

                # Calculer observations AMP
                amp_obs = build_amp_observations(
                    root_states,
                    dof_pos,
                    dof_vel,
                    key_pos,
                    self.cfg.local_root_obs
                )

                # Sauvegarder
                self._amp_obs_demo_buf[i, step] = amp_obs.squeeze(0)

        # Retourner au format plat (num_samples, num_amp_obs)
        return self._amp_obs_demo_buf.view(num_samples, -1)

    def _build_amp_obs_demo_buf(self, num_samples: int):
        """Construit le buffer pour les observations de démonstration.

        Args:
            num_samples: Nombre d'échantillons
        """
        self._amp_obs_demo_buf = torch.zeros(
            (num_samples, self._num_amp_obs_steps, NUM_OBS),
            device=self.device,
            dtype=torch.float,
        )

        print(f"[HumanoidAMP] Demo buffer créé: {num_samples} samples × {self.num_amp_obs} obs")
