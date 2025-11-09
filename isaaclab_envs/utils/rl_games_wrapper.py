# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper pour adapter DirectRLEnv au format RL-Games.

RL-Games attend le format Gym classique:
    obs, rewards, dones, infos = env.step(actions)

Isaac Lab DirectRLEnv retourne:
    obs_dict, rewards, terminated, truncated, infos = env.step(actions)

Ce wrapper adapte le format pour la compatibilité.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Any

from omni.isaac.lab.envs import DirectRLEnv


class RLGamesVecEnv:
    """Wrapper vectorisé pour adapter DirectRLEnv au format RL-Games.

    Adapte:
    - Format de retour step() (obs_dict → obs, terminated/truncated → dones)
    - Format de retour reset() (obs_dict → obs)
    - Ajoute observations AMP dans infos
    - Expose méthodes requises par RL-Games
    """

    def __init__(self, env: DirectRLEnv):
        """Initialise le wrapper.

        Args:
            env: Environnement DirectRLEnv à wrapper (ex: HumanoidAMPEnv)
        """
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device

        # Espaces d'observation et d'action
        # RL-Games utilise num_obs/num_acts au lieu de observation_space/action_space
        self.num_obs = env.num_observations
        self.num_acts = env.num_actions

        # Pour compatibilité avec certains codes RL-Games
        self.observation_space = type('obj', (object,), {
            'shape': (self.num_obs,)
        })()
        self.action_space = type('obj', (object,), {
            'shape': (self.num_acts,)
        })()

        # Viewer (pour debug AMP)
        self.viewer = None
        if hasattr(env, 'sim') and hasattr(env.sim, 'viewer'):
            self.viewer = env.sim.viewer

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Exécute un pas de simulation.

        Adapte le format DirectRLEnv → RL-Games.

        Args:
            actions: Actions à appliquer, shape (num_envs, num_acts)

        Returns:
            Tuple de (obs, rewards, dones, infos):
            - obs: Observations, shape (num_envs, num_obs)
            - rewards: Récompenses, shape (num_envs,)
            - dones: Flags done, shape (num_envs,)
            - infos: Dictionnaire d'informations supplémentaires
        """
        # Step DirectRLEnv
        obs_dict, rewards, terminated, truncated, infos = self.env.step(actions)

        # Extraire observations policy
        obs = obs_dict['policy']

        # Combiner terminated et truncated en dones
        # Pour RL-Games, done = terminated OR truncated
        dones = torch.logical_or(terminated, truncated)

        # Ajouter observations AMP dans infos (requis par AMPAgent)
        if hasattr(self.env, '_curr_amp_obs_buf'):
            infos['amp_obs'] = self.env._curr_amp_obs_buf
        else:
            # Fallback si pas d'observations AMP disponibles
            infos['amp_obs'] = torch.zeros(
                (self.num_envs, self.env.get_num_amp_obs()),
                device=self.device
            )

        # Ajouter flag terminate (utilisé par AMPAgent pour calcul next_values)
        infos['terminate'] = terminated

        return obs, rewards, dones, infos

    def reset(self, env_ids: torch.Tensor = None) -> torch.Tensor:
        """Reset les environnements.

        Args:
            env_ids: IDs des environnements à reset (None = tous)

        Returns:
            obs: Observations après reset, shape (num_envs, num_obs)
        """
        # Reset DirectRLEnv
        obs_dict, _ = self.env.reset(env_ids)

        # Extraire observations policy
        obs = obs_dict['policy']

        return obs

    def reset_done(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reset automatique des environnements done.

        Utilisé par certains agents RL-Games.

        Returns:
            Tuple de (obs, done_env_ids)
        """
        # Obtenir IDs des environnements terminés
        done_env_ids = self.env.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(done_env_ids) > 0:
            obs_dict, _ = self.env.reset(done_env_ids)
            obs = obs_dict['policy']
        else:
            # Aucun reset nécessaire, retourner obs actuelles
            obs = self.env.obs_buf

        return obs, done_env_ids

    # ========================================================================
    # Méthodes AMP (délégation vers l'environnement)
    # ========================================================================

    def fetch_amp_obs_demo(self, num_samples: int) -> torch.Tensor:
        """Récupère observations de démonstration pour entraîner le discriminateur.

        Args:
            num_samples: Nombre d'échantillons à générer

        Returns:
            Observations AMP de démonstration, shape (num_samples, num_amp_obs)
        """
        if hasattr(self.env, 'fetch_amp_obs_demo'):
            return self.env.fetch_amp_obs_demo(num_samples)
        else:
            raise NotImplementedError(
                f"Environnement {type(self.env).__name__} ne supporte pas fetch_amp_obs_demo"
            )

    def get_num_amp_obs(self) -> int:
        """Retourne le nombre d'observations AMP.

        Returns:
            Nombre de dimensions des observations AMP
        """
        if hasattr(self.env, 'get_num_amp_obs'):
            return self.env.get_num_amp_obs()
        else:
            raise NotImplementedError(
                f"Environnement {type(self.env).__name__} ne supporte pas get_num_amp_obs"
            )

    # ========================================================================
    # Propriétés et méthodes utilitaires
    # ========================================================================

    def get_number_of_agents(self) -> int:
        """Retourne le nombre d'agents par environnement.

        Pour la plupart des environnements Isaac Lab: 1 agent par env.

        Returns:
            Nombre d'agents (généralement 1)
        """
        return 1

    def get_env_info(self) -> Dict[str, Any]:
        """Retourne les informations sur l'environnement.

        Utilisé par RL-Games pour configurer les réseaux.

        Returns:
            Dictionnaire d'informations
        """
        info = {
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            'num_envs': self.num_envs,
            'num_observations': self.num_obs,
            'num_actions': self.num_acts,
        }

        # Ajouter infos AMP si disponibles
        if hasattr(self.env, 'get_num_amp_obs'):
            info['num_amp_obs'] = self.env.get_num_amp_obs()

        return info

    @property
    def unwrapped(self):
        """Retourne l'environnement non-wrappé.

        Returns:
            Environnement DirectRLEnv original
        """
        return self.env

    def close(self):
        """Ferme l'environnement."""
        if hasattr(self.env, 'close'):
            self.env.close()

    def seed(self, seed: int = None):
        """Définit la seed aléatoire.

        Args:
            seed: Seed à utiliser
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)


# ============================================================================
# Factory fonction pour créer l'environnement wrappé
# ============================================================================

def create_rlgames_env(env_name: str, cfg, **kwargs) -> RLGamesVecEnv:
    """Crée un environnement DirectRLEnv wrappé pour RL-Games.

    Args:
        env_name: Nom de l'environnement (ex: "HumanoidAMP")
        cfg: Configuration de l'environnement
        **kwargs: Arguments supplémentaires pour l'environnement

    Returns:
        Environnement wrappé compatible RL-Games
    """
    # Import dynamique de l'environnement
    if env_name == "HumanoidAMP":
        from isaaclab_envs.envs.humanoid_amp import HumanoidAMPEnv
        env = HumanoidAMPEnv(cfg, **kwargs)
    else:
        raise ValueError(f"Environnement inconnu: {env_name}")

    # Wrapper pour RL-Games
    wrapped_env = RLGamesVecEnv(env)

    return wrapped_env
