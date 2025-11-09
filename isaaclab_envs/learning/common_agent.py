# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Agent de base commun pour Isaac Lab.

Extension de l'agent A2C de RL-Games avec fonctionnalités supplémentaires
pour environnements Isaac Lab.
"""

from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.algos_torch import a2c_continuous

import torch
from torch import nn


class CommonAgent(a2c_continuous.A2CAgent):
    """Agent de base commun pour environnements Isaac Lab.

    Étend A2CAgent de RL-Games avec quelques fonctionnalités supplémentaires
    communes à tous les agents Isaac Lab.
    """

    def __init__(self, base_name, params):
        """Initialise l'agent commun.

        Args:
            base_name: Nom de base pour cet agent
            params: Dictionnaire de paramètres
        """
        super().__init__(base_name, params)

        # Liste des tenseurs à transférer dans le batch
        self.tensor_list = ['obses', 'actions', 'values', 'neglogpacs', 'mus', 'sigmas', 'returns']

    def _load_config_params(self, config):
        """Charge les paramètres depuis la configuration.

        Args:
            config: Dictionnaire de configuration
        """
        super()._load_config_params(config)

        # Pas de paramètres supplémentaires pour l'agent de base
        # Les classes dérivées peuvent étendre cette méthode

        return

    def _build_net_config(self):
        """Construit la configuration du réseau.

        Returns:
            Dictionnaire de configuration pour le réseau
        """
        config = super()._build_net_config()

        # Pas de configuration supplémentaire pour l'agent de base
        # Les classes dérivées peuvent étendre cette méthode

        return config

    def _preproc_obs(self, obs_batch):
        """Pré-traite les observations.

        Args:
            obs_batch: Batch d'observations

        Returns:
            Observations pré-traitées
        """
        if type(obs_batch) is dict:
            obs_batch = obs_batch['obs']

        if self.normalize_input:
            obs_batch = self.running_mean_std(obs_batch)

        return obs_batch

    def _eval_critic(self, obs):
        """Évalue le critique (value network).

        Args:
            obs: Observations (dict ou tensor)

        Returns:
            Valeurs prédites
        """
        self.model.eval()

        # Pré-traiter observations
        obs_batch = self._preproc_obs(obs)

        # Forward pass
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs_batch,
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            values = res_dict['values']

        return values

    def _env_reset_done(self):
        """Reset les environnements qui sont terminés.

        Returns:
            obs: Nouvelles observations
            done_env_ids: IDs des environnements resetés
        """
        # Trouver environnements terminés
        done_env_ids = self.dones.nonzero(as_tuple=False).squeeze(-1)

        # RL-Games reset automatiquement dans env_step
        # Donc ici on retourne juste les observations actuelles
        return self.obs, done_env_ids

    def env_step(self, actions):
        """Effectue un pas dans l'environnement.

        Args:
            actions: Actions à exécuter

        Returns:
            obs: Nouvelles observations
            rewards: Récompenses
            dones: Flags de terminaison
            infos: Informations supplémentaires
        """
        # Clipper les actions si nécessaire
        if self.has_action_bounds:
            actions = torch.clamp(actions, -1.0, 1.0)

        # Step dans l'environnement
        obs, rewards, dones, infos = self.vec_env.step(actions)

        # Normaliser valeur si activé
        if self.normalize_value:
            # Mettre à jour running stats avec les récompenses
            self.value_mean_std.train()
            self.value_mean_std(rewards.unsqueeze(-1))

        return obs, rewards, dones, infos

    def get_action_values(self, obs):
        """Obtient les actions et valeurs pour des observations.

        Args:
            obs: Observations (dict ou tensor)

        Returns:
            Dictionnaire contenant actions, valeurs, log probs, etc.
        """
        self.model.eval()

        # Pré-traiter observations
        obs_batch = self._preproc_obs(obs)

        # Forward pass
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs_batch,
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)

        return res_dict

    def get_masked_action_values(self, obs, masks):
        """Obtient les actions avec masques (pour espaces discrets).

        Args:
            obs: Observations
            masks: Masques d'actions

        Returns:
            Dictionnaire contenant actions, valeurs, etc.
        """
        # Cette méthode est pour les espaces d'action discrets
        # Pour AMP (continu), on utilise simplement get_action_values
        return self.get_action_values(obs)

    def discount_values(self, fdones, values, rewards, next_values):
        """Calcule les avantages avec Generalized Advantage Estimation (GAE).

        Args:
            fdones: Flags de terminaison (float)
            values: Valeurs prédites par le critique
            rewards: Récompenses
            next_values: Valeurs des états suivants

        Returns:
            Avantages calculés
        """
        lastgaelam = 0
        advantages = torch.zeros_like(rewards)

        # Dimensions: (horizon_length, num_envs, 1)
        horizon_length = rewards.shape[0]

        # GAE: calcul des avantages depuis la fin vers le début
        for t in reversed(range(horizon_length)):
            if t == horizon_length - 1:
                nextnonterminal = 1.0 - fdones[t]
                nextvalues = next_values[t]
            else:
                nextnonterminal = 1.0 - fdones[t]
                nextvalues = values[t + 1]

            # TD error: delta = r + gamma * V(s') - V(s)
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]

            # GAE: A = delta + gamma * tau * A(t+1)
            advantages[t] = lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam

        return advantages

    def train_actor_critic(self, input_dict):
        """Entraîne l'acteur et le critique.

        Args:
            input_dict: Dictionnaire d'entrée contenant les données du batch

        Returns:
            Dictionnaire contenant les métriques d'entraînement
        """
        # Calculer les gradients et faire la backprop
        self.calc_gradients(input_dict)

        # Retourner les résultats d'entraînement
        return self.train_result

    def bound_loss(self, mu):
        """Calcule la perte de bornes pour les actions.

        Pénalise les actions qui dépassent les bornes [-1, 1].

        Args:
            mu: Moyennes des distributions d'actions

        Returns:
            Perte de bornes
        """
        if self.bounds_loss_coef is not None and self.bounds_loss_coef > 0:
            # Pénaliser les actions en dehors de [-1, 1]
            soft_bound = 1.0
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
            mu_loss_low = torch.clamp_min(-soft_bound - mu, 0.0) ** 2
            b_loss = (mu_loss_high + mu_loss_low).sum(axis=-1)
            return b_loss.mean()
        else:
            return torch.tensor(0.0, device=self.ppo_device)

    def _actor_loss(self, old_action_log_probs, action_log_probs, advantage, curr_e_clip):
        """Calcule la perte de l'acteur (PPO clipped loss).

        Args:
            old_action_log_probs: Anciens log probabilities
            action_log_probs: Nouveaux log probabilities
            advantage: Avantages
            curr_e_clip: Coefficient de clipping epsilon

        Returns:
            Dictionnaire contenant la perte de l'acteur
        """
        # PPO clipped loss
        ratio = torch.exp(old_action_log_probs - action_log_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
        a_loss = torch.max(-surr1, -surr2).mean()

        # Clipfrac (fraction d'exemples clippés) - pour monitoring
        with torch.no_grad():
            clipfrac = torch.mean((torch.abs(ratio - 1.0) > curr_e_clip).float())

        info = {
            'actor_loss': a_loss,
            'actor_clipped': clipfrac
        }
        return info

    def _critic_loss(self, value_preds_batch, values, curr_e_clip, return_batch, clip_value):
        """Calcule la perte du critique.

        Args:
            value_preds_batch: Anciennes prédictions de valeur
            values: Nouvelles prédictions de valeur
            curr_e_clip: Coefficient de clipping
            return_batch: Returns calculés (avantages + valeurs)
            clip_value: Si True, clipper les valeurs

        Returns:
            Dictionnaire contenant la perte du critique
        """
        if clip_value:
            # Clipper les valeurs autour des anciennes prédictions
            value_pred_clipped = value_preds_batch + \
                (values - value_preds_batch).clamp(-curr_e_clip, curr_e_clip)
            value_losses = (values - return_batch) ** 2
            value_losses_clipped = (value_pred_clipped - return_batch) ** 2
            c_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            # MSE simple
            c_loss = ((return_batch - values) ** 2).mean()

        info = {
            'critic_loss': c_loss
        }
        return info
