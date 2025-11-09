# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Réseaux de neurones AMP pour Isaac Lab.

Ce fichier combine:
- amp_models.py: Modèle A2C avec discriminateur
- amp_network_builder.py: Builder pour construire les réseaux

Basé sur RL-Games, compatible avec Isaac Lab DirectRLEnv.
"""

import torch
import torch.nn as nn
import numpy as np

from rl_games.algos_torch.models import ModelA2CContinuousLogStd
from rl_games.algos_torch import torch_ext, layers, network_builder

# Échelle d'initialisation pour les logits du discriminateur
DISC_LOGIT_INIT_SCALE = 1.0


# ============================================================================
# Modèle AMP
# ============================================================================

class ModelAMPContinuous(ModelA2CContinuousLogStd):
    """Modèle A2C continu avec discriminateur AMP.

    Étend le modèle A2C standard de RL-Games en ajoutant:
    - Réseau discriminateur pour distinguer agent/démonstration
    - Évaluations discriminateur dans le forward pass (training)
    """

    def __init__(self, network):
        """Initialise le modèle AMP.

        Args:
            network: Builder de réseau (AMPBuilder)
        """
        super().__init__(network)

    def build(self, config):
        """Construit le réseau avec discriminateur.

        Args:
            config: Configuration du réseau contenant:
                - input_shape: Forme observations policy
                - amp_input_shape: Forme observations AMP (pour discriminateur)
                - normalize_value: Normaliser les valeurs
                - normalize_input: Normaliser les entrées
                - value_size: Taille de la sortie value (1 pour scalar)

        Returns:
            Instance de Network configurée
        """
        # Construire le réseau via AMPBuilder
        net = self.network_builder.build('amp', **config)

        # Debug: Afficher les noms de paramètres
        for name, _ in net.named_parameters():
            print(name)

        # Extraire config
        obs_shape = config['input_shape']
        normalize_value = config.get('normalize_value', False)
        normalize_input = config.get('normalize_input', False)
        value_size = config.get('value_size', 1)

        return self.Network(
            net,
            obs_shape=obs_shape,
            normalize_value=normalize_value,
            normalize_input=normalize_input,
            value_size=value_size
        )

    class Network(ModelA2CContinuousLogStd.Network):
        """Réseau A2C avec discriminateur AMP.

        Forward pass retourne:
        - Actions, valeurs, log_probs (standard A2C)
        - Logits discriminateur (training uniquement):
          - disc_agent_logit: Discriminateur sur observations agent
          - disc_agent_replay_logit: Discriminateur sur replay buffer
          - disc_demo_logit: Discriminateur sur démonstrations
        """

        def __init__(self, a2c_network, **kwargs):
            """Initialise le réseau.

            Args:
                a2c_network: Réseau A2C (de AMPBuilder.Network)
                **kwargs: Arguments supplémentaires pour parent
            """
            super().__init__(a2c_network, **kwargs)

        def forward(self, input_dict):
            """Forward pass avec évaluations discriminateur.

            Args:
                input_dict: Dictionnaire contenant:
                    - obs: Observations policy (N, num_obs)
                    - is_train: Mode entraînement (bool)
                    - amp_obs: Observations AMP agent (training)
                    - amp_obs_replay: Observations AMP replay (training)
                    - amp_obs_demo: Observations AMP démo (training)

            Returns:
                result: Dictionnaire contenant:
                    - actions, values, log_probs (standard)
                    - disc_agent_logit (training)
                    - disc_agent_replay_logit (training)
                    - disc_demo_logit (training)
            """
            is_train = input_dict.get('is_train', True)

            # Forward pass A2C standard
            result = super().forward(input_dict)

            # Évaluations discriminateur (training uniquement)
            if is_train:
                # Discriminateur sur observations agent actuelles
                amp_obs = input_dict['amp_obs']
                disc_agent_logit = self.a2c_network.eval_disc(amp_obs)
                result["disc_agent_logit"] = disc_agent_logit

                # Discriminateur sur observations replay buffer
                amp_obs_replay = input_dict['amp_obs_replay']
                disc_agent_replay_logit = self.a2c_network.eval_disc(amp_obs_replay)
                result["disc_agent_replay_logit"] = disc_agent_replay_logit

                # Discriminateur sur observations démonstration
                amp_demo_obs = input_dict['amp_obs_demo']
                disc_demo_logit = self.a2c_network.eval_disc(amp_demo_obs)
                result["disc_demo_logit"] = disc_demo_logit

            return result


# ============================================================================
# Builder Réseau AMP
# ============================================================================

class AMPBuilder(network_builder.A2CBuilder):
    """Builder pour construire réseaux A2C avec discriminateur AMP.

    Étend A2CBuilder de RL-Games en ajoutant:
    - Construction du réseau discriminateur
    - Évaluation discriminateur (eval_disc)
    """

    def __init__(self, **kwargs):
        """Initialise le builder AMP."""
        super().__init__(**kwargs)

    class Network(network_builder.A2CBuilder.Network):
        """Réseau A2C avec discriminateur.

        Architecture:
        - Actor: Observations → MLP → Actions (mu, sigma)
        - Critic: Observations → MLP → Value
        - Discriminateur: Observations AMP → MLP → Logit (real vs fake)
        """

        def __init__(self, params, **kwargs):
            """Initialise le réseau.

            Args:
                params: Paramètres de configuration du réseau
                **kwargs: Arguments supplémentaires dont:
                    - amp_input_shape: Forme observations AMP
                    - actions_num: Nombre d'actions
            """
            super().__init__(params, **kwargs)

            # Sigma fixe si learn_sigma=False
            if self.is_continuous:
                if not self.space_config['learn_sigma']:
                    actions_num = kwargs.get('actions_num')
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                    self.sigma = nn.Parameter(
                        torch.zeros(actions_num, requires_grad=False, dtype=torch.float32),
                        requires_grad=False
                    )
                    sigma_init(self.sigma)

            # Construire discriminateur
            amp_input_shape = kwargs.get('amp_input_shape')
            self._build_disc(amp_input_shape)

        def load(self, params):
            """Charge les paramètres du réseau.

            Args:
                params: Dictionnaire de paramètres contenant:
                    - disc: Configuration discriminateur
                        - units: Liste tailles couches [1024, 512]
                        - activation: Fonction d'activation ('relu')
                        - initializer: Initialisation poids
            """
            super().load(params)

            # Charger config discriminateur
            self._disc_units = params['disc']['units']
            self._disc_activation = params['disc']['activation']
            self._disc_initializer = params['disc']['initializer']

        def eval_critic(self, obs):
            """Évalue le critic (fonction de valeur).

            Args:
                obs: Observations, shape (N, num_obs)

            Returns:
                value: Valeurs estimées, shape (N, 1)
            """
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)
            value = self.value_act(self.value(c_out))
            return value

        def eval_disc(self, amp_obs):
            """Évalue le discriminateur.

            Le discriminateur distingue les observations agent (fake)
            des observations démonstration (real).

            Args:
                amp_obs: Observations AMP, shape (N, num_amp_obs)

            Returns:
                disc_logits: Logits discriminateur, shape (N, 1)
                    Valeur positive → probablement démonstration
                    Valeur négative → probablement agent
            """
            disc_mlp_out = self._disc_mlp(amp_obs)
            disc_logits = self._disc_logits(disc_mlp_out)
            return disc_logits

        def get_disc_logit_weights(self):
            """Retourne les poids de la couche logits (pour régularisation).

            Returns:
                Poids de disc_logits aplatis, shape (mlp_out_size,)
            """
            return torch.flatten(self._disc_logits.weight)

        def get_disc_weights(self):
            """Retourne tous les poids du discriminateur (pour régularisation).

            Returns:
                Liste de tenseurs de poids (MLP + logits)
            """
            weights = []

            # Poids MLP
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    weights.append(torch.flatten(m.weight))

            # Poids logits
            weights.append(torch.flatten(self._disc_logits.weight))

            return weights

        def _build_disc(self, input_shape):
            """Construit le réseau discriminateur.

            Architecture:
            - Input: Observations AMP (num_amp_obs)
            - Hidden: MLP avec plusieurs couches (ex: [1024, 512])
            - Output: Logit scalaire (1)

            Args:
                input_shape: Forme observations AMP, ex: [210] pour 2 steps × 105
            """
            # Construire MLP
            mlp_args = {
                'input_size': input_shape[0],
                'units': self._disc_units,
                'activation': self._disc_activation,
                'dense_func': torch.nn.Linear
            }
            self._disc_mlp = self._build_mlp(**mlp_args)

            # Couche finale: hidden → logit (1D)
            mlp_out_size = self._disc_units[-1]
            self._disc_logits = torch.nn.Linear(mlp_out_size, 1)

            # Initialiser poids MLP
            mlp_init = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            # Initialiser poids logits (uniform)
            torch.nn.init.uniform_(
                self._disc_logits.weight,
                -DISC_LOGIT_INIT_SCALE,
                DISC_LOGIT_INIT_SCALE
            )
            torch.nn.init.zeros_(self._disc_logits.bias)

    def build(self, name, **kwargs):
        """Construit le réseau.

        Args:
            name: Nom du réseau ('amp')
            **kwargs: Arguments de configuration

        Returns:
            Instance de AMPBuilder.Network
        """
        net = AMPBuilder.Network(self.params, **kwargs)
        return net
