# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dataset AMP pour Isaac Lab.

Extension de PPODataset de RL-Games avec shuffling aléatoire des indices.
"""

import torch
from rl_games.common import datasets


class AMPDataset(datasets.PPODataset):
    """Dataset pour entraînement AMP.

    Étend PPODataset en ajoutant un shuffling aléatoire des indices
    de batch pour améliorer la diversité des minibatches.
    """

    def __init__(self, batch_size, minibatch_size, is_discrete, is_rnn, device, seq_len):
        """Initialise le dataset AMP.

        Args:
            batch_size: Taille du batch complet
            minibatch_size: Taille des minibatches
            is_discrete: Si l'espace d'action est discret
            is_rnn: Si le modèle utilise un RNN
            device: Device PyTorch
            seq_len: Longueur des séquences (pour RNN)
        """
        super().__init__(batch_size, minibatch_size, is_discrete, is_rnn, device, seq_len)

        # Buffer pour indices aléatoires (shuffling)
        self._idx_buf = torch.randperm(batch_size)

    def update_mu_sigma(self, mu, sigma):
        """Met à jour mu et sigma (non implémenté).

        Args:
            mu: Moyenne
            sigma: Écart-type

        Raises:
            NotImplementedError: Méthode non utilisée pour AMP
        """
        raise NotImplementedError()

    def _get_item(self, idx):
        """Récupère un minibatch depuis le buffer.

        Args:
            idx: Index du minibatch

        Returns:
            Dictionnaire contenant les données du minibatch
        """
        # Calculer indices du minibatch
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        sample_idx = self._idx_buf[start:end]

        # Créer input_dict avec données échantillonnées
        input_dict = {}
        for k, v in self.values_dict.items():
            if k not in self.special_names and v is not None:
                input_dict[k] = v[sample_idx]

        # Shuffler buffer si fin du batch atteinte
        if end >= self.batch_size:
            self._shuffle_idx_buf()

        return input_dict

    def _shuffle_idx_buf(self):
        """Shuffle le buffer d'indices aléatoires."""
        self._idx_buf[:] = torch.randperm(self.batch_size)
