# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay buffer pour AMP.

Buffer circulaire simple pour stocker et échantillonner des observations AMP.
Utilisé pour stocker les observations agent et démonstration.
"""

import torch
import numpy as np


class ReplayBuffer:
    """Buffer de replay circulaire pour observations AMP.

    Stocke des observations dans un buffer circulaire et permet
    l'échantillonnage aléatoire pour l'entraînement du discriminateur.
    """

    def __init__(self, size: int, device: torch.device):
        """Initialise le replay buffer.

        Args:
            size: Taille maximale du buffer (nombre d'observations)
            device: Device PyTorch (CPU ou CUDA)
        """
        self._size = size
        self._device = device

        # Compteur total d'éléments stockés (peut dépasser size)
        self._total_count = 0

        # Position actuelle dans le buffer circulaire
        self._head_idx = 0

        # Buffer de données (initialisé à la première insertion)
        self._data_dict = None

    def get_buffer_size(self) -> int:
        """Retourne la taille maximale du buffer.

        Returns:
            Taille du buffer
        """
        return self._size

    def get_total_count(self) -> int:
        """Retourne le nombre total d'éléments stockés.

        Returns:
            Nombre total d'éléments (peut dépasser la taille du buffer)
        """
        return self._total_count

    def get_current_size(self) -> int:
        """Retourne le nombre actuel d'éléments dans le buffer.

        Returns:
            min(total_count, buffer_size)
        """
        return min(self._total_count, self._size)

    def store(self, data_dict: dict):
        """Stocke des données dans le buffer.

        Ajoute les données au buffer de manière circulaire.
        Si le buffer est plein, écrase les anciennes données.

        Args:
            data_dict: Dictionnaire de tenseurs à stocker
                      Ex: {'amp_obs': tensor de shape (N, obs_dim)}
        """
        # Initialiser le buffer à la première insertion
        if self._data_dict is None:
            self._data_dict = {}
            for key, val in data_dict.items():
                val_shape = val.shape
                # Créer buffer circulaire de taille (buffer_size, *val_shape[1:])
                data_shape = (self._size,) + val_shape[1:]
                self._data_dict[key] = torch.zeros(
                    data_shape,
                    dtype=val.dtype,
                    device=self._device
                )

        # Nombre d'éléments à ajouter
        num_samples = data_dict[list(data_dict.keys())[0]].shape[0]

        # Stocker dans le buffer circulaire
        for key, val in data_dict.items():
            assert key in self._data_dict, f"Clé {key} non trouvée dans le buffer"

            # Gérer le cas où l'insertion dépasse la fin du buffer
            end_idx = self._head_idx + num_samples
            if end_idx <= self._size:
                # Insertion simple
                self._data_dict[key][self._head_idx:end_idx] = val
            else:
                # Insertion en deux parties (wrap around)
                first_part_size = self._size - self._head_idx
                self._data_dict[key][self._head_idx:self._size] = val[:first_part_size]
                self._data_dict[key][0:end_idx - self._size] = val[first_part_size:]

        # Mettre à jour compteurs
        self._head_idx = (self._head_idx + num_samples) % self._size
        self._total_count += num_samples

        return

    def sample(self, num_samples: int) -> dict:
        """Échantillonne aléatoirement des données du buffer.

        Args:
            num_samples: Nombre d'échantillons à retourner

        Returns:
            Dictionnaire de tenseurs échantillonnés
            Ex: {'amp_obs': tensor de shape (num_samples, obs_dim)}
        """
        if self._data_dict is None:
            raise ValueError("Le buffer est vide, impossible d'échantillonner")

        # Taille actuelle du buffer
        current_size = self.get_current_size()

        if current_size == 0:
            raise ValueError("Le buffer est vide")

        # Échantillonner des indices aléatoires
        indices = torch.randint(
            0, current_size,
            (num_samples,),
            device=self._device
        )

        # Extraire les données
        sampled_dict = {}
        for key, val in self._data_dict.items():
            sampled_dict[key] = val[indices]

        return sampled_dict

    def clear(self):
        """Vide le buffer."""
        self._total_count = 0
        self._head_idx = 0
        self._data_dict = None
        return

    def __len__(self) -> int:
        """Retourne le nombre actuel d'éléments dans le buffer."""
        return self.get_current_size()
