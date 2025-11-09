# Copyright (c) 2018-2025, NVIDIA Corporation & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Bibliothèque de mouvements pour AMP (Adversarial Motion Priors).

Cette bibliothèque charge et gère les données de capture de mouvement pour l'entraînement AMP.
Elle fournit des fonctionnalités pour:
- Charger des mouvements depuis des fichiers .npy (SkeletonMotion)
- Échantillonner des mouvements et des temps aléatoires
- Interpoler entre les frames
- Convertir les rotations locales en positions DOF

IMPORTANT: Convention de quaternions
- Les fichiers .npy (poselib) utilisent: [x, y, z, w] (format original)
- Isaac Lab utilise: [w, x, y, z]
- Ce fichier effectue la conversion lors du chargement
"""

import numpy as np
import os
import torch
import yaml
from typing import List, Tuple

# Import de poselib (utilise format XYZW)
# NOTE: poselib n'est pas encore migré, nous utilisons l'ancien pour l'instant
try:
    from isaacgymenvs.tasks.amp.poselib.poselib.skeleton.skeleton3d import SkeletonMotion
    from isaacgymenvs.tasks.amp.poselib.poselib.core.rotation3d import (
        quat_mul_norm, quat_inverse, quat_angle_axis
    )
    POSELIB_AVAILABLE = True
except ImportError:
    POSELIB_AVAILABLE = False
    print("Warning: poselib not available, MotionLib will not function")

# Import des utilitaires Isaac Lab
from isaaclab_envs.utils.math import (
    to_torch, slerp, quat_to_exp_map, quat_to_angle_axis, normalize_angle
)


# Constantes pour le mapping DOF
# TODO: Ces valeurs devront être recalculées pour Isaac Lab (breadth-first ordering)
# Pour l'instant, nous utilisons les mêmes valeurs qu'Isaac Gym
DOF_BODY_IDS = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
DOF_OFFSETS = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 27, 28]


def convert_quat_xyzw_to_wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Convertit un quaternion de [x, y, z, w] vers [w, x, y, z].

    Args:
        quat_xyzw: Quaternion au format [x, y, z, w], shape (..., 4)

    Returns:
        Quaternion au format [w, x, y, z], shape (..., 4)
    """
    # Réorganiser: [x, y, z, w] -> [w, x, y, z]
    if len(quat_xyzw.shape) == 1:
        return torch.tensor([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
                          dtype=quat_xyzw.dtype, device=quat_xyzw.device)
    else:
        return torch.cat([quat_xyzw[..., 3:4], quat_xyzw[..., :3]], dim=-1)


def convert_quat_wxyz_to_xyzw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Convertit un quaternion de [w, x, y, z] vers [x, y, z, w].

    Args:
        quat_wxyz: Quaternion au format [w, x, y, z], shape (..., 4)

    Returns:
        Quaternion au format [x, y, z, w], shape (..., 4)
    """
    # Réorganiser: [w, x, y, z] -> [x, y, z, w]
    if len(quat_wxyz.shape) == 1:
        return torch.tensor([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
                          dtype=quat_wxyz.dtype, device=quat_wxyz.device)
    else:
        return torch.cat([quat_wxyz[..., 1:4], quat_wxyz[..., 0:1]], dim=-1)


class MotionLib:
    """Bibliothèque pour charger et gérer les mouvements de capture de motion.

    Cette classe charge des fichiers de mouvements (format SkeletonMotion) et fournit
    des méthodes pour échantillonner des états de mouvement pour l'entraînement AMP.
    """

    def __init__(self, motion_file: str, num_dofs: int, key_body_ids: np.ndarray, device: str):
        """Initialise la bibliothèque de mouvements.

        Args:
            motion_file: Chemin vers le fichier de mouvement (.npy ou .yaml)
            num_dofs: Nombre de degrés de liberté du robot
            key_body_ids: IDs des corps clés pour les observations
            device: Device PyTorch ('cuda:0', 'cpu', etc.)
        """
        if not POSELIB_AVAILABLE:
            raise RuntimeError("poselib n'est pas disponible. MotionLib ne peut pas être initialisé.")

        self._num_dof = num_dofs
        self._key_body_ids = key_body_ids
        self._device = device
        self._load_motions(motion_file)

        self.motion_ids = torch.arange(len(self._motions), dtype=torch.long, device=self._device)

    def num_motions(self) -> int:
        """Retourne le nombre de mouvements chargés."""
        return len(self._motions)

    def get_total_length(self) -> float:
        """Retourne la durée totale de tous les mouvements."""
        return sum(self._motion_lengths)

    def get_motion(self, motion_id: int):
        """Retourne un mouvement spécifique par son ID."""
        return self._motions[motion_id]

    def sample_motions(self, n: int) -> np.ndarray:
        """Échantillonne n IDs de mouvements selon leurs poids.

        Args:
            n: Nombre d'IDs à échantillonner

        Returns:
            Tableau d'IDs de mouvements, shape (n,)
        """
        m = self.num_motions()
        motion_ids = np.random.choice(m, size=n, replace=True, p=self._motion_weights)
        return motion_ids

    def sample_time(self, motion_ids: np.ndarray, truncate_time: float = None) -> np.ndarray:
        """Échantillonne des temps aléatoires pour les mouvements donnés.

        Args:
            motion_ids: IDs des mouvements, shape (n,)
            truncate_time: Temps à retrancher de la fin du mouvement

        Returns:
            Temps échantillonnés, shape (n,)
        """
        n = len(motion_ids)
        phase = np.random.uniform(low=0.0, high=1.0, size=motion_ids.shape)

        motion_len = self._motion_lengths[motion_ids]
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time

    def get_motion_length(self, motion_ids: np.ndarray) -> np.ndarray:
        """Retourne les longueurs des mouvements spécifiés."""
        return self._motion_lengths[motion_ids]

    def get_motion_state(
        self, motion_ids: np.ndarray, motion_times: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Récupère l'état du mouvement aux temps spécifiés.

        Args:
            motion_ids: IDs des mouvements, shape (n,)
            motion_times: Temps dans chaque mouvement, shape (n,)

        Returns:
            Tuple de (root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos)
            - root_pos: Position racine, shape (n, 3)
            - root_rot: Rotation racine [w, x, y, z], shape (n, 4)
            - dof_pos: Positions DOF, shape (n, num_dofs)
            - root_vel: Vélocité racine, shape (n, 3)
            - root_ang_vel: Vélocité angulaire racine, shape (n, 3)
            - dof_vel: Vélocités DOF, shape (n, num_dofs)
            - key_pos: Positions des corps clés, shape (n, num_key_bodies, 3)
        """
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()
        num_key_bodies = self._key_body_ids.shape[0]

        # Buffers pour les données numpy
        root_pos0 = np.empty([n, 3])
        root_pos1 = np.empty([n, 3])
        root_rot = np.empty([n, 4])
        root_rot0 = np.empty([n, 4])  # Format XYZW de poselib
        root_rot1 = np.empty([n, 4])  # Format XYZW de poselib
        root_vel = np.empty([n, 3])
        root_ang_vel = np.empty([n, 3])
        local_rot0 = np.empty([n, num_bodies, 4])  # Format XYZW de poselib
        local_rot1 = np.empty([n, num_bodies, 4])  # Format XYZW de poselib
        dof_vel = np.empty([n, self._num_dof])
        key_pos0 = np.empty([n, num_key_bodies, 3])
        key_pos1 = np.empty([n, num_key_bodies, 3])

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        # Extraction des données pour chaque mouvement unique
        unique_ids = np.unique(motion_ids)
        for uid in unique_ids:
            ids = np.where(motion_ids == uid)
            curr_motion = self._motions[uid]

            root_pos0[ids, :] = curr_motion.global_translation[frame_idx0[ids], 0].numpy()
            root_pos1[ids, :] = curr_motion.global_translation[frame_idx1[ids], 0].numpy()

            # Rotations au format XYZW de poselib
            root_rot0[ids, :] = curr_motion.global_rotation[frame_idx0[ids], 0].numpy()
            root_rot1[ids, :] = curr_motion.global_rotation[frame_idx1[ids], 0].numpy()

            local_rot0[ids, :, :] = curr_motion.local_rotation[frame_idx0[ids]].numpy()
            local_rot1[ids, :, :] = curr_motion.local_rotation[frame_idx1[ids]].numpy()

            root_vel[ids, :] = curr_motion.global_root_velocity[frame_idx0[ids]].numpy()
            root_ang_vel[ids, :] = curr_motion.global_root_angular_velocity[frame_idx0[ids]].numpy()

            key_pos0[ids, :, :] = curr_motion.global_translation[
                frame_idx0[ids][:, np.newaxis], self._key_body_ids[np.newaxis, :]
            ].numpy()
            key_pos1[ids, :, :] = curr_motion.global_translation[
                frame_idx1[ids][:, np.newaxis], self._key_body_ids[np.newaxis, :]
            ].numpy()

            dof_vel[ids, :] = curr_motion.dof_vels[frame_idx0[ids]]

        # Conversion en tenseurs PyTorch
        blend = to_torch(np.expand_dims(blend, axis=-1), device=self._device)

        root_pos0 = to_torch(root_pos0, device=self._device)
        root_pos1 = to_torch(root_pos1, device=self._device)

        # IMPORTANT: Conversion XYZW → WXYZ pour Isaac Lab
        root_rot0_xyzw = to_torch(root_rot0, device=self._device)
        root_rot1_xyzw = to_torch(root_rot1, device=self._device)
        root_rot0 = convert_quat_xyzw_to_wxyz(root_rot0_xyzw)
        root_rot1 = convert_quat_xyzw_to_wxyz(root_rot1_xyzw)

        root_vel = to_torch(root_vel, device=self._device)
        root_ang_vel = to_torch(root_ang_vel, device=self._device)

        # Conversion XYZW → WXYZ pour les rotations locales
        local_rot0_xyzw = to_torch(local_rot0, device=self._device)
        local_rot1_xyzw = to_torch(local_rot1, device=self._device)
        local_rot0 = convert_quat_xyzw_to_wxyz(local_rot0_xyzw)
        local_rot1 = convert_quat_xyzw_to_wxyz(local_rot1_xyzw)

        key_pos0 = to_torch(key_pos0, device=self._device)
        key_pos1 = to_torch(key_pos1, device=self._device)
        dof_vel = to_torch(dof_vel, device=self._device)

        # Interpolation
        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1

        # Slerp pour les quaternions (maintenant au format WXYZ)
        root_rot = slerp(root_rot0, root_rot1, blend)

        blend_exp = blend.unsqueeze(-1)
        key_pos = (1.0 - blend_exp) * key_pos0 + blend_exp * key_pos1

        local_rot = slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        dof_pos = self._local_rotation_to_dof(local_rot)

        return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos

    def _load_motions(self, motion_file: str):
        """Charge les mouvements depuis un fichier.

        Args:
            motion_file: Chemin vers le fichier (.npy ou .yaml avec liste)
        """
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []

        total_len = 0.0

        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)

        for f in range(num_motion_files):
            curr_file = motion_files[f]
            print(f"Loading {f + 1}/{num_motion_files} motion files: {curr_file}")

            curr_motion = SkeletonMotion.from_file(curr_file)
            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.tensor.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)

            # Calculer les vélocités DOF
            curr_dof_vels = self._compute_motion_dof_vels(curr_motion)
            curr_motion.dof_vels = curr_dof_vels

            self._motions.append(curr_motion)
            self._motion_lengths.append(curr_len)

            curr_weight = motion_weights[f]
            self._motion_weights.append(curr_weight)
            self._motion_files.append(curr_file)

        # Conversion en arrays numpy
        self._motion_lengths = np.array(self._motion_lengths)
        self._motion_weights = np.array(self._motion_weights)
        self._motion_weights /= np.sum(self._motion_weights)

        self._motion_fps = np.array(self._motion_fps)
        self._motion_dt = np.array(self._motion_dt)
        self._motion_num_frames = np.array(self._motion_num_frames)

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print(f"Loaded {num_motions} motions with a total length of {total_len:.3f}s.")

    def _fetch_motion_files(self, motion_file: str) -> Tuple[List[str], List[float]]:
        """Récupère la liste des fichiers de mouvement et leurs poids.

        Args:
            motion_file: Chemin vers un fichier .npy ou .yaml

        Returns:
            Tuple de (motion_files, motion_weights)
        """
        ext = os.path.splitext(motion_file)[1]

        if ext == ".yaml":
            dir_name = os.path.dirname(motion_file)
            motion_files = []
            motion_weights = []

            with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            motion_list = motion_config['motions']
            for motion_entry in motion_list:
                curr_file = motion_entry['file']
                curr_weight = motion_entry['weight']
                assert curr_weight >= 0

                curr_file = os.path.join(dir_name, curr_file)
                motion_weights.append(curr_weight)
                motion_files.append(curr_file)
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]

        return motion_files, motion_weights

    def _calc_frame_blend(
        self, time: np.ndarray, length: np.ndarray, num_frames: np.ndarray, dt: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calcule les indices de frames et le facteur de blend pour interpolation.

        Args:
            time: Temps dans le mouvement, shape (n,)
            length: Longueur du mouvement, shape (n,)
            num_frames: Nombre de frames, shape (n,)
            dt: Delta temps entre frames, shape (n,)

        Returns:
            Tuple de (frame_idx0, frame_idx1, blend)
        """
        phase = time / length
        phase = np.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).astype(int)
        frame_idx1 = np.minimum(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self) -> int:
        """Retourne le nombre de corps dans le squelette."""
        motion = self.get_motion(0)
        num_bodies = motion.num_joints
        return num_bodies

    def _compute_motion_dof_vels(self, motion) -> np.ndarray:
        """Calcule les vélocités DOF pour toutes les frames.

        Args:
            motion: Objet SkeletonMotion

        Returns:
            Vélocités DOF, shape (num_frames, num_dofs)
        """
        num_frames = motion.tensor.shape[0]
        dt = 1.0 / motion.fps
        dof_vels = []

        for f in range(num_frames - 1):
            local_rot0 = motion.local_rotation[f]  # Format XYZW
            local_rot1 = motion.local_rotation[f + 1]  # Format XYZW
            frame_dof_vel = self._local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
            dof_vels.append(frame_dof_vel)

        # Dupliquer la dernière vélocité
        dof_vels.append(dof_vels[-1])
        dof_vels = np.array(dof_vels)

        return dof_vels

    def _local_rotation_to_dof(self, local_rot: torch.Tensor) -> torch.Tensor:
        """Convertit les rotations locales en positions DOF.

        Args:
            local_rot: Rotations locales [w, x, y, z], shape (n, num_bodies, 4)

        Returns:
            Positions DOF, shape (n, num_dofs)
        """
        body_ids = DOF_BODY_IDS
        dof_offsets = DOF_OFFSETS

        n = local_rot.shape[0]
        dof_pos = torch.zeros((n, self._num_dof), dtype=torch.float, device=self._device)

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if joint_size == 3:
                # Articulation sphérique (3 DOF) → carte exponentielle
                joint_q = local_rot[:, body_id]  # [w, x, y, z]
                joint_exp_map = quat_to_exp_map(joint_q)
                dof_pos[:, joint_offset:(joint_offset + joint_size)] = joint_exp_map

            elif joint_size == 1:
                # Articulation à charnière (1 DOF) → angle
                joint_q = local_rot[:, body_id]  # [w, x, y, z]
                joint_theta, joint_axis = quat_to_angle_axis(joint_q)
                # Supposer que l'articulation est toujours le long de l'axe y
                joint_theta = joint_theta * joint_axis[..., 1]
                joint_theta = normalize_angle(joint_theta)
                dof_pos[:, joint_offset] = joint_theta

            else:
                print(f"Unsupported joint type with {joint_size} DOFs")
                assert False

        return dof_pos

    def _local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt: float) -> np.ndarray:
        """Calcule les vélocités DOF entre deux frames.

        Args:
            local_rot0: Rotations locales frame 0 (format XYZW de poselib)
            local_rot1: Rotations locales frame 1 (format XYZW de poselib)
            dt: Delta temps entre les frames

        Returns:
            Vélocités DOF, shape (num_dofs,)
        """
        body_ids = DOF_BODY_IDS
        dof_offsets = DOF_OFFSETS

        dof_vel = np.zeros([self._num_dof])

        # NOTE: quat_mul_norm et quat_inverse viennent de poselib (format XYZW)
        diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
        local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
        local_vel = local_vel.numpy()

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if joint_size == 3:
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset:(joint_offset + joint_size)] = joint_vel

            elif joint_size == 1:
                assert joint_size == 1
                joint_vel = local_vel[body_id]
                # Supposer que l'articulation est toujours le long de l'axe y
                dof_vel[joint_offset] = joint_vel[1]

            else:
                print(f"Unsupported joint type with {joint_size} DOFs")
                assert False

        return dof_vel
