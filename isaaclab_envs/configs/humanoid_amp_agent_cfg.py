# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration de l'agent PPO pour HumanoidAMP."""

from isaaclab.utils import configclass

from isaaclab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class HumanoidAMPPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration du runner PPO pour HumanoidAMP."""

    num_steps_per_env = 16  # horizon_length dans l'ancienne config
    max_iterations = 5000
    save_interval = 50
    experiment_name = "humanoid_amp"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[1024, 512],
        critic_hidden_dims=[1024, 512],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=5.0,  # critic_coef
        use_clipped_value_loss=False,  # clip_value
        clip_param=0.2,  # e_clip
        entropy_coef=0.0,
        num_learning_epochs=6,  # mini_epochs
        num_mini_batches=4,  # calculé à partir de minibatch_size
        learning_rate=5.0e-5,
        schedule="constant",  # lr_schedule
        gamma=0.99,
        lam=0.95,  # tau
        desired_kl=0.008,  # kl_threshold
        max_grad_norm=1.0,  # grad_norm
    )


@configclass
class HumanoidAMPCfg:
    """Configuration spécifique AMP pour HumanoidAMP."""

    # Buffers AMP
    amp_obs_demo_buffer_size = 200000
    amp_replay_buffer_size = 1000000
    amp_replay_keep_prob = 0.01

    # Batch sizes
    amp_batch_size = 512
    amp_minibatch_size = 4096

    # Coefficients AMP
    disc_coef = 5.0  # Coefficient du discriminateur
    disc_logit_reg = 0.05  # Régularisation logit
    disc_grad_penalty = 5.0  # Pénalité de gradient
    disc_reward_scale = 2.0  # Échelle de récompense du discriminateur
    disc_weight_decay = 0.0001  # Décroissance des poids

    # Normalisation
    normalize_amp_input = True

    # Poids des récompenses
    task_reward_w = 0.0  # Pas de récompense de tâche, uniquement AMP
    disc_reward_w = 1.0  # Récompense du discriminateur uniquement

    # Architecture du discriminateur
    disc_hidden_dims = [1024, 512]
    disc_activation = "relu"

    # Bounds loss (pour garder les actions dans les limites)
    bounds_loss_coef = 10.0
