#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script d'entraînement pour HumanoidAMP avec Isaac Lab.

Ce script initialise un environnement HumanoidAMP avec Isaac Lab,
le wrappe pour RL-Games, et lance l'entraînement AMP.

Usage:
    python train.py --task HumanoidAMP --num_envs 4096 --headless

Args:
    --task: Nom de la tâche (HumanoidAMP)
    --num_envs: Nombre d'environnements parallèles
    --headless: Mode sans interface graphique
    --checkpoint: Chemin vers un checkpoint à charger
    --play: Mode évaluation (pas d'entraînement)
"""

import argparse
import os
import sys
from datetime import datetime

# Ajouter le répertoire racine au PYTHONPATH
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

import torch
import numpy as np

# RL-Games
from rl_games.torch_runner import Runner
from rl_games.common import env_configurations, vecenv


def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Entraînement HumanoidAMP avec Isaac Lab")

    # Arguments environnement
    parser.add_argument("--task", type=str, default="HumanoidAMP",
                        help="Nom de la tâche à entraîner")
    parser.add_argument("--num_envs", type=int, default=4096,
                        help="Nombre d'environnements parallèles")
    parser.add_argument("--headless", action="store_true",
                        help="Lancer en mode headless (sans GUI)")

    # Arguments entraînement
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed aléatoire")
    parser.add_argument("--max_iterations", type=int, default=None,
                        help="Nombre maximum d'itérations d'entraînement")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Chemin vers checkpoint à charger")

    # Arguments RL-Games
    parser.add_argument("--play", action="store_true",
                        help="Mode évaluation (pas d'entraînement)")
    parser.add_argument("--sigma", type=float, default=None,
                        help="Écart-type pour exploration")

    # Arguments Isaac Lab
    parser.add_argument("--enable_cameras", action="store_true",
                        help="Activer les caméras (ralentit)")

    args = parser.parse_args()
    return args


def create_env_config(args):
    """Crée la configuration de l'environnement.

    Args:
        args: Arguments parsés

    Returns:
        Configuration de l'environnement
    """
    # Import de la config
    from isaaclab_envs.configs import HumanoidAMPEnvCfg

    # Créer config
    cfg = HumanoidAMPEnvCfg()

    # Adapter selon arguments
    cfg.scene.num_envs = args.num_envs

    # Seed
    if args.seed is not None:
        cfg.seed = args.seed

    return cfg


def create_rlgames_config(args):
    """Crée la configuration RL-Games.

    Args:
        args: Arguments parsés

    Returns:
        Dictionnaire de configuration RL-Games
    """
    config = {
        'params': {
            'seed': args.seed if args.seed is not None else 42,
            'algo': {
                'name': 'amp_continuous',  # Utilise AMPAgent
            },
            'model': {
                'name': 'amp_continuous',  # Utilise ModelAMPContinuous
            },
            'network': {
                'name': 'amp',  # Utilise AMPBuilder
                'separate': False,
                'space': {
                    'continuous': {
                        'mu_activation': 'None',
                        'sigma_activation': 'None',
                        'mu_init': {
                            'name': 'default',
                        },
                        'sigma_init': {
                            'name': 'const_initializer',
                            'val': -2.9,
                        },
                        'fixed_sigma': True,
                        'learn_sigma': False,
                    },
                },
                'mlp': {
                    'units': [1024, 512, 256],
                    'activation': 'relu',
                    'initializer': {
                        'name': 'default',
                    },
                },
                'disc': {
                    'units': [1024, 512],
                    'activation': 'relu',
                    'initializer': {
                        'name': 'default',
                    },
                },
            },
            'config': {
                'name': f'HumanoidAMP_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'env_name': 'rlgpu',
                'multi_gpu': False,
                'ppo': True,
                'mixed_precision': False,
                'normalize_input': True,
                'normalize_value': True,
                'normalize_advantage': True,
                'value_bootstrap': True,
                'num_actors': args.num_envs,
                'reward_shaper': {
                    'scale_value': 1.0,
                },
                'gamma': 0.99,
                'tau': 0.95,
                'learning_rate': 2e-5,
                'lr_schedule': 'adaptive',
                'schedule_type': 'standard',
                'kl_threshold': 0.016,
                'score_to_win': 20000,
                'max_epochs': args.max_iterations if args.max_iterations else 100000,
                'save_best_after': 100,
                'save_frequency': 100,
                'grad_norm': 1.0,
                'entropy_coef': 0.0,
                'truncate_grads': True,
                'e_clip': 0.2,
                'horizon_length': 32,
                'minibatch_size': 16384,
                'mini_epochs': 6,
                'critic_coef': 5,
                'clip_value': False,
                'bounds_loss_coef': 10,
                'amp_obs_demo_buffer_size': 200000,
                'amp_replay_buffer_size': 1000000,
                'amp_replay_keep_prob': 0.01,
                'amp_batch_size': 512,
                'amp_minibatch_size': 4096,
                'disc_coef': 5,
                'disc_logit_reg': 0.01,
                'disc_grad_penalty': 5,
                'disc_reward_scale': 2,
                'disc_weight_decay': 0.0001,
                'normalize_amp_input': True,
                'task_reward_w': 0.0,  # AMP pur: pas de récompense de tâche
                'disc_reward_w': 1.0,   # Seulement récompense discriminateur
            },
        },
    }

    # Mode play (évaluation)
    if args.play:
        config['params']['config']['player'] = {
            'deterministic': True,
            'games_num': 1000,
            'print_stats': True,
        }

    # Checkpoint
    if args.checkpoint:
        config['params']['load_checkpoint'] = True
        config['params']['load_path'] = args.checkpoint

    return config


def main():
    """Fonction principale."""
    # Parser arguments
    args = parse_args()

    print("=" * 80)
    print(f"Entraînement HumanoidAMP avec Isaac Lab")
    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"Num envs: {args.num_envs}")
    print(f"Headless: {args.headless}")
    print(f"Seed: {args.seed}")
    print("=" * 80)

    # Import des modules nécessaires
    from isaaclab_envs.utils import create_rlgames_env
    from isaaclab_envs.learning import ModelAMPContinuous, AMPBuilder, AMPAgent

    # Créer configuration environnement
    env_cfg = create_env_config(args)

    # Créer environnement wrappé
    print("\n[INFO] Création de l'environnement...")
    env = create_rlgames_env(args.task, env_cfg, headless=args.headless)
    print(f"[INFO] Environnement créé: {args.num_envs} environnements")

    # Enregistrer environnement dans RL-Games
    vecenv.register(
        'rlgpu',
        lambda config_name, num_actors, **kwargs: env
    )

    # Enregistrer modèle, builder et agent dans RL-Games
    from rl_games.algos_torch import model_builder
    from rl_games.common import object_factory

    model_builder.register_model('amp_continuous', ModelAMPContinuous)
    model_builder.register_network('amp', AMPBuilder)
    object_factory.register_builder('amp_continuous', lambda **kwargs: AMPAgent(**kwargs))

    # Créer configuration RL-Games
    rlg_config = create_rlgames_config(args)

    # Créer runner RL-Games
    print("\n[INFO] Initialisation RL-Games Runner...")
    runner = Runner()
    runner.load(rlg_config)

    # Lancer entraînement ou évaluation
    if args.play:
        print("\n[INFO] Lancement mode évaluation...")
        runner.run({'train': False, 'play': True})
    else:
        print("\n[INFO] Lancement entraînement...")
        runner.run({'train': True, 'play': False})

    print("\n[INFO] Terminé!")


if __name__ == "__main__":
    main()
