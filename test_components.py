#!/usr/bin/env python3

"""
Tests Unitaires pour Isaac Lab AMP Migration
Date: 2025-11-09

Ces tests peuvent être exécutés SANS Isaac Lab installé.
Ils testent uniquement les composants indépendants.

Usage:
    python test_components.py
"""

import sys
import torch
import numpy as np
from pathlib import Path

print("=" * 80)
print("  Tests Unitaires - Migration Isaac Lab AMP")
print("=" * 80)
print()

# ==============================================================================
# Configuration
# ==============================================================================

# Ajouter le chemin du projet
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"📍 Device: {device}")
if device == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
print()

# ==============================================================================
# Test 1: ReplayBuffer
# ==============================================================================

print("[1/5] Test ReplayBuffer...")

try:
    from isaaclab_envs.learning.replay_buffer import ReplayBuffer

    # Créer buffer
    buffer_size = 100
    buffer = ReplayBuffer(buffer_size, device=device)

    # Test store
    data = {'amp_obs': torch.randn(10, 105, device=device)}
    buffer.store(data)
    assert buffer.get_current_size() == 10, "Taille incorrecte après store"

    # Test sample
    samples = buffer.sample(5)
    assert samples['amp_obs'].shape == (5, 105), "Shape incorrecte après sample"

    # Test wrap-around
    for _ in range(15):
        buffer.store({'amp_obs': torch.randn(10, 105, device=device)})
    assert buffer.get_current_size() == buffer_size, "Buffer devrait être plein"

    # Test clear
    buffer.clear()
    assert buffer.get_current_size() == 0, "Buffer devrait être vide après clear"

    print("   ✅ ReplayBuffer fonctionne correctement")

except Exception as e:
    print(f"   ❌ ERREUR ReplayBuffer: {e}")
    sys.exit(1)

print()

# ==============================================================================
# Test 2: Fonctions Mathématiques
# ==============================================================================

print("[2/5] Test Fonctions Mathématiques...")

try:
    from isaaclab_envs.utils.math import (
        quat_mul, quat_conjugate, quat_rotate,
        normalize, normalize_angle
    )

    # Test quaternion multiplication
    q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)  # Identity [w, x, y, z]
    q2 = torch.tensor([[0.707, 0.707, 0.0, 0.0]], device=device)  # 90° rotation around x
    result = quat_mul(q1, q2)
    assert result.shape == (1, 4), "Shape incorrecte pour quat_mul"

    # Test quaternion conjugate
    q_conj = quat_conjugate(q2)
    assert q_conj[0, 0] == q2[0, 0], "w devrait rester identique"
    assert torch.allclose(q_conj[0, 1:], -q2[0, 1:]), "xyz devrait être négatif"

    # Test normalize
    vec = torch.tensor([[3.0, 4.0, 0.0]], device=device)
    vec_norm = normalize(vec)
    assert torch.allclose(torch.norm(vec_norm, dim=-1), torch.tensor(1.0)), "Norme devrait être 1"

    # Test normalize_angle
    angles = torch.tensor([3.5 * np.pi, -2.5 * np.pi], device=device)
    angles_norm = normalize_angle(angles)
    assert torch.all(angles_norm >= -np.pi), "Angles devraient être >= -π"
    assert torch.all(angles_norm <= np.pi), "Angles devraient être <= π"

    print("   ✅ Fonctions mathématiques fonctionnent correctement")

except Exception as e:
    print(f"   ❌ ERREUR Fonctions Math: {e}")
    sys.exit(1)

print()

# ==============================================================================
# Test 3: AMPDataset
# ==============================================================================

print("[3/5] Test AMPDataset...")

try:
    from isaaclab_envs.learning.amp_datasets import AMPDataset

    batch_size = 128
    minibatch_size = 32
    dataset = AMPDataset(
        batch_size=batch_size,
        minibatch_size=minibatch_size,
        is_discrete=False,
        is_rnn=False,
        device=device,
        seq_len=1
    )

    # Simuler des données
    dataset.values_dict = {
        'obs': torch.randn(batch_size, 105, device=device),
        'actions': torch.randn(batch_size, 21, device=device),
        'values': torch.randn(batch_size, 1, device=device),
    }

    # Test iteration
    dataset.update_values_dict(dataset.values_dict)
    num_batches = len(dataset)
    expected_batches = batch_size // minibatch_size
    assert num_batches == expected_batches, f"Devrait avoir {expected_batches} batches"

    # Test get item
    batch = dataset[0]
    assert 'obs' in batch, "Batch devrait contenir 'obs'"
    assert batch['obs'].shape[0] == minibatch_size, "Minibatch size incorrecte"

    print("   ✅ AMPDataset fonctionne correctement")

except Exception as e:
    print(f"   ❌ ERREUR AMPDataset: {e}")
    sys.exit(1)

print()

# ==============================================================================
# Test 4: Imports Principaux
# ==============================================================================

print("[4/5] Test Imports Principaux...")

imports_to_test = [
    ('isaaclab_envs.learning', 'AMPAgent'),
    ('isaaclab_envs.learning', 'CommonAgent'),
    ('isaaclab_envs.learning', 'ModelAMPContinuous'),
    ('isaaclab_envs.learning', 'AMPBuilder'),
    ('isaaclab_envs.utils.math', 'quat_mul'),
]

all_imports_ok = True
for module_name, class_name in imports_to_test:
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"   ✅ {module_name}.{class_name}")
    except Exception as e:
        print(f"   ❌ {module_name}.{class_name}: {e}")
        all_imports_ok = False

if not all_imports_ok:
    print("   ❌ Certains imports ont échoué")
    sys.exit(1)
else:
    print("   ✅ Tous les imports fonctionnent")

print()

# ==============================================================================
# Test 5: Structure des Fichiers
# ==============================================================================

print("[5/5] Test Structure des Fichiers...")

required_files = [
    'isaaclab_envs/__init__.py',
    'isaaclab_envs/configs/__init__.py',
    'isaaclab_envs/envs/__init__.py',
    'isaaclab_envs/envs/humanoid_amp/humanoid_amp_env.py',
    'isaaclab_envs/envs/humanoid_amp/mdp/observations.py',
    'isaaclab_envs/learning/amp_agent.py',
    'isaaclab_envs/learning/amp_network.py',
    'isaaclab_envs/learning/replay_buffer.py',
    'isaaclab_envs/utils/math.py',
    'isaaclab_envs/utils/motion_lib.py',
    'isaaclab_envs/scripts/train.py',
]

all_files_present = True
for file_path in required_files:
    full_path = project_root / file_path
    if full_path.exists():
        print(f"   ✅ {file_path}")
    else:
        print(f"   ❌ {file_path} (MANQUANT)")
        all_files_present = False

if not all_files_present:
    print("   ❌ Certains fichiers sont manquants")
    sys.exit(1)
else:
    print("   ✅ Tous les fichiers requis sont présents")

print()

# ==============================================================================
# Résumé
# ==============================================================================

print("=" * 80)
print("  ✅ TOUS LES TESTS RÉUSSIS !")
print("=" * 80)
print()
print("Composants testés:")
print("  • ReplayBuffer (store, sample, wrap-around)")
print("  • Fonctions mathématiques (quaternions, normalize)")
print("  • AMPDataset (iteration, batching)")
print("  • Imports de toutes les classes principales")
print("  • Structure complète des fichiers")
print()
print("Prochaines étapes:")
print("  1. Installer Isaac Lab (voir RUNPOD_GUIDE.md)")
print("  2. Tester avec environnements réels")
print("  3. Lancer entraînement court (10 iterations)")
print()
print("=" * 80)
