# Isaac Lab Environments - HumanoidAMP

Ce dossier contient le code migré d'Isaac Gym vers Isaac Lab pour l'environnement HumanoidAMP (Adversarial Motion Priors).

## 📁 Structure

```
isaaclab_envs/
├── configs/                    # Configurations Python (@configclass)
│   ├── scene_cfg.py           # Configuration de la scène
│   ├── humanoid_amp_env_cfg.py    # Configuration de l'environnement
│   └── humanoid_amp_agent_cfg.py  # Configuration PPO/AMP
├── envs/                      # Environnements
│   └── humanoid_amp/
│       ├── humanoid_amp_env.py    # Classe d'environnement principale
│       └── mdp/               # Définitions MDP
│           ├── observations.py
│           ├── rewards.py
│           ├── terminations.py
│           └── events.py
├── assets/                    # Configurations d'assets
│   └── humanoid_cfg.py       # ArticulationCfg pour humanoid
├── utils/                     # Utilitaires
│   ├── math.py               # Fonctions mathématiques (quaternions)
│   └── motion_lib.py         # Bibliothèque de mouvements
├── learning/                  # Algorithmes d'apprentissage
│   ├── amp_agent.py          # Agent AMP
│   └── amp_models.py         # Modèles AMP (discriminateur)
└── scripts/                   # Scripts d'entraînement
    ├── train.py              # Script d'entraînement
    └── play.py               # Script d'évaluation
```

## 🔄 Changements Majeurs par rapport à Isaac Gym

### 1. Conventions de Quaternions
- **Isaac Gym**: `[x, y, z, w]`
- **Isaac Lab**: `[w, x, y, z]`

### 2. Ordre des Joints
- **Isaac Gym**: Depth-first ordering
- **Isaac Lab**: Breadth-first ordering

### 3. API Tensor
- **Isaac Gym**: `gym.acquire_*()` + `gymtorch.wrap_tensor()`
- **Isaac Lab**: Accès direct via `ArticulationView`

### 4. Configuration
- **Isaac Gym**: YAML
- **Isaac Lab**: Python `@configclass`

### 5. Classe de Base
- **Isaac Gym**: `VecTask`
- **Isaac Lab**: `DirectRLEnv`

## 📝 Statut de Migration

✅ **Phase 1 Complétée**: Structure de base créée
- [x] Dossiers créés
- [x] Fichiers de configuration Python
- [x] ArticulationCfg pour humanoid
- [x] SceneCfg définie

⏳ **Prochaines Étapes**:
- [ ] Migration de la classe d'environnement
- [ ] Correction des conventions de quaternions
- [ ] Migration de motion_lib
- [ ] Migration des algorithmes AMP
- [ ] Tests et validation

## 🚀 Utilisation (après migration complète)

```python
# Entraînement
python scripts/train.py --task HumanoidAMP

# Évaluation
python scripts/play.py --task HumanoidAMP --checkpoint path/to/checkpoint.pth
```

## 📚 Références

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [Migration Guide](https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html)
- [AMP Paper](https://xbpeng.github.io/projects/AMP/)
