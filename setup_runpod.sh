#!/bin/bash

# ==============================================================================
# Script d'installation automatique Isaac Lab sur RunPod
# Date: 2025-11-09
# Usage: bash setup_runpod.sh
# ==============================================================================

set -e  # Arrêter si erreur

echo "════════════════════════════════════════════════════════════════"
echo "  Installation Isaac Lab AMP sur RunPod"
echo "════════════════════════════════════════════════════════════════"
echo ""

# ==============================================================================
# 1. Vérifications Préalables
# ==============================================================================

echo "[1/6] Vérifications préalables..."

# Vérifier GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERREUR: nvidia-smi non trouvé. GPU NVIDIA requis."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "✅ GPU détecté: $GPU_NAME"

# Vérifier Docker
if ! command -v docker &> /dev/null; then
    echo "❌ ERREUR: Docker non installé."
    exit 1
fi

DOCKER_VERSION=$(docker --version | awk '{print $3}' | tr -d ',')
echo "✅ Docker version: $DOCKER_VERSION"

# Vérifier nvidia-docker
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "❌ ERREUR: nvidia-docker non configuré correctement."
    exit 1
fi

echo "✅ nvidia-docker fonctionnel"
echo ""

# ==============================================================================
# 2. Pull Image Isaac Lab
# ==============================================================================

echo "[2/6] Téléchargement de l'image Isaac Lab..."
echo "⏳ Ceci peut prendre 5-10 minutes (~10-15 GB)..."
echo ""

ISAAC_LAB_IMAGE="nvcr.io/nvidia/isaac-lab:2.3.0"

if docker image inspect $ISAAC_LAB_IMAGE &> /dev/null; then
    echo "✅ Image Isaac Lab déjà présente"
else
    docker pull $ISAAC_LAB_IMAGE
    echo "✅ Image Isaac Lab téléchargée"
fi
echo ""

# ==============================================================================
# 3. Créer Workspace
# ==============================================================================

echo "[3/6] Création du workspace..."

WORKSPACE_DIR="$HOME/isaac-lab-workspace"
CACHE_DIR="$HOME/isaac-lab-cache"

mkdir -p $WORKSPACE_DIR
mkdir -p $CACHE_DIR

echo "✅ Workspace créé: $WORKSPACE_DIR"
echo "✅ Cache créé: $CACHE_DIR"
echo ""

# ==============================================================================
# 4. Vérifier Code Projet
# ==============================================================================

echo "[4/6] Vérification du code projet..."

# Chercher le dossier isaaclab_envs
if [ -d "./isaaclab_envs" ]; then
    PROJECT_DIR="$(pwd)"
    echo "✅ Code projet trouvé: $PROJECT_DIR"
elif [ -d "$WORKSPACE_DIR/IsaacGymEnvs/isaaclab_envs" ]; then
    PROJECT_DIR="$WORKSPACE_DIR/IsaacGymEnvs"
    echo "✅ Code projet trouvé: $PROJECT_DIR"
else
    echo "⚠️  Code projet non trouvé dans le répertoire actuel."
    echo "   Vous devez uploader le code dans: $WORKSPACE_DIR"
    echo ""
    echo "   Options:"
    echo "   1. Via Git: cd $WORKSPACE_DIR && git clone <votre-repo>"
    echo "   2. Via SCP: scp -r IsaacGymEnvs root@<runpod-ip>:$WORKSPACE_DIR"
    echo "   3. Via Upload RunPod Web Terminal"
    echo ""
    read -p "   Continuer sans code (y/N)? " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    PROJECT_DIR="$WORKSPACE_DIR"
fi
echo ""

# ==============================================================================
# 5. Créer Script de Lancement Container
# ==============================================================================

echo "[5/6] Création du script de lancement..."

LAUNCH_SCRIPT="$PROJECT_DIR/launch_isaac_lab.sh"

cat > $LAUNCH_SCRIPT << 'EOF'
#!/bin/bash

# Script de lancement container Isaac Lab
# Usage: bash launch_isaac_lab.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="$HOME/isaac-lab-cache"

echo "════════════════════════════════════════════════════════════════"
echo "  Lancement Container Isaac Lab AMP"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Project: $PROJECT_DIR"
echo "Cache:   $CACHE_DIR"
echo ""

docker run --name isaac-lab-amp \
  --gpus all \
  -it \
  --rm \
  --network=host \
  -e "ACCEPT_EULA=Y" \
  -e "PRIVACY_CONSENT=Y" \
  -v $PROJECT_DIR:/workspace \
  -v $CACHE_DIR:/isaac-sim/kit/cache:rw \
  nvcr.io/nvidia/isaac-lab:2.3.0 \
  /bin/bash -c "
    echo '════════════════════════════════════════════════════════════════';
    echo '  Container Isaac Lab Démarré';
    echo '════════════════════════════════════════════════════════════════';
    echo '';
    echo '📁 Workspace: /workspace';
    echo '🐍 Python: \$(python --version)';
    echo '🎮 GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)';
    echo '';
    echo 'Installation des dépendances...';
    cd /workspace;
    pip install -q rl-games tensorboardX 2>/dev/null;
    echo '✅ Dépendances installées';
    echo '';
    echo '════════════════════════════════════════════════════════════════';
    echo '  Prêt à utiliser !';
    echo '════════════════════════════════════════════════════════════════';
    echo '';
    echo 'Commandes utiles:';
    echo '  - Test imports:  python -c \"from isaaclab_envs.learning import AMPAgent; print(\u2705 OK!)\"';
    echo '  - Test rapide:   python isaaclab_envs/scripts/train.py --task HumanoidAMP --num_envs 512 --headless --max_iterations 10';
    echo '  - Entraînement:  python isaaclab_envs/scripts/train.py --task HumanoidAMP --num_envs 4096 --headless';
    echo '';
    /bin/bash
  "
EOF

chmod +x $LAUNCH_SCRIPT

echo "✅ Script de lancement créé: $LAUNCH_SCRIPT"
echo ""

# ==============================================================================
# 6. Créer Script de Test Rapide
# ==============================================================================

echo "[6/6] Création du script de test rapide..."

TEST_SCRIPT="$PROJECT_DIR/test_quick.sh"

cat > $TEST_SCRIPT << 'EOF'
#!/bin/bash

# Test rapide de la migration Isaac Lab
# Usage: bash test_quick.sh (depuis le container Isaac Lab)

set -e

echo "════════════════════════════════════════════════════════════════"
echo "  Test Rapide Migration Isaac Lab AMP"
echo "════════════════════════════════════════════════════════════════"
echo ""

# 1. Test imports
echo "[1/3] Test des imports..."
python -c "
from isaaclab_envs.learning import AMPAgent, ReplayBuffer
from isaaclab_envs.utils import create_rlgames_env
from isaaclab_envs.envs.humanoid_amp import HumanoidAMPEnv
print('✅ Tous les imports fonctionnent!')
"
echo ""

# 2. Test ReplayBuffer
echo "[2/3] Test ReplayBuffer..."
python -c "
import torch
from isaaclab_envs.learning import ReplayBuffer

buffer = ReplayBuffer(100, device='cpu')
buffer.store({'amp_obs': torch.randn(10, 105)})
samples = buffer.sample(5)
assert samples['amp_obs'].shape == (5, 105)
print('✅ ReplayBuffer fonctionne!')
"
echo ""

# 3. Test fichiers motion capture
echo "[3/3] Vérification des données motion capture..."
if [ -d "/workspace/assets/amp/motions" ]; then
    NUM_FILES=$(find /workspace/assets/amp/motions -name "*.npy" | wc -l)
    if [ $NUM_FILES -gt 0 ]; then
        echo "✅ $NUM_FILES fichiers motion capture trouvés"
    else
        echo "⚠️  Aucun fichier .npy trouvé dans assets/amp/motions/"
    fi
else
    echo "⚠️  Dossier assets/amp/motions/ non trouvé"
fi
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "  ✅ Tests Réussis !"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Prochaines étapes:"
echo "  1. Lancer entraînement test:"
echo "     python isaaclab_envs/scripts/train.py --task HumanoidAMP --num_envs 512 --headless --max_iterations 10"
echo ""
EOF

chmod +x $TEST_SCRIPT

echo "✅ Script de test créé: $TEST_SCRIPT"
echo ""

# ==============================================================================
# Résumé
# ==============================================================================

echo "════════════════════════════════════════════════════════════════"
echo "  ✅ Installation Terminée !"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  GPU:       $GPU_NAME"
echo "  Docker:    $DOCKER_VERSION"
echo "  Image:     $ISAAC_LAB_IMAGE"
echo "  Workspace: $PROJECT_DIR"
echo "  Cache:     $CACHE_DIR"
echo ""
echo "Prochaines étapes:"
echo ""
echo "  1. Uploader votre code (si pas encore fait):"
echo "     cd $WORKSPACE_DIR"
echo "     git clone https://github.com/<votre-repo>/isaaclab-amp.git"
echo ""
echo "  2. Lancer le container Isaac Lab:"
echo "     bash $LAUNCH_SCRIPT"
echo ""
echo "  3. Dans le container, tester:"
echo "     bash /workspace/test_quick.sh"
echo ""
echo "  4. Lancer l'entraînement:"
echo "     python isaaclab_envs/scripts/train.py --task HumanoidAMP --num_envs 512 --headless --max_iterations 10"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
