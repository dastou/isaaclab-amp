# 🚀 Guide RunPod - Déploiement Isaac Lab AMP

**Date**: 2025-11-09
**GPU**: A100 (80GB)
**Objectif**: Tester la migration Isaac Lab sur RunPod

---

## 📋 Ce que nous avons découvert

### ✅ RunPod est PARFAIT pour notre cas

1. **GPU A100 déjà configuré** ✅
   - Drivers NVIDIA pré-installés
   - CUDA Toolkit inclus
   - nvidia-docker automatiquement configuré

2. **Templates PyTorch prêts à l'emploi** ✅
   - PyTorch 2.4 + CUDA 12.4 disponible
   - Python 3.10/3.11 pré-installé
   - Pas de setup manuel nécessaire

3. **Isaac Lab supporte Docker** ✅
   - Image officielle : `nvcr.io/nvidia/isaac-lab:2.3.0`
   - Mode headless parfait pour serveur
   - Compatible A100

---

## 🎯 Plan d'Action (3 Étapes)

### **Étape 1**: Préparer le Pod RunPod (5 min)
### **Étape 2**: Installer Isaac Lab via Docker (10-15 min)
### **Étape 3**: Tester l'entraînement AMP (5-10 min)

**Temps total estimé**: 20-30 minutes

---

## 📦 Étape 1: Préparer le Pod RunPod

### A. Créer le Pod

1. **Connecte-toi sur RunPod** : https://www.runpod.io/

2. **Sélectionne le GPU**:
   - GPU: **A100 (80GB)** ✅ (tu l'as déjà)
   - Type: Pod

3. **Choisis le Template**:
   - **Option A** (Recommandée): `RunPod PyTorch 2.4`
   - **Option B**: `RunPod PyTorch` (dernière version)
   - **Option C**: Template Docker custom (on créera une image)

4. **Configure le Pod**:
   - **Container Disk**: 50GB minimum (Isaac Lab est ~10-15GB)
   - **Volume Disk** (optionnel): 20GB pour logs/checkpoints
   - **Expose Ports**: Pas nécessaire pour headless

5. **Démarre le Pod** → Attends qu'il soit "Running"

### B. Connecte-toi au Pod

**Option 1: Terminal Web RunPod** (Plus simple)
- Clique sur "Connect" → "Start Web Terminal"
- Terminal s'ouvre dans le navigateur

**Option 2: SSH** (Plus pro)
```bash
# Récupère la commande SSH depuis RunPod dashboard
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
```

### C. Vérifier l'Environnement

```bash
# 1. Vérifier GPU
nvidia-smi

# Doit afficher:
# - Tesla A100-SXM4-80GB
# - CUDA Version: 12.x
# - Driver Version: 535.x+

# 2. Vérifier Docker
docker --version
# Docker version 26.x ou plus

# 3. Vérifier nvidia-docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
# Doit afficher les infos GPU depuis le container
```

✅ **Si tout fonctionne, passe à l'Étape 2**

---

## 🐳 Étape 2: Installer Isaac Lab

### Option A: Docker Pull (Recommandé - Plus rapide)

```bash
# 1. Pull l'image officielle Isaac Lab
docker pull nvcr.io/nvidia/isaac-lab:2.3.0

# Temps: ~5-10 min (dépend de la connexion)
# Taille: ~10-15 GB

# 2. Créer un dossier pour le code
mkdir -p ~/isaac-lab-workspace
cd ~/isaac-lab-workspace

# 3. Uploader ton code (voir section Transfert Code ci-dessous)
```

### Option B: Build depuis Source (Plus long mais plus flexible)

```bash
# 1. Cloner Isaac Lab
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 2. Build l'image Docker
./docker/container.py start

# Temps: ~15-20 min
```

---

## 📤 Transfert du Code vers RunPod

### Méthode 1: Git (Recommandée)

**Sur ton PC Windows** (si pas déjà fait):
```bash
cd C:\Users\HP\Desktop\DIC2\DeepLearning\projet_matiere_dl\IsaacGymEnvs

# Initialiser Git
git init
git add .
git commit -m "Migration Isaac Lab complète"

# Push vers GitHub
git remote add origin https://github.com/<ton-username>/isaaclab-amp.git
git push -u origin main
```

**Sur RunPod**:
```bash
cd ~/isaac-lab-workspace
git clone https://github.com/<ton-username>/isaaclab-amp.git
cd isaaclab-amp
```

### Méthode 2: Upload Direct RunPod

1. **Zipper le projet** sur Windows:
   - Clique droit sur `IsaacGymEnvs` → "Compress to ZIP"

2. **Upload via RunPod**:
   - Dans le Web Terminal, clique sur "Upload" (icône en haut)
   - Sélectionne le fichier ZIP
   - Dézippe: `unzip IsaacGymEnvs.zip`

### Méthode 3: SCP (Si SSH configuré)

```bash
# Depuis Windows (PowerShell)
scp -P <runpod-port> -r IsaacGymEnvs root@<runpod-ip>:~/
```

---

## 🏃 Étape 3: Lancer Isaac Lab et Tester

### A. Démarrer le Container Isaac Lab

```bash
cd ~/isaac-lab-workspace/isaaclab-amp  # ou ton dossier

# Lancer le container avec ton code monté
docker run --name isaac-lab-amp \
  --gpus all \
  -it \
  --rm \
  --network=host \
  -e "ACCEPT_EULA=Y" \
  -e "PRIVACY_CONSENT=Y" \
  -v $(pwd):/workspace \
  -v ~/isaac-lab-cache:/isaac-sim/kit/cache:rw \
  nvcr.io/nvidia/isaac-lab:2.3.0 \
  /bin/bash
```

**Explication des options**:
- `--gpus all` : Donne accès au GPU A100
- `-v $(pwd):/workspace` : Monte ton code dans le container
- `-v ~/isaac-lab-cache:...` : Cache pour accélérer les démarrages
- `--rm` : Supprime le container à la sortie (propre)
- `--network=host` : Réseau partagé (optionnel)

### B. Dans le Container - Installer les Dépendances

```bash
# Tu es maintenant DANS le container Isaac Lab

# 1. Aller dans le workspace
cd /workspace

# 2. Installer RL-Games
pip install rl-games

# 3. Installer dépendances supplémentaires
pip install tensorboardX

# 4. Vérifier les imports (Test rapide)
python -c "from isaaclab_envs.learning import AMPAgent; print('✅ Imports OK!')"
```

### C. Lancer l'Entraînement de Test

```bash
# Test court: 512 env, 10 iterations, headless
python isaaclab_envs/scripts/train.py \
    --task HumanoidAMP \
    --num_envs 512 \
    --headless \
    --max_iterations 10 \
    --seed 42
```

**Ce qui va se passer**:
1. Création de 512 environnements humanoid
2. Chargement des données motion capture
3. Initialisation agent AMP
4. 10 itérations d'entraînement
5. Sauvegarde checkpoint

**Durée estimée**: 5-10 minutes

**Si ça marche**: 🎉 **Migration réussie !**

---

## 📊 Monitoring pendant l'Entraînement

### Depuis un autre Terminal RunPod

```bash
# Ouvrir un 2ème terminal

# Surveiller GPU
watch -n 1 nvidia-smi

# Surveiller les logs du container
docker logs -f isaac-lab-amp
```

### Métriques à observer

- **GPU Utilization**: Devrait être ~80-100%
- **Memory Used**: ~10-20 GB (sur 80 GB A100)
- **Power Draw**: ~300-400W
- **Temperature**: ~60-80°C

---

## 🐛 Troubleshooting

### Erreur: "No module named 'omni'"

**Cause**: Isaac Lab pas correctement installé dans le container

**Solution**:
```bash
# Utilise l'image officielle exactement comme indiqué
docker pull nvcr.io/nvidia/isaac-lab:2.3.0
```

### Erreur: "CUDA out of memory"

**Cause**: Trop d'environnements pour le GPU

**Solution**:
```bash
# Réduire num_envs
python isaaclab_envs/scripts/train.py --num_envs 256 --headless
```

### Erreur: "Failed to create simulation"

**Cause**: Mode headless pas activé correctement

**Solution**:
```bash
# Vérifier que --headless est bien présent
# ET que DISPLAY n'est pas défini
unset DISPLAY
python isaaclab_envs/scripts/train.py --headless ...
```

### Erreur: "Motion files not found"

**Cause**: Données motion capture pas transférées

**Solution**:
```bash
# Vérifier présence des fichiers
ls -lh /workspace/assets/amp/motions/

# Si vide, re-upload le dossier assets/
```

### Container trop lent au démarrage

**Cause**: Pas de cache

**Solution**:
```bash
# Créer le dossier cache
mkdir -p ~/isaac-lab-cache

# Utiliser -v pour monter le cache (déjà dans la commande docker run)
```

---

## ⚡ Optimisations RunPod

### 1. Utiliser un Volume Persistant

```bash
# Créer un volume sur RunPod (via dashboard)
# Taille: 50GB
# Type: Network Volume

# Monter dans le container
docker run ... -v /runpod-volume:/data ...
```

**Avantages**:
- Logs/checkpoints persistent entre redémarrages
- Pas de perte de données

### 2. Augmenter le Batch Size

```bash
# Avec A100 80GB, tu peux augmenter
python isaaclab_envs/scripts/train.py \
    --num_envs 4096 \  # Au lieu de 512
    --headless
```

**Performance**: ~2-3x plus rapide

### 3. Multi-GPU (si pod avec plusieurs A100)

```bash
# Vérifier nombre de GPUs
nvidia-smi --list-gpus

# Utiliser tous les GPUs
docker run --gpus all ...

# Dans train.py, RL-Games détectera automatiquement
```

---

## 💰 Estimation Coûts RunPod

### A100 80GB (Secure Cloud)
- **Prix**: ~$1.89/heure
- **Test (30 min)**: ~$0.95
- **Entraînement court (2h)**: ~$3.80
- **Entraînement complet (10h)**: ~$19

### A100 80GB (Community Cloud - Moins cher)
- **Prix**: ~$0.80-1.20/heure
- **Test (30 min)**: ~$0.40-0.60
- **Entraînement court (2h)**: ~$1.60-2.40

**Recommandation**: Utilise Community Cloud pour les tests

---

## 📝 Checklist Complète

### Avant de Commencer
- [ ] Compte RunPod créé
- [ ] Crédits ajoutés (~$5-10 pour tests)
- [ ] Code uploadé (Git ou ZIP)
- [ ] GPU A100 sélectionné

### Setup
- [ ] Pod démarré et "Running"
- [ ] Terminal connecté
- [ ] `nvidia-smi` fonctionne
- [ ] Docker installé et fonctionnel

### Installation
- [ ] Image Isaac Lab pullée
- [ ] Code transféré dans ~/isaac-lab-workspace
- [ ] Container démarré
- [ ] RL-Games installé
- [ ] Imports testés avec succès

### Test
- [ ] Entraînement court lancé (10 iterations)
- [ ] Pas d'erreurs
- [ ] Checkpoint sauvegardé
- [ ] Logs affichés correctement

### Validation
- [ ] GPU utilisé à ~80-100%
- [ ] Discriminateur fonctionne
- [ ] Récompenses calculées
- [ ] ✅ **MIGRATION VALIDÉE !**

---

## 🎉 Prochaines Étapes après Validation

Si le test fonctionne:

1. **Entraînement complet** (optionnel):
   ```bash
   python isaaclab_envs/scripts/train.py \
       --task HumanoidAMP \
       --num_envs 4096 \
       --headless \
       --max_iterations 5000
   ```

2. **Sauvegarder les résultats**:
   ```bash
   # Depuis le container
   tar -czf results.tar.gz runs/ logs/

   # Depuis RunPod (autre terminal)
   docker cp isaac-lab-amp:/workspace/results.tar.gz ~/

   # Download via RunPod dashboard
   ```

3. **Évaluation du modèle**:
   ```bash
   python isaaclab_envs/scripts/train.py \
       --task HumanoidAMP \
       --checkpoint runs/HumanoidAMP/model.pth \
       --play
   ```

---

## 📚 Ressources Utiles

### Documentation
- [Isaac Lab Docker Guide](https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html)
- [RunPod Documentation](https://docs.runpod.io/)
- [RL-Games GitHub](https://github.com/Denys88/rl_games)

### Support
- RunPod Discord: https://discord.gg/runpod
- Isaac Lab GitHub Issues: https://github.com/isaac-sim/IsaacLab/issues

### Notre Documentation
- `README.md` - Documentation principale
- `ETAT_FINAL_PROJET.md` - État complet du projet
- `RESUME_MIGRATION.md` - Résumé migration

---

## 🎯 Résumé des Commandes Clés

```bash
# ========================================
# SETUP INITIAL (une seule fois)
# ========================================

# 1. Pull image Isaac Lab
docker pull nvcr.io/nvidia/isaac-lab:2.3.0

# 2. Créer workspace et uploader code
mkdir -p ~/isaac-lab-workspace
cd ~/isaac-lab-workspace
git clone https://github.com/<ton-repo>.git
cd <ton-repo>

# ========================================
# LANCER LE CONTAINER
# ========================================

docker run --name isaac-lab-amp \
  --gpus all -it --rm --network=host \
  -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" \
  -v $(pwd):/workspace \
  -v ~/isaac-lab-cache:/isaac-sim/kit/cache:rw \
  nvcr.io/nvidia/isaac-lab:2.3.0 /bin/bash

# ========================================
# DANS LE CONTAINER
# ========================================

# Installer dépendances
pip install rl-games tensorboardX

# Test imports
python -c "from isaaclab_envs.learning import AMPAgent; print('✅ OK!')"

# Lancer entraînement test
python isaaclab_envs/scripts/train.py \
    --task HumanoidAMP \
    --num_envs 512 \
    --headless \
    --max_iterations 10
```

---

**Dernière mise à jour**: 2025-11-09
**Testé sur**: RunPod A100 80GB
**Statut**: Prêt à tester 🚀
