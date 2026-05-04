# 😊 Détection des Émotions Faciales en Temps Réel — 4 Modèles ML

Système de reconnaissance d'émotions faciales en temps réel comparant 4 architectures de Machine Learning : **InceptionV3**, **CNN V2**, **SVM** et **DenseNet**. Le projet inclut une interface web interactive avec détection via caméra.
<img width="349" height="221" alt="Screenshot 2026-05-04 213659" src="https://github.com/user-attachments/assets/68665922-92d7-44f7-b818-3ae36548eb6f" />

<img width="378" height="237" alt="Screenshot 2026-05-04 213714" src="https://github.com/user-attachments/assets/b2576ff0-4dfc-45bd-ad28-b15324153600" />

<img width="390" height="238" alt="Screenshot 2026-05-04 213727" src="https://github.com/user-attachments/assets/485da3f4-9803-4bf4-b60f-9e15c7ed29d9" />


##  Fonctionnalités

-  **Détection en Temps Réel** — Analyse des émotions via flux vidéo caméra
-  **4 Modèles Comparés** — InceptionV3, CNN V2, SVM (RBF), DenseNet121
-  **7 Émotions** — Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
-  **Scores de Confiance** — Probabilités affichées pour chaque émotion
-  **Sélection de Modèle** — Switch dynamique entre les 4 architectures
-  **Visualisations** — Matrices de confusion, courbes ROC/AUC, courbes d'apprentissage

##  Architecture du Pipeline

┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│   RAF-DB    │────▶│ Prétraitement│────▶│   Modèles   │────▶│  Interface  │
│  Dataset    │     │ (normalisation│     │  ML/DL      │     │   Web       │
│  15K images │     │ augmentation) │     │ 4 approches │     │ Temps réel  │
└─────────────┘     └──────────────┘     └─────────────┘     └─────────────┘


### Étapes détaillées :

| Étape | Description |
|-------|-------------|
| **Dataset** | RAF-DB (Real-world Affective Faces Database) — ~15,000 images |
| **Prétraitement** | Redimensionnement 128×128, normalisation, augmentation (rotation, zoom, flip) |
| **Équilibrage** | Undersampling classe majoritaire + oversampling classes minoritaires |
| **Entraînement** | 4 modèles entraînés et évalués sur les mêmes données |
| **Déploiement** | Interface web avec OpenCV pour capture vidéo en temps réel |

##  Modèles Implémentés

### Comparaison des Performances

| Modèle | Précision Test | AUC Moyenne | Temps d'Entraînement | Paramètres |
|--------|---------------|-------------|---------------------|------------|
| **InceptionV3** ⭐ | **85.59%** | **0.97-0.99** | ~252 min | ~22.9M |
| CNN V2 | 80.62% | 0.95-1.00 | ~240 min | ~4.9M |
| SVM (RBF) | 75.78% | 0.87-0.97 | ~17 min | ~10K |
| DenseNet121 | 73.01% | 0.91-0.98 | ~280 min | ~7.6M |

> **InceptionV3 retenu comme modèle optimal** — meilleur compromis précision/robustesse avec excellentes métriques AUC (0.97-0.99).

### Détails par Modèle

####  InceptionV3 (Transfer Learning)
- Base pré-entraînée sur ImageNet
- Couches personnalisées : Dropout 20% → Flatten → Dense 128 → Dense 128 → Dense 7 (softmax)
- Fine-tuning sur les dernières couches
- **AUC par classe** : Surprise (0.99), Fear (0.99), Happy (0.99), Angry (0.99), Disgust (0.97), Neutral (0.97), Sad (0.96)

####  CNN V2 (Architecture Personnalisée)
- 4 blocs Conv2D + MaxPooling (32 → 64 → 128 → 512 filtres)
- Flatten → Dense 512 → Dropout 50% → Dense 7 (softmax)
- Architecture légère et rapide à déployer

####  SVM avec Noyau RBF
- Extraction de features par pixels aplatis
- StandardScaler pour normalisation
- Comparaison des noyaux : Linear (60.31%), **RBF (75.78%)**, Polynomial (71.71%)

#### DenseNet121 (Connexions Denses)
- Base pré-entraînée sur ImageNet
- GlobalAveragePooling2D → Dense 512 → Dropout 40% → Dense 7 (softmax)
- Connexions denses entre couches pour éviter le vanishing gradient

##  Résultats Détaillés

### Matrice de Confusion — InceptionV3
<img width="519" height="438" alt="image" src="https://github.com/user-attachments/assets/71265e0c-69df-4163-8d98-dce1310161ba" />


**Observations** :
- Excellente reconnaissance : Happy (410/410), Neutral (402/402), Angry (359/359)
- Confusions mineures : Sad ↔ Neutral (99 cas), Disgust ↔ Happy (17 cas)

### Courbes ROC
<img width="520" height="405" alt="image" src="https://github.com/user-attachments/assets/cd459a1d-52e6-4347-bedb-bfd3f7a105ac" />


Toutes les courbes ROC s'élèvent rapidement vers 1.0, confirmant la qualité discriminante du modèle.

##  Installation & Utilisation

### Prérequis
- Python 3.8+
- Webcam fonctionnelle

### Installation

```bash
# Cloner le repository
git clone https://github.com/TON_USERNAME/emotion-detection-4models.git
cd emotion-detection-4models

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
