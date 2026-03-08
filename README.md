<div align="center">

# 🧠 EmotienSense — YOLOv8

**Reconnaissance faciale des émotions en temps réel, propulsée par YOLOv8 & Streamlit**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00CFCF?logo=data:image/svg+xml;base64,)](https://github.com/ultralytics/ultralytics)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Licence MIT](https://img.shields.io/badge/Licence-MIT-green)](LICENSE)

</div>

---

## 📋 Table des matières

- [À propos du projet](#-à-propos-du-projet)
- [Fonctionnalités](#-fonctionnalités)
- [Émotions détectées](#-émotions-détectées)
- [Architecture technique](#-architecture-technique)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Structure du projet](#-structure-du-projet)
- [Contribuer](#-contribuer)
- [Licence](#-licence)

---

## 🔍 À propos du projet

**EmotienSense** est une application web de reconnaissance faciale des émotions basée sur le modèle de détection d'objets **YOLOv8**. Elle permet d'analyser des images de visages et d'identifier en temps réel l'émotion dominante parmi sept états affectifs fondamentaux, avec affichage des probabilités pour chaque classe.

L'interface, construite avec **Streamlit**, propose un design moderne et épuré, disponible en thème clair ou sombre, et supporte l'analyse d'images individuelles ou en lot.

---

## ✨ Fonctionnalités

| Fonctionnalité | Description |
|---|---|
| 🖼️ **Analyse d'image unique** | Chargez une image et obtenez instantanément l'émotion détectée avec son niveau de confiance |
| 📁 **Analyse en lot** | Traitez plusieurs images simultanément et consultez les résultats côte à côte |
| 📊 **Distribution des probabilités** | Visualisation graphique des scores de confiance pour les 7 émotions |
| 🌙 **Thème sombre / ☀️ thème clair** | Interface adaptable selon les préférences de l'utilisateur |
| ⚡ **Inférence rapide** | Modèle optimisé ONNX pour des performances élevées |
| 🧠 **Mise en cache du modèle** | Chargement unique du modèle en mémoire via `@st.cache_resource` |

---

## 😶 Émotions détectées

Le modèle reconnaît **7 émotions fondamentales** basées sur le modèle d'Ekman :

| Émotion | Emoji | Description |
|---|---|---|
| **Surprise** | 😲 | Stimulus inattendu |
| **Peur** | 😨 | Menace perçue |
| **Dégoût** | 🤢 | Réponse aversive |
| **Joie** | 😄 | Affect positif |
| **Tristesse** | 😢 | Affect négatif |
| **Colère** | 😠 | Forte activation négative |
| **Neutre** | 😐 | Aucun affect prononcé |

---

## 🏗️ Architecture technique

```
Image d'entrée (PIL)
        │
        ▼
┌───────────────────┐
│  Prétraitement    │  Redimensionnement à 64×64 px
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Modèle YOLOv8    │  Classification (best.pt / best.onnx)
│  (Ultralytics)    │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Post-traitement  │  Extraction top-1, distribution des probabilités
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Interface        │  Affichage Streamlit (carte résultat + barres)
│  Streamlit        │
└───────────────────┘
```

- **Modèle** : YOLOv8 (classification), entraîné sur des images faciales 64×64 px
- **Format** : `.pt` (PyTorch) ou `.onnx` (ONNX Runtime) — le modèle `.pt` est prioritaire
- **Framework UI** : Streamlit avec CSS personnalisé (polices Google Fonts : Syne, DM Sans)

---

## 📦 Prérequis

- **Python** ≥ 3.9
- **pip** (gestionnaire de paquets Python)
- Les bibliothèques système suivantes (Linux) :
  - `libgl1`
  - `libglib2.0-0`

---

## 🚀 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/Obelisk999/EmotienSense-YOLOv8.git
cd EmotienSense-YOLOv8
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# ou
.venv\Scripts\activate         # Windows
```

### 3. Installer les dépendances système (Linux uniquement)

```bash
sudo apt-get install -y libgl1 libglib2.0-0
```

### 4. Installer les dépendances Python

```bash
pip install -r requirements.txt
```

---

## 🎮 Utilisation

### Lancer l'application

```bash
streamlit run streamlit-app.py
```

L'application s'ouvre automatiquement dans votre navigateur à l'adresse `http://localhost:8501`.

### Guide d'utilisation

1. **Sélectionnez le mode d'analyse** dans la barre latérale :
   - *Image unique* — pour analyser une seule photo
   - *Analyse en lot* — pour traiter plusieurs images à la fois

2. **Chargez vos images** via le composant de téléversement (formats supportés : JPG, JPEG, PNG, BMP, WEBP)

3. **Consultez les résultats** :
   - L'émotion dominante avec son score de confiance
   - La distribution complète des probabilités pour les 7 émotions

4. **Changez le thème** (🌙 Sombre / ☀️ Clair) depuis la barre latérale à tout moment

---

## 📁 Structure du projet

```
EmotienSense-YOLOv8/
│
├── streamlit-app.py      # Application principale Streamlit
├── best.pt               # Poids du modèle YOLOv8 (PyTorch)
├── best.onnx             # Poids du modèle exporté en ONNX
├── requirements.txt      # Dépendances Python
├── packages.txt          # Dépendances système (Streamlit Cloud)
└── README.md             # Documentation du projet
```

---

## 🤝 Contribuer

Les contributions sont les bienvenues ! Pour proposer une amélioration :

1. **Forkez** le dépôt
2. **Créez** une branche pour votre fonctionnalité : `git checkout -b feature/ma-fonctionnalite`
3. **Commitez** vos changements : `git commit -m "feat: ajout de ma fonctionnalité"`
4. **Poussez** votre branche : `git push origin feature/ma-fonctionnalite`
5. **Ouvrez** une Pull Request

Merci de respecter les conventions de nommage et d'ajouter des tests si nécessaire.

---

## 📄 Licence

Ce projet est distribué sous licence **MIT**. Voir le fichier [LICENSE](LICENSE) pour plus d'informations.

---

<div align="center">

Fait avec ❤️ par [Obelisk999](https://github.com/Obelisk999)

</div>