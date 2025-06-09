# 🦾 YOLOv8 Video Pre-Annotation Tool (Dockerized)

Ce projet permet de **pré-annoter automatiquement des vidéos** à l'aide de **YOLOv8**, et de générer des **images + fichiers de labels au format YOLO**, compatibles avec LabelImg ou un entraînement.

---

## 📦 Fonctionnalités

- 🔍 Détection automatique d'objets avec [YOLOv8](https://github.com/ultralytics/ultralytics)
- 🎥 Traitement de plusieurs vidéos
- 🖼️ Affichage en direct des frames annotées (optionnel)
- 💾 Génération uniquement si objets détectés
- 🐳 Totalement dockerisé (via `docker-compose`)
- ✅ Export des annotations au **format YOLO**

---

## 🗂️ Structure du projet



ia-labelling/
├── app/
│ ├── process.py # Script principal
│ └── requirements.txt
├── videos/ # Vidéos à traiter (.mp4, .avi, etc.)
├── output/
│ ├── images/ # Frames extraites avec objets
│ └── labels/ # Labels .txt (format YOLO)
├── Dockerfile
├── docker-compose.yml
├── run.sh # Script de lancement Linux/macOS


---

## 🚀 Lancement rapide

### 1. Préparer les vidéos

Place tes vidéos dans le dossier `videos/`.

### 2. Construire l'image Docker

```bash
docker-compose up --build
```

---

# Projet renommé : ia-video-labeller
