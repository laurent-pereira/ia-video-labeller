#!/bin/bash
echo "🖥️ Autorisation X11 (nécessaire pour l'affichage temps réel)..."
xhost +local:docker
echo "🚀 Lancement du traitement YOLO..."
docker-compose up
echo "🧹 Nettoyage..."
xhost -local:docker