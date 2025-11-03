# 🏠 HomeSecurity – AI-Powered Smart Surveillance

This project uses **YOLOv8 + FaceNet + ResNet50** for real-time
face and body recognition. It identifies family members (e.g., Mom)
and alerts via Telegram if unknown people enter the camera frame.

## 📂 Folder Structure

scripts/ → main detection scripts
data/models/ → PKL centroids (not pushed)
models/ → YOLO weights (not pushed)
.env → Telegram credentials

## 🚀 Run Locally

python scripts/test_body_signature_on_video.py

## 💡 Features

Person detection (YOLOv8)

Face recognition (custom YOLO + Facenet)

Telegram alert system

Raspberry Pi 5 ready
