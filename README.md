# 🌾 AI-Powered Crop Recommendation System

This project is an AI-based Crop Recommendation System that suggests the most suitable crop to cultivate based on soil and environmental conditions like nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, pH, and rainfall.

## 🔍 Overview

- **Model**: Custom PyTorch Neural Network
- **Frontend**: Simple HTML form (`index.html`)
- **Backend API**: Python with FastAPI (`api.py`)
- **Deployment**: Hosted on [Render](https://render.com)
- **Dataset**: `Crop_recommendation.csv`

---

## 🚀 Features

- One-hot encoded crop labels
- Standardized feature scaling
- Multilayer neural network with ReLU activations
- Log-likelihood loss (CrossEntropy) for classification
- Achieves ~99% training and 97% test accuracy

---

## 🧠 How It Works

1. **User Inputs**: Soil/environment values in the frontend.
2. **Backend**: FastAPI receives data, runs inference using the trained `.pth` model.
3. **Response**: Returns predicted crop name with high confidence.

---

## 📁 Project Structure

```bash
.
├── api.py                  # FastAPI backend
├── index.html              # Frontend interface
├── crop_recommendation_model.pth  # Trained PyTorch model
├── requirements.txt        # Python dependencies
├── start.sh                # Startup script for Render
├── .vscode/settings.json   # Optional: editor settings
└── README.md               # You're reading this!
