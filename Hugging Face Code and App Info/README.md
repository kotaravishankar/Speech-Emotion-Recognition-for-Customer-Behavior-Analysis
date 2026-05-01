---
title: Speech Emotion Recognition
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.34.2
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# 🎙️ Speech Emotion Recognition — CNN-BiLSTM

A deep-learning app that classifies speech audio into one of five emotions:
**Angry · Fear · Happy · Neutral · Sad**

## Model

**Architecture:** CNN-BiLSTM Hybrid (dual-input)
- Sequential branch: 1-D CNN → Bidirectional LSTM, fed a `(128, 40)` MFCC matrix
- Scalar branch: Dense network, fed 7 hand-crafted audio features
- Outputs concatenated → softmax over 5 emotions

**Dataset:** [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) (~6,000 utterances)

## Files

| File | Required | Purpose |
|------|:--------:|---------|
| `app.py` | ✅ | Gradio backend |
| `requirements.txt` | ✅ | Python dependencies |
| `CNN_bilstm_hybrid_model.h5` | ✅ | Trained Keras model |
| `seq_scaler.pkl` | ⭐ | StandardScaler for MFCC branch (improves accuracy) |
| `scaler.pkl` | ⭐ | StandardScaler for scalar branch (improves accuracy) |

If the `.pkl` scalers are missing, the app falls back to per-sample normalisation.
Predictions will still vary correctly but accuracy will be lower.
