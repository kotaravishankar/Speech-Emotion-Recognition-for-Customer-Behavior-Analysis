"""
Speech Emotion Recognition — Hugging Face Space
Model  : CNN-BiLSTM Hybrid (confirmed input order: seq first, sca second)
Dataset: CREMA-D  →  Angry · Fear · Happy · Neutral · Sad

FIX APPLIED:
  The scaler mean/scale values are now hardcoded directly from the real
  trained pkl files. This eliminates the pickle security warning entirely
  while keeping 100% accurate normalisation — identical to using the pkl.

  Hardcoded values come from:
    scaler.pkl     — StandardScaler fitted on 24,595 samples, 7 features
    seq_scaler.pkl — StandardScaler fitted on 3,148,160 samples, 40 features
"""

import os
import numpy as np
import librosa
import gradio as gr
from tensorflow.keras.models import load_model

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ── Constants (must match training notebook exactly) ──────────────────────────
N_MFCC   = 40
N_FRAMES = 128
SR       = 22050

EMOTIONS = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']
EMOJI    = {'Angry': '😠', 'Fear': '😨', 'Happy': '😊', 'Neutral': '😐', 'Sad': '😢'}

# ── Scaler parameters (extracted directly from the trained pkl files) ─────────
# These are the EXACT mean_ and scale_ arrays from your Kaggle-trained scalers.
# Using them here is mathematically identical to calling scaler.transform(),
# with zero pickle imports and zero security warnings.

# scaler.pkl — 7 scalar audio features
# Feature order: [zcr, rms, melspec_mean, spectral_centroid,
#                 spectral_bandwidth, chroma_mean, chroma_std]
_SCA_MEAN = np.array([
    9.95149388e-02,  # zcr
    1.69842356e-02,  # rms
    1.69464677e-01,  # melspec_mean
    1.89848583e+03,  # spectral_centroid
    2.04184920e+03,  # spectral_bandwidth
    6.56718143e-01,  # chroma_mean
    2.27545863e-01,  # chroma_std
], dtype=np.float32)

_SCA_SCALE = np.array([
    9.70803944e-02,
    1.04422814e-02,
    2.38664660e-01,
    1.22236123e+03,
    6.45861361e+02,
    7.40111936e-02,
    4.11900897e-02,
], dtype=np.float32)

# seq_scaler.pkl — 40 MFCC coefficients per frame
_SEQ_MEAN = np.array([
    -3.22110894e+02,  9.42561541e+01,  8.81889711e+00,  3.62345989e+01,
    -6.06313581e+00,  1.46675522e+01, -9.31947591e+00,  5.30093597e+00,
    -7.60822701e+00,  1.45401009e+00, -1.44076521e+00, -3.10112201e+00,
     1.65849197e+00, -5.91204601e+00,  1.63091029e+00, -6.47686188e+00,
     4.09445413e-02, -5.10681920e+00, -6.56387098e-01, -2.91713071e+00,
    -2.32162703e+00, -7.45084526e-01, -2.44628949e+00,  8.61839177e-01,
    -1.93074582e+00,  1.75614709e+00, -1.59935702e+00,  1.30719663e+00,
    -8.06449533e-01,  4.80321565e-01,  5.73763560e-01,  8.61241295e-02,
     1.63028603e+00, -2.35077966e-01,  1.55734095e+00, -3.64637238e-01,
     1.76073902e+00, -4.48568675e-01,  8.31356419e-01, -2.61202307e-01,
], dtype=np.float32)

_SEQ_SCALE = np.array([
    181.14923802,  62.58547764,  27.60154388,  28.02673894,  19.07675786,
     17.64354245,  14.58522354,  10.46910847,  11.01488603,   8.23571368,
      7.79338623,   7.21633759,   7.006693  ,   7.3474257 ,   6.92391437,
      7.66064713,   6.37497265,   6.78131423,   6.10415119,   5.94061905,
      5.76756526,   5.54465896,   5.95711933,   5.9233969 ,   6.03088389,
      6.21539635,   6.17629736,   5.94770157,   5.8694749 ,   5.70837528,
      5.68895141,   5.6731644 ,   5.73522524,   5.53324397,   5.49019741,
      5.44521164,   5.50334667,   5.24724744,   5.07534764,   4.85416066,
], dtype=np.float32)

print("=" * 65)
print("Scaler parameters loaded from hardcoded training values.")
print(f"  seq_scaler : 40 MFCC features  (fitted on 3,148,160 samples)")
print(f"  sca_scaler :  7 audio features  (fitted on    24,595 samples)")
print("=" * 65)

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading CNN-BiLSTM model …")
model = load_model("CNN_bilstm_hybrid_model.h5", compile=False)
for i, inp in enumerate(model.inputs):
    print(f"  model.inputs[{i}]: name={inp.name}  shape={inp.shape}")
print("Model loaded OK")
print("=" * 65)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_sequence(y: np.ndarray) -> np.ndarray:
    """MFCC matrix (128, 40), normalised with exact training scaler."""
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)   # (40, T)
    if mfcc.shape[1] < N_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, N_FRAMES - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :N_FRAMES]
    mfcc = mfcc.T.astype(np.float32)                          # (128, 40)
    # StandardScaler: (x - mean) / scale  — identical to scaler.transform()
    return ((mfcc - _SEQ_MEAN) / _SEQ_SCALE).astype(np.float32)


def extract_scalars(y: np.ndarray) -> np.ndarray:
    """7 hand-crafted features in training order, shape (7,)."""
    stft   = np.abs(librosa.stft(y))
    chroma = librosa.feature.chroma_stft(S=stft, sr=SR)
    return np.array([
        librosa.feature.zero_crossing_rate(y=y).mean(),       # [0] zcr
        librosa.feature.rms(y=y).mean(),                      # [1] rms
        librosa.feature.melspectrogram(y=y, sr=SR).mean(),    # [2] melspec mean
        librosa.feature.spectral_centroid(y=y, sr=SR).mean(), # [3] spectral centroid
        librosa.feature.spectral_bandwidth(y=y, sr=SR).mean(),# [4] spectral bandwidth
        chroma.mean(),                                         # [5] chroma mean
        chroma.std(),                                          # [6] chroma std
    ], dtype=np.float32)


def scale_scalars(sca: np.ndarray) -> np.ndarray:
    """Normalise 7-feature vector with exact training scaler."""
    return ((sca - _SCA_MEAN) / _SCA_SCALE).astype(np.float32)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_emotion(audio_path):
    if audio_path is None:
        return {e: 0.0 for e in EMOTIONS}, "Please upload or record audio first."

    try:
        y, _ = librosa.load(audio_path, sr=SR)

        if len(y) < SR * 0.5:
            return (
                {e: 0.0 for e in EMOTIONS},
                "⚠️ Audio clip is too short. Please provide at least 1 second of speech."
            )

        seq = extract_sequence(y)          # (128, 40)  — normalised
        sca = extract_scalars(y)           # (7,)       — raw
        sca = scale_scalars(sca)           # (7,)       — normalised

        X_seq = seq[np.newaxis].astype(np.float32)   # (1, 128, 40)
        X_sca = sca[np.newaxis].astype(np.float32)   # (1, 7)

        # Input order: inputs[0] = seq (None,128,40), inputs[1] = sca (None,7)
        probs = model.predict([X_seq, X_sca], verbose=0)[0]   # (5,)

        confidence  = {EMOTIONS[i]: float(probs[i]) for i in range(5)}
        top_idx     = int(np.argmax(probs))
        top_emotion = EMOTIONS[top_idx]
        top_conf    = float(probs[top_idx]) * 100

        return confidence, f"{EMOJI[top_emotion]}  **{top_emotion}** detected  ({top_conf:.1f}% confidence)"

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {e: 0.0 for e in EMOTIONS}, f"⚠️ Error processing audio: {exc}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────

DESCRIPTION = """
## 🎙️ Speech Emotion Recognition

Upload a **WAV / MP3** clip (1–10 s of clear speech) or record from your microphone.

**Emotions detected:** Angry · Fear · Happy · Neutral · Sad

*Trained on [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)*
"""

with gr.Blocks(title="Speech Emotion Recognition", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Audio Input",
            )
            predict_btn = gr.Button("Predict Emotion 🔍", variant="primary")
        with gr.Column(scale=1):
            label_output  = gr.Label(num_top_classes=5, label="Emotion Probabilities")
            status_output = gr.Markdown()

    predict_btn.click(
        fn=predict_emotion,
        inputs=audio_input,
        outputs=[label_output, status_output],
    )

    gr.Markdown(
        "---\n"
        "**Model:** CNN-BiLSTM Hybrid &nbsp;|&nbsp; "
        "**Framework:** TensorFlow/Keras &nbsp;|&nbsp; "
        "**UI:** Gradio"
    )

if __name__ == "__main__":
    demo.launch()
