# =============================================================================
# PASTE THIS AS THE VERY LAST CELL IN YOUR KAGGLE TRAINING NOTEBOOK
# Run once after your training cells have finished.
#
# WHAT IT DOES:
#   1. Saves seq_scaler and sca_scaler (the two StandardScaler objects fitted
#      during training) as .pkl files in Kaggle Output.
#   2. Runs a full sanity check — predicts on a real CREMA-D file and tells
#      you if the output is correct before you upload anything.
#
# WHY SCALERS ARE ESSENTIAL:
#   Your model was trained on StandardScaler-normalised inputs.
#   Without them the 7 scalar features span wildly different magnitudes:
#     spectral_centroid  ≈ 2100    (40,000× larger than melspec mean ≈ 0.00005)
#   Any fallback normalisation that treats all features the same produces a
#   near-identical vector for every clip → model always predicts Sad (99.98%).
#
# WHAT TO DO AFTER RUNNING:
#   1. Look in the Kaggle Output panel (right side)
#   2. Download BOTH files:  seq_scaler.pkl  and  scaler.pkl
#   3. Upload both to your Hugging Face Space (same folder as app.py)
#   4. Space auto-restarts → check logs for "[OK] seq_scaler.pkl loaded"
# =============================================================================

import joblib
import numpy as np
import librosa

# ── 1. EXPORT SCALERS ──────────────────────────────────────────────────────────
# These two objects must already be in memory from your training cell:
#   seq_scaler  = StandardScaler fitted on X_seq_final reshaped to (-1, 40)
#   sca_scaler  = StandardScaler fitted on X_sca_final shape (-1, 7)
#
# NOTE: In some notebooks the variable is called 'scaler' not 'sca_scaler'.
#       Adjust the names below to match your actual variable names.

joblib.dump(seq_scaler, "/kaggle/working/seq_scaler.pkl")
print(f"Saved: seq_scaler.pkl  — {seq_scaler.n_features_in_} features (should be 40)")

joblib.dump(sca_scaler, "/kaggle/working/scaler.pkl")
print(f"Saved: scaler.pkl      — {sca_scaler.n_features_in_} features (should be 7)")

# ── 2. VALIDATION ─────────────────────────────────────────────────────────────
# Confirm feature counts match what app.py expects
assert seq_scaler.n_features_in_ == 40, \
    f"seq_scaler has {seq_scaler.n_features_in_} features — expected 40. Check your training cell."
assert sca_scaler.n_features_in_ == 7, \
    f"sca_scaler has {sca_scaler.n_features_in_} features — expected 7. Check your training cell."
print("\nFeature count validation: PASSED ✓")

# ── 3. SANITY CHECK — run one prediction with the exported scalers ─────────────
N_MFCC   = 40
N_FRAMES = 128
SR       = 22050
# Alphabetical order = LabelEncoder order
EMOTIONS = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']

# Use any CREMA-D file labelled ANG (Angry) for the check.
# If this exact path doesn't exist in your dataset, change it to any .wav you have.
TEST_WAV = "/kaggle/input/datasets/ejlok1/cremad/AudioWAV/1001_DFA_ANG_XX.wav"
EXPECTED = "Angry"   # ← change if you use a different file

def _extract_seq(path):
    y, _ = librosa.load(path, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    if mfcc.shape[1] < N_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, N_FRAMES - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :N_FRAMES]
    mfcc = mfcc.T.astype(np.float32)                   # (128, 40)
    return seq_scaler.transform(mfcc)[np.newaxis]       # (1, 128, 40)

def _extract_sca(path):
    y, _ = librosa.load(path, sr=SR)
    stft   = np.abs(librosa.stft(y))
    chroma = librosa.feature.chroma_stft(S=stft, sr=SR)
    features = np.array([
        librosa.feature.zero_crossing_rate(y=y).mean(),
        librosa.feature.rms(y=y).mean(),
        librosa.feature.melspectrogram(y=y, sr=SR).mean(),
        librosa.feature.spectral_centroid(y=y, sr=SR).mean(),
        librosa.feature.spectral_bandwidth(y=y, sr=SR).mean(),
        chroma.mean(),
        chroma.std(),
    ], dtype=np.float32)
    return sca_scaler.transform(features.reshape(1, -1))  # (1, 7)

print(f"\nRunning sanity check on: {TEST_WAV}")
Xs = _extract_seq(TEST_WAV)
Xf = _extract_sca(TEST_WAV)

print(f"  Sequence input shape : {Xs.shape}  ← must be (1, 128, 40)")
print(f"  Scalar input shape   : {Xf.shape}   ← must be (1, 7)")

# NOTE: In your notebook the model variable might be called cnn_lstm_model,
#       model, or hybrid_model. Adjust the name below.
probs = cnn_lstm_model.predict([Xs, Xf], verbose=0)[0]
pred  = EMOTIONS[np.argmax(probs)]
conf  = probs.max() * 100

print(f"\nPredicted : {pred}  ({conf:.1f}% confidence)")
print(f"Expected  : {EXPECTED}")

if pred == EXPECTED:
    print("\nSANITY CHECK: PASSED ✓  — scalers are working correctly")
    print("You are safe to upload seq_scaler.pkl and scaler.pkl to your Space.")
else:
    print("\nSANITY CHECK: FAILED ✗")
    print("Predicted the wrong emotion. Possible causes:")
    print("  1. Variable name mismatch — check seq_scaler / sca_scaler names above")
    print("  2. Feature extraction order — make sure the 7 features are in the same")
    print("     order here as in your training cell")
    print("  3. Wrong model variable — check the cnn_lstm_model name above")
    print("  4. Scaler fitted on wrong data — re-run training then re-run this cell")

# ── 4. DOWNLOAD REMINDER ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Files saved to Kaggle Output:")
print("  /kaggle/working/seq_scaler.pkl")
print("  /kaggle/working/scaler.pkl")
print("\nNext steps:")
print("  1. Download both files from the Kaggle Output panel (right side)")
print("  2. Upload both to your Hugging Face Space alongside app.py")
print("  3. Wait for the Space to restart (~1 min)")
print("  4. Check Space logs for '[OK] seq_scaler.pkl loaded' and")
print("     '[OK] scaler.pkl loaded'")
print("  5. Test with a WAV file — predictions should now vary correctly")
print("=" * 60)
