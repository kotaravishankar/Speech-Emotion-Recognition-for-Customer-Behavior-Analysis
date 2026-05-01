# =============================================================================
# PASTE THIS AS A NEW CELL AT THE VERY END OF YOUR KAGGLE NOTEBOOK
# Run it once after your training cells have finished.
#
# WHAT IT DOES:
#   Saves the two StandardScaler objects that were fitted during training
#   (seq_scaler and sca_scaler) as .pkl files in Kaggle Output.
#
# WHY YOU NEED THESE:
#   Without these scalers, the app has to guess how to normalise the
#   audio features. The 7 scalar features span wildly different scales
#   (spectral_centroid ~2000, melspec_mean ~0.00005) so any guessed
#   normalisation produces garbage input to the model, which then
#   collapses to predicting "Sad" for every audio clip.
#
# STEPS AFTER RUNNING:
#   1. Look in the Kaggle Output panel (right side of screen)
#   2. Download BOTH files:  seq_scaler.pkl  and  scaler.pkl
#   3. Upload both files to your Hugging Face Space (same folder as app.py)
#   4. The Space auto-restarts and switches to accurate mode
# =============================================================================

import joblib

# seq_scaler: fitted on X_seq_final reshaped to (-1, 40) in your training cell
# sca_scaler: fitted on X_sca_final shape (-1, 7) in your training cell
# If you get NameError here, re-run the training cell first.

joblib.dump(seq_scaler, "/kaggle/working/seq_scaler.pkl")
print("Saved: seq_scaler.pkl  (MFCC branch - 40 features)")

joblib.dump(sca_scaler, "/kaggle/working/scaler.pkl")
print("Saved: scaler.pkl      (scalar branch - 7 features)")

# Quick sanity check - verify the scalers have the right number of features
print(f"\nseq_scaler expects {seq_scaler.n_features_in_} features per frame  (should be 40)")
print(f"sca_scaler expects {sca_scaler.n_features_in_} features             (should be 7)")

print("\nDone! Download both files from the Kaggle Output panel on the right.")
print("Upload to your Hugging Face Space alongside app.py and the .h5 model.")
