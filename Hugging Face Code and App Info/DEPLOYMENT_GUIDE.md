# 🚀 Step-by-Step: Deploy Speech Emotion Recognition on Hugging Face Spaces

---

## Prerequisites

- A free [Hugging Face](https://huggingface.co) account
- Git installed on your computer (`git --version` to check)
- Your trained model file: `emotion_cnn_bilstm_hybrid_model.h5`

---

## STEP 0 — Export the Scaler from Your Training Notebook

Before anything else, go back to your Kaggle notebook and **add the export cell**
provided in `export_scaler_snippet.py`. Run it to produce two files:

| File | Purpose |
|------|---------|
| `emotion_cnn_bilstm_hybrid_model.h5` | Trained CNN-BiLSTM model |
| `scaler.pkl` | Fitted StandardScaler for the scalar feature branch |

Download both from Kaggle Output → your local machine.

---

## STEP 1 — Create a New Hugging Face Space

1. Go to [https://huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in the form:
   - **Space name:** `speech-emotion-recognition` (or any name you like)
   - **License:** MIT
   - **SDK:** Select **Gradio**
   - **Visibility:** Public (required for free tier GPU/CPU)
3. Click **Create Space**.

You will land on an empty Space page.

---

## STEP 2 — Clone the Space Locally

Open your terminal and run:

```bash
# Replace YOUR_USERNAME with your HF username and YOUR_SPACE_NAME with what you chose
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME
```

You should see an empty folder (or one with a stub README.md).

---

## STEP 3 — Copy Your App Files Into the Folder

Copy everything from the `ser_app/` folder you received into the cloned repo:

```
YOUR_SPACE_NAME/
├── app.py                                  ← Gradio backend
├── requirements.txt                        ← Python dependencies
├── README.md                               ← Space metadata card
├── emotion_cnn_bilstm_hybrid_model.h5      ← Your trained model (from Kaggle)
└── scaler.pkl                              ← Exported scaler (from Kaggle)
```

> ⚠️ **The .h5 file is large (~100–300 MB).** If it exceeds 100 MB you must use
> Git LFS (see Step 3b below).

### Step 3b — Enable Git LFS for large files (required if model > 100 MB)

```bash
# Install Git LFS once
git lfs install

# Track .h5 and .pkl files
git lfs track "*.h5"
git lfs track "*.pkl"

# This creates a .gitattributes file — add it to git
git add .gitattributes
```

---

## STEP 4 — Verify `app.py` Feature Extraction Matches Your Notebook

Open `app.py` and confirm these constants match your training code exactly:

```python
N_MFCC    = 40        # matches N_MFCC in your notebook
N_FRAMES  = 128       # matches N_FRAMES in your notebook
SR        = 22050     # matches the sample rate used by librosa.load()
HOP_LEN   = 512       # matches hop_length in librosa calls
EMOTIONS  = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']  # must match LabelEncoder order
```

Also double-check the **scalar feature list** in `extract_scalar_features()`.
The output vector length must exactly match what the model was trained on.
Run the sanity-check cell in `export_scaler_snippet.py` to get the exact length.

---

## STEP 5 — Commit and Push

```bash
git add .
git commit -m "Initial deployment: CNN-BiLSTM Speech Emotion Recognition"
git push
```

If prompted for credentials, use your **Hugging Face username** and a
**User Access Token** (create one at Settings → Access Tokens → New Token,
role = Write).

---

## STEP 6 — Watch the Build

1. Go to your Space URL:
   `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`
2. Click the **Logs** tab to watch the build in real time.
3. Hugging Face will:
   - Install everything in `requirements.txt`
   - Start `app.py` (which loads the model)
   - Expose the Gradio UI publicly

Build time is usually **3–8 minutes** on first run.

---

## STEP 7 — Test the Live App

Once the Space shows **"Running"**:

1. Upload a short WAV file (2–10 seconds of speech).
2. Click **Predict Emotion 🔍**.
3. You should see a probability bar chart and the top predicted emotion.

To test with CREMA-D samples, download a few `.wav` files from
[the dataset](https://github.com/CheyneyComputerScience/CREMA-D) and use them.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Build fails with `ModuleNotFoundError` | Check `requirements.txt` — pin the version that matches your notebook |
| `ValueError: Input 0 is incompatible with layer` | Feature vector length mismatch — re-run the sanity-check snippet |
| `OSError: Unable to open file` | Model file not pushed — check Git LFS is set up |
| Predictions are random / wrong | Scaler not applied or EMOTIONS list order is wrong |
| Space stays "Building" > 15 min | Free CPU tier is slow; check Logs for the actual error |

---

## Sharing the Link for Your Report

Once live, your Space URL (e.g. `https://huggingface.co/spaces/yourname/speech-emotion-recognition`)
is the **code/demo link** you paste into your assessment report.

Include in your report:
- The Space URL
- A screenshot of the running app
- Your GitHub repo link (optional but recommended) for the notebook

---

## Optional Extras (for higher marks)

- **Add example audio files** to `gr.Examples([...])` in `app.py` so markers can test immediately
- **Display a confusion matrix image** as a static figure in the Gradio layout
- **Show MFCC spectrogram** of the uploaded audio alongside the prediction
- **Add a `requirements.txt` pin test** in your notebook to verify reproducibility
