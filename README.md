# 🔊 Real-Time Hazard Sound Detection and Alert System

A complete AI-powered project that **detects hazardous sounds in real time** using a
CNN trained on the [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)
dataset and streams live audio classification through a modern Streamlit UI.

---

## 🎯 Features

| Feature | Details |
|---|---|
| **Dataset** | UrbanSound8K (real audio, no dummy data) |
| **Classes** | `dog_bark`, `gun_shot`, `siren`, `normal` |
| **Feature Extraction** | Log-Mel Spectrogram (128 × 173) via librosa |
| **Model** | 4-block CNN with BatchNorm & Dropout (TensorFlow/Keras) |
| **Real-time input** | sounddevice microphone recording |
| **UI** | Streamlit with live confidence bars, alert banners |
| **Alert system** | Red banner + synthesised beep for `gun_shot` / `siren` |

---

## 📁 Project Structure

```
Hazard-system-/
├── app.py                    # Streamlit web application
├── train_model.py            # Training script
├── requirements.txt          # Python dependencies
├── model/
│   ├── model.h5              # Saved best model (created after training)
│   └── training_history.png  # Loss/accuracy curves (created after training)
├── utils/
│   ├── __init__.py
│   ├── feature_extraction.py # Log-Mel Spectrogram & MFCC extraction
│   ├── dataset_loader.py     # UrbanSound8K loader + label encoding
│   └── alert.py              # Synthesised alert sound
└── dataset/
    └── UrbanSound8K/         # ← place dataset here (see below)
        ├── audio/
        │   ├── fold1/
        │   │   └── *.wav
        │   └── … fold10/
        └── metadata/
            └── UrbanSound8K.csv
```

---

## ⚙️ Setup

### 1 · Install Python dependencies

```bash
pip install -r requirements.txt
```

> Tested with Python 3.9 – 3.11.  TensorFlow 2.12+ is required.

### 2 · Download the UrbanSound8K dataset

1. Visit <https://urbansounddataset.weebly.com/urbansound8k.html> and fill in the
   short registration form.
2. Download the archive and extract it.
3. Move / copy the extracted folder so it matches the structure shown above:

   ```
   dataset/
   └── UrbanSound8K/
       ├── audio/fold1/ … fold10/
       └── metadata/UrbanSound8K.csv
   ```

### 3 · Train the model

```bash
python train_model.py
```

The script will:
- Load and filter the dataset (dog_bark, gun_shot, siren + normal class)
- Extract Log-Mel Spectrograms for every clip
- Train the CNN with EarlyStopping and ModelCheckpoint
- Save the best weights to `model/model.h5`
- Save training curves to `model/training_history.png`

> ⏱ Full training takes ~20–40 minutes on CPU (faster on GPU).

### 4 · Launch the Streamlit UI

```bash
streamlit run app.py
```

Open <http://localhost:8501> in your browser.

---

## 🖥️ Streamlit UI

| Tab | Description |
|---|---|
| **🎙️ Live Detection** | Records microphone audio in configurable chunks, classifies each chunk and shows smoothed predictions |
| **📁 Upload Audio** | Analyse any `.wav / .mp3 / .ogg / .flac` file offline |

Both tabs display:
- **Detection result** with coloured banner
- **Confidence score**
- **Per-class probability bars**
- **Red alert + beep sound** for `gun_shot` and `siren`

---

## 🧠 Model Architecture

```
Input  (128 × 173 × 1)   ← Log-Mel Spectrogram
  Conv2D(32)  → BN → ReLU → MaxPool → Dropout(0.25)
  Conv2D(64)  → BN → ReLU → MaxPool → Dropout(0.25)
  Conv2D(128) → BN → ReLU → MaxPool → Dropout(0.25)
  Conv2D(256) → BN → ReLU → GlobalAvgPool
  Dense(256)  → BN → ReLU → Dropout(0.5)
  Dense(128)  → ReLU      → Dropout(0.3)
Output Dense(4, softmax)  ← dog_bark | gun_shot | siren | normal
```

---

## ⚠️ Error Handling

| Situation | Response |
|---|---|
| Dataset not found | Clear message with download instructions; `train_model.py` exits |
| Model not found | Streamlit shows instructions to run `train_model.py` |
| Corrupted audio file | Logged warning; file is skipped during training |
| sounddevice unavailable | Alert playback fails silently; upload tab still works |

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `librosa` | Audio loading, feature extraction |
| `tensorflow` | CNN model training & inference |
| `scikit-learn` | Label encoding, train/test split |
| `sounddevice` | Live microphone recording |
| `streamlit` | Web UI |
| `pandas` | Metadata CSV handling |
| `tqdm` | Progress bars during feature extraction |

---

## 🎓 BCA Final Year Project

This project demonstrates an end-to-end machine learning pipeline:
**data acquisition → feature engineering → model training → real-time inference → web deployment**.
