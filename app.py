"""
app.py  –  Real-Time Hazard Sound Detection and Alert System
=============================================================
Streamlit UI that supports:
  • Live microphone recording with real-time classification
  • Audio file upload for offline analysis
  • Visual alert (red banner) for gun_shot and siren
  • Audio alert playback in-browser

Run with:
    streamlit run app.py
"""

import os
import sys
import io
import time
import threading
import queue

import numpy as np
import streamlit as st

# ── Project path setup ───────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "model.h5")
LABEL_NAMES = ["dog_bark", "gun_shot", "siren", "normal"]
HAZARD_ALERTS = {"gun_shot", "siren"}  # classes that trigger red alert

# ── Lazy imports (heavy libs) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the Keras model once and cache it."""
    if not os.path.isfile(MODEL_PATH):
        return None
    import tensorflow as tf  # noqa: PLC0415
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def get_extractor():
    from utils.feature_extraction import FeatureExtractor  # noqa: PLC0415
    return FeatureExtractor()


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────

def predict(model, extractor, samples: np.ndarray):
    """
    Given raw audio samples (float32, mono, at extractor.sample_rate),
    return (label, confidence, all_probs).
    """
    feat = extractor.extract_from_samples(samples)          # (n_mels, T, 1)
    feat_batch = feat[np.newaxis, ...]                      # (1, n_mels, T, 1)
    probs = model.predict(feat_batch, verbose=0)[0]         # (4,)
    idx = int(np.argmax(probs))
    return LABEL_NAMES[idx], float(probs[idx]), probs


def smooth_predictions(history: list, window: int = 5):
    """
    Average probabilities over the last *window* frames and return the
    smoothed label and confidence.
    """
    if not history:
        return "—", 0.0, np.zeros(len(LABEL_NAMES))
    recent = history[-window:]
    avg = np.mean(recent, axis=0)
    idx = int(np.argmax(avg))
    return LABEL_NAMES[idx], float(avg[idx]), avg


# ─────────────────────────────────────────────────────────────────────────────
# Real-time recording thread
# ─────────────────────────────────────────────────────────────────────────────

class AudioListener:
    """
    Background thread that records audio in chunks and feeds predictions into
    a queue.
    """

    def __init__(self, model, extractor, result_queue: queue.Queue, chunk_duration: float = 2.0):
        self.model = model
        self.extractor = extractor
        self.result_queue = result_queue
        self.chunk_duration = chunk_duration
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        try:
            import sounddevice as sd  # noqa: PLC0415
        except ImportError:
            self.result_queue.put({"error": "sounddevice not installed"})
            return

        sr = self.extractor.sample_rate
        chunk_samples = int(sr * self.chunk_duration)

        while not self._stop_event.is_set():
            try:
                audio = sd.rec(
                    chunk_samples,
                    samplerate=sr,
                    channels=1,
                    dtype="float32",
                )
                sd.wait()
                samples = audio[:, 0]
                label, conf, probs = predict(self.model, self.extractor, samples)
                self.result_queue.put(
                    {"label": label, "confidence": conf, "probs": probs.tolist()}
                )
            except Exception as exc:  # noqa: BLE001
                self.result_queue.put({"error": str(exc)})
                break


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

EMOJI = {
    "dog_bark": "🐕",
    "gun_shot": "🔫",
    "siren":    "🚨",
    "normal":   "✅",
    "—":        "🎙️",
}

LABEL_COLOR = {
    "dog_bark": "#f0a500",
    "gun_shot": "#e53935",
    "siren":    "#e53935",
    "normal":   "#43a047",
    "—":        "#9e9e9e",
}


def render_confidence_bar(label: str, conf: float, all_probs: np.ndarray):
    """Display per-class probability bars."""
    st.markdown("#### Per-class probabilities")
    for i, lbl in enumerate(LABEL_NAMES):
        p = float(all_probs[i]) if len(all_probs) == len(LABEL_NAMES) else 0.0
        bar_html = (
            f'<div style="margin:4px 0">'
            f'<span style="width:90px;display:inline-block;font-size:0.85rem">{EMOJI.get(lbl,"")}&nbsp;{lbl}</span>'
            f'<div style="display:inline-block;width:{int(p*200)}px;height:14px;'
            f'background:{LABEL_COLOR.get(lbl,"#aaa")};border-radius:3px;vertical-align:middle"></div>'
            f'&nbsp;<span style="font-size:0.8rem">{p*100:.1f}%</span>'
            f"</div>"
        )
        st.markdown(bar_html, unsafe_allow_html=True)


def render_alert(label: str, confidence: float):
    """Render a prominent coloured banner for the detected sound."""
    color = LABEL_COLOR.get(label, "#9e9e9e")
    emoji = EMOJI.get(label, "🔊")
    conf_pct = f"{confidence * 100:.1f}%"

    if label in HAZARD_ALERTS:
        st.markdown(
            f"""
            <div style="
                background:{color};color:white;padding:18px 24px;
                border-radius:10px;text-align:center;font-size:1.4rem;
                font-weight:bold;animation:blink 1s step-start infinite;
                box-shadow:0 4px 15px rgba(229,57,53,0.5)">
                ⚠️ HAZARD ALERT — {emoji} {label.upper()}<br>
                <span style="font-size:1rem">Confidence: {conf_pct}</span>
            </div>
            <style>
            @keyframes blink {{
                0%,100%{{opacity:1}} 50%{{opacity:0.6}}
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="
                background:{color};color:white;padding:14px 20px;
                border-radius:10px;text-align:center;font-size:1.2rem;
                font-weight:600">
                {emoji} Detected: {label.replace("_"," ").title()}<br>
                <span style="font-size:0.9rem">Confidence: {conf_pct}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main Streamlit app
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Hazard Sound Detection",
        page_icon="🔊",
        layout="wide",
    )

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        """
        <h1 style="text-align:center;color:#1565c0">
        🔊 Real-Time Hazard Sound Detection & Alert System
        </h1>
        <p style="text-align:center;color:#555;font-size:1.05rem">
        Powered by a CNN trained on <b>UrbanSound8K</b> · Classes: dog_bark, gun_shot, siren, normal
        </p>
        <hr>
        """,
        unsafe_allow_html=True,
    )

    # ── Model loading ─────────────────────────────────────────────────────────
    with st.spinner("Loading model …"):
        model = load_model()

    if model is None:
        st.error(
            "⚠️ **Model not found!**\n\n"
            f"Expected path: `{MODEL_PATH}`\n\n"
            "Please train the model first:\n"
            "```bash\npython train_model.py\n```"
        )
        st.stop()

    extractor = get_extractor()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/000000/speaker.png",
            width=80,
        )
        st.markdown("## ⚙️ Settings")
        chunk_duration = st.slider(
            "Recording chunk (seconds)", min_value=1, max_value=5, value=2
        )
        smooth_window = st.slider(
            "Smoothing window (frames)", min_value=1, max_value=10, value=5
        )
        st.markdown("---")
        st.markdown("### 🏷️ Class Labels")
        for lbl in LABEL_NAMES:
            st.markdown(f"{EMOJI.get(lbl,'')} **{lbl}**")
        st.markdown("---")
        history_img = os.path.join(PROJECT_ROOT, "model", "training_history.png")
        if os.path.isfile(history_img):
            st.markdown("### 📈 Training History")
            st.image(history_img, use_column_width=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_live, tab_upload = st.tabs(["🎙️ Live Detection", "📁 Upload Audio"])

    # ════════════════════════════════════════════════════════════════════════
    # Tab 1 – Live detection
    # ════════════════════════════════════════════════════════════════════════
    with tab_live:
        st.markdown("### 🎤 Live Microphone Detection")
        st.info(
            "Click **▶ Start Listening** to begin real-time detection. "
            "The system records audio in short chunks and classifies each one. "
            "Predictions are smoothed over multiple frames for stability."
        )

        # Session-state initialisation
        if "listening" not in st.session_state:
            st.session_state.listening = False
        if "pred_history" not in st.session_state:
            st.session_state.pred_history = []
        if "listener" not in st.session_state:
            st.session_state.listener = None
        if "result_queue" not in st.session_state:
            st.session_state.result_queue = queue.Queue()

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("▶ Start Listening", disabled=st.session_state.listening, use_container_width=True):
                st.session_state.listening = True
                st.session_state.pred_history = []
                q = queue.Queue()
                st.session_state.result_queue = q
                listener = AudioListener(model, extractor, q, chunk_duration)
                st.session_state.listener = listener
                listener.start()
                st.rerun()

        with col_btn2:
            if st.button("⏹ Stop Listening", disabled=not st.session_state.listening, use_container_width=True):
                st.session_state.listening = False
                if st.session_state.listener:
                    st.session_state.listener.stop()
                    st.session_state.listener = None
                st.rerun()

        # Status indicator
        if st.session_state.listening:
            st.markdown(
                '<span style="color:green;font-weight:bold">● LISTENING …</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span style="color:grey">● Idle</span>',
                unsafe_allow_html=True,
            )

        # Poll the result queue and update UI
        detection_placeholder = st.empty()
        bars_placeholder = st.empty()
        alert_audio_placeholder = st.empty()

        if st.session_state.listening:
            q = st.session_state.result_queue
            try:
                result = q.get_nowait()
            except queue.Empty:
                result = None

            if result:
                if "error" in result:
                    st.error(f"Recording error: {result['error']}")
                    st.session_state.listening = False
                else:
                    st.session_state.pred_history.append(result["probs"])
                    label, conf, avg_probs = smooth_predictions(
                        st.session_state.pred_history, window=smooth_window
                    )
                    with detection_placeholder.container():
                        render_alert(label, conf)
                    with bars_placeholder.container():
                        render_confidence_bar(label, conf, avg_probs)

                    # Play alert audio for hazard sounds
                    if label in HAZARD_ALERTS:
                        from utils.alert import read_alert_wav_bytes  # noqa: PLC0415
                        alert_bytes = read_alert_wav_bytes()
                        with alert_audio_placeholder.container():
                            st.audio(alert_bytes, format="audio/wav", autoplay=True)

            # Auto-refresh while listening
            time.sleep(0.3)
            st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # Tab 2 – Upload audio
    # ════════════════════════════════════════════════════════════════════════
    with tab_upload:
        st.markdown("### 📁 Analyse an Audio File")
        st.info(
            "Upload a `.wav` or `.mp3` file.  "
            "The same feature extraction pipeline used during training will be applied."
        )

        uploaded = st.file_uploader(
            "Choose an audio file", type=["wav", "mp3", "ogg", "flac"]
        )

        if uploaded is not None:
            # Play the uploaded file
            st.audio(uploaded, format=f"audio/{uploaded.name.split('.')[-1]}")

            with st.spinner("Analysing …"):
                try:
                    import librosa  # noqa: PLC0415

                    audio_bytes = uploaded.read()
                    buf = io.BytesIO(audio_bytes)
                    samples, sr = librosa.load(buf, sr=extractor.sample_rate, mono=True)
                    label, conf, probs = predict(model, extractor, samples)

                    st.markdown("---")
                    render_alert(label, conf)
                    st.markdown("")
                    render_confidence_bar(label, conf, probs)

                    if label in HAZARD_ALERTS:
                        from utils.alert import read_alert_wav_bytes  # noqa: PLC0415
                        alert_bytes = read_alert_wav_bytes()
                        st.markdown("🔔 **Alert sound:**")
                        st.audio(alert_bytes, format="audio/wav", autoplay=True)

                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not process file: {exc}")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <hr>
        <p style="text-align:center;color:#999;font-size:0.8rem">
        Real-Time Hazard Sound Detection &amp; Alert System · BCA Final Year Project
        </p>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
