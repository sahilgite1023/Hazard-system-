"""
Alert utilities: generates and plays a beep alert sound.

The alert tone is synthesised on the fly with NumPy + sounddevice so the
project works without any bundled audio assets.  A WAV file (alert.wav) is
also written to disk the first time it is generated, so Streamlit's
st.audio() can play it in the browser.
"""

import os
import wave

import numpy as np


ALERT_WAV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "alert.wav")


def _generate_alert_samples(
    freq: float = 880.0,
    duration: float = 1.0,
    sample_rate: int = 22050,
    volume: float = 0.8,
) -> np.ndarray:
    """Return a 1-D float32 array containing a two-tone alert beep."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Two-tone siren: alternates between freq and freq*1.5
    half = len(t) // 2
    tone = np.concatenate(
        [
            np.sin(2 * np.pi * freq * t[:half]),
            np.sin(2 * np.pi * freq * 1.5 * t[half:]),
        ]
    )
    # Fade in/out to avoid clicks
    fade = int(sample_rate * 0.05)
    tone[:fade] *= np.linspace(0, 1, fade)
    tone[-fade:] *= np.linspace(1, 0, fade)
    return (tone * volume).astype(np.float32)


def ensure_alert_wav(path: str = ALERT_WAV_PATH) -> str:
    """
    Create alert.wav if it does not exist yet.

    Returns the absolute path to the WAV file.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        samples = _generate_alert_samples()
        sample_rate = 22050
        pcm = (samples * 32767).astype(np.int16)
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
    return path


def play_alert() -> None:
    """
    Play the alert tone through the default audio output device.

    Falls back silently if sounddevice is unavailable (e.g. headless server).
    """
    try:
        import sounddevice as sd  # noqa: PLC0415

        samples = _generate_alert_samples()
        sd.play(samples, samplerate=22050)
        sd.wait()
    except Exception:  # noqa: BLE001
        pass


def read_alert_wav_bytes(path: str = ALERT_WAV_PATH) -> bytes:
    """Return the raw bytes of alert.wav for use with st.audio()."""
    path = ensure_alert_wav(path)
    with open(path, "rb") as fh:
        return fh.read()
