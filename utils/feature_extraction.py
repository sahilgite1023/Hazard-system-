"""
Feature extraction utilities for audio classification.

Extracts Log-Mel Spectrograms and MFCCs from audio files and raw waveforms
using librosa, with consistent shape for CNN model input.
"""

import numpy as np
import librosa


# ── Audio / spectrogram constants ──────────────────────────────────────────
SAMPLE_RATE = 22050      # Hz
DURATION = 4.0           # seconds per clip
N_MELS = 128             # mel-filter banks
N_MFCC = 40              # MFCC coefficients
N_FFT = 1024             # FFT window size
HOP_LENGTH = 512         # hop between frames
MAX_FRAMES = int(np.ceil(SAMPLE_RATE * DURATION / HOP_LENGTH))  # 173


class FeatureExtractor:
    """Extracts audio features suitable for the CNN hazard-detection model."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        duration: float = DURATION,
        n_mels: int = N_MELS,
        n_mfcc: int = N_MFCC,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_frames = int(np.ceil(sample_rate * duration / hop_length))

    # ── low-level helpers ────────────────────────────────────────────────────

    def _load_audio(self, file_path: str) -> np.ndarray:
        """Load and resample audio; pad/trim to fixed duration."""
        samples, _ = librosa.load(
            file_path,
            sr=self.sample_rate,
            duration=self.duration,
            mono=True,
        )
        target_len = int(self.sample_rate * self.duration)
        if len(samples) < target_len:
            samples = np.pad(samples, (0, target_len - len(samples)))
        else:
            samples = samples[:target_len]
        return samples

    def _fix_frames(self, spec: np.ndarray) -> np.ndarray:
        """Ensure spectrogram has exactly self.max_frames time columns."""
        if spec.shape[1] < self.max_frames:
            pad = self.max_frames - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad)))
        else:
            spec = spec[:, : self.max_frames]
        return spec

    # ── public API ──────────────────────────────────────────────────────────

    def log_mel_spectrogram(self, samples: np.ndarray) -> np.ndarray:
        """Return normalised log-Mel spectrogram of shape (n_mels, max_frames)."""
        mel = librosa.feature.melspectrogram(
            y=samples,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = self._fix_frames(log_mel)
        # Normalise to [0, 1]
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-9)
        return log_mel.astype(np.float32)

    def mfcc(self, samples: np.ndarray) -> np.ndarray:
        """Return normalised MFCC array of shape (n_mfcc, max_frames)."""
        mfcc = librosa.feature.mfcc(
            y=samples,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        mfcc = self._fix_frames(mfcc)
        # Standardise
        mean = mfcc.mean(axis=1, keepdims=True)
        std = mfcc.std(axis=1, keepdims=True) + 1e-9
        mfcc = (mfcc - mean) / std
        return mfcc.astype(np.float32)

    def extract_from_file(self, file_path: str) -> np.ndarray:
        """
        Extract Log-Mel Spectrogram from a file.

        Returns
        -------
        np.ndarray
            Shape ``(n_mels, max_frames, 1)`` — ready for the CNN.
        """
        samples = self._load_audio(file_path)
        log_mel = self.log_mel_spectrogram(samples)
        return log_mel[:, :, np.newaxis]  # add channel dim

    def extract_from_samples(self, samples: np.ndarray) -> np.ndarray:
        """
        Extract Log-Mel Spectrogram from a raw waveform (real-time path).

        Parameters
        ----------
        samples : np.ndarray
            1-D float32 audio waveform, already at ``self.sample_rate``.

        Returns
        -------
        np.ndarray
            Shape ``(n_mels, max_frames, 1)`` — ready for the CNN.
        """
        target_len = int(self.sample_rate * self.duration)
        if len(samples) < target_len:
            samples = np.pad(samples, (0, target_len - len(samples)))
        else:
            samples = samples[:target_len]

        log_mel = self.log_mel_spectrogram(samples.astype(np.float32))
        return log_mel[:, :, np.newaxis]

    @property
    def input_shape(self) -> tuple:
        """CNN input shape: (n_mels, max_frames, 1)."""
        return (self.n_mels, self.max_frames, 1)
