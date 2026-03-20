"""Spike encoders for multi-modal sensor data."""
import numpy as np


class AudioEncoder:
    """Rate coding encoder for audio signals.

    For each timestep t, spike[t] = 1 if abs(signal[t]) > threshold else 0.
    Threshold is set to mean + 0.5 * std of the signal.
    """

    def encode(self, signal: np.ndarray) -> np.ndarray:
        """Encode audio signal to spike train.

        Args:
            signal: Array of shape (T,) or (T, channels).

        Returns:
            Binary spike array of the same shape.
        """
        signal = np.asarray(signal, dtype=float)
        threshold = np.mean(np.abs(signal)) + 0.5 * np.std(np.abs(signal))
        spikes = (np.abs(signal) > threshold).astype(np.float32)
        return spikes


class EventCameraEncoder:
    """Direct encoding for event camera frames.

    Flattens (H, W) event frame and thresholds at 0 to produce binary spikes.
    """

    def encode(self, event_frame: np.ndarray) -> np.ndarray:
        """Encode event camera frame to flat binary spike array.

        Args:
            event_frame: Array of shape (H, W) with values in [-1, 1].

        Returns:
            Flat binary array of size H*W.
        """
        event_frame = np.asarray(event_frame, dtype=float)
        flat = event_frame.flatten()
        spikes = (flat > 0).astype(np.float32)
        return spikes


class IMUEncoder:
    """Temporal coding encoder for IMU data.

    Encodes rate-of-change of IMU readings into spike timing.
    Spikes when normalised change exceeds threshold.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def encode(self, imu: np.ndarray) -> np.ndarray:
        """Encode IMU readings to spike array.

        Args:
            imu: Array of shape (T, 6) with accelerometer (xyz) and gyro (xyz).

        Returns:
            Spike array of shape (T, 6).
        """
        imu = np.asarray(imu, dtype=float)
        T, D = imu.shape  # D should be 6

        # Compute rate of change (diff), pad first row with zeros
        diff = np.zeros_like(imu)
        diff[1:] = np.abs(np.diff(imu, axis=0))

        # Normalise per channel to [0, 1]
        max_diff = np.max(diff, axis=0, keepdims=True)
        max_diff = np.where(max_diff == 0, 1.0, max_diff)
        norm_diff = diff / max_diff

        spikes = (norm_diff > self.threshold).astype(np.float32)
        return spikes
