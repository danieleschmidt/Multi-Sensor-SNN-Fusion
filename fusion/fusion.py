"""Multi-Sensor SNN Fusion — orchestrates encoders, reservoir, and readout."""
import numpy as np

from .encoders import AudioEncoder, EventCameraEncoder, IMUEncoder
from .reservoir import LiquidStatePool
from .readout import LinearReadout


class MultiSensorFusion:
    """Fuses binaural audio, event camera, and IMU data via a Liquid State Machine.

    Pipeline:
        1. Encode each modality into spike trains.
        2. Concatenate encoded spikes as input to reservoir.
        3. Run reservoir over time steps.
        4. Use mean reservoir state as feature vector.
        5. Train / predict with LinearReadout.

    Args:
        audio_len: Number of audio timesteps per sample (T_audio).
        event_shape: Shape (H, W) of event camera frames.
        imu_len: Number of IMU timesteps per sample (T_imu).
        n_neurons: Reservoir size.
        alpha: Ridge regression regularisation.
        seed: Random seed.
    """

    def __init__(
        self,
        audio_len: int = 100,
        event_shape: tuple[int, int] = (8, 8),
        imu_len: int = 50,
        n_neurons: int = 100,
        alpha: float = 1e-3,
        seed: int = 42,
    ):
        self.audio_len = audio_len
        self.event_shape = event_shape
        self.imu_len = imu_len

        self.audio_enc = AudioEncoder()
        self.event_enc = EventCameraEncoder()
        self.imu_enc = IMUEncoder()

        # Compute reservoir input dimensionality
        audio_dim = audio_len          # 1D spikes per timestep (mono after mean)
        event_dim = event_shape[0] * event_shape[1]  # flat event frame
        imu_dim = 6                    # 6-DOF IMU
        # We concatenate all modality features, then repeat over max T
        self.input_dim = audio_dim + event_dim + imu_dim

        self.reservoir = LiquidStatePool(
            n_neurons=n_neurons,
            input_dim=self.input_dim,
            seed=seed,
        )
        self.readout = LinearReadout(alpha=alpha)

    def _encode_sample(
        self,
        audio: np.ndarray,
        event_frame: np.ndarray,
        imu: np.ndarray,
    ) -> np.ndarray:
        """Encode a single sample and return concatenated feature vector.

        Each modality is encoded then flattened; the result is a single
        1D vector (mean over time for time-varying modalities).

        Args:
            audio: (T_audio,) or (T_audio, ch) audio signal.
            event_frame: (H, W) event camera frame.
            imu: (T_imu, 6) IMU readings.

        Returns:
            Feature vector of shape (input_dim,).
        """
        # Audio: mean over time → (T_audio,) scalar per timestep → mean → scalar
        audio_spikes = self.audio_enc.encode(audio)
        if audio_spikes.ndim > 1:
            audio_spikes = audio_spikes.mean(axis=1)
        audio_feat = audio_spikes  # (T_audio,)

        # Event camera: flat binary (H*W,)
        event_feat = self.event_enc.encode(event_frame)

        # IMU: mean over time → (6,)
        imu_spikes = self.imu_enc.encode(imu)
        imu_feat = imu_spikes.mean(axis=0)

        return np.concatenate([audio_feat, event_feat, imu_feat])

    def _encode_and_run(
        self,
        audio: np.ndarray,
        event_frame: np.ndarray,
        imu: np.ndarray,
    ) -> np.ndarray:
        """Encode sample and run through reservoir, returning mean state.

        Args:
            audio: (T_audio,) audio signal.
            event_frame: (H, W) event camera frame.
            imu: (T_imu, 6) IMU readings.

        Returns:
            Mean reservoir state of shape (n_neurons,).
        """
        feat = self._encode_sample(audio, event_frame, imu)
        # Treat the feature vector as a single-step input to the reservoir
        # Reshape to (1, input_dim) for run()
        states = self.reservoir.run(feat.reshape(1, -1))  # (1, n_neurons)
        return states.mean(axis=0)  # (n_neurons,)

    def fit(
        self,
        audios: list,
        event_frames: list,
        imus: list,
        labels: np.ndarray,
    ) -> "MultiSensorFusion":
        """Fit fusion model on training data.

        Args:
            audios: List of N audio arrays, each (T_audio,).
            event_frames: List of N event frames, each (H, W).
            imus: List of N IMU arrays, each (T_imu, 6).
            labels: Integer class labels (N,).

        Returns:
            self
        """
        features = []
        for audio, ef, imu in zip(audios, event_frames, imus):
            features.append(self._encode_and_run(audio, ef, imu))
        features = np.stack(features)
        self.readout.fit(features, labels)
        return self

    def predict(
        self,
        audios: list,
        event_frames: list,
        imus: list,
    ) -> np.ndarray:
        """Predict class labels.

        Args:
            audios: List of N audio arrays.
            event_frames: List of N event frames.
            imus: List of N IMU arrays.

        Returns:
            Predicted labels (N,).
        """
        features = []
        for audio, ef, imu in zip(audios, event_frames, imus):
            features.append(self._encode_and_run(audio, ef, imu))
        features = np.stack(features)
        return self.readout.predict(features)
