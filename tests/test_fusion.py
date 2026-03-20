"""Tests for Multi-Sensor SNN Fusion."""
import numpy as np
import pytest

from fusion.encoders import AudioEncoder, EventCameraEncoder, IMUEncoder
from fusion.reservoir import LiquidStatePool
from fusion.readout import LinearReadout
from fusion.fusion import MultiSensorFusion


# ──────────────────────────────────────────────
# AudioEncoder tests
# ──────────────────────────────────────────────

def test_audio_encoder_shape_1d():
    enc = AudioEncoder()
    signal = np.random.randn(50)
    spikes = enc.encode(signal)
    assert spikes.shape == (50,), f"Expected (50,), got {spikes.shape}"


def test_audio_encoder_shape_2d():
    enc = AudioEncoder()
    signal = np.random.randn(50, 2)
    spikes = enc.encode(signal)
    assert spikes.shape == (50, 2), f"Expected (50, 2), got {spikes.shape}"


def test_audio_encoder_binary():
    enc = AudioEncoder()
    signal = np.random.randn(100)
    spikes = enc.encode(signal)
    unique = np.unique(spikes)
    assert set(unique).issubset({0.0, 1.0}), f"Non-binary values: {unique}"


# ──────────────────────────────────────────────
# EventCameraEncoder tests
# ──────────────────────────────────────────────

def test_event_camera_encoder_1d():
    enc = EventCameraEncoder()
    frame = np.random.uniform(-1, 1, (8, 8))
    spikes = enc.encode(frame)
    assert spikes.ndim == 1, "Output should be 1D"


def test_event_camera_encoder_shape():
    enc = EventCameraEncoder()
    H, W = 10, 12
    frame = np.random.uniform(-1, 1, (H, W))
    spikes = enc.encode(frame)
    assert spikes.shape == (H * W,), f"Expected ({H*W},), got {spikes.shape}"


def test_event_camera_encoder_binary():
    enc = EventCameraEncoder()
    frame = np.random.uniform(-1, 1, (6, 6))
    spikes = enc.encode(frame)
    unique = np.unique(spikes)
    assert set(unique).issubset({0.0, 1.0}), f"Non-binary values: {unique}"


# ──────────────────────────────────────────────
# IMUEncoder tests
# ──────────────────────────────────────────────

def test_imu_encoder_shape():
    enc = IMUEncoder()
    imu = np.random.randn(30, 6)
    spikes = enc.encode(imu)
    assert spikes.shape == (30, 6), f"Expected (30, 6), got {spikes.shape}"


def test_imu_encoder_binary():
    enc = IMUEncoder()
    imu = np.random.randn(30, 6)
    spikes = enc.encode(imu)
    unique = np.unique(spikes)
    assert set(unique).issubset({0.0, 1.0}), f"Non-binary values: {unique}"


# ──────────────────────────────────────────────
# LiquidStatePool tests
# ──────────────────────────────────────────────

def test_reservoir_state_shape():
    pool = LiquidStatePool(n_neurons=100, input_dim=10)
    u = np.random.randn(10)
    state = pool.step(u)
    assert state.shape == (100,), f"Expected (100,), got {state.shape}"


def test_reservoir_state_changes():
    pool = LiquidStatePool(n_neurons=100, input_dim=10)
    u = np.ones(10)
    state0 = pool.x.copy()
    state1 = pool.step(u)
    assert not np.allclose(state0, state1), "Reservoir state should change after step"


def test_reservoir_run_shape():
    T = 20
    pool = LiquidStatePool(n_neurons=100, input_dim=5)
    seq = np.random.randn(T, 5)
    states = pool.run(seq)
    assert states.shape == (T, 100), f"Expected ({T}, 100), got {states.shape}"


# ──────────────────────────────────────────────
# LinearReadout tests
# ──────────────────────────────────────────────

def test_readout_predict_shape():
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((60, 50))
    y_train = np.array([i % 3 for i in range(60)])
    X_test = rng.standard_normal((20, 50))

    ro = LinearReadout()
    ro.fit(X_train, y_train)
    preds = ro.predict(X_test)
    assert preds.shape == (20,), f"Expected (20,), got {preds.shape}"


def test_readout_predict_valid_classes():
    rng = np.random.default_rng(1)
    X_train = rng.standard_normal((60, 50))
    y_train = np.array([i % 3 for i in range(60)])
    X_test = rng.standard_normal((10, 50))

    ro = LinearReadout()
    ro.fit(X_train, y_train)
    preds = ro.predict(X_test)
    assert set(preds).issubset({0, 1, 2}), f"Invalid class predictions: {set(preds)}"


# ──────────────────────────────────────────────
# MultiSensorFusion end-to-end tests
# ──────────────────────────────────────────────

def _make_dataset(n=100, seed=0):
    """Generate synthetic 3-class multimodal data."""
    rng = np.random.default_rng(seed)
    labels = np.array([i % 3 for i in range(n)])
    audios, events, imus = [], [], []
    for label in labels:
        # Each class has a different mean — makes the task learnable
        offset = label * 2.0
        audios.append(rng.standard_normal(100) + offset)
        events.append(rng.uniform(-1, 1, (8, 8)) + offset * 0.1)
        imus.append(rng.standard_normal((50, 6)) + offset * 0.5)
    return audios, events, imus, labels


def test_fusion_end_to_end():
    audios, events, imus, labels = _make_dataset(n=30, seed=42)
    model = MultiSensorFusion(audio_len=100, event_shape=(8, 8), imu_len=50, seed=42)
    model.fit(audios, events, imus, labels)
    preds = model.predict(audios, events, imus)
    assert preds.shape == (30,), f"Expected (30,), got {preds.shape}"


def test_fusion_accuracy():
    """Fusion should achieve > 60% accuracy on separable 3-class synthetic data."""
    n = 100
    audios, events, imus, labels = _make_dataset(n=n, seed=7)
    model = MultiSensorFusion(audio_len=100, event_shape=(8, 8), imu_len=50, seed=7)
    model.fit(audios, events, imus, labels)
    preds = model.predict(audios, events, imus)
    acc = np.mean(preds == labels)
    assert acc > 0.60, f"Accuracy {acc:.2%} is below 60% threshold"
