#!/usr/bin/env python3
"""Demo: Multi-Sensor SNN Fusion on synthetic multimodal data.

Generates 100 samples across 3 classes, trains the fusion model,
and reports classification accuracy.
"""
import numpy as np
from fusion import MultiSensorFusion


def make_dataset(n: int = 100, seed: int = 0):
    """Generate synthetic multimodal data with 3 balanced classes."""
    rng = np.random.default_rng(seed)
    labels = np.array([i % 3 for i in range(n)])
    audios, events, imus = [], [], []
    for label in labels:
        offset = label * 2.0
        audios.append(rng.standard_normal(100) + offset)
        events.append(rng.uniform(-1, 1, (8, 8)) + offset * 0.1)
        imus.append(rng.standard_normal((50, 6)) + offset * 0.5)
    return audios, events, imus, labels


def main():
    print("Multi-Sensor SNN Fusion Demo")
    print("=" * 40)

    # Generate data
    audios, events, imus, labels = make_dataset(n=100, seed=42)
    n = len(labels)
    split = int(0.8 * n)

    train_a, test_a = audios[:split], audios[split:]
    train_e, test_e = events[:split], events[split:]
    train_i, test_i = imus[:split], imus[split:]
    train_l, test_l = labels[:split], labels[split:]

    print(f"Train samples: {split}  |  Test samples: {n - split}")
    print(f"Classes: {sorted(set(labels.tolist()))}")
    print()

    # Build and train model
    model = MultiSensorFusion(
        audio_len=100,
        event_shape=(8, 8),
        imu_len=50,
        n_neurons=100,
        alpha=1e-3,
        seed=42,
    )
    print("Training fusion model...")
    model.fit(train_a, train_e, train_i, train_l)

    # Evaluate
    preds = model.predict(test_a, test_e, test_i)
    acc = np.mean(preds == test_l)
    print(f"Test accuracy: {acc:.1%}")

    # Also train accuracy
    train_preds = model.predict(train_a, train_e, train_i)
    train_acc = np.mean(train_preds == train_l)
    print(f"Train accuracy: {train_acc:.1%}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
