# Multi-Sensor SNN Fusion

A **Liquid State Machine** (Echo State Network) that fuses binaural audio, event camera, and IMU/tactile data into a unified spiking neural representation for multimodal classification.

## Architecture

```
Audio signal  ──[Rate Coding]──────────────┐
                                            ├──► Liquid State Pool ──► Linear Readout ──► Class
Event frame   ──[Direct Encoding]──────────┤         (Reservoir)        (Ridge Regression)
                                            │
IMU (accel+gyro)──[Temporal Coding]────────┘
```

### Encoders

| Modality | Encoder | Method |
|----------|---------|--------|
| Audio | `AudioEncoder` | Rate coding — spike if amplitude > mean + 0.5σ |
| Event camera | `EventCameraEncoder` | Direct — flatten frame, threshold at 0 |
| IMU (6-DOF) | `IMUEncoder` | Temporal — spike on normalised rate-of-change |

### Reservoir

`LiquidStatePool` — random recurrent network (N=100 neurons), spectral radius 0.9, fixed weights. State update: `x = tanh(W_res @ x + W_in @ u)`.

### Readout

`LinearReadout` — ridge regression trained on mean reservoir states: `W = (XᵀX + λI)⁻¹ XᵀY`.

## Quick Start

```bash
pip install numpy
python demo.py
```

## Run Tests

```bash
pytest tests/
```

## Package Structure

```
fusion/
  encoders.py   # AudioEncoder, EventCameraEncoder, IMUEncoder
  reservoir.py  # LiquidStatePool
  readout.py    # LinearReadout
  fusion.py     # MultiSensorFusion
demo.py
tests/
  test_fusion.py
requirements.txt
```

## License

MIT
