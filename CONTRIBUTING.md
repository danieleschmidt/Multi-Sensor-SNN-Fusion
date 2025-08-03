# Contributing to Multi-Sensor SNN-Fusion

Thank you for your interest in contributing to the Multi-Sensor SNN-Fusion project! This document provides guidelines for contributing to our neuromorphic computing framework.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Hardware Contributions](#hardware-contributions)
- [Dataset Contributions](#dataset-contributions)
- [Community](#community)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- Git with LFS for dataset management
- Experience with PyTorch and neuromorphic computing concepts

### Quick Start

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/Multi-Sensor-SNN-Fusion`
3. Install dependencies: `pip install -e .`
4. Run tests: `pytest tests/`
5. Create a feature branch: `git checkout -b feature/your-feature`

## Development Setup

### Environment Setup

```bash
# Create conda environment
conda create -n snn-fusion python=3.10
conda activate snn-fusion

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Download test datasets
python scripts/download_datasets.py --dataset test
```

### Hardware Setup (Optional)

For neuromorphic hardware development:

```bash
# Intel Loihi 2
source /opt/intel/loihi2/setup.sh
pip install lava-nc

# BrainChip Akida
pip install akida

# SpiNNaker
pip install spynnaker
```

## Contributing Process

### 1. Issue Creation

- Check existing issues before creating new ones
- Use issue templates for bug reports and feature requests
- Include relevant system information and reproduction steps
- Tag issues appropriately (bug, enhancement, hardware, dataset)

### 2. Pull Request Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/descriptive-name
   ```

2. **Make Changes**
   - Follow coding standards
   - Add tests for new functionality
   - Update documentation
   - Ensure backwards compatibility

3. **Test Changes**
   ```bash
   pytest tests/
   python tests/integration/test_hardware.py  # Hardware tests
   ```

4. **Submit Pull Request**
   - Use clear, descriptive title
   - Reference related issues
   - Include performance benchmarks for algorithmic changes
   - Add hardware compatibility notes

### 3. Review Process

- All PRs require review from maintainers
- Hardware-specific changes need hardware team approval
- Algorithm changes require research team review
- Breaking changes need project lead approval

## Code Standards

### Python Style

- Follow PEP 8 with 88-character line limit
- Use Black for formatting: `black src/ tests/`
- Type hints required for public APIs
- Docstrings in Google style

```python
def process_spikes(
    spike_train: torch.Tensor,
    time_window: float = 50.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Process spike trains for neuromorphic inference.
    
    Args:
        spike_train: Input spike data [batch, time, neurons]
        time_window: Processing window in milliseconds
        device: Target computation device
        
    Returns:
        Processed spike features [batch, features]
        
    Raises:
        ValueError: If spike_train has incorrect dimensions
    """
```

### Neuromorphic-Specific Guidelines

- Use event-driven processing patterns
- Minimize synchronous operations
- Document temporal dynamics clearly
- Include hardware compatibility notes
- Profile memory usage for large networks

### Performance Requirements

- Latency: <1ms for inference on target hardware
- Memory: Document peak memory usage
- Energy: Profile power consumption on neuromorphic chips
- Scalability: Test with 1000+ neuron networks

## Testing Guidelines

### Test Categories

1. **Unit Tests** (`tests/unit/`)
   - Individual function/class testing
   - Mock hardware dependencies
   - Fast execution (<5 minutes total)

2. **Integration Tests** (`tests/integration/`)
   - End-to-end pipeline testing
   - Multi-modal data processing
   - Hardware abstraction layer

3. **Hardware Tests** (`tests/hardware/`)
   - Platform-specific validation
   - Performance benchmarking
   - Power consumption measurement

4. **Performance Tests** (`tests/performance/`)
   - Latency benchmarks
   - Throughput measurement
   - Memory profiling

### Test Requirements

```python
# Example test structure
def test_lsm_forward_pass():
    """Test LSM forward pass with various input sizes."""
    lsm = LiquidStateMachine(n_inputs=64, n_reservoir=1000)
    
    # Test multiple batch sizes
    for batch_size in [1, 8, 32]:
        inputs = torch.randn(batch_size, 100, 64)
        outputs, states = lsm(inputs, return_states=True)
        
        assert outputs.shape == (batch_size, lsm.n_outputs)
        assert states is not None
        assert not torch.isnan(outputs).any()
```

### Hardware Testing

```python
@pytest.mark.loihi
def test_loihi_deployment():
    """Test model deployment on Intel Loihi 2."""
    if not has_loihi_hardware():
        pytest.skip("Loihi hardware not available")
    
    model = create_test_model()
    loihi_model = deploy_to_loihi(model)
    
    # Test inference
    test_input = generate_spike_input()
    output = loihi_model.run(test_input)
    
    assert output.shape == expected_shape
    assert latency < 1.0  # milliseconds
```

## Documentation

### API Documentation

- All public functions require docstrings
- Include mathematical formulations for algorithms
- Provide usage examples
- Document hardware-specific behavior

### Tutorials and Examples

- Jupyter notebooks for new features
- Step-by-step implementation guides
- Hardware deployment examples
- Performance optimization tips

### Architecture Documentation

- Update ADRs for significant changes
- Document design decisions
- Include performance trade-offs
- Maintain compatibility matrices

## Hardware Contributions

### Supported Platforms

1. **Intel Loihi 2**
   - Spiking neural network acceleration
   - On-chip learning support
   - Event-driven computation

2. **BrainChip Akida**
   - Edge AI acceleration
   - Ultra-low power inference
   - Quantized networks

3. **SpiNNaker 2**
   - Massively parallel simulation
   - Real-time neural modeling
   - Large-scale networks

### Adding New Hardware Support

1. Create hardware abstraction in `src/snn_fusion/hardware/`
2. Implement deployment pipeline
3. Add performance benchmarks
4. Update documentation and examples
5. Include in CI/CD testing

## Dataset Contributions

### Dataset Standards

- **Format**: HDF5 with standardized schema
- **Metadata**: Comprehensive sensor specifications
- **Synchronization**: Sub-millisecond timing accuracy
- **Quality**: Validated sensor calibration
- **Ethics**: Privacy and consent compliance

### Contributing New Datasets

1. **Proposal**: Submit dataset proposal issue
2. **Collection**: Follow data collection protocols
3. **Processing**: Use standardized preprocessing pipeline
4. **Validation**: Run quality assurance checks
5. **Documentation**: Create dataset description
6. **License**: Ensure appropriate licensing

### MAVEN Dataset Schema

```python
dataset_structure = {
    'metadata': {
        'sample_rate': int,
        'duration': float,
        'sensors': dict,
        'labels': list,
    },
    'audio': {
        'binaural': np.ndarray,  # [time, 2]
        'sample_rate': int,
    },
    'vision': {
        'events': np.ndarray,    # [time, x, y, polarity]
        'resolution': tuple,
    },
    'tactile': {
        'imu': np.ndarray,       # [time, 6]
        'sample_rate': int,
    },
    'annotations': {
        'timestamps': np.ndarray,
        'labels': np.ndarray,
    }
}
```

## Community

### Communication Channels

- **GitHub Discussions**: Technical discussions and Q&A
- **Discord**: Real-time community chat
- **Mailing List**: Development updates and announcements
- **Monthly Meetings**: Open community calls

### Getting Help

1. Check existing documentation and examples
2. Search GitHub issues for similar problems
3. Ask questions in GitHub Discussions
4. Join community Discord for real-time help

### Reporting Issues

#### Bug Reports

Use the bug report template and include:
- System information (OS, Python version, hardware)
- Minimal reproduction example
- Expected vs actual behavior
- Error messages and stack traces
- Hardware configuration (if applicable)

#### Feature Requests

Use the feature request template and include:
- Clear description of the feature
- Use cases and motivation
- Proposed implementation approach
- Hardware compatibility considerations
- Performance impact analysis

### Recognition

Contributors are recognized through:
- GitHub contributor acknowledgments
- Research paper co-authorship (for significant contributions)
- Conference presentation opportunities
- Beta access to new hardware platforms

## Maintenance

### Release Process

1. Version updates follow semantic versioning
2. Hardware compatibility testing
3. Performance regression testing
4. Documentation updates
5. Community notification

### Long-term Support

- LTS versions maintained for 2 years
- Security updates for all supported versions
- Hardware driver compatibility maintenance
- Migration guides for breaking changes

---

## Questions?

For questions about contributing, please:
1. Check the FAQ in our documentation
2. Search existing GitHub issues
3. Start a discussion in GitHub Discussions
4. Contact maintainers via email

We appreciate your contributions to advancing neuromorphic computing research!