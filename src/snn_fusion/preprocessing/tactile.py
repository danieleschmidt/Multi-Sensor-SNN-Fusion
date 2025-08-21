"""
Tactile Preprocessing for Neuromorphic Computing

Implements IMU processing, wavelet transforms, and tactile spike encoding
for neuromorphic multi-modal sensor fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import pywt


class WaveletTransform(nn.Module):
    """
    Wavelet transform for tactile and IMU signal processing.
    Provides time-frequency decomposition optimized for spike encoding.
    """
    
    def __init__(
        self,
        wavelet: str = 'db4',
        levels: int = 4,
        mode: str = 'symmetric',
        threshold: float = 0.1,
    ):
        super().__init__()
        self.wavelet = wavelet
        self.levels = levels
        self.mode = mode
        self.threshold = threshold
        
        # Pre-compute wavelet filters for efficiency
        self.wavelet_obj = pywt.Wavelet(wavelet)
        
    def _wavelet_decompose(self, signal: np.ndarray) -> List[np.ndarray]:
        """Perform wavelet decomposition on 1D signal."""
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.levels, mode=self.mode)
        return coeffs
    
    def _extract_features(self, coeffs: List[np.ndarray]) -> np.ndarray:
        """Extract features from wavelet coefficients."""
        features = []
        
        for level, coeff in enumerate(coeffs):
            # Statistical features
            features.extend([
                np.mean(np.abs(coeff)),  # Mean absolute value
                np.std(coeff),           # Standard deviation
                np.sqrt(np.mean(coeff**2)),  # RMS
                np.max(np.abs(coeff)),   # Maximum absolute value
            ])
            
            # Energy in different frequency bands
            energy = np.sum(coeff**2)
            features.append(energy)
            
            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(coeff)) != 0)
            features.append(zero_crossings / len(coeff))
        
        return np.array(features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply wavelet transform to input signals.
        
        Args:
            x: Input tensor [B, Channels, Time] or [B, Time]
            
        Returns:
            Wavelet features [B, Features] or [B, Channels, Features]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        batch_size, channels, time_steps = x.shape
        all_features = []
        
        for b in range(batch_size):
            batch_features = []
            
            for c in range(channels):
                signal = x[b, c].cpu().numpy()
                
                # Wavelet decomposition
                coeffs = self._wavelet_decompose(signal)
                
                # Feature extraction
                features = self._extract_features(coeffs)
                batch_features.append(features)
            
            all_features.append(np.stack(batch_features))
        
        result = torch.tensor(np.stack(all_features), dtype=torch.float32, device=x.device)
        
        if result.size(1) == 1:
            result = result.squeeze(1)  # Remove channel dim if single channel
        
        return result


class IMUEncoder(nn.Module):
    """
    IMU data encoder for converting accelerometer and gyroscope data
    to spike-based representations for neuromorphic processing.
    """
    
    def __init__(
        self,
        sampling_rate: float = 1000.0,  # Hz
        window_size: int = 100,
        overlap: int = 50,
        feature_dim: int = 64,
        accel_range: Tuple[float, float] = (-16.0, 16.0),  # g
        gyro_range: Tuple[float, float] = (-2000.0, 2000.0),  # deg/s
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.overlap = overlap
        self.feature_dim = feature_dim
        self.accel_range = accel_range
        self.gyro_range = gyro_range
        
        # Feature extraction layers
        self.accel_conv = nn.Conv1d(3, feature_dim // 2, kernel_size=5, padding=2)
        self.gyro_conv = nn.Conv1d(3, feature_dim // 2, kernel_size=5, padding=2)
        self.fusion_layer = nn.Linear(feature_dim, feature_dim)
        
        # Wavelet transform for time-frequency analysis
        self.wavelet_transform = WaveletTransform()
    
    def _preprocess_imu(self, accel: torch.Tensor, gyro: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess IMU data with filtering and normalization."""
        # Normalize to [-1, 1] range
        accel_norm = 2 * (accel - self.accel_range[0]) / (self.accel_range[1] - self.accel_range[0]) - 1
        gyro_norm = 2 * (gyro - self.gyro_range[0]) / (self.gyro_range[1] - self.gyro_range[0]) - 1
        
        # Apply low-pass filter to remove noise
        accel_filt = F.avg_pool1d(accel_norm.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        gyro_filt = F.avg_pool1d(gyro_norm.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        
        return accel_filt, gyro_filt
    
    def _extract_motion_features(self, accel: torch.Tensor, gyro: torch.Tensor) -> torch.Tensor:
        """Extract motion-specific features from IMU data."""
        batch_size, channels, time_steps = accel.shape
        
        # Magnitude features
        accel_mag = torch.sqrt(torch.sum(accel**2, dim=1, keepdim=True))
        gyro_mag = torch.sqrt(torch.sum(gyro**2, dim=1, keepdim=True))
        
        # Jerk (derivative of acceleration)
        accel_jerk = torch.diff(accel, dim=2)
        jerk_mag = torch.sqrt(torch.sum(accel_jerk**2, dim=1, keepdim=True))
        
        # Angular acceleration
        gyro_accel = torch.diff(gyro, dim=2)
        
        # Combine features
        motion_features = torch.cat([
            accel, gyro, accel_mag, gyro_mag,
            F.pad(jerk_mag, (1, 0)),  # Pad to match time dimension
            F.pad(gyro_accel, (1, 0), value=0)
        ], dim=1)
        
        return motion_features
    
    def _windowed_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from windowed segments."""
        batch_size, channels, time_steps = x.shape
        
        # Create sliding windows
        step_size = self.window_size - self.overlap
        num_windows = (time_steps - self.window_size) // step_size + 1
        
        windowed_features = []
        
        for i in range(num_windows):
            start_idx = i * step_size
            end_idx = start_idx + self.window_size
            
            window = x[:, :, start_idx:end_idx]  # [B, C, W]
            
            # Statistical features per window
            features = torch.cat([
                torch.mean(window, dim=2),
                torch.std(window, dim=2),
                torch.min(window, dim=2)[0],
                torch.max(window, dim=2)[0],
                torch.median(window, dim=2)[0],
            ], dim=1)
            
            windowed_features.append(features)
        
        return torch.stack(windowed_features, dim=2)  # [B, Features, Windows]
    
    def forward(self, accel: torch.Tensor, gyro: torch.Tensor) -> torch.Tensor:
        """
        Process IMU data and extract features.
        
        Args:
            accel: Accelerometer data [B, 3, T] (x, y, z axes)
            gyro: Gyroscope data [B, 3, T] (x, y, z axes)
            
        Returns:
            IMU features [B, Features]
        """
        # Preprocess
        accel_proc, gyro_proc = self._preprocess_imu(accel, gyro)
        
        # Extract motion features
        motion_features = self._extract_motion_features(accel_proc, gyro_proc)
        
        # Convolutional feature extraction
        accel_conv_features = self.accel_conv(accel_proc)
        gyro_conv_features = self.gyro_conv(gyro_proc)
        
        # Pool temporal dimension
        accel_pooled = F.adaptive_avg_pool1d(accel_conv_features, 1).squeeze(2)
        gyro_pooled = F.adaptive_avg_pool1d(gyro_conv_features, 1).squeeze(2)
        
        # Combine features
        combined_features = torch.cat([accel_pooled, gyro_pooled], dim=1)
        
        # Final fusion layer
        output_features = self.fusion_layer(combined_features)
        output_features = torch.tanh(output_features)  # Bounded activation
        
        return output_features


class TactileSpikeEncoder(nn.Module):
    """
    Converts tactile sensor data to spike trains for neuromorphic processing.
    Supports various tactile modalities including force, pressure, and texture.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_neurons: int = 100,
        encoding_method: str = 'population',
        spike_rate: float = 100.0,  # Hz
        time_steps: int = 100,
        adaptation_factor: float = 0.95,
        refractory_period: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.encoding_method = encoding_method
        self.spike_rate = spike_rate
        self.time_steps = time_steps
        self.adaptation_factor = adaptation_factor
        self.refractory_period = refractory_period
        
        # Population coding parameters
        if encoding_method == 'population':
            self.centers = nn.Parameter(torch.linspace(-1, 1, num_neurons))
            self.widths = nn.Parameter(torch.ones(num_neurons) * 0.2)
        
        # Adaptation mechanism
        self.register_buffer('adaptation_state', torch.ones(num_neurons))
        self.register_buffer('refractory_state', torch.zeros(num_neurons))
    
    def _population_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Encode using population of neurons with overlapping receptive fields."""
        batch_size, features = x.shape
        
        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(-1)  # [B, F, 1]
        centers_expanded = self.centers.unsqueeze(0).unsqueeze(0)  # [1, 1, N]
        widths_expanded = self.widths.unsqueeze(0).unsqueeze(0)  # [1, 1, N]
        
        # Gaussian tuning curves
        response = torch.exp(-((x_expanded - centers_expanded) ** 2) / (2 * widths_expanded ** 2))
        
        # Apply adaptation
        response = response * self.adaptation_state.unsqueeze(0).unsqueeze(0)
        
        # Convert to spike probabilities
        spike_prob = response * self.spike_rate / 1000.0  # Hz to prob per ms
        
        # Generate spikes over time
        spikes = []
        for t in range(self.time_steps):
            # Check refractory period
            can_spike = (self.refractory_state == 0)
            
            # Generate spikes
            random_vals = torch.rand_like(response)
            spike_t = (random_vals < spike_prob).float()
            
            # Apply refractory period
            spike_t = spike_t * can_spike.unsqueeze(0).unsqueeze(0)
            
            # Update refractory state
            self.refractory_state = torch.maximum(
                self.refractory_state - 1,
                torch.zeros_like(self.refractory_state)
            )
            self.refractory_state += (spike_t.sum(dim=(0, 1)) > 0) * self.refractory_period
            
            # Update adaptation
            spike_occurred = (spike_t.sum(dim=(0, 1)) > 0)
            self.adaptation_state *= torch.where(
                spike_occurred,
                torch.full_like(self.adaptation_state, self.adaptation_factor),
                torch.ones_like(self.adaptation_state)
            )
            
            spikes.append(spike_t)
        
        return torch.stack(spikes, dim=-1)  # [B, F, N, T]
    
    def _rate_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Rate-based encoding where spike frequency encodes stimulus intensity."""
        # Normalize input to [0, 1]
        x_norm = torch.sigmoid(x)  # Soft normalization
        
        # Scale to spike rates
        spike_rates = x_norm * self.spike_rate / 1000.0
        
        # Generate Poisson spikes
        spikes = []
        for t in range(self.time_steps):
            random_vals = torch.rand_like(x_norm)
            spike_t = (random_vals < spike_rates).float()
            spikes.append(spike_t)
        
        return torch.stack(spikes, dim=-1)  # [B, F, T]
    
    def _temporal_contrast_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Encode based on temporal changes in stimulus."""
        if not hasattr(self, 'prev_input'):
            self.register_buffer('prev_input', torch.zeros_like(x))
        
        # Calculate temporal difference
        diff = x - self.prev_input
        self.prev_input = x.detach()
        
        # Positive and negative changes
        pos_change = F.relu(diff)
        neg_change = F.relu(-diff)
        
        # Convert to spike probabilities
        pos_spikes = self._rate_encoding(pos_change)
        neg_spikes = self._rate_encoding(neg_change)
        
        # Combine as different channels
        return torch.stack([pos_spikes, neg_spikes], dim=1)  # [B, 2, F, T]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert tactile features to spike trains.
        
        Args:
            x: Input tactile features [B, Features]
            
        Returns:
            Spike trains with method-dependent dimensions
        """
        if self.encoding_method == 'population':
            return self._population_encoding(x)
        elif self.encoding_method == 'rate':
            return self._rate_encoding(x)
        elif self.encoding_method == 'temporal_contrast':
            return self._temporal_contrast_encoding(x)
        else:
            raise ValueError(f"Unknown encoding method: {self.encoding_method}")


class VibrotactileSensor(nn.Module):
    """
    Processes vibrotactile sensor data for texture recognition and surface analysis.
    Implements frequency domain analysis and spike-based encoding.
    """
    
    def __init__(
        self,
        sampling_rate: float = 10000.0,  # High sampling for vibration
        frequency_bands: List[Tuple[float, float]] = None,
        filter_order: int = 4,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.filter_order = filter_order
        
        if frequency_bands is None:
            # Default frequency bands for texture analysis
            self.frequency_bands = [
                (0, 50),      # Low frequency - pressure/force
                (50, 200),    # Mid-low - coarse texture
                (200, 500),   # Mid - fine texture  
                (500, 1000),  # High - surface roughness
                (1000, 2000), # Very high - micro-texture
            ]
        else:
            self.frequency_bands = frequency_bands
    
    def _apply_bandpass_filter(self, signal: torch.Tensor, low_freq: float, high_freq: float) -> torch.Tensor:
        """Apply bandpass filter to isolate frequency band."""
        # Simple butterworth-like filter implementation
        # For production, would use scipy.signal equivalents
        
        # Normalize frequencies
        nyquist = self.sampling_rate / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        # Create simple FIR bandpass filter
        filter_length = 51
        n = torch.arange(filter_length, dtype=torch.float32, device=signal.device)
        n_center = filter_length // 2
        
        # Ideal bandpass response
        h = torch.zeros(filter_length, device=signal.device)
        for i, freq_norm in enumerate([low_norm, high_norm]):
            if freq_norm > 0:
                sinc_vals = torch.sin(2 * np.pi * freq_norm * (n - n_center)) / (np.pi * (n - n_center))
                sinc_vals[n_center] = 2 * freq_norm
                h += (-1)**i * sinc_vals
        
        # Apply Hamming window
        window = 0.54 - 0.46 * torch.cos(2 * np.pi * n / (filter_length - 1))
        h = h * window
        h = h / h.sum()  # Normalize
        
        # Apply filter via convolution
        h_expanded = h.unsqueeze(0).unsqueeze(0)  # [1, 1, L]
        signal_expanded = signal.unsqueeze(1)  # [B, 1, T]
        
        filtered = F.conv1d(signal_expanded, h_expanded, padding=filter_length//2)
        return filtered.squeeze(1)
    
    def _extract_spectral_features(self, signal: torch.Tensor) -> torch.Tensor:
        """Extract spectral features from vibrotactile signal."""
        batch_size, time_steps = signal.shape
        features = []
        
        for low_freq, high_freq in self.frequency_bands:
            # Apply bandpass filter
            filtered_signal = self._apply_bandpass_filter(signal, low_freq, high_freq)
            
            # Extract features from filtered signal
            band_features = torch.cat([
                torch.mean(torch.abs(filtered_signal), dim=1, keepdim=True),  # Mean amplitude
                torch.std(filtered_signal, dim=1, keepdim=True),              # Variability
                torch.max(torch.abs(filtered_signal), dim=1, keepdim=True)[0], # Peak amplitude
                torch.sqrt(torch.mean(filtered_signal**2, dim=1, keepdim=True)), # RMS
            ], dim=1)
            
            features.append(band_features)
        
        return torch.cat(features, dim=1)  # [B, Features]
    
    def forward(self, vibration_signal: torch.Tensor) -> torch.Tensor:
        """
        Process vibrotactile sensor data.
        
        Args:
            vibration_signal: Raw vibration signal [B, Time]
            
        Returns:
            Spectral features [B, Features]
        """
        return self._extract_spectral_features(vibration_signal)