"""
Vision Preprocessing for Neuromorphic Computing

Implements event-based vision processing, Gabor filtering, and visual spike encoding
for neuromorphic multi-modal sensor fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any

# Optional OpenCV import
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class GaborFilters(nn.Module):
    """
    Gabor filter bank for extracting oriented edge features from visual data.
    Optimized for neuromorphic vision processing.
    """
    
    def __init__(
        self,
        num_orientations: int = 8,
        num_scales: int = 4,
        kernel_size: int = 31,
        sigma_x: float = 2.0,
        sigma_y: float = 2.0,
        lambda_wave: float = 4.0,
        gamma: float = 0.5,
    ):
        super().__init__()
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        self.kernel_size = kernel_size
        
        # Generate Gabor filter bank
        filters = self._generate_gabor_bank(
            num_orientations, num_scales, kernel_size,
            sigma_x, sigma_y, lambda_wave, gamma
        )
        
        self.register_buffer('filters', filters)
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=num_orientations * num_scales,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.conv.weight.data = filters.unsqueeze(1)
        
    def _generate_gabor_bank(
        self, 
        num_orientations: int,
        num_scales: int,
        kernel_size: int,
        sigma_x: float,
        sigma_y: float,
        lambda_wave: float,
        gamma: float
    ) -> torch.Tensor:
        """Generate a bank of Gabor filters with different orientations and scales."""
        filters = []
        
        for scale in range(num_scales):
            scale_factor = 1.5 ** scale
            current_lambda = lambda_wave * scale_factor
            current_sigma_x = sigma_x * scale_factor
            current_sigma_y = sigma_y * scale_factor
            
            for orientation in range(num_orientations):
                theta = orientation * np.pi / num_orientations
                
                # Generate Gabor kernel
                kernel = self._gabor_kernel(
                    kernel_size, current_sigma_x, current_sigma_y,
                    theta, current_lambda, gamma
                )
                filters.append(kernel)
        
        return torch.stack(filters)
    
    def _gabor_kernel(
        self,
        size: int,
        sigma_x: float,
        sigma_y: float,
        theta: float,
        lambda_wave: float,
        gamma: float
    ) -> torch.Tensor:
        """Generate a single Gabor kernel."""
        # Create coordinate grids
        y, x = torch.meshgrid(
            torch.arange(size, dtype=torch.float32) - size // 2,
            torch.arange(size, dtype=torch.float32) - size // 2,
            indexing='ij'
        )
        
        # Rotate coordinates
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        
        # Generate Gabor filter
        gaussian = torch.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma_x**2))
        sinusoid = torch.cos(2 * np.pi * x_theta / lambda_wave)
        
        gabor = gaussian * sinusoid
        
        # Normalize to zero mean
        gabor = gabor - gabor.mean()
        
        return gabor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gabor filter bank to input image."""
        # Ensure input is grayscale
        if x.dim() == 3:  # [H, W, C]
            x = x.mean(dim=-1, keepdim=True)
        if x.dim() == 4 and x.size(1) > 1:  # [B, C, H, W]
            x = x.mean(dim=1, keepdim=True)
        
        return self.conv(x)


class EventEncoder(nn.Module):
    """
    Event-based vision encoder for converting standard images to event-like spikes.
    Implements temporal difference encoding for neuromorphic processing.
    """
    
    def __init__(
        self,
        threshold: float = 0.1,
        temporal_window: int = 10,
        spatial_filter_size: int = 3,
    ):
        super().__init__()
        self.threshold = threshold
        self.temporal_window = temporal_window
        self.spatial_filter_size = spatial_filter_size
        
        # Spatial difference filter (like DVS camera)
        self.register_buffer('prev_frame', None)
        
        # Laplacian filter for edge detection
        laplacian = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('laplacian', laplacian)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convert input to event-like representation.
        
        Args:
            x: Input tensor [B, C, H, W] or [B, T, C, H, W]
            
        Returns:
            Dictionary with 'events', 'polarity', and 'timestamps'
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)  # Add temporal dimension
        
        batch_size, time_steps, channels, height, width = x.shape
        
        # Convert to grayscale if needed
        if channels > 1:
            x = x.mean(dim=2, keepdim=True)
        
        events = []
        polarities = []
        timestamps = []
        
        for t in range(time_steps):
            current_frame = x[:, t, 0]  # [B, H, W]
            
            if self.prev_frame is not None:
                # Temporal difference
                diff = current_frame - self.prev_frame
                
                # Apply spatial filtering
                diff_filtered = F.conv2d(
                    diff.unsqueeze(1), 
                    self.laplacian, 
                    padding=1
                )
                
                # Generate events based on threshold
                pos_events = (diff_filtered > self.threshold).float()
                neg_events = (diff_filtered < -self.threshold).float()
                
                # Combine events
                event_tensor = pos_events - neg_events
                polarity_tensor = torch.sign(diff_filtered) * (
                    torch.abs(diff_filtered) > self.threshold
                ).float()
                
                events.append(event_tensor)
                polarities.append(polarity_tensor)
                timestamps.append(torch.full_like(event_tensor, t))
            
            self.prev_frame = current_frame.detach()
        
        if events:
            return {
                'events': torch.stack(events, dim=1),
                'polarity': torch.stack(polarities, dim=1),
                'timestamps': torch.stack(timestamps, dim=1)
            }
        else:
            # Return zeros for first frame
            zero_events = torch.zeros(batch_size, 1, 1, height, width, device=x.device)
            return {
                'events': zero_events,
                'polarity': zero_events,
                'timestamps': zero_events
            }


class VisualSpikeEncoder(nn.Module):
    """
    Converts visual features to spike trains for neuromorphic processing.
    Implements population coding and temporal encoding strategies.
    """
    
    def __init__(
        self,
        input_features: int,
        num_neurons: int = 100,
        encoding_method: str = 'population',
        spike_rate: float = 100.0,  # Hz
        time_steps: int = 100,
        min_value: float = 0.0,
        max_value: float = 1.0,
    ):
        super().__init__()
        self.input_features = input_features
        self.num_neurons = num_neurons
        self.encoding_method = encoding_method
        self.spike_rate = spike_rate
        self.time_steps = time_steps
        self.min_value = min_value
        self.max_value = max_value
        
        if encoding_method == 'population':
            # Population coding with Gaussian tuning curves
            centers = torch.linspace(min_value, max_value, num_neurons)
            self.register_buffer('centers', centers)
            self.sigma = (max_value - min_value) / (num_neurons * 0.8)
        
    def population_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Encode using population of neurons with Gaussian tuning curves."""
        # x shape: [B, Features] -> [B, Features, Neurons]
        x_expanded = x.unsqueeze(-1)  # [B, Features, 1]
        centers_expanded = self.centers.unsqueeze(0).unsqueeze(0)  # [1, 1, Neurons]
        
        # Gaussian response
        response = torch.exp(-((x_expanded - centers_expanded) ** 2) / (2 * self.sigma ** 2))
        
        # Convert to spike probabilities
        spike_probs = response * self.spike_rate / 1000.0  # Convert Hz to probability per ms
        
        # Generate spikes over time
        spikes = []
        for t in range(self.time_steps):
            random_vals = torch.rand_like(response)
            spike_t = (random_vals < spike_probs).float()
            spikes.append(spike_t)
        
        return torch.stack(spikes, dim=-1)  # [B, Features, Neurons, Time]
    
    def rate_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Encode using rate coding - higher values = higher spike rates."""
        # Normalize input to [0, 1]
        x_norm = (x - self.min_value) / (self.max_value - self.min_value)
        x_norm = torch.clamp(x_norm, 0.0, 1.0)
        
        # Convert to spike rates
        spike_rates = x_norm * self.spike_rate / 1000.0  # Hz to probability per ms
        
        # Generate spikes
        spikes = []
        for t in range(self.time_steps):
            random_vals = torch.rand_like(x_norm)
            spike_t = (random_vals < spike_rates).float()
            spikes.append(spike_t)
        
        return torch.stack(spikes, dim=-1)  # [B, Features, Time]
    
    def temporal_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Encode using temporal patterns - spike timing encodes information."""
        # Normalize input to [0, 1]
        x_norm = (x - self.min_value) / (self.max_value - self.min_value)
        x_norm = torch.clamp(x_norm, 0.0, 1.0)
        
        # Calculate spike times based on input value
        # Lower values spike earlier, higher values spike later
        spike_times = (1.0 - x_norm) * (self.time_steps - 1)
        spike_times = torch.round(spike_times).long()
        
        # Create spike trains
        batch_size, features = x.shape
        spikes = torch.zeros(batch_size, features, self.time_steps, device=x.device)
        
        # Set spikes at computed times
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, features)
        feature_indices = torch.arange(features).unsqueeze(0).expand(batch_size, -1)
        
        spikes[batch_indices, feature_indices, spike_times] = 1.0
        
        return spikes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode visual features as spikes.
        
        Args:
            x: Input features [B, Features] or [B, C, H, W]
            
        Returns:
            Spike trains [B, Features, Time] or [B, Features, Neurons, Time]
        """
        # Flatten spatial dimensions if needed
        if x.dim() > 2:
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
        
        if self.encoding_method == 'population':
            return self.population_encoding(x)
        elif self.encoding_method == 'rate':
            return self.rate_encoding(x)
        elif self.encoding_method == 'temporal':
            return self.temporal_encoding(x)
        else:
            raise ValueError(f"Unknown encoding method: {self.encoding_method}")


class DVSEventProcessor(nn.Module):
    """
    Process Dynamic Vision Sensor (DVS) events for neuromorphic processing.
    Handles real DVS event data format and conversion to spike trains.
    """
    
    def __init__(
        self,
        sensor_size: Tuple[int, int] = (346, 260),
        time_window_ms: float = 50.0,
        spatial_downsample: int = 1,
    ):
        super().__init__()
        self.sensor_size = sensor_size
        self.time_window_ms = time_window_ms
        self.spatial_downsample = spatial_downsample
        
        self.output_size = (
            sensor_size[0] // spatial_downsample,
            sensor_size[1] // spatial_downsample
        )
    
    def events_to_spike_tensor(
        self,
        events: Dict[str, np.ndarray],
        time_steps: int = 100
    ) -> torch.Tensor:
        """
        Convert DVS events to spike tensor representation.
        
        Args:
            events: Dict with 'x', 'y', 't', 'p' arrays
            time_steps: Number of temporal bins
            
        Returns:
            Spike tensor [2, H, W, T] (2 for ON/OFF channels)
        """
        x = events['x']
        y = events['y']
        t = events['t']
        p = events['p']  # Polarity: 1 for ON, 0 for OFF
        
        # Normalize time to [0, time_steps-1]
        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t_norm = ((t - t_min) / (t_max - t_min) * (time_steps - 1)).astype(int)
        else:
            t_norm = np.zeros_like(t, dtype=int)
        
        # Downsample spatial coordinates
        x_down = x // self.spatial_downsample
        y_down = y // self.spatial_downsample
        
        # Filter out events outside bounds
        valid_mask = (
            (x_down >= 0) & (x_down < self.output_size[0]) &
            (y_down >= 0) & (y_down < self.output_size[1]) &
            (t_norm >= 0) & (t_norm < time_steps)
        )
        
        x_down = x_down[valid_mask]
        y_down = y_down[valid_mask]
        t_norm = t_norm[valid_mask]
        p_filt = p[valid_mask]
        
        # Create spike tensor
        spike_tensor = torch.zeros(2, self.output_size[0], self.output_size[1], time_steps)
        
        # Fill spike tensor
        for i in range(len(x_down)):
            channel = int(p_filt[i])  # 0 for OFF, 1 for ON
            spike_tensor[channel, x_down[i], y_down[i], t_norm[i]] = 1.0
        
        return spike_tensor
    
    def forward(self, events: Dict[str, np.ndarray], time_steps: int = 100) -> torch.Tensor:
        """Process DVS events into spike tensor format."""
        return self.events_to_spike_tensor(events, time_steps)