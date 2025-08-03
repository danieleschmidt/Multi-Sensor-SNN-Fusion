"""
Audio Preprocessing for Neuromorphic Processing

Implements cochlear models and binaural processing for converting
audio signals to spike-based representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from typing import Tuple, Optional, Dict, Any
import math


class CochlearModel(nn.Module):
    """
    Biologically-inspired cochlear model for audio spike encoding.
    
    Converts binaural audio signals into spike trains using gammatone
    filterbanks and hair cell dynamics modeling.
    """
    
    def __init__(
        self,
        sample_rate: int = 48000,
        n_channels: int = 64,
        freq_range: Tuple[float, float] = (100.0, 8000.0),
        q_factor: float = 9.26,
        tau_hair_cell: float = 0.5,
        threshold_adapt: bool = True,
        refractory_period: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize cochlear model.
        
        Args:
            sample_rate: Audio sampling rate (Hz)
            n_channels: Number of frequency channels
            freq_range: Frequency range (low, high) in Hz
            q_factor: Q factor for gammatone filters
            tau_hair_cell: Hair cell time constant (ms)
            threshold_adapt: Enable adaptive thresholding
            refractory_period: Refractory period (ms)
            device: Computation device
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.freq_range = freq_range
        self.q_factor = q_factor
        self.tau_hair_cell = tau_hair_cell
        self.threshold_adapt = threshold_adapt
        self.refractory_period = refractory_period
        self.device = device or torch.device('cpu')
        
        # Create gammatone filterbank
        self.center_freqs = self._create_center_frequencies()
        self.gammatone_filters = self._create_gammatone_filters()
        
        # Hair cell dynamics
        self.hair_cell_lpf = self._create_hair_cell_filter()
        
        # Adaptive thresholding
        if threshold_adapt:
            self.register_buffer(
                'adaptive_thresholds',
                torch.ones(n_channels, device=self.device) * 0.1
            )
            self.threshold_adaptation_rate = 0.01
        
        # Refractory state
        self.register_buffer(
            'refractory_timers',
            torch.zeros(n_channels, device=self.device)
        )
        
        self.to(self.device)
    
    def _create_center_frequencies(self) -> torch.Tensor:
        """Create logarithmically spaced center frequencies."""
        freq_min, freq_max = self.freq_range
        
        # ERB (Equivalent Rectangular Bandwidth) scale
        erb_min = self._hz_to_erb(freq_min)
        erb_max = self._hz_to_erb(freq_max)
        
        erb_freqs = torch.linspace(erb_min, erb_max, self.n_channels, device=self.device)
        center_freqs = self._erb_to_hz(erb_freqs)
        
        return center_freqs
    
    def _hz_to_erb(self, freq_hz: float) -> float:
        """Convert frequency in Hz to ERB scale."""
        return 21.4 * np.log10(0.00437 * freq_hz + 1.0)
    
    def _erb_to_hz(self, erb: torch.Tensor) -> torch.Tensor:
        """Convert ERB scale to frequency in Hz."""
        return (10 ** (erb / 21.4) - 1.0) / 0.00437
    
    def _create_gammatone_filters(self) -> nn.ModuleList:
        """Create gammatone filter bank."""
        filters = nn.ModuleList()
        
        for center_freq in self.center_freqs:
            filter_module = GammatoneFilter(
                center_freq=center_freq.item(),
                sample_rate=self.sample_rate,
                q_factor=self.q_factor,
                device=self.device,
            )
            filters.append(filter_module)
        
        return filters
    
    def _create_hair_cell_filter(self) -> nn.Module:
        """Create hair cell low-pass filter."""
        # Simple first-order low-pass filter
        cutoff_freq = 1000.0 / (2 * math.pi * self.tau_hair_cell)  # Convert time constant to frequency
        return LowPassFilter(cutoff_freq, self.sample_rate, self.device)
    
    def forward(self, audio_signal: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process audio signal through cochlear model.
        
        Args:
            audio_signal: Input audio [batch_size, n_samples] or [batch_size, 2, n_samples] for stereo
            
        Returns:
            spike_trains: Output spikes [batch_size, n_samples, n_channels * n_ears]
            states: Dictionary of internal states
        """
        if audio_signal.dim() == 2:
            # Mono audio - duplicate for binaural
            audio_signal = audio_signal.unsqueeze(1).repeat(1, 2, 1)
        elif audio_signal.dim() == 3 and audio_signal.shape[1] == 1:
            # Mono audio in stereo format
            audio_signal = audio_signal.repeat(1, 2, 1)
        
        batch_size, n_ears, n_samples = audio_signal.shape
        
        # Process each ear separately
        all_spikes = []
        channel_responses = []
        
        for ear in range(n_ears):
            ear_audio = audio_signal[:, ear, :]  # [batch_size, n_samples]
            ear_spikes, ear_responses = self._process_ear(ear_audio)
            all_spikes.append(ear_spikes)
            channel_responses.append(ear_responses)
        
        # Concatenate binaural channels
        spike_trains = torch.cat(all_spikes, dim=-1)  # [batch_size, n_samples, n_channels * 2]
        
        states = {
            'channel_responses': torch.stack(channel_responses, dim=1),  # [batch_size, 2, n_samples, n_channels]
            'center_frequencies': self.center_freqs,
            'adaptive_thresholds': self.adaptive_thresholds.clone() if self.threshold_adapt else None,
        }
        
        return spike_trains, states
    
    def _process_ear(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process single ear audio through cochlear model."""
        batch_size, n_samples = audio.shape
        
        # Apply gammatone filterbank
        channel_responses = torch.zeros(batch_size, n_samples, self.n_channels, device=self.device)
        
        for i, gammatone_filter in enumerate(self.gammatone_filters):
            filtered_audio = gammatone_filter(audio)
            channel_responses[:, :, i] = filtered_audio
        
        # Hair cell processing (rectification + low-pass filtering)
        rectified = torch.relu(channel_responses)
        hair_cell_output = self.hair_cell_lpf(rectified)
        
        # Spike generation with adaptive thresholding
        spikes = self._generate_spikes(hair_cell_output)
        
        return spikes, hair_cell_output
    
    def _generate_spikes(self, hair_cell_output: torch.Tensor) -> torch.Tensor:
        """Generate spikes from hair cell output using adaptive thresholding."""
        batch_size, n_samples, n_channels = hair_cell_output.shape
        
        spikes = torch.zeros_like(hair_cell_output)
        
        for t in range(n_samples):
            # Current hair cell activity
            current_activity = hair_cell_output[:, t, :]  # [batch_size, n_channels]
            
            # Apply adaptive thresholds
            if self.threshold_adapt:
                thresholds = self.adaptive_thresholds.unsqueeze(0)  # [1, n_channels]
            else:
                thresholds = torch.ones(1, n_channels, device=self.device) * 0.1
            
            # Check refractory period
            refractory_mask = self.refractory_timers > 0
            
            # Generate spikes
            spike_mask = (current_activity > thresholds) & ~refractory_mask.unsqueeze(0)
            spikes[:, t, :] = spike_mask.float()
            
            # Update refractory timers
            self.refractory_timers[spike_mask.any(dim=0)] = self.refractory_period
            self.refractory_timers = torch.clamp(self.refractory_timers - 1, min=0)
            
            # Update adaptive thresholds
            if self.threshold_adapt:
                # Increase threshold for spiking channels
                spike_increment = spike_mask.float().mean(dim=0) * self.threshold_adaptation_rate
                self.adaptive_thresholds += spike_increment
                
                # Decay thresholds gradually
                self.adaptive_thresholds *= 0.9999
                self.adaptive_thresholds = torch.clamp(self.adaptive_thresholds, min=0.01, max=1.0)
        
        return spikes
    
    def reset_state(self) -> None:
        """Reset all internal states."""
        if self.threshold_adapt:
            self.adaptive_thresholds.fill_(0.1)
        self.refractory_timers.fill_(0.0)


class GammatoneFilter(nn.Module):
    """
    Gammatone filter implementation for cochlear modeling.
    """
    
    def __init__(
        self,
        center_freq: float,
        sample_rate: int,
        q_factor: float = 9.26,
        order: int = 4,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize gammatone filter.
        
        Args:
            center_freq: Center frequency (Hz)
            sample_rate: Sampling rate (Hz)
            q_factor: Q factor for bandwidth
            order: Filter order
            device: Computation device
        """
        super().__init__()
        
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.q_factor = q_factor
        self.order = order
        self.device = device or torch.device('cpu')
        
        # Compute filter parameters
        self.bandwidth = center_freq / q_factor
        self.dt = 1.0 / sample_rate
        
        # Create impulse response
        self.impulse_response = self._create_impulse_response()
        
        self.to(self.device)
    
    def _create_impulse_response(self) -> torch.Tensor:
        """Create gammatone impulse response."""
        # Length of impulse response (4 time constants)
        tau = 1.0 / (2 * math.pi * self.bandwidth)
        duration = 4 * tau
        n_samples = int(duration * self.sample_rate)
        
        t = torch.arange(n_samples, device=self.device, dtype=torch.float) * self.dt
        
        # Gammatone impulse response
        envelope = (t ** (self.order - 1)) * torch.exp(-2 * math.pi * self.bandwidth * t)
        carrier = torch.cos(2 * math.pi * self.center_freq * t)
        
        impulse_response = envelope * carrier
        
        # Normalize
        impulse_response = impulse_response / impulse_response.abs().max()
        
        return impulse_response
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gammatone filter via convolution."""
        # Ensure input is 2D [batch_size, n_samples]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size, n_samples = x.shape
        
        # Pad input for convolution
        padding = len(self.impulse_response) - 1
        x_padded = F.pad(x, (padding, 0))
        
        # Apply convolution
        filtered = F.conv1d(
            x_padded.unsqueeze(1),  # Add channel dimension
            self.impulse_response.view(1, 1, -1),  # Filter weights
            padding=0
        ).squeeze(1)  # Remove channel dimension
        
        # Trim to original length
        filtered = filtered[:, :n_samples]
        
        return filtered


class LowPassFilter(nn.Module):
    """First-order low-pass filter for hair cell modeling."""
    
    def __init__(
        self,
        cutoff_freq: float,
        sample_rate: int,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.device = device or torch.device('cpu')
        
        # Compute filter coefficient
        dt = 1.0 / sample_rate
        tau = 1.0 / (2 * math.pi * cutoff_freq)
        self.alpha = dt / (dt + tau)
        
        # State variable
        self.register_buffer('state', torch.tensor(0.0, device=self.device))
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply low-pass filter."""
        if x.dim() == 2:
            batch_size, n_samples = x.shape
            output = torch.zeros_like(x)
            
            for t in range(n_samples):
                self.state = self.alpha * x[:, t] + (1 - self.alpha) * self.state
                output[:, t] = self.state
            
            return output
        elif x.dim() == 3:
            batch_size, n_samples, n_channels = x.shape
            output = torch.zeros_like(x)
            
            # Initialize state for each channel
            if self.state.numel() == 1:
                self.state = self.state.expand(batch_size, n_channels).contiguous()
            
            for t in range(n_samples):
                self.state = self.alpha * x[:, t, :] + (1 - self.alpha) * self.state
                output[:, t, :] = self.state
            
            return output
        else:
            raise ValueError(f"Unsupported input dimensionality: {x.dim()}")


class BiauralProcessor(nn.Module):
    """
    Binaural audio processing for spatial hearing.
    
    Extracts interaural time differences (ITD) and interaural level 
    differences (ILD) for spatial audio processing.
    """
    
    def __init__(
        self,
        max_itd_samples: int = 32,
        n_itd_channels: int = 16,
        n_ild_channels: int = 8,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize binaural processor.
        
        Args:
            max_itd_samples: Maximum ITD in samples
            n_itd_channels: Number of ITD-selective channels
            n_ild_channels: Number of ILD-selective channels
            device: Computation device
        """
        super().__init__()
        
        self.max_itd_samples = max_itd_samples
        self.n_itd_channels = n_itd_channels
        self.n_ild_channels = n_ild_channels
        self.device = device or torch.device('cpu')
        
        # ITD delay lines
        self.itd_delays = torch.linspace(
            -max_itd_samples, max_itd_samples, n_itd_channels, device=self.device
        ).round().int()
        
        # ILD level differences
        self.ild_preferences = torch.linspace(
            -1.0, 1.0, n_ild_channels, device=self.device
        )
        
        self.to(self.device)
    
    def forward(
        self, 
        left_spikes: torch.Tensor, 
        right_spikes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process binaural spike trains.
        
        Args:
            left_spikes: Left ear spikes [batch_size, n_samples, n_freq_channels]
            right_spikes: Right ear spikes [batch_size, n_samples, n_freq_channels]
            
        Returns:
            itd_response: ITD-selective responses [batch_size, n_samples, n_itd_channels]
            ild_response: ILD-selective responses [batch_size, n_samples, n_ild_channels]
        """
        batch_size, n_samples, n_freq_channels = left_spikes.shape
        
        # Compute ITD responses
        itd_response = self._compute_itd_response(left_spikes, right_spikes)
        
        # Compute ILD responses  
        ild_response = self._compute_ild_response(left_spikes, right_spikes)
        
        return itd_response, ild_response
    
    def _compute_itd_response(
        self, 
        left_spikes: torch.Tensor, 
        right_spikes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ITD-selective responses using cross-correlation."""
        batch_size, n_samples, n_freq_channels = left_spikes.shape
        
        itd_responses = torch.zeros(
            batch_size, n_samples, self.n_itd_channels, device=self.device
        )
        
        for i, delay in enumerate(self.itd_delays):
            # Apply delay to appropriate ear
            if delay >= 0:
                # Delay left ear
                left_delayed = F.pad(left_spikes, (0, 0, delay.item(), 0))[:, :n_samples, :]
                right_ref = right_spikes
            else:
                # Delay right ear
                left_ref = left_spikes
                right_delayed = F.pad(right_spikes, (0, 0, -delay.item(), 0))[:, :n_samples, :]
                right_ref = right_delayed
                left_ref = left_spikes
            
            # Cross-correlation (simplified as element-wise product)
            if delay >= 0:
                correlation = left_delayed * right_ref
            else:
                correlation = left_ref * right_delayed
            
            # Sum across frequency channels
            itd_responses[:, :, i] = correlation.sum(dim=-1)
        
        return itd_responses
    
    def _compute_ild_response(
        self, 
        left_spikes: torch.Tensor, 
        right_spikes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ILD-selective responses."""
        batch_size, n_samples, n_freq_channels = left_spikes.shape
        
        # Compute instantaneous level differences
        left_level = left_spikes.sum(dim=-1, keepdim=True)  # [batch_size, n_samples, 1]
        right_level = right_spikes.sum(dim=-1, keepdim=True)
        
        # Normalize levels
        total_level = left_level + right_level + 1e-6  # Avoid division by zero
        ild = (left_level - right_level) / total_level  # Range [-1, 1]
        
        # Compute responses for each ILD preference
        ild_responses = torch.zeros(
            batch_size, n_samples, self.n_ild_channels, device=self.device
        )
        
        for i, preference in enumerate(self.ild_preferences):
            # Gaussian tuning around preferred ILD
            response = torch.exp(-0.5 * ((ild.squeeze(-1) - preference) / 0.3) ** 2)
            ild_responses[:, :, i] = response
        
        return ild_responses


class AudioSpikeEncoder(nn.Module):
    """
    Complete audio preprocessing pipeline for neuromorphic processing.
    """
    
    def __init__(
        self,
        sample_rate: int = 48000,
        n_cochlear_channels: int = 64,
        enable_binaural: bool = True,
        enable_preprocessing: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize audio spike encoder.
        
        Args:
            sample_rate: Audio sampling rate
            n_cochlear_channels: Number of cochlear channels per ear
            enable_binaural: Enable binaural processing
            enable_preprocessing: Enable audio preprocessing
            device: Computation device
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_cochlear_channels = n_cochlear_channels
        self.enable_binaural = enable_binaural
        self.enable_preprocessing = enable_preprocessing
        self.device = device or torch.device('cpu')
        
        # Cochlear model
        self.cochlear_model = CochlearModel(
            sample_rate=sample_rate,
            n_channels=n_cochlear_channels,
            device=device,
        )
        
        # Binaural processor
        if enable_binaural:
            self.binaural_processor = BiauralProcessor(device=device)
        
        self.to(self.device)
    
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete audio preprocessing pipeline.
        
        Args:
            audio: Raw audio signal [batch_size, n_samples] or [batch_size, 2, n_samples]
            
        Returns:
            spike_data: Dictionary containing various spike representations
        """
        # Cochlear processing
        cochlear_spikes, cochlear_states = self.cochlear_model(audio)
        
        result = {
            'cochlear_spikes': cochlear_spikes,
            'cochlear_states': cochlear_states,
        }
        
        # Binaural processing
        if self.enable_binaural and audio.shape[1] == 2:
            # Split stereo channels
            left_spikes = cochlear_spikes[:, :, :self.n_cochlear_channels]
            right_spikes = cochlear_spikes[:, :, self.n_cochlear_channels:]
            
            itd_response, ild_response = self.binaural_processor(left_spikes, right_spikes)
            
            result.update({
                'itd_response': itd_response,
                'ild_response': ild_response,
                'left_spikes': left_spikes,
                'right_spikes': right_spikes,
            })
        
        return result