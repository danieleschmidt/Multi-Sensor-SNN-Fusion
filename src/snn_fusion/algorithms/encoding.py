"""
Spike Encoding Algorithms

Implements various spike encoding methods for converting sensory data
into spike trains suitable for neuromorphic processing.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from enum import Enum

try:
    import scipy.signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class EncodingType(Enum):
    """Types of spike encoding."""
    RATE = "rate"
    TEMPORAL = "temporal"
    POPULATION = "population"
    RANK_ORDER = "rank_order"
    COCHLEAR = "cochlear"
    DVS = "dvs"
    TACTILE = "tactile"


@dataclass
class SpikeData:
    """Spike train data structure."""
    spike_times: np.ndarray  # Spike timing in ms
    neuron_ids: np.ndarray   # Neuron identifiers
    duration: float          # Total duration in ms
    n_neurons: int          # Number of neurons
    metadata: Optional[Dict[str, Any]] = None


class SpikeEncoder(ABC):
    """
    Abstract base class for spike encoders.
    
    Provides common interface for converting sensory data into
    spike trains for neuromorphic processing.
    """
    
    def __init__(
        self,
        n_neurons: int,
        duration: float = 100.0,  # ms
        dt: float = 1.0,         # ms
    ):
        """
        Initialize spike encoder.
        
        Args:
            n_neurons: Number of encoding neurons
            duration: Encoding duration in ms
            dt: Time resolution in ms
        """
        self.n_neurons = n_neurons
        self.duration = duration
        self.dt = dt
        self.logger = logging.getLogger(__name__)
        
        # Time vector
        self.time_vector = np.arange(0, duration, dt)
        self.n_timesteps = len(self.time_vector)
    
    @abstractmethod
    def encode(self, data: np.ndarray) -> SpikeData:
        """
        Encode input data into spike trains.
        
        Args:
            data: Input data to encode
            
        Returns:
            Encoded spike data
        """
        pass
    
    def encode_batch(self, data_batch: np.ndarray) -> List[SpikeData]:
        """
        Encode batch of data samples.
        
        Args:
            data_batch: Batch of input data
            
        Returns:
            List of encoded spike data
        """
        return [self.encode(data) for data in data_batch]
    
    def _create_spike_data(
        self,
        spike_matrix: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SpikeData:
        """
        Create SpikeData from binary spike matrix.
        
        Args:
            spike_matrix: Binary matrix (neurons x time)
            metadata: Optional metadata
            
        Returns:
            SpikeData object
        """
        spike_times = []
        neuron_ids = []
        
        for neuron_id in range(spike_matrix.shape[0]):
            spike_indices = np.where(spike_matrix[neuron_id, :])[0]
            spike_times.extend(spike_indices * self.dt)
            neuron_ids.extend([neuron_id] * len(spike_indices))
        
        return SpikeData(
            spike_times=np.array(spike_times),
            neuron_ids=np.array(neuron_ids),
            duration=self.duration,
            n_neurons=self.n_neurons,
            metadata=metadata or {}
        )


class RateEncoder(SpikeEncoder):
    """
    Rate-based spike encoder using Poisson processes.
    
    Encodes input values as Poisson spike trains with rates
    proportional to input magnitudes.
    """
    
    def __init__(
        self,
        n_neurons: int,
        duration: float = 100.0,
        dt: float = 1.0,
        max_rate: float = 100.0,  # Hz
        min_rate: float = 0.0,    # Hz
    ):
        """
        Initialize rate encoder.
        
        Args:
            n_neurons: Number of encoding neurons
            duration: Encoding duration in ms
            dt: Time resolution in ms
            max_rate: Maximum firing rate in Hz
            min_rate: Minimum firing rate in Hz
        """
        super().__init__(n_neurons, duration, dt)
        self.max_rate = max_rate
        self.min_rate = min_rate
    
    def encode(self, data: np.ndarray) -> SpikeData:
        """
        Encode data using rate coding.
        
        Args:
            data: Input data (1D array)
            
        Returns:
            Encoded spike data
        """
        try:
            # Normalize data to [0, 1]
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
            
            # Map to firing rates
            rates = self.min_rate + data_norm * (self.max_rate - self.min_rate)
            
            # Ensure we have enough neurons
            if len(rates) > self.n_neurons:
                # Downsample or group data
                rates = self._downsample_data(rates)
            elif len(rates) < self.n_neurons:
                # Repeat or interpolate data
                rates = self._upsample_data(rates)
            
            # Generate Poisson spike trains
            spike_matrix = np.zeros((self.n_neurons, self.n_timesteps))
            
            for neuron_id in range(self.n_neurons):
                rate_hz = rates[neuron_id]
                prob_per_ms = rate_hz / 1000.0  # Convert Hz to probability per ms
                
                # Generate spikes based on Poisson process
                random_values = np.random.random(self.n_timesteps)
                spike_matrix[neuron_id, :] = random_values < prob_per_ms
            
            metadata = {
                'encoding_type': 'rate',
                'rates': rates,
                'max_rate': self.max_rate,
                'min_rate': self.min_rate,
            }
            
            return self._create_spike_data(spike_matrix, metadata)
            
        except Exception as e:
            self.logger.error(f"Failed to encode data with rate encoder: {e}")
            raise
    
    def _downsample_data(self, data: np.ndarray) -> np.ndarray:
        """Downsample data to match number of neurons."""
        # Simple binning approach
        bin_size = len(data) // self.n_neurons
        downsampled = []
        
        for i in range(self.n_neurons):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size if i < self.n_neurons - 1 else len(data)
            downsampled.append(np.mean(data[start_idx:end_idx]))
        
        return np.array(downsampled)
    
    def _upsample_data(self, data: np.ndarray) -> np.ndarray:
        """Upsample data to match number of neurons."""
        # Linear interpolation
        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, self.n_neurons)
        upsampled = np.interp(x_new, x_old, data)
        
        return upsampled


class TemporalEncoder(SpikeEncoder):
    """
    Temporal spike encoder using time-to-first-spike coding.
    
    Encodes input values as spike timing, where larger values
    generate earlier spikes.
    """
    
    def __init__(
        self,
        n_neurons: int,
        duration: float = 100.0,
        dt: float = 1.0,
        encoding_window: float = 50.0,  # ms
    ):
        """
        Initialize temporal encoder.
        
        Args:
            n_neurons: Number of encoding neurons
            duration: Total duration in ms
            dt: Time resolution in ms
            encoding_window: Time window for encoding in ms
        """
        super().__init__(n_neurons, duration, dt)
        self.encoding_window = encoding_window
    
    def encode(self, data: np.ndarray) -> SpikeData:
        """
        Encode data using temporal coding.
        
        Args:
            data: Input data (1D array)
            
        Returns:
            Encoded spike data
        """
        try:
            # Normalize data to [0, 1]
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
            
            # Adjust data size to match neurons
            if len(data_norm) > self.n_neurons:
                data_norm = self._downsample_data(data_norm)
            elif len(data_norm) < self.n_neurons:
                data_norm = self._upsample_data(data_norm)
            
            # Calculate spike times (larger values -> earlier spikes)
            spike_times = self.encoding_window * (1.0 - data_norm)
            
            # Create spike matrix
            spike_matrix = np.zeros((self.n_neurons, self.n_timesteps))
            
            for neuron_id in range(self.n_neurons):
                spike_time = spike_times[neuron_id]
                spike_timestep = int(spike_time / self.dt)
                
                if 0 <= spike_timestep < self.n_timesteps:
                    spike_matrix[neuron_id, spike_timestep] = 1
            
            metadata = {
                'encoding_type': 'temporal',
                'spike_times': spike_times,
                'encoding_window': self.encoding_window,
            }
            
            return self._create_spike_data(spike_matrix, metadata)
            
        except Exception as e:
            self.logger.error(f"Failed to encode data with temporal encoder: {e}")
            raise
    
    def _downsample_data(self, data: np.ndarray) -> np.ndarray:
        """Downsample data to match number of neurons."""
        bin_size = len(data) // self.n_neurons
        return np.array([np.mean(data[i*bin_size:(i+1)*bin_size]) for i in range(self.n_neurons)])
    
    def _upsample_data(self, data: np.ndarray) -> np.ndarray:
        """Upsample data to match number of neurons."""
        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, self.n_neurons)
        return np.interp(x_new, x_old, data)


class CochlearEncoder(SpikeEncoder):
    """
    Cochlear-inspired spike encoder for audio signals.
    
    Implements frequency decomposition similar to the cochlea,
    followed by spike generation based on energy in each band.
    """
    
    def __init__(
        self,
        n_neurons: int,
        duration: float = 100.0,
        dt: float = 1.0,
        sample_rate: int = 16000,
        freq_range: Tuple[float, float] = (80, 8000),  # Hz
    ):
        """
        Initialize cochlear encoder.
        
        Args:
            n_neurons: Number of encoding neurons (frequency channels)
            duration: Encoding duration in ms
            dt: Time resolution in ms
            sample_rate: Audio sample rate in Hz
            freq_range: Frequency range (min_freq, max_freq) in Hz
        """
        super().__init__(n_neurons, duration, dt)
        self.sample_rate = sample_rate
        self.freq_range = freq_range
        
        # Create frequency channels (logarithmic spacing)
        self.center_freqs = np.logspace(
            np.log10(freq_range[0]),
            np.log10(freq_range[1]),
            n_neurons
        )
        
        # Create filterbank if scipy is available
        if SCIPY_AVAILABLE:
            self.filterbank = self._create_filterbank()
        else:
            self.filterbank = None
            self.logger.warning("Scipy not available - using simplified cochlear encoding")
    
    def encode(self, audio_data: np.ndarray) -> SpikeData:
        """
        Encode audio data using cochlear model.
        
        Args:
            audio_data: Audio signal (1D array)
            
        Returns:
            Encoded spike data
        """
        try:
            # Resample audio to match duration if needed
            target_samples = int(self.duration * self.sample_rate / 1000.0)
            if len(audio_data) != target_samples:
                audio_data = self._resample_audio(audio_data, target_samples)
            
            # Apply filterbank
            if self.filterbank is not None:
                filtered_signals = self._apply_filterbank(audio_data)
            else:
                filtered_signals = self._simple_frequency_decomposition(audio_data)
            
            # Generate spikes based on energy in each channel
            spike_matrix = self._energy_to_spikes(filtered_signals)
            
            metadata = {
                'encoding_type': 'cochlear',
                'center_frequencies': self.center_freqs,
                'sample_rate': self.sample_rate,
                'freq_range': self.freq_range,
            }
            
            return self._create_spike_data(spike_matrix, metadata)
            
        except Exception as e:
            self.logger.error(f"Failed to encode audio with cochlear encoder: {e}")
            raise
    
    def _create_filterbank(self):
        """Create gammatone filterbank for cochlear processing."""
        if not SCIPY_AVAILABLE:
            return None
        
        # Simplified gammatone filterbank
        filterbank = []
        
        for center_freq in self.center_freqs:
            # Create bandpass filter around center frequency
            bandwidth = center_freq * 0.25  # 25% bandwidth
            low_freq = max(center_freq - bandwidth/2, 1)
            high_freq = min(center_freq + bandwidth/2, self.sample_rate/2 - 1)
            
            # Butterworth bandpass filter
            sos = scipy.signal.butter(
                4, [low_freq, high_freq],
                btype='band',
                fs=self.sample_rate,
                output='sos'
            )
            filterbank.append(sos)
        
        return filterbank
    
    def _apply_filterbank(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply filterbank to audio signal."""
        if not SCIPY_AVAILABLE or self.filterbank is None:
            return self._simple_frequency_decomposition(audio_data)
        
        filtered_signals = np.zeros((self.n_neurons, len(audio_data)))
        
        for i, sos in enumerate(self.filterbank):
            try:
                filtered = scipy.signal.sosfilt(sos, audio_data)
                filtered_signals[i, :] = filtered
            except Exception as e:
                self.logger.warning(f"Filter {i} failed: {e}")
                # Fallback to original signal
                filtered_signals[i, :] = audio_data
        
        return filtered_signals
    
    def _simple_frequency_decomposition(self, audio_data: np.ndarray) -> np.ndarray:
        """Simple frequency decomposition using FFT."""
        # Use overlapping windows for frequency analysis
        window_size = min(1024, len(audio_data) // 4)
        hop_size = window_size // 2
        
        filtered_signals = np.zeros((self.n_neurons, len(audio_data)))
        
        # Create frequency bins
        freq_bins = np.fft.fftfreq(window_size, 1/self.sample_rate)
        positive_freqs = freq_bins[:window_size//2]
        
        for i, center_freq in enumerate(self.center_freqs):
            # Find closest frequency bin
            freq_idx = np.argmin(np.abs(positive_freqs - center_freq))
            bandwidth = max(1, window_size // 32)  # Bandwidth in bins
            
            # Process windows
            for start in range(0, len(audio_data) - window_size, hop_size):
                window = audio_data[start:start + window_size]
                
                # Apply window function
                windowed = window * np.hanning(len(window))
                
                # FFT
                fft_data = np.fft.fft(windowed)
                
                # Extract energy in frequency band
                start_bin = max(0, freq_idx - bandwidth//2)
                end_bin = min(len(fft_data)//2, freq_idx + bandwidth//2)
                
                energy = np.sum(np.abs(fft_data[start_bin:end_bin])**2)
                
                # Assign energy to output
                end_sample = min(start + window_size, len(audio_data))
                filtered_signals[i, start:end_sample] += energy / window_size
        
        return filtered_signals
    
    def _energy_to_spikes(self, filtered_signals: np.ndarray) -> np.ndarray:
        """Convert energy signals to spike trains."""
        # Calculate time windows for spike generation
        samples_per_timestep = int(self.sample_rate * self.dt / 1000.0)
        
        spike_matrix = np.zeros((self.n_neurons, self.n_timesteps))
        
        for neuron_id in range(self.n_neurons):
            signal = filtered_signals[neuron_id, :]
            
            # Calculate energy in each time window
            for t in range(self.n_timesteps):
                start_sample = t * samples_per_timestep
                end_sample = min(start_sample + samples_per_timestep, len(signal))
                
                if end_sample > start_sample:
                    energy = np.mean(signal[start_sample:end_sample]**2)
                    
                    # Convert energy to spike probability
                    # Higher energy -> higher spike probability
                    spike_prob = np.clip(energy / (np.max(signal)**2 + 1e-8), 0, 1)
                    
                    # Generate spike
                    if np.random.random() < spike_prob:
                        spike_matrix[neuron_id, t] = 1
        
        return spike_matrix
    
    def _resample_audio(self, audio_data: np.ndarray, target_samples: int) -> np.ndarray:
        """Resample audio to target length."""
        if SCIPY_AVAILABLE:
            # Use scipy resampling
            return scipy.signal.resample(audio_data, target_samples)
        else:
            # Simple linear interpolation
            x_old = np.linspace(0, 1, len(audio_data))
            x_new = np.linspace(0, 1, target_samples)
            return np.interp(x_new, x_old, audio_data)


class DVSEncoder(SpikeEncoder):
    """
    Dynamic Vision Sensor (DVS) inspired encoder for visual data.
    
    Encodes changes in visual intensity as ON/OFF spike events,
    mimicking the behavior of event-based cameras.
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        duration: float = 100.0,
        dt: float = 1.0,
        threshold: float = 0.1,
        polarity: bool = True,
    ):
        """
        Initialize DVS encoder.
        
        Args:
            width: Image width
            height: Image height
            duration: Encoding duration in ms
            dt: Time resolution in ms
            threshold: Change detection threshold
            polarity: Whether to encode both ON and OFF events
        """
        n_neurons = width * height * (2 if polarity else 1)
        super().__init__(n_neurons, duration, dt)
        
        self.width = width
        self.height = height
        self.threshold = threshold
        self.polarity = polarity
        
        # Previous frame for change detection
        self.previous_frame = None
    
    def encode(self, image_sequence: np.ndarray) -> SpikeData:
        """
        Encode image sequence using DVS model.
        
        Args:
            image_sequence: Sequence of images (time, height, width)
            
        Returns:
            Encoded spike data
        """
        try:
            if image_sequence.ndim == 2:
                # Single image - create artificial sequence
                image_sequence = np.expand_dims(image_sequence, 0)
            
            # Resize images if needed
            if image_sequence.shape[1] != self.height or image_sequence.shape[2] != self.width:
                image_sequence = self._resize_images(image_sequence)
            
            # Generate spikes from image changes
            spike_matrix = self._detect_changes(image_sequence)
            
            metadata = {
                'encoding_type': 'dvs',
                'width': self.width,
                'height': self.height,
                'threshold': self.threshold,
                'polarity': self.polarity,
            }
            
            return self._create_spike_data(spike_matrix, metadata)
            
        except Exception as e:
            self.logger.error(f"Failed to encode images with DVS encoder: {e}")
            raise
    
    def _detect_changes(self, image_sequence: np.ndarray) -> np.ndarray:
        """Detect intensity changes and generate spikes."""
        n_frames = image_sequence.shape[0]
        
        # Initialize spike matrix
        if self.polarity:
            spike_matrix = np.zeros((self.width * self.height * 2, self.n_timesteps))
        else:
            spike_matrix = np.zeros((self.width * self.height, self.n_timesteps))
        
        # Process each frame
        for frame_idx in range(n_frames):
            current_frame = image_sequence[frame_idx]
            
            if self.previous_frame is not None:
                # Calculate intensity changes
                change = current_frame - self.previous_frame
                
                # Generate spikes for significant changes
                self._generate_change_spikes(change, frame_idx, spike_matrix)
            
            self.previous_frame = current_frame.copy()
        
        return spike_matrix
    
    def _generate_change_spikes(
        self,
        change: np.ndarray,
        frame_idx: int,
        spike_matrix: np.ndarray,
    ) -> None:
        """Generate spikes for intensity changes."""
        # Map frame index to time index
        time_idx = min(frame_idx * self.n_timesteps // 10, self.n_timesteps - 1)
        
        for y in range(self.height):
            for x in range(self.width):
                pixel_change = change[y, x]
                
                if abs(pixel_change) > self.threshold:
                    # Calculate neuron index
                    pixel_idx = y * self.width + x
                    
                    if self.polarity:
                        if pixel_change > 0:
                            # ON event
                            neuron_idx = pixel_idx * 2
                        else:
                            # OFF event
                            neuron_idx = pixel_idx * 2 + 1
                    else:
                        neuron_idx = pixel_idx
                    
                    # Generate spike
                    if neuron_idx < spike_matrix.shape[0]:
                        spike_matrix[neuron_idx, time_idx] = 1
    
    def _resize_images(self, image_sequence: np.ndarray) -> np.ndarray:
        """Resize image sequence to target dimensions."""
        # Simple nearest neighbor resampling
        resized = np.zeros((image_sequence.shape[0], self.height, self.width))
        
        for i, frame in enumerate(image_sequence):
            # Calculate scaling factors
            y_scale = frame.shape[0] / self.height
            x_scale = frame.shape[1] / self.width
            
            for y in range(self.height):
                for x in range(self.width):
                    src_y = int(y * y_scale)
                    src_x = int(x * x_scale)
                    
                    src_y = min(src_y, frame.shape[0] - 1)
                    src_x = min(src_x, frame.shape[1] - 1)
                    
                    resized[i, y, x] = frame[src_y, src_x]
        
        return resized


class TactileEncoder(SpikeEncoder):
    """
    Tactile sensor encoder for pressure and texture data.
    
    Encodes tactile sensor readings into spike trains based on
    pressure magnitude and temporal changes.
    """
    
    def __init__(
        self,
        n_sensors: int,
        duration: float = 100.0,
        dt: float = 1.0,
        pressure_sensitivity: float = 0.1,
        temporal_sensitivity: float = 0.05,
    ):
        """
        Initialize tactile encoder.
        
        Args:
            n_sensors: Number of tactile sensors
            duration: Encoding duration in ms
            dt: Time resolution in ms
            pressure_sensitivity: Sensitivity to pressure magnitude
            temporal_sensitivity: Sensitivity to pressure changes
        """
        super().__init__(n_sensors * 2, duration, dt)  # 2 neurons per sensor
        
        self.n_sensors = n_sensors
        self.pressure_sensitivity = pressure_sensitivity
        self.temporal_sensitivity = temporal_sensitivity
        
        # Previous readings for change detection
        self.previous_readings = None
    
    def encode(self, tactile_data: np.ndarray) -> SpikeData:
        """
        Encode tactile sensor data.
        
        Args:
            tactile_data: Tactile readings (time, sensors) or (sensors,)
            
        Returns:
            Encoded spike data
        """
        try:
            if tactile_data.ndim == 1:
                # Single reading - create temporal sequence
                tactile_data = np.tile(tactile_data, (self.n_timesteps, 1))
            
            # Ensure correct number of sensors
            if tactile_data.shape[1] != self.n_sensors:
                tactile_data = self._adjust_sensor_count(tactile_data)
            
            # Generate spikes
            spike_matrix = self._encode_tactile_spikes(tactile_data)
            
            metadata = {
                'encoding_type': 'tactile',
                'n_sensors': self.n_sensors,
                'pressure_sensitivity': self.pressure_sensitivity,
                'temporal_sensitivity': self.temporal_sensitivity,
            }
            
            return self._create_spike_data(spike_matrix, metadata)
            
        except Exception as e:
            self.logger.error(f"Failed to encode tactile data: {e}")
            raise
    
    def _encode_tactile_spikes(self, tactile_data: np.ndarray) -> np.ndarray:
        """Encode tactile readings into spike trains."""
        # Each sensor has 2 neurons: pressure magnitude and pressure change
        spike_matrix = np.zeros((self.n_sensors * 2, self.n_timesteps))
        
        for t in range(self.n_timesteps):
            time_idx = min(t, tactile_data.shape[0] - 1)
            current_readings = tactile_data[time_idx, :]
            
            for sensor_idx in range(self.n_sensors):
                pressure = current_readings[sensor_idx]
                
                # Pressure magnitude neuron
                magnitude_neuron = sensor_idx * 2
                pressure_rate = pressure * self.pressure_sensitivity
                if np.random.random() < pressure_rate:
                    spike_matrix[magnitude_neuron, t] = 1
                
                # Pressure change neuron
                if self.previous_readings is not None:
                    change = abs(pressure - self.previous_readings[sensor_idx])
                    change_neuron = sensor_idx * 2 + 1
                    change_rate = change * self.temporal_sensitivity
                    
                    if np.random.random() < change_rate:
                        spike_matrix[change_neuron, t] = 1
            
            self.previous_readings = current_readings.copy()
        
        return spike_matrix
    
    def _adjust_sensor_count(self, tactile_data: np.ndarray) -> np.ndarray:
        """Adjust data to match expected number of sensors."""
        current_sensors = tactile_data.shape[1]
        
        if current_sensors > self.n_sensors:
            # Downsample
            indices = np.linspace(0, current_sensors - 1, self.n_sensors, dtype=int)
            return tactile_data[:, indices]
        else:
            # Upsample by interpolation
            x_old = np.linspace(0, 1, current_sensors)
            x_new = np.linspace(0, 1, self.n_sensors)
            
            upsampled = np.zeros((tactile_data.shape[0], self.n_sensors))
            for t in range(tactile_data.shape[0]):
                upsampled[t, :] = np.interp(x_new, x_old, tactile_data[t, :])
            
            return upsampled


class MultiModalEncoder:
    """
    Multi-modal encoder combining different sensory modalities.
    
    Coordinates encoding of audio, visual, and tactile data into
    synchronized spike trains for cross-modal processing.
    """
    
    def __init__(
        self,
        audio_config: Optional[Dict[str, Any]] = None,
        visual_config: Optional[Dict[str, Any]] = None,
        tactile_config: Optional[Dict[str, Any]] = None,
        duration: float = 100.0,
        dt: float = 1.0,
    ):
        """
        Initialize multi-modal encoder.
        
        Args:
            audio_config: Audio encoder configuration
            visual_config: Visual encoder configuration
            tactile_config: Tactile encoder configuration
            duration: Encoding duration in ms
            dt: Time resolution in ms
        """
        self.duration = duration
        self.dt = dt
        self.logger = logging.getLogger(__name__)
        
        # Initialize encoders
        self.encoders = {}
        self.neuron_offsets = {}
        self.total_neurons = 0
        
        if audio_config:
            self.encoders['audio'] = CochlearEncoder(
                duration=duration, dt=dt, **audio_config
            )
            self.neuron_offsets['audio'] = self.total_neurons
            self.total_neurons += self.encoders['audio'].n_neurons
        
        if visual_config:
            self.encoders['visual'] = DVSEncoder(
                duration=duration, dt=dt, **visual_config
            )
            self.neuron_offsets['visual'] = self.total_neurons
            self.total_neurons += self.encoders['visual'].n_neurons
        
        if tactile_config:
            self.encoders['tactile'] = TactileEncoder(
                duration=duration, dt=dt, **tactile_config
            )
            self.neuron_offsets['tactile'] = self.total_neurons
            self.total_neurons += self.encoders['tactile'].n_neurons
        
        self.logger.info(f"Initialized multi-modal encoder with {self.total_neurons} neurons")
    
    def encode(
        self,
        audio_data: Optional[np.ndarray] = None,
        visual_data: Optional[np.ndarray] = None,
        tactile_data: Optional[np.ndarray] = None,
    ) -> SpikeData:
        """
        Encode multi-modal data.
        
        Args:
            audio_data: Audio signal data
            visual_data: Visual image data
            tactile_data: Tactile sensor data
            
        Returns:
            Combined spike data
        """
        try:
            all_spike_times = []
            all_neuron_ids = []
            
            # Encode each modality
            if audio_data is not None and 'audio' in self.encoders:
                audio_spikes = self.encoders['audio'].encode(audio_data)
                offset = self.neuron_offsets['audio']
                
                all_spike_times.extend(audio_spikes.spike_times)
                all_neuron_ids.extend(audio_spikes.neuron_ids + offset)
            
            if visual_data is not None and 'visual' in self.encoders:
                visual_spikes = self.encoders['visual'].encode(visual_data)
                offset = self.neuron_offsets['visual']
                
                all_spike_times.extend(visual_spikes.spike_times)
                all_neuron_ids.extend(visual_spikes.neuron_ids + offset)
            
            if tactile_data is not None and 'tactile' in self.encoders:
                tactile_spikes = self.encoders['tactile'].encode(tactile_data)
                offset = self.neuron_offsets['tactile']
                
                all_spike_times.extend(tactile_spikes.spike_times)
                all_neuron_ids.extend(tactile_spikes.neuron_ids + offset)
            
            # Combine all spikes
            combined_spikes = SpikeData(
                spike_times=np.array(all_spike_times),
                neuron_ids=np.array(all_neuron_ids),
                duration=self.duration,
                n_neurons=self.total_neurons,
                metadata={
                    'encoding_type': 'multi_modal',
                    'modalities': list(self.encoders.keys()),
                    'neuron_offsets': self.neuron_offsets,
                }
            )
            
            return combined_spikes
            
        except Exception as e:
            self.logger.error(f"Failed to encode multi-modal data: {e}")
            raise
    
    def get_modality_neurons(self, modality: str) -> Optional[Tuple[int, int]]:
        """
        Get neuron range for specific modality.
        
        Args:
            modality: Modality name ('audio', 'visual', 'tactile')
            
        Returns:
            Tuple of (start_neuron, end_neuron) or None
        """
        if modality not in self.encoders:
            return None
        
        start_neuron = self.neuron_offsets[modality]
        end_neuron = start_neuron + self.encoders[modality].n_neurons
        
        return start_neuron, end_neuron