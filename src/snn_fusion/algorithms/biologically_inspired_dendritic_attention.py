"""
Biologically-Inspired Dendritic Processing for Neuromorphic Attention

Advanced neuromorphic architecture that incorporates realistic dendritic computation
principles to enhance attention mechanisms beyond current approaches. Implements
compartmentalized attention processing, non-linear integration of multi-modal inputs,
and biologically-plausible plasticity mechanisms.

Research Innovation:
- Compartmentalized dendritic processing mimicking biological neurons
- Non-linear integration of multi-modal inputs via active dendrites
- Dendritic spike propagation with realistic electrophysiology
- Branch-specific plasticity and adaptation mechanisms
- Active dendritic conductances (Ca2+, Na+, K+ channels)
- Dendritic computation for predictive coding and attention

Publication Status: Novel biologically-inspired neuromorphic computing approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import math
from dataclasses import dataclass
from enum import Enum

from .temporal_spike_attention import TemporalSpikeAttention
from ..models.neurons import AdaptiveLIF


class DendriticCompartmentType(Enum):
    """Types of dendritic compartments with different properties."""
    PROXIMAL = "proximal"      # Close to soma, fast integration
    DISTAL = "distal"          # Far from soma, non-linear processing
    APICAL = "apical"          # Apical dendrites, top-down modulation
    BASAL = "basal"            # Basal dendrites, feedforward integration


@dataclass
class DendriticProperties:
    """Properties of dendritic compartments."""
    compartment_type: DendriticCompartmentType
    length: float                    # μm
    diameter: float                  # μm
    resistance: float                # MΩ·cm
    capacitance: float              # μF/cm²
    active_conductances: Dict[str, float]  # Channel densities
    distance_from_soma: float       # μm
    branch_order: int              # Dendritic branch order


class ActiveConductance:
    """Model of active dendritic conductances (ion channels)."""
    
    def __init__(
        self,
        channel_type: str,
        max_conductance: float,
        reversal_potential: float,
        activation_params: Dict[str, float],
        inactivation_params: Optional[Dict[str, float]] = None,
    ):
        self.channel_type = channel_type
        self.max_conductance = max_conductance  # mS/cm²
        self.reversal_potential = reversal_potential  # mV
        self.activation_params = activation_params
        self.inactivation_params = inactivation_params or {}
        
        # State variables
        self.activation_state = 0.0
        self.inactivation_state = 1.0 if inactivation_params else None
    
    def compute_current(self, voltage: float, dt: float = 0.1) -> float:
        """Compute ionic current based on Hodgkin-Huxley formulation."""
        # Update activation
        self._update_gating_variable(
            voltage, dt, 'activation',
            self.activation_params, 'activation_state'
        )
        
        # Update inactivation if present
        if self.inactivation_params and self.inactivation_state is not None:
            self._update_gating_variable(
                voltage, dt, 'inactivation',
                self.inactivation_params, 'inactivation_state'
            )
        
        # Compute current: I = g_max * m^p * h^q * (V - E_rev)
        conductance = self.max_conductance * (self.activation_state ** self.activation_params.get('power', 1))
        
        if self.inactivation_state is not None:
            conductance *= (self.inactivation_state ** self.inactivation_params.get('power', 1))
        
        current = conductance * (voltage - self.reversal_potential)
        return current
    
    def _update_gating_variable(
        self,
        voltage: float,
        dt: float,
        gate_type: str,
        params: Dict[str, float],
        state_attr: str,
    ) -> None:
        """Update gating variable using first-order kinetics."""
        # Sigmoid activation/inactivation curves
        v_half = params.get('v_half', -50.0)
        slope = params.get('slope', 5.0)
        tau_max = params.get('tau_max', 10.0)
        
        # Steady-state value
        inf = 1.0 / (1.0 + math.exp((v_half - voltage) / slope))
        
        # Time constant (simplified)
        tau = tau_max / (1.0 + abs((voltage - v_half) / 50.0))
        
        # Update state variable
        current_state = getattr(self, state_attr)
        new_state = inf + (current_state - inf) * math.exp(-dt / tau)
        setattr(self, state_attr, new_state)


class DendriticCompartment(nn.Module):
    """Individual dendritic compartment with realistic biophysics."""
    
    def __init__(
        self,
        properties: DendriticProperties,
        input_size: int,
        synaptic_density: float = 0.5,  # Synapses per μm²
    ):
        super().__init__()
        
        self.properties = properties
        self.input_size = input_size
        self.synaptic_density = synaptic_density
        
        # Morphological properties
        surface_area = math.pi * properties.diameter * properties.length
        self.membrane_capacitance = surface_area * properties.capacitance
        self.membrane_resistance = properties.resistance / surface_area
        
        # Number of synaptic inputs
        self.n_synapses = max(1, int(surface_area * synaptic_density))
        
        # Synaptic weights and delays
        self.synaptic_weights = nn.Parameter(
            torch.randn(self.n_synapses, input_size) * 0.1
        )
        self.synaptic_delays = nn.Parameter(
            torch.rand(self.n_synapses) * 5.0 + 1.0  # 1-6 ms delays
        )
        
        # Active conductances
        self.active_conductances = self._initialize_conductances()
        
        # Dendritic integration parameters
        self.integration_nonlinearity = self._create_integration_function()
        
        # Plasticity mechanisms
        self.branch_plasticity = self._initialize_plasticity()
        
        # State variables
        self.voltage = -65.0  # mV
        self.calcium_concentration = 0.1  # μM
        self.synaptic_efficacies = torch.ones(self.n_synapses)
        
        # Spike detection
        self.spike_threshold = -40.0  # mV for dendritic spikes
        self.last_spike_time = -1000.0
    
    def _initialize_conductances(self) -> Dict[str, ActiveConductance]:
        """Initialize active conductances based on compartment type."""
        conductances = {}
        
        if self.properties.compartment_type == DendriticCompartmentType.PROXIMAL:
            # Fast Na+ and K+ channels for rapid integration
            conductances['na_fast'] = ActiveConductance(
                'sodium_fast', 120.0, 50.0,
                {'v_half': -40.0, 'slope': 5.0, 'tau_max': 1.0, 'power': 3},
                {'v_half': -60.0, 'slope': -5.0, 'tau_max': 10.0, 'power': 1}
            )
            conductances['k_delayed'] = ActiveConductance(
                'potassium_delayed', 36.0, -77.0,
                {'v_half': -50.0, 'slope': 5.0, 'tau_max': 5.0, 'power': 4}
            )
            
        elif self.properties.compartment_type == DendriticCompartmentType.DISTAL:
            # Non-linear integration with Ca2+ channels
            conductances['ca_l'] = ActiveConductance(
                'calcium_l', 2.0, 120.0,
                {'v_half': -30.0, 'slope': 8.0, 'tau_max': 20.0, 'power': 2},
                {'v_half': -80.0, 'slope': -5.0, 'tau_max': 100.0, 'power': 1}
            )
            conductances['ca_n'] = ActiveConductance(
                'calcium_n', 1.5, 120.0,
                {'v_half': -20.0, 'slope': 6.0, 'tau_max': 15.0, 'power': 2}
            )
            conductances['k_ca'] = ActiveConductance(
                'potassium_ca', 8.0, -77.0,
                {'v_half': 0.0, 'slope': 10.0, 'tau_max': 50.0, 'power': 2}  # Ca2+-dependent
            )
            
        elif self.properties.compartment_type == DendriticCompartmentType.APICAL:
            # Top-down modulation channels
            conductances['h_current'] = ActiveConductance(
                'hyperpolarization_activated', 0.5, -30.0,
                {'v_half': -80.0, 'slope': -10.0, 'tau_max': 200.0, 'power': 1}
            )
            conductances['ca_t'] = ActiveConductance(
                'calcium_t', 1.0, 120.0,
                {'v_half': -60.0, 'slope': 5.0, 'tau_max': 5.0, 'power': 2},
                {'v_half': -80.0, 'slope': -3.0, 'tau_max': 30.0, 'power': 1}
            )
        
        return conductances
    
    def _create_integration_function(self) -> nn.Module:
        """Create non-linear integration function for synaptic inputs."""
        if self.properties.compartment_type == DendriticCompartmentType.DISTAL:
            # Non-linear integration with saturation
            return nn.Sequential(
                nn.Linear(self.input_size, 32),
                nn.Tanh(),
                nn.Linear(32, 16),
                nn.Sigmoid(),
                nn.Linear(16, 1),
            )
        else:
            # Linear integration for proximal compartments
            return nn.Linear(self.input_size, 1)
    
    def _initialize_plasticity(self) -> Dict[str, nn.Module]:
        """Initialize branch-specific plasticity mechanisms."""
        plasticity_mechanisms = {}
        
        # Hebbian plasticity
        plasticity_mechanisms['hebbian'] = nn.Sequential(
            nn.Linear(2, 8),  # Pre and post activity
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        
        # Calcium-dependent plasticity
        plasticity_mechanisms['calcium'] = nn.Sequential(
            nn.Linear(3, 8),  # Pre, post, and Ca2+ concentration
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Tanh(),  # Can be positive or negative
        )
        
        # Metaplasticity (plasticity of plasticity)
        plasticity_mechanisms['meta'] = nn.Sequential(
            nn.Linear(4, 8),  # Activity history
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        
        return plasticity_mechanisms
    
    def forward(
        self,
        synaptic_inputs: torch.Tensor,
        contextual_modulation: Optional[torch.Tensor] = None,
        dt: float = 0.1,
        return_detailed_states: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process synaptic inputs through dendritic compartment.
        
        Args:
            synaptic_inputs: [batch_size, time_steps, input_size]
            contextual_modulation: Optional top-down modulation
            dt: Integration time step (ms)
            return_detailed_states: Return detailed biophysical states
            
        Returns:
            compartment_output: Processed dendritic output
            states: Detailed compartment states
        """
        batch_size, time_steps, input_size = synaptic_inputs.shape
        
        # Initialize outputs and states
        outputs = []
        voltage_trace = []
        calcium_trace = []
        spike_times = []
        plasticity_changes = []
        
        for t in range(time_steps):
            # Current synaptic input
            current_input = synaptic_inputs[:, t, :]  # [batch_size, input_size]
            
            # Synaptic integration with delays (simplified)
            synaptic_current = self._compute_synaptic_current(current_input, t * dt)
            
            # Active conductance currents
            active_currents = self._compute_active_currents(dt)
            
            # Contextual modulation
            if contextual_modulation is not None:
                modulation_current = self._apply_contextual_modulation(
                    contextual_modulation[:, t] if contextual_modulation.dim() > 1 else contextual_modulation
                )
            else:
                modulation_current = 0.0
            
            # Total membrane current
            total_current = synaptic_current + active_currents + modulation_current
            
            # Update membrane voltage
            dv_dt = (-self.voltage + 65.0) / (self.membrane_resistance * self.membrane_capacitance) + \
                   total_current / self.membrane_capacitance
            
            self.voltage += dv_dt * dt
            
            # Update calcium concentration (simplified)
            ca_influx = self._compute_calcium_influx()
            ca_decay = -self.calcium_concentration / 100.0  # 100 ms time constant
            self.calcium_concentration += (ca_influx + ca_decay) * dt
            self.calcium_concentration = max(0.1, self.calcium_concentration)  # Minimum basal level
            
            # Check for dendritic spike
            spike_occurred = self._detect_dendritic_spike(t * dt)
            if spike_occurred:
                spike_times.append(t * dt)
                self._handle_dendritic_spike()
            
            # Apply plasticity
            plasticity_change = self._update_synaptic_plasticity(
                current_input, self.voltage, self.calcium_concentration, dt
            )
            plasticity_changes.append(plasticity_change)
            
            # Non-linear integration of inputs
            integrated_output = self._integrate_dendritic_inputs(
                current_input, self.voltage, self.calcium_concentration
            )
            
            outputs.append(integrated_output)
            voltage_trace.append(self.voltage)
            calcium_trace.append(self.calcium_concentration)
        
        # Stack temporal outputs
        compartment_output = torch.stack(outputs, dim=1)  # [batch_size, time_steps, output_size]
        
        # Compile states
        states = {
            'voltage_trace': torch.tensor(voltage_trace),
            'calcium_trace': torch.tensor(calcium_trace),
            'spike_times': spike_times,
            'plasticity_changes': torch.stack(plasticity_changes) if plasticity_changes else None,
            'n_dendritic_spikes': len(spike_times),
            'average_voltage': sum(voltage_trace) / len(voltage_trace),
            'peak_calcium': max(calcium_trace),
            'synaptic_efficacies': self.synaptic_efficacies.clone(),
        }
        
        if return_detailed_states:
            states.update({
                'active_conductances': {k: v.__dict__ for k, v in self.active_conductances.items()},
                'membrane_properties': {
                    'resistance': self.membrane_resistance,
                    'capacitance': self.membrane_capacitance,
                    'surface_area': math.pi * self.properties.diameter * self.properties.length,
                },
                'synaptic_weights': self.synaptic_weights.detach(),
                'integration_nonlinearity_type': str(type(self.integration_nonlinearity)),
            })
        
        return compartment_output, states
    
    def _compute_synaptic_current(self, inputs: torch.Tensor, current_time: float) -> float:
        """Compute synaptic current with realistic kinetics."""
        batch_size = inputs.shape[0]
        total_current = 0.0
        
        # Sample synaptic inputs (simplified - one per batch)
        for synapse_idx in range(min(self.n_synapses, inputs.shape[1])):
            # Synaptic delay
            delay = self.synaptic_delays[synapse_idx].item()
            if current_time >= delay:
                # Synaptic weight
                weight = self.synaptic_weights[synapse_idx].mean().item()
                efficacy = self.synaptic_efficacies[synapse_idx].item()
                
                # Input strength (average over batch)
                input_strength = inputs[:, min(synapse_idx, inputs.shape[1]-1)].mean().item()
                
                # Synaptic current with AMPA-like kinetics
                current = weight * efficacy * input_strength * self._ampa_kinetics(current_time - delay)
                total_current += current
        
        return total_current
    
    def _ampa_kinetics(self, time_since_spike: float) -> float:
        """AMPA receptor kinetics."""
        if time_since_spike < 0:
            return 0.0
        
        # Dual exponential: rise and decay
        tau_rise = 0.5  # ms
        tau_decay = 5.0  # ms
        
        amplitude = math.exp(-time_since_spike / tau_decay) - math.exp(-time_since_spike / tau_rise)
        return max(0.0, amplitude)
    
    def _compute_active_currents(self, dt: float) -> float:
        """Compute currents from active conductances."""
        total_current = 0.0
        
        for conductance in self.active_conductances.values():
            # Special handling for Ca2+-dependent channels
            if conductance.channel_type == 'potassium_ca':
                # Modify activation based on calcium concentration
                ca_factor = self.calcium_concentration / (self.calcium_concentration + 1.0)
                conductance.activation_state = ca_factor
            
            current = conductance.compute_current(self.voltage, dt)
            total_current += current
        
        return total_current
    
    def _apply_contextual_modulation(self, modulation: torch.Tensor) -> float:
        """Apply contextual/top-down modulation."""
        # Modulation affects synaptic efficacy and excitability
        modulation_strength = modulation.mean().item() if modulation.dim() > 0 else modulation.item()
        
        # Multiplicative modulation of membrane excitability
        modulation_current = modulation_strength * (self.voltage + 65.0) * 0.1
        
        return modulation_current
    
    def _compute_calcium_influx(self) -> float:
        """Compute calcium influx from voltage-gated channels."""
        ca_influx = 0.0
        
        for name, conductance in self.active_conductances.items():
            if 'calcium' in name:
                # Calcium current contributes to intracellular Ca2+
                ca_current = conductance.compute_current(self.voltage)
                if ca_current > 0:  # Influx
                    ca_influx += ca_current * 0.01  # Conversion factor
        
        return ca_influx
    
    def _detect_dendritic_spike(self, current_time: float) -> bool:
        """Detect dendritic spike based on voltage threshold."""
        if self.voltage > self.spike_threshold and (current_time - self.last_spike_time) > 2.0:
            self.last_spike_time = current_time
            return True
        return False
    
    def _handle_dendritic_spike(self) -> None:
        """Handle dendritic spike generation."""
        # Reset voltage (simplified)
        self.voltage = -50.0  # Partial reset, not full like somatic spikes
        
        # Calcium influx during spike
        self.calcium_concentration += 2.0  # μM increase
        
        # Activity-dependent changes
        self._update_spike_dependent_plasticity()
    
    def _update_spike_dependent_plasticity(self) -> None:
        """Update plasticity based on dendritic spike."""
        # Increase synaptic efficacies for recently active synapses
        activity_window = 10.0  # ms
        
        # Simplified: boost all synapses (in practice, would track individual timings)
        boost_factor = 1.02
        self.synaptic_efficacies = torch.clamp(
            self.synaptic_efficacies * boost_factor,
            0.1, 2.0  # Bounds on efficacy
        )
    
    def _update_synaptic_plasticity(
        self,
        pre_activity: torch.Tensor,
        post_voltage: float,
        calcium: float,
        dt: float,
    ) -> torch.Tensor:
        """Update synaptic plasticity based on activity and calcium."""
        batch_size = pre_activity.shape[0]
        
        # Average pre-synaptic activity
        pre_activity_mean = pre_activity.mean().item()
        
        # Post-synaptic activity (voltage-based)
        post_activity = 1.0 / (1.0 + math.exp(-(post_voltage + 55.0) / 10.0))
        
        # Hebbian plasticity
        hebbian_input = torch.tensor([pre_activity_mean, post_activity])
        hebbian_change = self.branch_plasticity['hebbian'](hebbian_input).item()
        
        # Calcium-dependent plasticity
        calcium_input = torch.tensor([pre_activity_mean, post_activity, calcium])
        calcium_change = self.branch_plasticity['calcium'](calcium_input).item()
        
        # Combined plasticity change
        total_change = (hebbian_change * 0.01 + calcium_change * 0.005) * dt
        
        # Update synaptic weights
        with torch.no_grad():
            self.synaptic_weights.data += total_change * torch.randn_like(self.synaptic_weights) * 0.1
            
            # Weight bounds
            self.synaptic_weights.data = torch.clamp(self.synaptic_weights.data, -2.0, 2.0)
        
        return torch.tensor([total_change])
    
    def _integrate_dendritic_inputs(
        self,
        inputs: torch.Tensor,
        voltage: float,
        calcium: float,
    ) -> torch.Tensor:
        """Integrate dendritic inputs with non-linear processing."""
        batch_size = inputs.shape[0]
        
        # Non-linear integration
        integrated = self.integration_nonlinearity(inputs)
        
        # Voltage-dependent modulation
        voltage_modulation = 1.0 + 0.1 * torch.tanh(torch.tensor((voltage + 65.0) / 20.0))
        integrated = integrated * voltage_modulation
        
        # Calcium-dependent modulation (for distal compartments)
        if self.properties.compartment_type == DendriticCompartmentType.DISTAL:
            calcium_modulation = 1.0 + 0.2 * torch.sigmoid(torch.tensor((calcium - 0.5) * 5.0))
            integrated = integrated * calcium_modulation
        
        return integrated


class BiologicalNeuronModel(nn.Module):
    """Complete biological neuron model with multiple dendritic compartments."""
    
    def __init__(
        self,
        input_size: int,
        compartment_config: Dict[DendriticCompartmentType, int],  # Number of each compartment type
        soma_size: int = 32,
        axon_delay: float = 2.0,  # ms
    ):
        super().__init__()
        
        self.input_size = input_size
        self.compartment_config = compartment_config
        self.soma_size = soma_size
        self.axon_delay = axon_delay
        
        # Create dendritic compartments
        self.dendritic_tree = self._create_dendritic_tree()
        
        # Somatic integration
        total_dendritic_outputs = sum(compartment_config.values())
        self.soma = AdaptiveLIF(
            n_neurons=soma_size,
            tau_mem=15.0,
            tau_adapt=80.0,
        )
        
        # Dendritic-to-soma integration
        self.dendritic_integration = nn.Linear(total_dendritic_outputs, soma_size)
        
        # Axonal transmission
        self.axon_delay_buffer = []
        self.max_delay_steps = int(axon_delay / 0.1)  # Assuming 0.1 ms time steps
    
    def _create_dendritic_tree(self) -> nn.ModuleDict:
        """Create dendritic compartments based on configuration."""
        compartments = nn.ModuleDict()
        
        for compartment_type, count in self.compartment_config.items():
            for i in range(count):
                # Define compartment properties
                properties = self._get_compartment_properties(compartment_type, i)
                
                compartment_name = f"{compartment_type.value}_{i}"
                compartment = DendriticCompartment(
                    properties=properties,
                    input_size=self.input_size,
                    synaptic_density=self._get_synaptic_density(compartment_type),
                )
                
                compartments[compartment_name] = compartment
        
        return compartments
    
    def _get_compartment_properties(
        self,
        compartment_type: DendriticCompartmentType,
        index: int,
    ) -> DendriticProperties:
        """Get properties for specific compartment type."""
        if compartment_type == DendriticCompartmentType.PROXIMAL:
            return DendriticProperties(
                compartment_type=compartment_type,
                length=50.0 + index * 10.0,  # μm
                diameter=2.0,  # μm
                resistance=100.0,  # MΩ·cm
                capacitance=1.0,  # μF/cm²
                active_conductances={'na': 50.0, 'k': 30.0},
                distance_from_soma=20.0 + index * 30.0,
                branch_order=1,
            )
        elif compartment_type == DendriticCompartmentType.DISTAL:
            return DendriticProperties(
                compartment_type=compartment_type,
                length=80.0 + index * 20.0,
                diameter=1.0,
                resistance=200.0,
                capacitance=1.0,
                active_conductances={'ca_l': 2.0, 'ca_n': 1.5, 'k_ca': 5.0},
                distance_from_soma=200.0 + index * 100.0,
                branch_order=3 + index,
            )
        elif compartment_type == DendriticCompartmentType.APICAL:
            return DendriticProperties(
                compartment_type=compartment_type,
                length=200.0,
                diameter=3.0,
                resistance=80.0,
                capacitance=1.0,
                active_conductances={'h': 0.5, 'ca_t': 1.0},
                distance_from_soma=300.0,
                branch_order=1,
            )
        else:  # BASAL
            return DendriticProperties(
                compartment_type=compartment_type,
                length=60.0 + index * 15.0,
                diameter=1.5,
                resistance=120.0,
                capacitance=1.0,
                active_conductances={'na': 20.0, 'k': 15.0},
                distance_from_soma=50.0 + index * 40.0,
                branch_order=2,
            )
    
    def _get_synaptic_density(self, compartment_type: DendriticCompartmentType) -> float:
        """Get synaptic density for compartment type."""
        density_map = {
            DendriticCompartmentType.PROXIMAL: 1.0,  # synapses per μm²
            DendriticCompartmentType.DISTAL: 0.3,
            DendriticCompartmentType.APICAL: 0.5,
            DendriticCompartmentType.BASAL: 0.8,
        }
        return density_map[compartment_type]
    
    def forward(
        self,
        inputs: torch.Tensor,
        contextual_inputs: Optional[Dict[str, torch.Tensor]] = None,
        return_dendritic_states: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process inputs through complete biological neuron model.
        
        Args:
            inputs: [batch_size, time_steps, input_size]
            contextual_inputs: Optional contextual modulation for different compartments
            return_dendritic_states: Return detailed dendritic states
            
        Returns:
            somatic_output: Final neuron output
            neuron_states: Comprehensive neuron states
        """
        batch_size, time_steps, input_size = inputs.shape
        
        # Process inputs through dendritic compartments
        dendritic_outputs = []
        dendritic_states = {}
        
        for compartment_name, compartment in self.dendritic_tree.items():
            # Get contextual modulation for this compartment
            context = None
            if contextual_inputs and compartment_name in contextual_inputs:
                context = contextual_inputs[compartment_name]
            
            # Process through compartment
            compartment_output, states = compartment(
                inputs,
                contextual_modulation=context,
                return_detailed_states=return_dendritic_states,
            )
            
            dendritic_outputs.append(compartment_output)
            dendritic_states[compartment_name] = states
        
        # Integrate dendritic outputs
        if dendritic_outputs:
            # Concatenate dendritic outputs
            combined_dendritic = torch.cat(dendritic_outputs, dim=-1)
            
            # Integrate to soma
            somatic_input = self.dendritic_integration(combined_dendritic)
        else:
            somatic_input = torch.zeros(batch_size, time_steps, self.soma_size)
        
        # Process through soma
        somatic_spikes = []
        for t in range(time_steps):
            current_input = somatic_input[:, t, :]
            spikes, soma_states = self.soma(current_input)
            somatic_spikes.append(spikes)
        
        somatic_output = torch.stack(somatic_spikes, dim=1)
        
        # Apply axonal delay
        delayed_output = self._apply_axonal_delay(somatic_output)
        
        # Compile neuron states
        neuron_states = {
            'dendritic_states': dendritic_states,
            'somatic_states': soma_states,
            'dendritic_integration': combined_dendritic.detach() if dendritic_outputs else None,
            'n_dendritic_compartments': len(self.dendritic_tree),
            'total_dendritic_spikes': sum(
                states.get('n_dendritic_spikes', 0) for states in dendritic_states.values()
            ),
            'somatic_spike_rate': somatic_output.mean().item(),
        }
        
        return delayed_output, neuron_states
    
    def _apply_axonal_delay(self, spikes: torch.Tensor) -> torch.Tensor:
        """Apply axonal conduction delay."""
        batch_size, time_steps, output_size = spikes.shape
        
        # Add to delay buffer
        self.axon_delay_buffer.append(spikes)
        
        # Maintain buffer size
        while len(self.axon_delay_buffer) > self.max_delay_steps:
            self.axon_delay_buffer.pop(0)
        
        # Return delayed spikes
        if len(self.axon_delay_buffer) >= self.max_delay_steps:
            return self.axon_delay_buffer[0]
        else:
            # Not enough delay accumulated, return zeros
            return torch.zeros_like(spikes)


class BiologicallyInspiredDendriticAttention(nn.Module):
    """
    Complete biologically-inspired dendritic attention system.
    
    Integrates multiple biological neuron models with dendritic computation
    for enhanced multi-modal attention processing.
    """
    
    def __init__(
        self,
        input_size: int,
        n_neurons: int = 64,
        modalities: List[str] = ['audio', 'vision', 'tactile'],
        compartment_specialization: Dict[str, List[DendriticCompartmentType]] = None,
        attention_heads: int = 8,
        predictive_coding_strength: float = 0.2,
    ):
        """
        Initialize biologically-inspired dendritic attention.
        
        Args:
            input_size: Input feature size
            n_neurons: Number of biological neurons
            modalities: Sensory modalities to process
            compartment_specialization: Compartment types for each modality
            attention_heads: Number of attention heads
            predictive_coding_strength: Strength of predictive coding
        """
        super().__init__()
        
        self.input_size = input_size
        self.n_neurons = n_neurons
        self.modalities = modalities
        self.attention_heads = attention_heads
        self.predictive_coding_strength = predictive_coding_strength
        
        # Default compartment specialization
        if compartment_specialization is None:
            compartment_specialization = {
                'audio': [DendriticCompartmentType.PROXIMAL, DendriticCompartmentType.BASAL],
                'vision': [DendriticCompartmentType.DISTAL, DendriticCompartmentType.APICAL],
                'tactile': [DendriticCompartmentType.BASAL, DendriticCompartmentType.PROXIMAL],
            }
        
        # Create biological neurons for each modality
        self.biological_neurons = nn.ModuleDict()
        
        for modality in modalities:
            compartment_config = {
                comp_type: 2 for comp_type in compartment_specialization.get(
                    modality, [DendriticCompartmentType.PROXIMAL, DendriticCompartmentType.BASAL]
                )
            }
            
            neuron_population = nn.ModuleList([
                BiologicalNeuronModel(
                    input_size=input_size,
                    compartment_config=compartment_config,
                    soma_size=32,
                ) for _ in range(n_neurons // len(modalities))
            ])
            
            self.biological_neurons[modality] = neuron_population
        
        # Cross-modal dendritic interactions
        self.cross_modal_dendrites = self._create_cross_modal_connections()
        
        # Predictive coding system
        self.predictive_coding = self._create_predictive_coding_system()
        
        # Attention mechanisms
        self.dendritic_attention_heads = nn.ModuleList([
            self._create_dendritic_attention_head() for _ in range(attention_heads)
        ])
        
        # Output integration
        total_neurons = sum(len(pop) for pop in self.biological_neurons.values())
        self.output_integration = nn.Sequential(
            nn.Linear(total_neurons * 32, 128),  # 32 is soma size
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )
    
    def _create_cross_modal_connections(self) -> nn.ModuleDict:
        """Create cross-modal dendritic connections."""
        connections = nn.ModuleDict()
        
        for i, source_modality in enumerate(self.modalities):
            for j, target_modality in enumerate(self.modalities):
                if i != j:
                    connection_name = f"{source_modality}_to_{target_modality}"
                    connections[connection_name] = nn.Sequential(
                        nn.Linear(32, 16),  # Soma size to connection strength
                        nn.Sigmoid(),
                        nn.Linear(16, 32),
                    )
        
        return connections
    
    def _create_predictive_coding_system(self) -> nn.ModuleDict:
        """Create predictive coding system for each modality."""
        predictive_systems = nn.ModuleDict()
        
        for modality in self.modalities:
            predictive_systems[f"{modality}_predictor"] = nn.LSTM(
                input_size=32,  # Soma size
                hidden_size=64,
                num_layers=2,
                batch_first=True,
            )
            
            predictive_systems[f"{modality}_error"] = nn.Linear(64, 32)
        
        return predictive_systems
    
    def _create_dendritic_attention_head(self) -> nn.Module:
        """Create dendritic-specific attention head."""
        return nn.MultiheadAttention(
            embed_dim=32,  # Soma size
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )
    
    def forward(
        self,
        multi_modal_inputs: Dict[str, torch.Tensor],
        return_dendritic_analysis: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process multi-modal inputs through biologically-inspired attention.
        
        Args:
            multi_modal_inputs: Dictionary of modality inputs
            return_dendritic_analysis: Return detailed dendritic analysis
            
        Returns:
            attention_output: Final attention output
            biological_analysis: Detailed biological processing analysis
        """
        batch_size = next(iter(multi_modal_inputs.values())).shape[0]
        time_steps = next(iter(multi_modal_inputs.values())).shape[1]
        
        # Process each modality through biological neurons
        modality_outputs = {}
        dendritic_analyses = {}
        
        for modality, inputs in multi_modal_inputs.items():
            neuron_population = self.biological_neurons[modality]
            
            # Process through each neuron in population
            population_outputs = []
            population_states = []
            
            for neuron in neuron_population:
                # Cross-modal contextual inputs
                contextual_inputs = self._generate_contextual_inputs(
                    modality, multi_modal_inputs
                )
                
                neuron_output, neuron_states = neuron(
                    inputs,
                    contextual_inputs=contextual_inputs,
                    return_dendritic_states=return_dendritic_analysis,
                )
                
                population_outputs.append(neuron_output)
                population_states.append(neuron_states)
            
            # Stack population outputs
            modality_output = torch.stack(population_outputs, dim=-2)  # [batch, time, n_neurons, soma_size]
            modality_outputs[modality] = modality_output
            dendritic_analyses[modality] = population_states
        
        # Predictive coding
        prediction_errors = {}
        
        for modality in self.modalities:
            predictor = self.predictive_coding[f"{modality}_predictor"]
            error_network = self.predictive_coding[f"{modality}_error"]
            
            # Get current activity
            current_activity = modality_outputs[modality].mean(dim=-2)  # Average over neurons
            
            # Generate prediction
            prediction, _ = predictor(current_activity)
            
            # Compute prediction error
            error = error_network(prediction) - current_activity
            prediction_errors[modality] = error * self.predictive_coding_strength
        
        # Apply dendritic attention
        attended_outputs = []
        
        for head_idx, attention_head in enumerate(self.dendritic_attention_heads):
            # Combine modality outputs for attention
            combined_activity = torch.cat([
                modality_outputs[mod].mean(dim=-2) for mod in self.modalities
            ], dim=-1)  # [batch, time, total_features]
            
            # Self-attention with dendritic processing
            attended, attention_weights = attention_head(
                combined_activity, combined_activity, combined_activity
            )
            
            attended_outputs.append(attended)
        
        # Integrate attention outputs
        if attended_outputs:
            combined_attention = torch.stack(attended_outputs, dim=-1).mean(dim=-1)
        else:
            combined_attention = torch.cat([
                modality_outputs[mod].mean(dim=-2) for mod in self.modalities
            ], dim=-1)
        
        # Apply prediction errors
        for i, modality in enumerate(self.modalities):
            start_idx = i * 32  # Assuming 32 features per modality
            end_idx = start_idx + 32
            combined_attention[:, :, start_idx:end_idx] += prediction_errors[modality]
        
        # Final output integration
        attention_output = self.output_integration(combined_attention.mean(dim=1))  # Pool over time
        
        # Compile biological analysis
        biological_analysis = {
            'dendritic_analyses': dendritic_analyses,
            'prediction_errors': prediction_errors,
            'total_dendritic_spikes': sum([
                sum([state.get('total_dendritic_spikes', 0) for state in states])
                for states in dendritic_analyses.values()
            ]),
            'somatic_spike_rates': {
                modality: torch.stack([
                    torch.tensor([state.get('somatic_spike_rate', 0.0) for state in states])
                    for states in [dendritic_analyses[modality]]
                ]).mean().item()
                for modality in self.modalities
            },
            'cross_modal_interactions': self._analyze_cross_modal_interactions(modality_outputs),
            'predictive_coding_effectiveness': self._analyze_predictive_coding(prediction_errors),
        }
        
        if return_dendritic_analysis:
            biological_analysis.update({
                'compartment_activities': self._analyze_compartment_activities(dendritic_analyses),
                'plasticity_changes': self._analyze_plasticity_changes(dendritic_analyses),
                'calcium_dynamics': self._analyze_calcium_dynamics(dendritic_analyses),
            })
        
        return attention_output, biological_analysis
    
    def _generate_contextual_inputs(
        self,
        target_modality: str,
        all_inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Generate contextual inputs from other modalities."""
        contextual_inputs = {}
        
        for source_modality, inputs in all_inputs.items():
            if source_modality != target_modality:
                connection_name = f"{source_modality}_to_{target_modality}"
                if connection_name in self.cross_modal_dendrites:
                    # Generate contextual signal
                    context_network = self.cross_modal_dendrites[connection_name]
                    context = context_network(inputs.mean(dim=-1, keepdim=True))
                    
                    # Map to dendritic compartments (simplified)
                    contextual_inputs[f"context_{source_modality}"] = context
        
        return contextual_inputs
    
    def _analyze_cross_modal_interactions(
        self,
        modality_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Analyze cross-modal interactions."""
        interactions = {}
        
        modality_list = list(modality_outputs.keys())
        
        for i, mod1 in enumerate(modality_list):
            for j, mod2 in enumerate(modality_list[i+1:], i+1):
                # Compute cross-correlation
                output1 = modality_outputs[mod1].mean(dim=(1, 2))  # Average over time and neurons
                output2 = modality_outputs[mod2].mean(dim=(1, 2))
                
                correlation = F.cosine_similarity(output1, output2, dim=-1).mean().item()
                interactions[f"{mod1}_{mod2}_correlation"] = correlation
        
        return interactions
    
    def _analyze_predictive_coding(
        self,
        prediction_errors: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Analyze predictive coding effectiveness."""
        analysis = {}
        
        for modality, errors in prediction_errors.items():
            error_magnitude = torch.abs(errors).mean().item()
            error_consistency = 1.0 / (1.0 + errors.var().item())
            
            analysis[f"{modality}_error_magnitude"] = error_magnitude
            analysis[f"{modality}_error_consistency"] = error_consistency
        
        return analysis
    
    def _analyze_compartment_activities(
        self,
        dendritic_analyses: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Dict[str, float]]:
        """Analyze activities of different compartment types."""
        compartment_activities = {}
        
        for modality, neuron_states_list in dendritic_analyses.items():
            modality_activities = {}
            
            for neuron_states in neuron_states_list:
                dendritic_states = neuron_states.get('dendritic_states', {})
                
                for compartment_name, states in dendritic_states.items():
                    compartment_type = compartment_name.split('_')[0]  # Extract type
                    
                    if compartment_type not in modality_activities:
                        modality_activities[compartment_type] = []
                    
                    avg_voltage = states.get('average_voltage', -65.0)
                    modality_activities[compartment_type].append(avg_voltage)
            
            # Average activities by compartment type
            compartment_activities[modality] = {
                comp_type: np.mean(voltages) for comp_type, voltages in modality_activities.items()
            }
        
        return compartment_activities
    
    def _analyze_plasticity_changes(
        self,
        dendritic_analyses: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, float]:
        """Analyze plasticity changes across the network."""
        total_plasticity_changes = []
        
        for modality, neuron_states_list in dendritic_analyses.items():
            for neuron_states in neuron_states_list:
                dendritic_states = neuron_states.get('dendritic_states', {})
                
                for compartment_name, states in dendritic_states.items():
                    plasticity_changes = states.get('plasticity_changes')
                    if plasticity_changes is not None:
                        total_plasticity_changes.extend(plasticity_changes.flatten().tolist())
        
        if total_plasticity_changes:
            return {
                'mean_plasticity_change': np.mean(total_plasticity_changes),
                'plasticity_variance': np.var(total_plasticity_changes),
                'total_plasticity_events': len(total_plasticity_changes),
            }
        else:
            return {'mean_plasticity_change': 0.0, 'plasticity_variance': 0.0, 'total_plasticity_events': 0}
    
    def _analyze_calcium_dynamics(
        self,
        dendritic_analyses: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Dict[str, float]]:
        """Analyze calcium dynamics across compartments."""
        calcium_analysis = {}
        
        for modality, neuron_states_list in dendritic_analyses.items():
            modality_calcium = []
            peak_calcium_values = []
            
            for neuron_states in neuron_states_list:
                dendritic_states = neuron_states.get('dendritic_states', {})
                
                for compartment_name, states in dendritic_states.items():
                    calcium_trace = states.get('calcium_trace')
                    if calcium_trace is not None:
                        modality_calcium.extend(calcium_trace.tolist())
                        peak_calcium_values.append(states.get('peak_calcium', 0.1))
            
            if modality_calcium:
                calcium_analysis[modality] = {
                    'mean_calcium': np.mean(modality_calcium),
                    'calcium_variance': np.var(modality_calcium),
                    'peak_calcium_average': np.mean(peak_calcium_values),
                    'calcium_events': sum(1 for ca in modality_calcium if ca > 1.0),  # Above threshold
                }
            else:
                calcium_analysis[modality] = {
                    'mean_calcium': 0.1, 'calcium_variance': 0.0,
                    'peak_calcium_average': 0.1, 'calcium_events': 0
                }
        
        return calcium_analysis


# Research validation utilities
class BiologicalValidityBenchmark:
    """Benchmark biological validity of dendritic processing."""
    
    @staticmethod
    def validate_dendritic_properties(
        model: BiologicallyInspiredDendriticAttention,
        reference_data: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        """Validate dendritic properties against biological data."""
        # Default reference values from neuroscience literature
        if reference_data is None:
            reference_data = {
                'proximal_resistance': 100.0,  # MΩ·cm
                'distal_resistance': 200.0,
                'membrane_capacitance': 1.0,  # μF/cm²
                'spike_threshold': -40.0,  # mV
                'resting_potential': -65.0,
                'calcium_baseline': 0.1,  # μM
            }
        
        validation_scores = {}
        
        # Sample a neuron for validation
        sample_modality = list(model.biological_neurons.keys())[0]
        sample_neuron = model.biological_neurons[sample_modality][0]
        
        # Check compartment properties
        for compartment_name, compartment in sample_neuron.dendritic_tree.items():
            properties = compartment.properties
            
            # Resistance validation
            if properties.compartment_type == DendriticCompartmentType.PROXIMAL:
                ref_resistance = reference_data['proximal_resistance']
            else:
                ref_resistance = reference_data['distal_resistance']
            
            resistance_error = abs(properties.resistance - ref_resistance) / ref_resistance
            validation_scores[f'{compartment_name}_resistance'] = 1.0 - min(1.0, resistance_error)
            
            # Morphology validation (simplified)
            reasonable_length = 20.0 <= properties.length <= 500.0
            reasonable_diameter = 0.5 <= properties.diameter <= 5.0
            
            validation_scores[f'{compartment_name}_morphology'] = (
                float(reasonable_length) + float(reasonable_diameter)
            ) / 2.0
        
        # Overall biological validity score
        validation_scores['overall_validity'] = np.mean(list(validation_scores.values()))
        
        return validation_scores
    
    @staticmethod
    def benchmark_computational_efficiency(
        model: BiologicallyInspiredDendriticAttention,
        test_data: Dict[str, torch.Tensor],
        n_trials: int = 10,
    ) -> Dict[str, float]:
        """Benchmark computational efficiency vs biological complexity."""
        import time
        
        model.eval()
        processing_times = []
        memory_usage = []
        
        with torch.no_grad():
            for trial in range(n_trials):
                start_time = time.time()
                
                # Forward pass
                output, analysis = model(test_data, return_dendritic_analysis=True)
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # Estimate memory usage
                model_memory = sum(p.numel() * 4 for p in model.parameters()) / (1024 ** 2)  # MB
                memory_usage.append(model_memory)
        
        return {
            'average_processing_time': np.mean(processing_times),
            'processing_time_std': np.std(processing_times),
            'memory_usage_mb': np.mean(memory_usage),
            'biological_complexity_score': BiologicalValidityBenchmark._compute_complexity_score(analysis),
        }
    
    @staticmethod
    def _compute_complexity_score(analysis: Dict[str, Any]) -> float:
        """Compute biological complexity score."""
        complexity_factors = []
        
        # Dendritic spike activity
        total_spikes = analysis.get('total_dendritic_spikes', 0)
        complexity_factors.append(min(1.0, total_spikes / 1000.0))  # Normalize
        
        # Cross-modal interactions
        interactions = analysis.get('cross_modal_interactions', {})
        if interactions:
            interaction_strength = np.mean([abs(v) for v in interactions.values()])
            complexity_factors.append(interaction_strength)
        
        # Plasticity activity
        plasticity = analysis.get('plasticity_changes', {})
        plasticity_events = plasticity.get('total_plasticity_events', 0)
        complexity_factors.append(min(1.0, plasticity_events / 100.0))
        
        return np.mean(complexity_factors) if complexity_factors else 0.0