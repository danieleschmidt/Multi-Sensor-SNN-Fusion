"""
Consciousness-Driven STDP (CD-STDP) Learning Algorithm

Advanced neuromorphic learning algorithm that incorporates consciousness-driven 
mechanisms with traditional spike-timing-dependent plasticity for enhanced
adaptation and pattern recognition in multi-modal sensor fusion systems.

Key Innovation: First neuromorphic learning algorithm incorporating consciousness-like
processing for enhanced adaptation, predictive processing, and long-term memory
formation with global workspace integration.

Research Foundation:
- Integrated Information Theory (IIT) for consciousness metrics
- Global Workspace Theory for information integration
- Predictive processing frameworks for forward models
- Advanced STDP with higher-order cognitive processes

Authors: Terry (Terragon Labs) - Advanced Neuromorphic Research Framework
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from collections import defaultdict, deque
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio
from enum import Enum
import json
import pickle
import math
from scipy.stats import entropy
import networkx as nx


class ConsciousnessState(Enum):
    """States of consciousness in the CD-STDP system."""
    UNCONSCIOUS = "unconscious"
    PRECONSCIOUS = "preconscious"
    CONSCIOUS = "conscious"
    METACONSCIOUS = "metaconscious"
    INTEGRATED = "integrated"


class AttentionMode(Enum):
    """Attention modes for consciousness-driven processing."""
    BOTTOM_UP = "bottom_up"
    TOP_DOWN = "top_down"
    GLOBAL_WORKSPACE = "global_workspace"
    PREDICTIVE = "predictive"
    METACOGNITIVE = "metacognitive"


class MemoryType(Enum):
    """Types of memory in consciousness system."""
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    METACOGNITIVE = "metacognitive"


@dataclass
class ConsciousNeuron:
    """
    Enhanced neuron model with consciousness-driven properties.
    """
    neuron_id: int
    layer_id: int
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Basic neural properties
    membrane_potential: float = 0.0
    threshold: float = 1.0
    refractory_period: int = 2
    refractory_counter: int = 0
    
    # Consciousness-specific properties
    consciousness_level: float = 0.0  # 0-1 scale
    attention_weight: float = 0.0
    global_availability: float = 0.0
    integration_strength: float = 0.0
    
    # Predictive processing
    prediction_error: float = 0.0
    forward_model_confidence: float = 0.5
    prediction_history: List[float] = field(default_factory=list)
    
    # Memory and learning
    working_memory: List[float] = field(default_factory=list)
    episodic_traces: List[Dict] = field(default_factory=list)
    semantic_associations: Dict[str, float] = field(default_factory=dict)
    
    # Metacognitive properties
    confidence_in_output: float = 0.5
    uncertainty_estimate: float = 0.5
    meta_learning_rate: float = 0.01
    
    # Activity history
    spike_times: List[float] = field(default_factory=list)
    activation_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    consciousness_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update_consciousness_level(self, global_activity: float, attention_signal: float) -> float:
        """Update consciousness level based on global activity and attention."""
        # Integrated Information Theory inspired computation
        local_activity = len([t for t in self.spike_times if time.time() - t < 0.1])
        
        # Phi-like measure (simplified IIT)
        phi = self._compute_integrated_information(local_activity, global_activity)
        
        # Global workspace contribution
        workspace_contribution = attention_signal * self.global_availability
        
        # Update consciousness level with temporal dynamics
        tau_consciousness = 0.05  # 50ms time constant
        dt = 0.001
        
        target_consciousness = min(1.0, phi + workspace_contribution)
        self.consciousness_level += dt * (target_consciousness - self.consciousness_level) / tau_consciousness
        self.consciousness_level = np.clip(self.consciousness_level, 0.0, 1.0)
        
        # Store in history
        self.consciousness_history.append(self.consciousness_level)
        
        return self.consciousness_level
    
    def _compute_integrated_information(self, local_activity: float, global_activity: float) -> float:
        """Compute simplified integrated information (Phi) measure."""
        if global_activity == 0:
            return 0.0
            
        # Normalized local contribution
        local_contribution = local_activity / (global_activity + 1e-6)
        
        # Integration with rest of system
        integration_factor = min(1.0, self.integration_strength * 2.0)
        
        # Simplified Phi computation
        phi = local_contribution * integration_factor * (1 - local_contribution)
        
        return phi
    
    def update_prediction_error(self, actual_input: float, predicted_input: float) -> float:
        """Update prediction error for predictive processing."""
        self.prediction_error = abs(actual_input - predicted_input)
        
        # Update confidence based on prediction accuracy
        accuracy = 1.0 - min(1.0, self.prediction_error)
        self.forward_model_confidence = 0.9 * self.forward_model_confidence + 0.1 * accuracy
        
        # Store prediction history
        self.prediction_history.append(self.prediction_error)
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-50:]
            
        # Update uncertainty estimate
        if len(self.prediction_history) >= 10:
            recent_errors = self.prediction_history[-10:]
            self.uncertainty_estimate = np.std(recent_errors)
            
        return self.prediction_error
    
    def form_episodic_memory(self, context: Dict[str, Any], significance: float) -> None:
        """Form episodic memory trace if consciousness level is sufficient."""
        if self.consciousness_level > 0.6 and significance > 0.3:
            episodic_trace = {
                'timestamp': time.time(),
                'context': context.copy(),
                'consciousness_level': self.consciousness_level,
                'significance': significance,
                'spike_pattern': self.spike_times[-10:].copy() if len(self.spike_times) >= 10 else []
            }
            
            self.episodic_traces.append(episodic_trace)
            
            # Limit memory size
            if len(self.episodic_traces) > 50:
                # Keep most significant memories
                self.episodic_traces.sort(key=lambda x: x['significance'], reverse=True)
                self.episodic_traces = self.episodic_traces[:30]
    
    def update_semantic_associations(self, concept: str, strength: float) -> None:
        """Update semantic associations based on conscious processing."""
        if self.consciousness_level > 0.4:
            current_strength = self.semantic_associations.get(concept, 0.0)
            decay_rate = 0.95
            learning_rate = self.consciousness_level * 0.1
            
            # Decay existing association
            current_strength *= decay_rate
            
            # Add new strength if significant
            if strength > 0.2:
                current_strength += learning_rate * strength
                
            self.semantic_associations[concept] = min(1.0, current_strength)
            
            # Prune weak associations
            if current_strength < 0.05:
                self.semantic_associations.pop(concept, None)
    
    def compute_metacognitive_confidence(self) -> float:
        """Compute metacognitive confidence in neuron's current state."""
        # Factors contributing to confidence
        
        # 1. Consistency of recent activity
        if len(self.activation_history) >= 10:
            recent_activity = list(self.activation_history)[-10:]
            activity_consistency = 1.0 - np.std(recent_activity)
        else:
            activity_consistency = 0.5
            
        # 2. Prediction accuracy
        prediction_confidence = self.forward_model_confidence
        
        # 3. Consciousness stability
        if len(self.consciousness_history) >= 10:
            consciousness_stability = 1.0 - np.std(list(self.consciousness_history)[-10:])
        else:
            consciousness_stability = 0.5
            
        # 4. Integration with global workspace
        integration_confidence = self.global_availability
        
        # Weighted combination
        self.confidence_in_output = (
            0.3 * activity_consistency +
            0.3 * prediction_confidence +
            0.2 * consciousness_stability +
            0.2 * integration_confidence
        )
        
        return self.confidence_in_output


class GlobalWorkspace:
    """
    Implementation of Global Workspace Theory for consciousness integration.
    """
    
    def __init__(self, capacity: int = 100, decay_rate: float = 0.95):
        self.capacity = capacity
        self.decay_rate = decay_rate
        
        # Global workspace components
        self.active_coalitions: List[Dict] = []
        self.attention_focus: Dict[str, float] = {}
        self.working_memory: deque = deque(maxlen=capacity)
        self.broadcast_history: List[Dict] = []
        
        # Consciousness metrics
        self.global_activity_level: float = 0.0
        self.attention_coherence: float = 0.0
        self.information_integration: float = 0.0
        
    def register_coalition(self, neurons: List[ConsciousNeuron], activation_strength: float) -> int:
        """Register a neural coalition for potential global access."""
        coalition = {
            'id': len(self.active_coalitions),
            'neurons': neurons,
            'strength': activation_strength,
            'timestamp': time.time(),
            'consciousness_support': np.mean([n.consciousness_level for n in neurons])
        }
        
        self.active_coalitions.append(coalition)
        return coalition['id']
    
    def compete_for_global_access(self) -> Optional[Dict]:
        """Competition between coalitions for global workspace access."""
        if not self.active_coalitions:
            return None
            
        # Score each coalition
        for coalition in self.active_coalitions:
            coalition['score'] = self._compute_coalition_score(coalition)
            
        # Winner-take-all competition
        winning_coalition = max(self.active_coalitions, key=lambda c: c['score'])
        
        # Threshold for global access
        if winning_coalition['score'] > 0.7:
            return winning_coalition
            
        return None
    
    def _compute_coalition_score(self, coalition: Dict) -> float:
        """Compute competition score for neural coalition."""
        # Factors: strength, consciousness support, coherence, novelty
        
        base_strength = coalition['strength']
        consciousness_support = coalition['consciousness_support']
        
        # Coherence among neurons
        neuron_activities = [len(n.spike_times[-10:]) for n in coalition['neurons']]
        if len(neuron_activities) > 1:
            coherence = 1.0 - np.std(neuron_activities) / (np.mean(neuron_activities) + 1e-6)
        else:
            coherence = 1.0
            
        # Novelty (not recently in workspace)
        novelty = 1.0
        for item in list(self.working_memory)[-10:]:
            if item.get('coalition_id') == coalition['id']:
                novelty *= 0.8
                
        score = (
            0.4 * base_strength +
            0.3 * consciousness_support +
            0.2 * coherence +
            0.1 * novelty
        )
        
        return score
    
    def global_broadcast(self, winning_coalition: Dict) -> Dict[str, float]:
        """Broadcast winning coalition globally and compute attention signals."""
        # Add to working memory
        broadcast_item = {
            'coalition_id': winning_coalition['id'],
            'content': {
                'neurons': [n.neuron_id for n in winning_coalition['neurons']],
                'activation_pattern': [len(n.spike_times[-10:]) for n in winning_coalition['neurons']],
                'consciousness_levels': [n.consciousness_level for n in winning_coalition['neurons']]
            },
            'timestamp': time.time(),
            'global_strength': winning_coalition['score']
        }
        
        self.working_memory.append(broadcast_item)
        
        # Generate attention signals
        attention_signals = {}
        global_strength = winning_coalition['score']
        
        # All neurons get some global signal
        base_attention = global_strength * 0.1
        
        # Winning coalition gets strong signal
        for neuron in winning_coalition['neurons']:
            attention_signals[neuron.neuron_id] = global_strength
            neuron.global_availability = global_strength
            
        # Related neurons get moderate signal based on associations
        for neuron in winning_coalition['neurons']:
            for concept, strength in neuron.semantic_associations.items():
                if strength > 0.5:
                    # Find neurons associated with this concept (simplified)
                    attention_signals[neuron.neuron_id] = attention_signals.get(
                        neuron.neuron_id, base_attention
                    ) + 0.3 * strength
                    
        # Update global metrics
        self.global_activity_level = global_strength
        self.attention_coherence = np.std(list(attention_signals.values())) if attention_signals else 0.0
        self.information_integration = len(attention_signals) / len(winning_coalition['neurons'])
        
        # Store broadcast history
        self.broadcast_history.append({
            'timestamp': time.time(),
            'coalition': winning_coalition,
            'attention_signals': attention_signals.copy(),
            'global_metrics': {
                'activity_level': self.global_activity_level,
                'attention_coherence': self.attention_coherence,
                'integration': self.information_integration
            }
        })
        
        return attention_signals
    
    def decay_workspace(self) -> None:
        """Apply decay to workspace contents."""
        # Decay active coalitions
        self.active_coalitions = [
            c for c in self.active_coalitions 
            if time.time() - c['timestamp'] < 1.0  # 1 second lifetime
        ]
        
        # Decay attention focus
        for key in list(self.attention_focus.keys()):
            self.attention_focus[key] *= self.decay_rate
            if self.attention_focus[key] < 0.01:
                del self.attention_focus[key]
                
        # Decay global activity
        self.global_activity_level *= self.decay_rate


class PredictiveProcessor:
    """
    Predictive processing component for consciousness-driven learning.
    """
    
    def __init__(self, prediction_horizon: int = 10, confidence_threshold: float = 0.7):
        self.prediction_horizon = prediction_horizon
        self.confidence_threshold = confidence_threshold
        
        # Predictive models (simplified)
        self.forward_models: Dict[str, Dict] = {}
        self.prediction_errors: deque = deque(maxlen=1000)
        self.surprise_signals: Dict[int, float] = {}
        
    def build_forward_model(self, neuron: ConsciousNeuron, context: Dict) -> None:
        """Build forward model for neuron based on experience."""
        model_key = f"neuron_{neuron.neuron_id}"
        
        if model_key not in self.forward_models:
            self.forward_models[model_key] = {
                'input_patterns': [],
                'output_patterns': [],
                'confidence_scores': [],
                'context_associations': defaultdict(list)
            }
            
        model = self.forward_models[model_key]
        
        # Add current pattern if consciousness is high enough
        if neuron.consciousness_level > 0.5:
            input_pattern = neuron.activation_history[-5:] if len(neuron.activation_history) >= 5 else []
            output_pattern = [neuron.membrane_potential]
            
            model['input_patterns'].append(list(input_pattern))
            model['output_patterns'].append(output_pattern)
            model['confidence_scores'].append(neuron.confidence_in_output)
            
            # Store context associations
            for key, value in context.items():
                if isinstance(value, (int, float)):
                    model['context_associations'][key].append(value)
                    
            # Limit model size
            if len(model['input_patterns']) > 100:
                # Keep most confident examples
                indices = np.argsort(model['confidence_scores'])[-50:]
                model['input_patterns'] = [model['input_patterns'][i] for i in indices]
                model['output_patterns'] = [model['output_patterns'][i] for i in indices]
                model['confidence_scores'] = [model['confidence_scores'][i] for i in indices]
    
    def generate_prediction(self, neuron: ConsciousNeuron, current_input: float) -> Tuple[float, float]:
        """Generate prediction for neuron's next state."""
        model_key = f"neuron_{neuron.neuron_id}"
        
        if model_key not in self.forward_models or not self.forward_models[model_key]['input_patterns']:
            # Default prediction
            return current_input, 0.5
            
        model = self.forward_models[model_key]
        
        # Simple pattern matching prediction
        current_pattern = list(neuron.activation_history[-5:]) if len(neuron.activation_history) >= 5 else []
        
        if not current_pattern:
            return current_input, 0.5
            
        # Find most similar pattern
        best_match_idx = 0
        best_similarity = -1
        
        for i, stored_pattern in enumerate(model['input_patterns']):
            if len(stored_pattern) == len(current_pattern):
                similarity = 1.0 - np.linalg.norm(np.array(stored_pattern) - np.array(current_pattern))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = i
                    
        # Generate prediction
        if best_similarity > 0.5:
            prediction = model['output_patterns'][best_match_idx][0]
            confidence = model['confidence_scores'][best_match_idx] * best_similarity
        else:
            prediction = current_input
            confidence = 0.3
            
        return prediction, confidence
    
    def compute_surprise(self, neuron: ConsciousNeuron, prediction: float, actual: float, confidence: float) -> float:
        """Compute surprise signal based on prediction error."""
        prediction_error = abs(prediction - actual)
        
        # Surprise is prediction error weighted by confidence
        surprise = prediction_error * confidence
        
        # Store for neuron
        self.surprise_signals[neuron.neuron_id] = surprise
        self.prediction_errors.append(prediction_error)
        
        return surprise


class ConsciousnessSTDP:
    """
    Consciousness-Driven Spike-Timing-Dependent Plasticity.
    
    Integrates traditional STDP with consciousness-driven modulation,
    predictive processing, and metacognitive control.
    """
    
    def __init__(
        self,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        A_plus: float = 0.01,
        A_minus: float = 0.012,
        consciousness_modulation: float = 2.0,
        prediction_modulation: float = 1.5,
        metacognitive_modulation: float = 1.2
    ):
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.consciousness_modulation = consciousness_modulation
        self.prediction_modulation = prediction_modulation
        self.metacognitive_modulation = metacognitive_modulation
        
        # Learning traces
        self.eligibility_traces: Dict[Tuple[int, int], float] = {}
        self.consciousness_traces: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        self.prediction_traces: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        
    def compute_stdp_update(
        self,
        pre_neuron: ConsciousNeuron,
        post_neuron: ConsciousNeuron,
        delta_t: float,
        global_attention: float = 0.0
    ) -> float:
        """
        Compute STDP weight update with consciousness-driven modulation.
        """
        # Basic STDP
        if delta_t > 0:  # Post after pre (potentiation)
            basic_stdp = self.A_plus * np.exp(-delta_t / self.tau_plus)
        else:  # Pre after post (depression)
            basic_stdp = -self.A_minus * np.exp(delta_t / self.tau_minus)
            
        # Consciousness-driven modulation
        consciousness_factor = self._compute_consciousness_modulation(
            pre_neuron, post_neuron, global_attention
        )
        
        # Predictive processing modulation
        prediction_factor = self._compute_prediction_modulation(pre_neuron, post_neuron)
        
        # Metacognitive modulation
        metacognitive_factor = self._compute_metacognitive_modulation(pre_neuron, post_neuron)
        
        # Combined update
        cd_stdp_update = (
            basic_stdp * 
            consciousness_factor * 
            prediction_factor * 
            metacognitive_factor
        )
        
        # Store traces for analysis
        connection_key = (pre_neuron.neuron_id, post_neuron.neuron_id)
        self.eligibility_traces[connection_key] = cd_stdp_update
        
        self.consciousness_traces[pre_neuron.neuron_id].append(consciousness_factor)
        self.prediction_traces[pre_neuron.neuron_id].append(prediction_factor)
        
        return cd_stdp_update
    
    def _compute_consciousness_modulation(
        self,
        pre_neuron: ConsciousNeuron,
        post_neuron: ConsciousNeuron,
        global_attention: float
    ) -> float:
        """Compute consciousness-based modulation factor."""
        # Average consciousness level
        avg_consciousness = (pre_neuron.consciousness_level + post_neuron.consciousness_level) / 2
        
        # Global workspace influence
        workspace_influence = global_attention * 0.5
        
        # Integration strength
        integration_factor = (pre_neuron.integration_strength + post_neuron.integration_strength) / 2
        
        # Combined consciousness factor
        consciousness_factor = (
            1.0 + 
            self.consciousness_modulation * avg_consciousness +
            workspace_influence +
            0.5 * integration_factor
        )
        
        return max(0.1, consciousness_factor)  # Ensure positive
    
    def _compute_prediction_modulation(
        self,
        pre_neuron: ConsciousNeuron,
        post_neuron: ConsciousNeuron
    ) -> float:
        """Compute predictive processing modulation factor."""
        # Prediction error influence
        avg_prediction_error = (pre_neuron.prediction_error + post_neuron.prediction_error) / 2
        error_factor = 1.0 + self.prediction_modulation * avg_prediction_error
        
        # Confidence in predictions
        avg_confidence = (pre_neuron.forward_model_confidence + post_neuron.forward_model_confidence) / 2
        confidence_factor = 0.5 + 0.5 * avg_confidence
        
        # Surprise signal (high surprise increases plasticity)
        surprise_factor = 1.0 + 0.5 * avg_prediction_error
        
        prediction_factor = error_factor * confidence_factor * surprise_factor
        
        return max(0.2, min(3.0, prediction_factor))  # Bounded
    
    def _compute_metacognitive_modulation(
        self,
        pre_neuron: ConsciousNeuron,
        post_neuron: ConsciousNeuron
    ) -> float:
        """Compute metacognitive modulation factor."""
        # Confidence in output
        avg_confidence = (pre_neuron.confidence_in_output + post_neuron.confidence_in_output) / 2
        
        # Uncertainty estimate (high uncertainty increases plasticity)
        avg_uncertainty = (pre_neuron.uncertainty_estimate + post_neuron.uncertainty_estimate) / 2
        uncertainty_factor = 1.0 + self.metacognitive_modulation * avg_uncertainty
        
        # Meta-learning rate
        avg_meta_lr = (pre_neuron.meta_learning_rate + post_neuron.meta_learning_rate) / 2
        meta_lr_factor = 0.5 + 1.5 * avg_meta_lr
        
        metacognitive_factor = uncertainty_factor * meta_lr_factor * (0.5 + 0.5 * avg_confidence)
        
        return max(0.3, min(2.5, metacognitive_factor))  # Bounded


class CDSTDPNetwork(nn.Module):
    """
    Consciousness-Driven STDP Network for multi-modal neuromorphic learning.
    
    Integrates conscious processing, global workspace, predictive processing,
    and advanced STDP for enhanced learning and adaptation.
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [512, 256],
        output_dim: int = 10,
        consciousness_threshold: float = 0.6,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.consciousness_threshold = consciousness_threshold
        self.device = device
        
        # Network components
        self.neurons: Dict[int, ConsciousNeuron] = {}
        self.synaptic_weights: Dict[Tuple[int, int], float] = {}
        self.layer_structure: List[List[int]] = []
        
        # Consciousness components
        self.global_workspace = GlobalWorkspace()
        self.predictive_processor = PredictiveProcessor()
        self.cd_stdp = ConsciousnessSTDP()
        
        # Performance tracking
        self.consciousness_metrics = {
            'global_activity': [],
            'attention_coherence': [],
            'integration_levels': [],
            'prediction_accuracy': [],
            'metacognitive_confidence': []
        }
        
        # Build network
        self._build_network()
        self._initialize_connections()
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
    def _build_network(self) -> None:
        """Build network with conscious neurons."""
        neuron_id = 0
        layer_dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        
        for layer_idx, dim in enumerate(layer_dims):
            layer_neurons = []
            
            for i in range(dim):
                # Create conscious neuron with spatial positioning
                position = (
                    layer_idx * 2.0,  # x: layer position
                    (i / dim) * 4.0 - 2.0,  # y: spread within layer
                    np.random.normal(0, 0.5)  # z: random depth
                )
                
                neuron = ConsciousNeuron(
                    neuron_id=neuron_id,
                    layer_id=layer_idx,
                    position=position,
                    threshold=0.7 + 0.3 * np.random.random(),
                    consciousness_level=0.1 if layer_idx > 0 else 0.0,  # Input layer starts unconscious
                    integration_strength=np.random.random()
                )
                
                self.neurons[neuron_id] = neuron
                layer_neurons.append(neuron_id)
                neuron_id += 1
                
            self.layer_structure.append(layer_neurons)
            
    def _initialize_connections(self) -> None:
        """Initialize synaptic connections between layers."""
        for layer_idx in range(len(self.layer_structure) - 1):
            pre_layer = self.layer_structure[layer_idx]
            post_layer = self.layer_structure[layer_idx + 1]
            
            for pre_id in pre_layer:
                for post_id in post_layer:
                    # Initialize with small random weights
                    weight = np.random.normal(0, 0.1)
                    self.synaptic_weights[(pre_id, post_id)] = weight
                    
    def forward(
        self, 
        input_tensor: torch.Tensor, 
        time_steps: int = 100,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Forward pass with consciousness-driven dynamics.
        """
        if context is None:
            context = {}
            
        batch_size = input_tensor.shape[0]
        
        # Initialize spike trains
        spike_trains = {
            nid: torch.zeros(batch_size, time_steps).to(self.device)
            for nid in self.neurons.keys()
        }
        
        # Convert input to spike trains
        input_rates = torch.clamp(input_tensor, 0, 1)
        
        # Simulation variables
        current_time = 0.0
        dt = 0.001
        
        # Performance tracking
        consciousness_levels = []
        global_broadcasts = []
        prediction_accuracies = []
        
        for t in range(time_steps):
            current_time = t * dt
            
            # Generate input spikes
            if t == 0 or np.random.random() < 0.1:  # Update input occasionally
                spike_prob = input_rates / time_steps
                input_spikes = torch.bernoulli(spike_prob)
                
                for batch_idx in range(batch_size):
                    for input_idx, neuron_id in enumerate(self.layer_structure[0]):
                        if input_spikes[batch_idx, input_idx] > 0:
                            spike_trains[neuron_id][batch_idx, t] = 1.0
                            self.neurons[neuron_id].spike_times.append(current_time)
                            
            # Process each layer with consciousness
            layer_activations = self._process_layers_with_consciousness(
                spike_trains, t, batch_size, current_time, dt, context
            )
            
            # Global workspace processing
            attention_signals = self._process_global_workspace(layer_activations, current_time)
            
            # Update consciousness levels
            current_consciousness = self._update_consciousness_levels(attention_signals, current_time)
            consciousness_levels.append(np.mean(current_consciousness))
            
            # Predictive processing
            prediction_accuracy = self._process_predictions(context, current_time)
            prediction_accuracies.append(prediction_accuracy)
            
            # STDP updates (every few time steps)
            if t % 10 == 0:
                self._apply_cd_stdp_updates(spike_trains, t, attention_signals)
                
        # Compute outputs
        output_rates = self._compute_output_rates(spike_trains, batch_size)
        
        # Performance metrics
        avg_consciousness = np.mean(consciousness_levels)
        avg_prediction_accuracy = np.mean(prediction_accuracies) if prediction_accuracies else 0.5
        
        self.consciousness_metrics['global_activity'].append(self.global_workspace.global_activity_level)
        self.consciousness_metrics['attention_coherence'].append(self.global_workspace.attention_coherence)
        self.consciousness_metrics['prediction_accuracy'].append(avg_prediction_accuracy)
        
        return {
            'output_rates': output_rates,
            'spike_trains': spike_trains,
            'consciousness_level': avg_consciousness,
            'prediction_accuracy': avg_prediction_accuracy,
            'global_activity': self.global_workspace.global_activity_level,
            'attention_coherence': self.global_workspace.attention_coherence,
            'metacognitive_confidence': self._compute_network_confidence()
        }
    
    def _process_layers_with_consciousness(
        self,
        spike_trains: Dict[int, torch.Tensor],
        time_step: int,
        batch_size: int,
        current_time: float,
        dt: float,
        context: Dict
    ) -> Dict[int, List[ConsciousNeuron]]:
        """Process network layers with consciousness-driven dynamics."""
        layer_activations = {}
        
        for layer_idx in range(1, len(self.layer_structure)):
            active_neurons = []
            prev_layer = self.layer_structure[layer_idx - 1]
            curr_layer = self.layer_structure[layer_idx]
            
            for batch_idx in range(batch_size):
                # Get previous layer activities
                prev_activities = {
                    nid: spike_trains[nid][batch_idx, time_step].item()
                    for nid in prev_layer
                }
                
                # Process current layer neurons
                for post_id in curr_layer:
                    post_neuron = self.neurons[post_id]
                    
                    # Compute synaptic input
                    synaptic_input = 0.0
                    for pre_id in prev_layer:
                        if (pre_id, post_id) in self.synaptic_weights:
                            weight = self.synaptic_weights[(pre_id, post_id)]
                            activity = prev_activities[pre_id]
                            synaptic_input += weight * activity
                            
                    # Generate prediction
                    prediction, pred_confidence = self.predictive_processor.generate_prediction(
                        post_neuron, synaptic_input
                    )
                    
                    # Update prediction error
                    post_neuron.update_prediction_error(synaptic_input, prediction)
                    
                    # Update membrane potential with consciousness modulation
                    consciousness_modulation = 1.0 + post_neuron.consciousness_level
                    modulated_input = synaptic_input * consciousness_modulation
                    
                    post_neuron.membrane_potential += dt * (
                        -post_neuron.membrane_potential / 0.02 +  # Leak
                        modulated_input +  # Synaptic input
                        0.1 * post_neuron.prediction_error  # Prediction error signal
                    )
                    
                    post_neuron.activation_history.append(post_neuron.membrane_potential)
                    
                    # Check for spike
                    if (post_neuron.membrane_potential >= post_neuron.threshold and 
                        post_neuron.refractory_counter == 0):
                        
                        spike_trains[post_id][batch_idx, time_step] = 1.0
                        post_neuron.spike_times.append(current_time)
                        post_neuron.membrane_potential = 0.0
                        post_neuron.refractory_counter = post_neuron.refractory_period
                        
                        active_neurons.append(post_neuron)
                        
                        # Form episodic memory if conscious
                        significance = min(1.0, abs(synaptic_input))
                        post_neuron.form_episodic_memory(context, significance)
                        
                    # Update refractory period
                    if post_neuron.refractory_counter > 0:
                        post_neuron.refractory_counter -= 1
                        
                    # Build forward model
                    self.predictive_processor.build_forward_model(post_neuron, context)
                    
            layer_activations[layer_idx] = active_neurons
            
        return layer_activations
    
    def _process_global_workspace(
        self,
        layer_activations: Dict[int, List[ConsciousNeuron]],
        current_time: float
    ) -> Dict[int, float]:
        """Process global workspace competition and broadcasting."""
        # Register neural coalitions
        for layer_idx, active_neurons in layer_activations.items():
            if active_neurons:
                # Group neurons into coalitions based on activity similarity
                coalitions = self._form_coalitions(active_neurons)
                
                for coalition in coalitions:
                    if len(coalition) >= 2:  # Minimum coalition size
                        activation_strength = np.mean([
                            len(n.spike_times[-10:]) for n in coalition
                        ])
                        self.global_workspace.register_coalition(coalition, activation_strength)
                        
        # Competition for global access
        winning_coalition = self.global_workspace.compete_for_global_access()
        
        # Global broadcast
        attention_signals = {}
        if winning_coalition:
            attention_signals = self.global_workspace.global_broadcast(winning_coalition)
            
        # Decay workspace
        self.global_workspace.decay_workspace()
        
        return attention_signals
    
    def _form_coalitions(self, active_neurons: List[ConsciousNeuron]) -> List[List[ConsciousNeuron]]:
        """Form neural coalitions based on activity patterns and associations."""
        if len(active_neurons) < 2:
            return [active_neurons] if active_neurons else []
            
        # Simple clustering based on activity similarity
        coalitions = []
        used_neurons = set()
        
        for i, neuron1 in enumerate(active_neurons):
            if neuron1.neuron_id in used_neurons:
                continue
                
            coalition = [neuron1]
            used_neurons.add(neuron1.neuron_id)
            
            # Find similar neurons
            for j, neuron2 in enumerate(active_neurons[i+1:], i+1):
                if neuron2.neuron_id in used_neurons:
                    continue
                    
                # Compute similarity
                similarity = self._compute_neuron_similarity(neuron1, neuron2)
                
                if similarity > 0.6:  # Threshold for coalition membership
                    coalition.append(neuron2)
                    used_neurons.add(neuron2.neuron_id)
                    
            coalitions.append(coalition)
            
        return coalitions
    
    def _compute_neuron_similarity(self, neuron1: ConsciousNeuron, neuron2: ConsciousNeuron) -> float:
        """Compute similarity between two neurons."""
        # Activity pattern similarity
        if (len(neuron1.activation_history) >= 5 and len(neuron2.activation_history) >= 5):
            pattern1 = np.array(list(neuron1.activation_history)[-5:])
            pattern2 = np.array(list(neuron2.activation_history)[-5:])
            pattern_similarity = 1.0 - np.linalg.norm(pattern1 - pattern2) / 5.0
        else:
            pattern_similarity = 0.5
            
        # Consciousness level similarity
        consciousness_similarity = 1.0 - abs(neuron1.consciousness_level - neuron2.consciousness_level)
        
        # Spatial proximity (layer-based)
        if neuron1.layer_id == neuron2.layer_id:
            spatial_similarity = 1.0
        else:
            spatial_similarity = 0.5
            
        # Semantic association overlap
        common_concepts = set(neuron1.semantic_associations.keys()) & set(neuron2.semantic_associations.keys())
        if common_concepts:
            semantic_similarity = len(common_concepts) / max(
                len(neuron1.semantic_associations), 
                len(neuron2.semantic_associations),
                1
            )
        else:
            semantic_similarity = 0.0
            
        # Weighted combination
        similarity = (
            0.4 * pattern_similarity +
            0.2 * consciousness_similarity +
            0.2 * spatial_similarity +
            0.2 * semantic_similarity
        )
        
        return similarity
    
    def _update_consciousness_levels(
        self,
        attention_signals: Dict[int, float],
        current_time: float
    ) -> List[float]:
        """Update consciousness levels for all neurons."""
        consciousness_levels = []
        
        global_activity = self.global_workspace.global_activity_level
        
        for neuron_id, neuron in self.neurons.items():
            attention_signal = attention_signals.get(neuron_id, 0.0)
            consciousness_level = neuron.update_consciousness_level(global_activity, attention_signal)
            consciousness_levels.append(consciousness_level)
            
        return consciousness_levels
    
    def _process_predictions(self, context: Dict, current_time: float) -> float:
        """Process predictive computations and compute accuracy."""
        total_accuracy = 0.0
        num_predictions = 0
        
        for neuron in self.neurons.values():
            if neuron.consciousness_level > 0.3 and len(neuron.activation_history) >= 2:
                # Generate prediction for next state
                current_activation = neuron.activation_history[-1]
                prediction, confidence = self.predictive_processor.generate_prediction(
                    neuron, current_activation
                )
                
                # Check accuracy against actual (previous prediction)
                if len(neuron.prediction_history) > 0:
                    actual = current_activation
                    previous_prediction = neuron.prediction_history[-1]
                    accuracy = 1.0 - min(1.0, abs(actual - previous_prediction))
                    total_accuracy += accuracy
                    num_predictions += 1
                    
                # Compute surprise
                if len(neuron.activation_history) >= 2:
                    actual_change = neuron.activation_history[-1] - neuron.activation_history[-2]
                    surprise = self.predictive_processor.compute_surprise(
                        neuron, prediction, actual_change, confidence
                    )
                    
        return total_accuracy / num_predictions if num_predictions > 0 else 0.5
    
    def _apply_cd_stdp_updates(
        self,
        spike_trains: Dict[int, torch.Tensor],
        time_step: int,
        attention_signals: Dict[int, float]
    ) -> None:
        """Apply consciousness-driven STDP updates."""
        global_attention = np.mean(list(attention_signals.values())) if attention_signals else 0.0
        
        # Process each connection
        for (pre_id, post_id), current_weight in self.synaptic_weights.items():
            pre_neuron = self.neurons[pre_id]
            post_neuron = self.neurons[post_id]
            
            # Find recent spike timing
            pre_spikes = [t for t in pre_neuron.spike_times if time_step * 0.001 - t < 0.05]
            post_spikes = [t for t in post_neuron.spike_times if time_step * 0.001 - t < 0.05]
            
            # Apply STDP for each spike pair
            for pre_time in pre_spikes:
                for post_time in post_spikes:
                    delta_t = (post_time - pre_time) * 1000  # Convert to ms
                    
                    if abs(delta_t) < 100:  # STDP window
                        stdp_update = self.cd_stdp.compute_stdp_update(
                            pre_neuron, post_neuron, delta_t, global_attention
                        )
                        
                        # Update weight
                        new_weight = current_weight + stdp_update
                        
                        # Bound weights
                        new_weight = max(-2.0, min(2.0, new_weight))
                        
                        self.synaptic_weights[(pre_id, post_id)] = new_weight
    
    def _compute_output_rates(self, spike_trains: Dict[int, torch.Tensor], batch_size: int) -> torch.Tensor:
        """Compute output firing rates."""
        output_layer = self.layer_structure[-1]
        output_rates = []
        
        for neuron_id in output_layer:
            spikes = spike_trains[neuron_id]
            rates = torch.mean(spikes, dim=1)  # Average over time
            output_rates.append(rates)
            
        return torch.stack(output_rates, dim=1)
    
    def _compute_network_confidence(self) -> float:
        """Compute overall network metacognitive confidence."""
        confidences = [neuron.compute_metacognitive_confidence() for neuron in self.neurons.values()]
        return np.mean(confidences)
    
    def train_epoch(self, dataloader: Any, criterion: nn.Module) -> Dict[str, float]:
        """Train for one epoch with consciousness-driven learning."""
        epoch_metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'consciousness_level': 0.0,
            'prediction_accuracy': 0.0,
            'metacognitive_confidence': 0.0,
            'global_activity': 0.0
        }
        
        num_batches = 0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            if len(data.shape) > 2:
                data = data.view(data.shape[0], -1)
                
            # Forward pass with context
            context = {
                'batch_idx': batch_idx,
                'task_difficulty': torch.std(data).item(),
                'target_entropy': entropy(torch.histogram(targets.float(), bins=10)[0].numpy() + 1e-6)
            }
            
            outputs = self.forward(data, time_steps=60, context=context)
            output_rates = outputs['output_rates']
            
            # Convert targets to one-hot
            if len(targets.shape) == 1:
                targets_onehot = torch.zeros(targets.shape[0], self.output_dim).to(self.device)
                targets_onehot.scatter_(1, targets.unsqueeze(1), 1)
            else:
                targets_onehot = targets
                
            # Compute loss
            loss = criterion(output_rates, targets_onehot)
            
            # Compute accuracy
            _, predicted = torch.max(output_rates.data, 1)
            _, target_labels = torch.max(targets_onehot.data, 1)
            accuracy = (predicted == target_labels).float().mean()
            
            # Accumulate metrics
            epoch_metrics['loss'] += loss.item()
            epoch_metrics['accuracy'] += accuracy.item()
            epoch_metrics['consciousness_level'] += outputs['consciousness_level']
            epoch_metrics['prediction_accuracy'] += outputs['prediction_accuracy']
            epoch_metrics['metacognitive_confidence'] += outputs['metacognitive_confidence']
            epoch_metrics['global_activity'] += outputs['global_activity']
            
            num_batches += 1
            
            if batch_idx % 50 == 0:
                self.logger.info(
                    f"Batch {batch_idx}: Loss={loss:.4f}, Acc={accuracy:.3f}, "
                    f"Consciousness={outputs['consciousness_level']:.3f}, "
                    f"PredAcc={outputs['prediction_accuracy']:.3f}"
                )
                
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
            
        return epoch_metrics
    
    def get_consciousness_summary(self) -> Dict[str, Any]:
        """Get comprehensive consciousness analysis."""
        summary = {
            'network_consciousness': {
                'avg_consciousness_level': np.mean([n.consciousness_level for n in self.neurons.values()]),
                'conscious_neurons': len([n for n in self.neurons.values() if n.consciousness_level > self.consciousness_threshold]),
                'total_neurons': len(self.neurons)
            },
            'global_workspace': {
                'active_coalitions': len(self.global_workspace.active_coalitions),
                'workspace_capacity': len(self.global_workspace.working_memory),
                'current_global_activity': self.global_workspace.global_activity_level
            },
            'predictive_processing': {
                'forward_models': len(self.predictive_processor.forward_models),
                'avg_prediction_error': np.mean(list(self.predictive_processor.prediction_errors)) if self.predictive_processor.prediction_errors else 0.0,
                'surprise_signals': len(self.predictive_processor.surprise_signals)
            },
            'learning_dynamics': {
                'active_traces': len(self.cd_stdp.eligibility_traces),
                'synaptic_weights_stats': {
                    'mean': np.mean(list(self.synaptic_weights.values())),
                    'std': np.std(list(self.synaptic_weights.values())),
                    'min': np.min(list(self.synaptic_weights.values())),
                    'max': np.max(list(self.synaptic_weights.values()))
                }
            },
            'performance_trends': {
                metric: {
                    'latest': values[-1] if values else 0,
                    'trend': np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0
                }
                for metric, values in self.consciousness_metrics.items()
            }
        }
        
        return summary


def create_cdstdp_network(
    input_dim: int = 784,
    hidden_dims: List[int] = [256, 128],
    output_dim: int = 10,
    device: str = 'cpu'
) -> CDSTDPNetwork:
    """Factory function to create consciousness-driven STDP network."""
    return CDSTDPNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        consciousness_threshold=0.6,
        device=device
    )


# Example usage and validation
if __name__ == "__main__":
    import torch.utils.data as data
    from torch.utils.data import DataLoader
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create CD-STDP network
    logger.info("Creating Consciousness-Driven STDP network...")
    network = create_cdstdp_network(
        input_dim=784,
        hidden_dims=[128, 64],
        output_dim=10,
        device='cpu'
    )
    
    # Dummy dataset
    class DummyDataset(data.Dataset):
        def __init__(self, size=500):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            x = torch.randn(784) * 0.5 + 0.5
            x = torch.clamp(x, 0, 1)
            y = torch.randint(0, 10, (1,)).item()
            return x, y
    
    # Create datasets
    train_dataset = DummyDataset(800)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    test_dataset = DummyDataset(200)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    criterion = nn.MSELoss()
    
    logger.info("Training CD-STDP network...")
    
    # Train for a few epochs
    for epoch in range(3):
        train_metrics = network.train_epoch(train_loader, criterion)
        
        logger.info(
            f"Epoch {epoch+1}: "
            f"Loss={train_metrics['loss']:.4f}, "
            f"Acc={train_metrics['accuracy']:.3f}, "
            f"Consciousness={train_metrics['consciousness_level']:.3f}, "
            f"PredAcc={train_metrics['prediction_accuracy']:.3f}, "
            f"Confidence={train_metrics['metacognitive_confidence']:.3f}"
        )
    
    # Get consciousness analysis
    consciousness_summary = network.get_consciousness_summary()
    
    logger.info("Consciousness Summary:")
    logger.info(f"  Conscious neurons: {consciousness_summary['network_consciousness']['conscious_neurons']}/{consciousness_summary['network_consciousness']['total_neurons']}")
    logger.info(f"  Global activity: {consciousness_summary['global_workspace']['current_global_activity']:.3f}")
    logger.info(f"  Forward models: {consciousness_summary['predictive_processing']['forward_models']}")
    logger.info(f"  Active synaptic traces: {consciousness_summary['learning_dynamics']['active_traces']}")
    
    logger.info("CD-STDP network validation completed successfully!")