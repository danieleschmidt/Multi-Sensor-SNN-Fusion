"""Liquid State Machine reservoir (Echo State Network)."""
import numpy as np


class LiquidStatePool:
    """Random recurrent reservoir with fixed weights.

    W_res is scaled to spectral radius 0.9 for stable dynamics.
    W_in connects input to reservoir neurons.

    Args:
        n_neurons: Number of reservoir neurons (N).
        input_dim: Dimensionality of input at each timestep.
        spectral_radius: Target spectral radius for W_res (default 0.9).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_neurons: int = 100,
        input_dim: int = 1,
        spectral_radius: float = 0.9,
        seed: int = 42,
    ):
        self.n_neurons = n_neurons
        self.input_dim = input_dim
        rng = np.random.default_rng(seed)

        # Recurrent weight matrix
        W_res = rng.standard_normal((n_neurons, n_neurons))
        eigenvalues = np.linalg.eigvals(W_res)
        current_sr = np.max(np.abs(eigenvalues))
        self.W_res = (W_res / current_sr) * spectral_radius

        # Input weight matrix
        self.W_in = rng.standard_normal((n_neurons, input_dim)) * 0.1

        # Initial state
        self.x = np.zeros(n_neurons)

    def reset(self):
        """Reset reservoir state to zero."""
        self.x = np.zeros(self.n_neurons)

    def step(self, u: np.ndarray) -> np.ndarray:
        """Advance reservoir by one timestep.

        Args:
            u: Input vector of shape (input_dim,).

        Returns:
            Updated reservoir state of shape (n_neurons,).
        """
        u = np.asarray(u, dtype=float).flatten()
        self.x = np.tanh(self.W_res @ self.x + self.W_in @ u)
        return self.x.copy()

    def run(self, spike_sequence: np.ndarray) -> np.ndarray:
        """Run reservoir over a sequence of inputs.

        Args:
            spike_sequence: Array of shape (T, input_dim).

        Returns:
            Array of reservoir states shape (T, n_neurons).
        """
        spike_sequence = np.asarray(spike_sequence, dtype=float)
        T = spike_sequence.shape[0]
        states = np.zeros((T, self.n_neurons))
        self.reset()
        for t in range(T):
            states[t] = self.step(spike_sequence[t])
        return states
