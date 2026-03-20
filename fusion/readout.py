"""Linear readout layer trained via ridge regression."""
import numpy as np


class LinearReadout:
    """Ridge regression readout for reservoir states.

    Solves W = (X^T X + lambda I)^{-1} X^T Y using pseudoinverse.

    Args:
        alpha: Ridge regularisation parameter (default 1e-3).
    """

    def __init__(self, alpha: float = 1e-3):
        self.alpha = alpha
        self.W: np.ndarray | None = None
        self.n_classes: int = 0

    def fit(self, states: np.ndarray, labels: np.ndarray) -> "LinearReadout":
        """Fit the readout layer.

        Args:
            states: Feature matrix of shape (n_samples, n_features).
            labels: Integer class labels of shape (n_samples,).

        Returns:
            self
        """
        states = np.asarray(states, dtype=float)
        labels = np.asarray(labels, dtype=int)
        n_samples, n_features = states.shape
        self.n_classes = int(labels.max()) + 1

        # One-hot encode labels
        Y = np.zeros((n_samples, self.n_classes))
        Y[np.arange(n_samples), labels] = 1.0

        # Ridge regression: W = (X^T X + alpha I)^{-1} X^T Y
        A = states.T @ states + self.alpha * np.eye(n_features)
        B = states.T @ Y
        self.W = np.linalg.solve(A, B).T  # shape (n_classes, n_features)
        return self

    def predict(self, states: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            states: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted class labels of shape (n_samples,).
        """
        if self.W is None:
            raise RuntimeError("LinearReadout must be fitted before calling predict.")
        states = np.asarray(states, dtype=float)
        scores = states @ self.W.T  # (n_samples, n_classes)
        return np.argmax(scores, axis=1)
