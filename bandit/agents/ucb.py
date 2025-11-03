from typing import Optional
import numpy as np

class UCBAgent:
    """
    Upper Confidence Bound (UCB) agent for k-armed bandit problems.
    
    Selects actions using:
        A_t = argmax_a [ Q(a) + c * sqrt(ln(t) / N(a)) ]
    
    Supports both sample-average and constant step-size updates.
    
    Args:
        k (int): Number of actions (arms).
        c (float): Exploration coefficient.
        step_size (float | None): Constant step size α; if None, uses 1/N[a].
        seed (int | None): Random seed for reproducibility.
    """
    def __init__(
        self,
        k: int,
        c: float = 2.0,
        step_size: Optional[float] = None,
        seed: Optional[int] = None
    ):
        self.k = k
        self.c = c
        self.alpha = step_size
        self.rng = np.random.default_rng(seed)
        self.reset()
    
    def reset(self) -> None:
        """Reset Q-values, counts, and timestep."""
        self.Q = np.zeros(self.k, dtype=float)
        self.N = np.zeros(self.k, dtype=int)
        self.t = 0
        self.last_action = None
    
    def select_action(self) -> int:
        """
        Choose an action using UCB policy.
        
        Returns:
            int: Selected action index.
        """
        self.t += 1
        ucb_values = np.array([
            q + self.c * np.sqrt(np.log(self.t) / (n if n > 0 else 1e-5))
            for q, n in zip(self.Q, self.N)
        ])
        action = int(np.argmax(ucb_values))
        self.last_action = action
        return action
    
    def update(self, action: int, reward: float) -> None:
        """Update the estimated value of the chosen action."""
        self.N[action] += 1
        alpha = self.alpha if self.alpha is not None else 1 / self.N[action]
        self.Q[action] += alpha * (reward - self.Q[action])
    
    def __repr__(self) -> str:
        parts = [f"k={self.k}", f"c={self.c}"]
        if self.alpha is not None:
            parts.append(f"α={self.alpha}")
        return f"UCBAgent({', '.join(parts)})"
