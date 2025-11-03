from typing import Optional, Literal
import numpy as np

class EpsilonGreedyAgent:
    """
    ε-Greedy agent for k-armed bandit problems.
    Supports:
        - Fixed ε
        - ε decay (linear, exponential, reciprocal)
        - Sample-average or constant step-size updates
        - Optimistic initialization
    
    Args:
        k (int): Number of actions (arms).
        epsilon (float): Initial exploration rate.
        decay_type (Literal['none', 'linear', 'exponential', 'reciprocal']): 
            Type of ε decay schedule.
        decay_rate (float): Rate parameter for ε decay.
        epsilon_min (float): Minimum allowed ε.
        initial_value (float): Initial Q-value for all arms.
        step_size (float | None): Constant step size (if None → 1/N).
        seed (int | None): Random seed.
    """
    def __init__(
        self,
        k: int,
        epsilon: float = 0.1,
        decay_type: Literal['none', 'linear', 'exponential', 'reciprocal'] = 'none',
        decay_rate: float = 0.001,
        epsilon_min: float = 0.01,
        initial_value: float = 0.0,
        step_size: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        self.k = k
        self.epsilon = epsilon
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        self.epsilon_min = epsilon_min
        self.alpha = step_size
        self.initial_value = initial_value
        self.rng = np.random.default_rng(seed)
        self.reset()
    
    def reset(self) -> None:
        """Reset Q-values, epsilon and counts."""
        self.Q = np.full(self.k, self.initial_value, dtype=float)
        self.N = np.zeros(self.k, dtype=int)
        self.last_action = None
        self.t = 0
        self.epsilon_current = self.epsilon
        self.epsilon_history = []
    
    def _update_epsilon(self) -> None:
        """Update ε based on decay type."""
        self.epsilon_history.append(self.epsilon_current)
        if self.decay_type == 'none':
            return
        self.t += 1
        if self.decay_type == 'linear':
            self.epsilon_current = max(self.epsilon_min, self.epsilon - self.decay_rate * self.t)
        elif self.decay_type == 'exponential':
            self.epsilon_current = max(self.epsilon_min, self.epsilon * np.exp(-self.decay_rate * self.t))
        elif self.decay_type == 'reciprocal':
            self.epsilon_current = max(self.epsilon_min, self.epsilon / (1 + self.decay_rate * self.t))
    
    def select_action(self) -> int:
        """
        Choose an action using ε-greedy policy.
        Returns:
            int: Selected action index.
        """
        # use current epsilon value for this decision
        if self.rng.random() < self.epsilon_current:
            # explore
            action = self.rng.integers(self.k)  
        else:
            # exploit (break ties randomly)
            max_value = np.max(self.Q)
            candidates = np.flatnonzero(self.Q == max_value)
            action = int(self.rng.choice(candidates))
        self.last_action = action
        # decay epsilon after the action so the initial epsilon is used at t=0
        self._update_epsilon()
        return action
    
    def update(self, action: int, reward: float) -> None:
        """Update the estimated value of the chosen action."""
        self.N[action] += 1
        alpha = self.alpha if self.alpha is not None else 1 / self.N[action]
        self.Q[action] += alpha * (reward - self.Q[action])
    
    def __repr__(self) -> str:
        parts = [f"k={self.k}"]
        
        if self.epsilon is not None:
            parts.append(f"ε={self.epsilon}")
        if self.decay_type != 'none':
            parts.append(f"decay={self.decay_type}")
            parts.append(f"decay_rate={self.decay_rate}")
        if self.initial_value != 0.0:
            parts.append(f"init_Q={self.initial_value}")
        if self.alpha is not None:
            parts.append(f"α={self.alpha}")
        
        return f"EpsilonGreedyAgent({', '.join(parts)})"
