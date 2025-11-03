
from typing import Optional
import numpy as np

class GradientBanditAgent:
    """
    Gradient Bandit agent for k-armed bandit problems.
    
    Uses preferences H(a) and softmax probabilities:
        π(a) = exp(H(a)) / sum_b exp(H(b))
    
    Updates preferences using reward baseline (optional):
        H(a) <- H(a) + α * (R - baseline) * (1 if a==action else -π(a))
    
    Args:
        k (int): Number of actions (arms).
        step_size (float): Learning rate α.
        use_baseline (bool): Whether to use average reward as baseline.
        seed (int | None): Random seed for reproducibility.
    """
    def __init__(
        self,
        k: int,
        step_size: float = 0.1,
        use_baseline: bool = True,
        seed: Optional[int] = None
    ):
        self.k = k
        self.alpha = step_size
        self.use_baseline = use_baseline
        self.rng = np.random.default_rng(seed)
        self.reset()
    
    def reset(self) -> None:
        """Reset preferences, probabilities, and average reward."""
        self.H = np.zeros(self.k, dtype=float)  # preferences
        self.pi = np.full(self.k, 1/self.k, dtype=float)
        self.last_action = None
        self.avg_reward = 0.0
        self.t = 0
    
    def select_action(self) -> int:
        """
        Choose an action using softmax probabilities derived from preferences.
        
        Returns:
            int: Selected action index.
        """
        expH = np.exp(self.H - np.max(self.H))  # stability
        self.pi = expH / np.sum(expH)
        action = int(self.rng.choice(self.k, p=self.pi))
        self.last_action = action
        return action
    
    def update(self, action: int, reward: float) -> None:
        """Update preferences based on reward and optional baseline."""
        self.t += 1
        if self.use_baseline:
            self.avg_reward += (reward - self.avg_reward) / self.t
            baseline = self.avg_reward
        else:
            baseline = 0.0
        
        for a in range(self.k):
            self.H[a] += self.alpha * (reward - baseline) * (1 if a == action else -self.pi[a])
    
    def __repr__(self) -> str:
        return f"GradientBanditAgent(k={self.k}, α={self.alpha}, baseline={self.use_baseline})"
