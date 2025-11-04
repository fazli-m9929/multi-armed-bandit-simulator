import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

class KArmedBandit:
    """
    Simulates a k-armed bandit environment where each arm's rewards are drawn 
    from a normal (Gaussian) distribution with its own mean and standard deviation.
    
    This environment is typically used to study exploration vs. exploitation 
    strategies (e.g., epsilon-greedy, UCB, gradient bandits).
    
    Args:
        k (int): Number of arms (bandits) in the environment.
        mean_range (Tuple[float, float]): Range (min, max) from which each arm's mean is sampled uniformly.
        std_range (Tuple[float, float]): Range (min, max) from which each arm's standard deviation is sampled uniformly.
        seed (int | None): Optional random seed for reproducibility.
    
    Attributes:
        k (int): Number of arms.
        means (np.ndarray): Mean reward for each arm.
        stds (np.ndarray): Standard deviation of rewards for each arm.
    """
    
    def __init__(
        self,
        k: int = 10,
        mean_range: Tuple[float, float] = (-2.0, 2.0),
        std_range: Tuple[float, float] = (0.5, 1.5),
        seed: int | None = None,
    ) -> None:
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.means = self.rng.uniform(*mean_range, size=k)
        self.stds = self.rng.uniform(*std_range, size=k)
    
    def sample_rewards(self, n: int = 1000) -> List[np.ndarray]:
        """
        Generate `n` reward samples for each arm according to its Gaussian distribution.
        
        Args:
            n (int): Number of reward samples per arm.
        
        Returns:
            List[np.ndarray]: A list of `k` NumPy arrays, 
            where each array contains `n` samples drawn from N(mean, std).
        """
        return [
            self.rng.normal(loc=mu, scale=sigma, size=n)
            for mu, sigma in zip(self.means, self.stds)
        ]
    
    def pull(self, arm: int) -> float:
        """
        Pull a specific arm once and get a stochastic reward.
        
        Args:
            arm (int): Index of the arm to pull (0 ≤ arm < k).
        
        Returns:
            float: Reward sampled from N(mean_arm, std_arm).
        """
        if not 0 <= arm < self.k:
            raise ValueError(f"Arm index {arm} out of range [0, {self.k - 1}]")
        return float(self.rng.normal(self.means[arm], self.stds[arm]))
    
    def optimal_arm(self) -> int:
        """
        Return the index of the arm with the highest mean (best expected reward).
        
        Returns:
            int: Index of the optimal arm.
        """
        return int(np.argmax(self.means))
    
    def plot_distributions(self, n_samples: int = 2000) -> None:
        """
        Plot the reward distributions of each arm as violins.
        Similar style to Sutton & Barto (Figure 2.1).
        
        Args:
            n_samples (int): Number of samples drawn from each arm to form the violins.
        """
        samples = [
            np.random.normal(self.means[i], self.stds[i], size=n_samples)
            for i in range(self.k)
        ]
        
        plt.figure(figsize=(9, 6))
        parts = plt.violinplot(
            samples, positions=np.arange(1, self.k + 1),
            showmeans=False, showextrema=False
        )
        
        for pc in parts['bodies']:
            pc.set_facecolor('gray')
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        
        # Plot the true means
        for i, mu in enumerate(self.means, 1):
            plt.hlines(mu, i - 0.25, i + 0.25, colors='k', lw=2)
            plt.text(i + 0.2, mu, f"$q_*( {i} )$", va='center', fontsize=10)
        
        plt.axhline(0, color='k', linestyle='--', lw=1)
        plt.xticks(np.arange(1, self.k + 1))
        plt.xlabel("Action", fontsize=12)
        plt.ylabel("Reward distribution", fontsize=12)
        plt.title("Reward Distributions for Each Action")
        plt.tight_layout()
        plt.show()
    
    def __repr__(self) -> str:
        return (
            f"KArmedBandit("
            f"k={self.k}, "
            f"means={np.round(self.means, 2)}, "
            f"stds={np.round(self.stds, 2)})"
        )
