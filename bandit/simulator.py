import numpy as np
import matplotlib.pyplot as plt
from .env import KArmedBandit
from .agents import EpsilonGreedyAgent, GradientBanditAgent, UCBAgent
from typing import Union, List

RLAgents = Union[EpsilonGreedyAgent, GradientBanditAgent, UCBAgent]

class BanditSimulator:
    """
    Runs multiple agents on k-armed bandit environments to evaluate learning behavior.
    
    Args:
        env_cls (type): Environment class (e.g., KArmedBandit).
        agents (list): List of agent instances.
        runs (int): Number of independent runs.
        steps (int): Number of steps per run.
        env_kwargs (dict): Parameters passed to the environment constructor.
        seed (int | None): Optional seed for reproducibility.
    """
    
    def __init__(
        self,
        env_cls,
        agents: List[RLAgents],
        runs: int = 2000,
        steps: int = 1000,
        env_kwargs: dict | None = None,
        seed: int | None = None,
    ):
        self.env_cls = env_cls
        self.agents = agents
        self.runs = runs
        self.steps = steps
        self.env_kwargs = env_kwargs or {}
        self.seed = seed
    
    def run(self):
        """Run all agents through all runs, returning rewards and optimal action rates."""
        all_rewards = np.zeros((len(self.agents), self.runs, self.steps))
        optimal_action_counts = np.zeros_like(all_rewards, dtype=bool)
        
        for i, agent in enumerate(self.agents):
            for r in range(self.runs):
                rng = np.random.default_rng(None if self.seed is None else self.seed + r)
                env: KArmedBandit = self.env_cls(seed=rng.integers(1e9), **self.env_kwargs)
                agent.reset()
                optimal_arm = env.optimal_arm()
                
                for t in range(self.steps):
                    action = agent.select_action()
                    reward = env.pull(action)
                    agent.update(action, reward)
                    all_rewards[i, r, t] = reward
                    optimal_action_counts[i, r, t] = (action == optimal_arm)
        
        return all_rewards, optimal_action_counts
    
    @staticmethod
    def plot_results(agents, all_rewards, optimal_action_counts, smooth=10):
        """Plot mean reward and optimal action % over time."""
        steps = all_rewards.shape[-1]
        x = np.arange(steps)
        
        plt.figure(figsize=(14, 5))
        
        # --- Average reward plot ---
        plt.subplot(1, 2, 1)
        for i, agent in enumerate(agents):
            avg_reward = all_rewards[i].mean(axis=0)
            if smooth > 1:
                avg_reward = np.convolve(avg_reward, np.ones(smooth) / smooth, mode="same")
            plt.plot(x, avg_reward, label=str(agent))
        plt.title("Average Reward")
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.legend()
        
        # --- Optimal action selection plot ---
        plt.subplot(1, 2, 2)
        for i, agent in enumerate(agents):
            optimal_rate = optimal_action_counts[i].mean(axis=0) * 100
            if smooth > 1:
                optimal_rate = np.convolve(optimal_rate, np.ones(smooth) / smooth, mode="same")
            plt.plot(x, optimal_rate, label=str(agent))
        plt.title("% Optimal Action")
        plt.xlabel("Steps")
        plt.ylabel("Percentage")
        plt.legend()
        
        plt.tight_layout()
        plt.show()
