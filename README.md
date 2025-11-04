# Multi-Armed Bandit Simulator

A clean, modular, lightweight simulator for testing and comparing different **multi-armed bandit** strategies, implemented in Python. It includes key exploration methods like **ε-greedy**, **UCB**, and **Gradient Bandit**, along with plotting and performance evaluation tools.

---

## 🚀 Overview

This project simulates the **k-armed bandit problem**, a cornerstone of reinforcement learning, to analyze how agents balance **exploration vs. exploitation**.

You can:

* Simulate stochastic reward environments.
* Test multiple agent strategies with or without decay mechanisms.
* Track reward evolution and optimal action percentages.
* Visualize and compare performance.

---

## 📁 Project Structure

```
.
├── bandit/
│   ├── __init__.py
│   ├── env.py                  # KArmedBandit environment class
│   ├── simulator.py            # Experiment runner for agents
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── epsilon_greedy.py   # ε-greedy strategy (fixed, decayed, optimistic, etc.)
│   │   ├── ucb.py              # Upper Confidence Bound (UCB) agent
│   │   └── gradient_bandit.py  # Policy gradient-based agent
│   └── __pycache__/
├── notebooks/
│   └── main.ipynb              # Interactive tests and visualization
├── pyproject.toml              # uv project configuration
├── README.md
└── uv.lock                     # dependency lock file
```

---

## ⚙️ Environment

The **KArmedBandit** simulates a stochastic reward process:

```python
KArmedBandit(k=10, mean_range=(-2, 2), std_range=(0.5, 1.5))
```

Each arm has its own Gaussian reward distribution. You can call:

```python
reward = env.pull(arm)
```

---

## 🧠 Agents Tested

The following configurations were tested in `main.ipynb`:

```python
agent_configs = [
    {"type": "epsilon", "k": 10, "epsilon": 0.3},
    {"type": "epsilon", "k": 10, "epsilon": 0.01},
    {"type": "epsilon", "k": 10, "epsilon": 0.3, "initial_value": 5.0},
    {"type": "epsilon", "k": 10, "epsilon": 0.3, "step_size": 0.1},
    {"type": "epsilon", "k": 10, "epsilon": 0.6, "decay_type": "exponential", "decay_rate": 0.005, "epsilon_min": 0.01},
    {"type": "ucb", "k": 10, "c": 2.0},
    {"type": "gradient", "k": 10, "step_size": 0.1, "use_baseline": True},
]
```

All agents were instantiated dynamically and tested under identical environments for fair comparison.

---

## 📊 Results Overview

Experiments were run with `steps=1000` and `runs=200`.

* **Exponential decay** ε-greedy showed smoother convergence and strong balance between exploration and exploitation.
* **Optimistic initialization** boosted early exploration but stabilized slower.
* **UCB** achieved consistent results but was more conservative.
* **Gradient Bandit** converged cleanly, favoring stable reward arms over time.

---

## 🧩 Visualization

Plots included:

* Average reward over time (smoothed)
* Optimal action percentage

Example (from the notebook):

```python
plot_results(agents, avg_rewards, optimal_actions)
```

---

## 🚀 Getting Started

```bash
uv sync
uv run notebooks/main.ipynb
```

Or open the notebook in VS Code / Jupyter and explore interactively.

---

## 🧾 License

MIT License — free to use, modify, and extend.

## Author

**MohammadReza Fazli**
