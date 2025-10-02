# Bilevel Fishery Optimization

Bilevel optimization framework for sustainable fishery management using Evolution Strategies to find optimal regulatory mechanisms that balance economic returns with ecosystem preservation.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## Overview

- **Outer loop (ES)**: Searches for optimal regulatory mechanisms (quotas, fines, bans)
- **Inner loop (IPPO)**: Trains strategic fisherman agents under each candidate mechanism
- **Environment**: Multi-agent fishery with Lotka-Volterra ecosystem dynamics

## Installation

**Prerequisites:** Python 3.12+, [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)

```bash
git clone git@github.com:deep-introspection/bilevel-fishery.git
cd bilevel-fishery
uv sync
```

Optional: Configure W&B for experiment tracking
```bash
wandb login
```

## Quick Start

```bash
# Basic run (10 outer iterations, 100 inner iterations per candidate)
uv run python main.py

# Quick test (reduced iterations for testing)
uv run python main.py --outer-iters 2 --inner-iters 10 --pop-size 4

# With automatic visualization (enables trajectory recording)
uv run python main.py --trace-episodes 3 --outer-iters 5

# With W&B experiment tracking
uv run python main.py --wandb --wandb-project my-project
```

## Usage

### Running Experiments

```bash
uv run python main.py [options]
```

**Key Options:**
- `--outer-iters N`: ES optimization iterations (default: 10)
- `--inner-iters N`: PPO training iterations per candidate (default: 100)
- `--pop-size N`: ES population size (default: 8)
- `--eval-episodes N`: Evaluation episodes per candidate (default: 5)
- `--sustain-weight W`: Sustainability penalty weight (default: 5.0)
- `--sus-threshold T`: Fish collapse threshold (default: 0.1)
- `--trace-episodes N`: Episodes to record for visualization (default: 0, disabled)
- `--workers N`: Parallel workers (default: 2)
- `--wandb`: Enable W&B logging

**Examples:**
```bash
# Sustainability-focused experiment
uv run python main.py --sustain-weight 10.0 --sus-threshold 0.15 --outer-iters 15

# High-resolution with visualization
uv run python main.py --outer-iters 20 --inner-iters 200 --pop-size 12 \
  --eval-episodes 10 --trace-episodes 5 --workers 4

# Production run with tracking
uv run python main.py --outer-iters 50 --inner-iters 500 --pop-size 16 \
  --workers 8 --wandb --wandb-project fishery-prod
```

### Visualization

**Automatic (Recommended):** Run experiments with `--trace-episodes > 0` to auto-generate visualizations in the experiment directory.

**Manual (Optional):** Use standalone script for post-hoc analysis:
```bash
# List available experiments
uv run python visualize_experiments.py --list

# Visualize latest experiment
uv run python visualize_experiments.py --latest --summary-only

# Visualize specific iteration
uv run python visualize_experiments.py --latest --iteration 0

# Visualize all iterations
uv run python visualize_experiments.py --latest --all-iterations
```

**Note:** `visualize_experiments.py` is a convenience wrapper. All visualization functions are in `visualization.py` and can be used directly in Python/notebooks.

## Project Structure

```
bilevel-fishery/
├── main.py                   # Experiment orchestration and CLI
├── config.py                 # Configuration and constants
├── environment.py            # Multi-agent fishery environment
├── mechanism.py              # Regulatory mechanism parameters
├── evolution_strategies.py   # ES optimization
├── evaluation.py             # Sustainability metrics
├── training.py               # PPO algorithm configuration
├── visualization.py          # Core visualization functions
├── visualize_experiments.py  # Optional: standalone visualization CLI
├── utils.py                  # File I/O and logging helpers
├── pyproject.toml            # Dependencies and metadata
└── README.md
```

**Module Overview:**
- **config.py**: Centralized defaults and hyperparameters
- **environment.py**: `FisheryEnvFixed` with Lotka-Volterra dynamics
- **mechanism.py**: Parameter mapping between unit hypercube and mechanism space
- **evolution_strategies.py**: ES optimizer with rank-based fitness shaping
- **evaluation.py**: Metrics computation and trajectory recording
- **training.py**: PPO algorithm builders with Ray/RLlib
- **visualization.py**: Plotting functions (ecosystem, actions, summaries)
- **utils.py**: File operations, W&B integration, experiment management
- **main.py**: Complete bilevel optimization pipeline

## Output Structure

Experiments create timestamped directories: `runs/bilevel_YYYYMMDD_HHMMSS/`

```
runs/bilevel_YYYYMMDD_HHMMSS/
├── experiment_summary.csv        # Per-iteration results
├── experiment_results.json       # Final optimized mechanism
├── experiment_summary.png        # Optimization progress (if --trace-episodes > 0)
├── outer_0/
│   ├── candidates.csv           # All evaluated candidates
│   ├── train_cand_*.csv         # Training curves
│   ├── best_trace_0.json        # Trajectories (if --trace-episodes > 0)
│   ├── best_trace_0.csv
│   ├── ecosystem_dynamics.png   # Generated when --trace-episodes > 0
│   ├── fishermen_actions.png
│   └── combined_analysis.png
└── checkpoint_*/                 # Final trained model
```

## Development

**Test components:**
```bash
# Test environment
python -c "from environment import FisheryEnvFixed; env = FisheryEnvFixed(); print(env.reset())"

# Test mechanism mapping
python -c "from mechanism import *; import numpy as np; print(map_unit_vector_to_mechanism(np.random.random(5)))"

# Test ES sampling
python -c "from evolution_strategies import *; import numpy as np; print(sample_es_population(np.ones(5)*0.5, 0.1, 4, np.random.default_rng()))"
```

**Common modifications:**
- Add mechanism parameter → Update `MechanismParameters` in `mechanism.py`
- Change sustainability metrics → Edit `EvaluationMetrics` in `evaluation.py`
- Modify ES algorithm → Update functions in `evolution_strategies.py`
- Customize environment → Extend `FisheryEnvFixed` in `environment.py`
- Adjust training → Modify `build_ppo_algorithm()` in `training.py`

## Dependencies

Core: Ray/RLlib (≥2.40.0), Gymnasium (≥0.29.0), NumPy (≥1.24.0), Matplotlib (≥3.7.0), Seaborn (≥0.12.0), WandB (≥0.15.0, optional)

See `pyproject.toml` for complete list.

## Troubleshooting

**Ray initialization fails:** `ray stop` to kill existing instances, check system resources

**Import errors:** `uv sync` to install dependencies, use `uv run` for all commands

**Slow training:** Reduce `--workers`, `--pop-size`, `--inner-iters`, or `--horizon`

**Memory issues:** Reduce `--pop-size`, `--workers`, `--eval-episodes`, or `--trace-episodes`

**Visualization errors:** Ensure `--trace-episodes > 0` when running experiments, check `MPLBACKEND=Agg`

## License

BSD 3-Clause License - see [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@software{bilevel_fishery,
  title = {Bilevel Optimization for Sustainable Fishery Management},
  author = {Guillaume Dumas},
  year = {2025},
  url = {https://github.com/deep-introspection/bilevel-fishery}
}
```
