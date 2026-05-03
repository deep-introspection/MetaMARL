# Bilevel Fishery Example

This folder contains the main fishery experiment: a bilevel optimisation framework that searches for fishing regulations (quota, fine, stock threshold) that keep fish populations stable while preserving fishermen's welfare. The outer regulator uses Evolution Strategies (ES) to explore the space of possible rules; the inner fishermen learn their best response to those rules using multi-agent reinforcement learning (APPO via Ray RLlib).

## How to run

From the repository root, after running `uv sync` once:

```bash
# Default experiment — V0 mechanism, PPO inner loop
./run.sh

# V1 experiment — risk-sensitive continuous penalties, APPO inner loop
./run.sh examples/bilevel_fishery/main_appo_one_mechanism_v1.yaml
```

The `./run.sh` script activates the virtual environment and calls:

```bash
python -m examples.bilevel_fishery.main --config <config.yaml>
```

Do not use `uv run` directly — it conflicts with Ray's internal environment variable setup.

## Files in this folder

| File | What it is |
|---|---|
| `config.yaml` | Default config for the V0 experiment (PPO, binary fine/ban) |
| `main_appo_one_mechanism_v1.yaml` | Config for the V1 experiment (APPO, risk-sensitive penalties) |
| `mechanism.py` | V0 mechanism: quota, fine amount, ban period, catch probability |
| `mechanism_v1.py` | V1 mechanism: quota, target stock, continuous risk penalty |
| `regulated_env.py` | V0 inner-loop environment — the world that fishing agents live in |
| `regulated_env_v1.py` | V1 inner-loop environment — uses the risk-penalty enforcement |
| `regulator_env.py` | Outer-loop environment — wraps the inner APPO run and computes ES fitness |
| `bilevel.py` | YAML config loader — parses a config file into a full `BilevelConfig` |
| `main.py` | Entry point called by `run.sh` |
| `main_appo_one_mechanism_v1.py` | Programmatic equivalent of the V1 YAML config |
| `sandbox_tutorial.ipynb` | Interactive tutorial — start here if you are new to the project |

## Config parameters — `config.yaml`

### Top-level experiment

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `experiment.name` | string | `bilevel_fishery` | Name tag for the run (used in logging) |
| `experiment.seed` | int | `0` | Global random seed for reproducibility |

### Ray cluster

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `ray.device` | string | `cpu` | Compute device. Use `gpu` only if a CUDA GPU is available and Ray is configured for it |
| `ray.num_cpus` | int | `4` | Total CPU cores made available to Ray |
| `ray.omp_threads` | int | `1` | OpenMP thread count per worker. Keep at 1 to avoid over-subscription |
| `ray.logging_level` | string | `ERROR` | Ray log verbosity. Set to `INFO` for debugging, `ERROR` for quiet runs |
| `ray.runtime_env.excludes` | list | — | Directories excluded from Ray's file sync (large or irrelevant folders) |

### Mechanism

The mechanism is the set of regulatory rules that the outer ES loop is searching over. V0 and V1 differ in which parameters exist.

#### V0 mechanism (`mechanism.space: FisheryMechanismSpace`)

| Parameter | Type | Range | Meaning |
|---|---|---|---|
| `mechanism.scaling.max_fine` | float | — | Upper bound on the fine amount. Used to normalise `fine_amount` into [0, 1] for ES |
| `mechanism.scaling.max_ban` | int | — | Upper bound on ban duration in steps. Used for normalisation |
| `mechanism.default.fixed_quota` | float | [0, 1] | Hard harvest cap per agent per step, expressed as a fraction of `max_fish` |
| `mechanism.default.prop_quota` | float | [0, 1] | Proportional quota factor. The effective quota is `min(fixed_quota, prop_quota * fish_norm)`. Tightens the cap when stock is low |
| `mechanism.default.min_stock` | float | [0, 1] | Normalised fish stock level below which all fishing is prohibited |
| `mechanism.default.fine_amount` | float | [0, max_fine] | Penalty multiplied by the violation magnitude when a breach is detected |
| `mechanism.default.ban_period` | int | [0, max_ban] | Number of steps an agent is banned from fishing after a detected violation |
| `mechanism.default.catch_prob` | float | [0, 1] | Probability that a violation is detected at any given step (stochastic enforcement) |

#### V1 mechanism (`mechanism.space: FisheryMechanismSpaceV1`) — `main_appo_one_mechanism_v1.yaml`

| Parameter | Type | Range | Meaning |
|---|---|---|---|
| `mechanism.default.fixed_quota` | float | [0, 1] | Same as V0: hard per-step harvest cap |
| `mechanism.default.prop_quota` | float | [0, 1] | Same as V0: stock-proportional quota multiplier |
| `mechanism.default.min_stock` | float | [0, 1] | Stock floor below which fishing is prohibited |
| `mechanism.default.target_stock` | float | [0, 1] | Desired stock level. Agents are penalised for fishing that is predicted to push the stock below this value |
| `mechanism.default.fine_amount` | float | [0, max_fine] | Base fine for quota violations |
| `mechanism.default.risk_penalty_scale` | float | [0, max_fine] | Overall magnitude of the continuous risk penalty. Larger values create stronger deterrence against fishing near collapse |
| `mechanism.default.risk_penalty_power` | float | [1, 5] | Exponent of the stock-shortfall term. A value of 2 gives a quadratic penalty that grows fast near collapse. Higher values concentrate the deterrence even closer to the threshold |

### Training schedule

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `training.outer_iters` | int | `100` | Number of ES generations. Each generation evaluates the full population of mechanism candidates |

### Outer loop (ES regulator)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `outer.optimizer` | string | `ES` | Must be `ES`. Only Evolution Strategies is supported for the outer loop |
| `outer.training.sigma` | float | `0.3` | Initial standard deviation of the ES search distribution. Controls how broadly ES explores the mechanism space at the start. Larger values explore more aggressively but may miss fine structure |
| `outer.training.mean_lr` | float | `0.1` | Learning rate for updating the ES mean (step size in the direction of improving mechanisms) |
| `outer.training.sigma_lr` | float | `0.05` | Learning rate for adapting `sigma` over time. ES can shrink `sigma` as it converges |
| `outer.training.min_sigma` | float | `0.001` | Floor on `sigma` — prevents the search from collapsing to a point prematurely |
| `outer.training.max_sigma` | float | `0.5` | Ceiling on `sigma` — prevents runaway exploration |
| `outer.environment.env` | string | `FisheryRegulatorEnv` | Class name of the outer-loop environment (resolves via the component registry) |
| `outer.environment.horizon` | int | `200` | Number of inner-loop steps used for fitness evaluation per mechanism |
| `outer.environment.train_iters` | int | `50` | Number of APPO training iterations run per ES mechanism evaluation. More iterations give agents more time to learn, but slow down each ES generation |
| `outer.environment.env_config.ecology_cfg.sus_weight` | float | `1.0` | Weight applied to the sustainability penalty in the fitness function: `fitness = mean_reward - sus_weight * sustainability_penalty` |
| `outer.environment.env_config.ecology_cfg.sus_threshold` | float | `0.1` | Normalised fish stock level considered a "collapse". Steps where fish falls below this threshold contribute to the sustainability penalty |
| `outer.environment.env_config.ecology_cfg.max_fish` | float | `2.0` | Carrying capacity for fish. Used to de-normalise trajectory values for visualisation |
| `outer.environment.env_config.ecology_cfg.max_algae` | float | `2.0` | Carrying capacity for algae. Used for de-normalisation in visualisation |

### Inner loop (APPO fishing agents)

#### Ecology parameters

These parameters define the predator-prey dynamics of the fishery environment. They appear in the Lotka-Volterra equations:

```
dX/dt = delta * X * Y  -  gamma * X  -  H(t)     (fish)
dY/dt = alpha * Y  -  beta * Y * X                (algae)
```

| Parameter | Symbol | Default | Meaning |
|---|---|---|---|
| `ecology_cfg.alpha` | α | `0.5` | Intrinsic algae growth rate. Higher values mean algae recovers faster |
| `ecology_cfg.beta` | β | `0.1` | Rate at which fish consume algae. Higher values deplete algae faster |
| `ecology_cfg.delta` | δ | `0.2` | Rate at which fish benefit from eating algae (growth coupling). Higher values mean fish grow faster when algae is abundant |
| `ecology_cfg.gamma` | γ | `0.4` | Natural fish mortality rate. Higher values mean fish die faster in the absence of food |
| `ecology_cfg.dt` | Δt | `0.01` | Euler integration step size. Smaller values give more accurate dynamics but increase simulation cost. For `horizon=200`, `dt=0.01` simulates 2 time units |
| `ecology_cfg.fish_init` | — | `0.5` | Initial fish biomass at the start of each episode (before log-normal jitter) |
| `ecology_cfg.algae_init` | — | `1.0` | Initial algae biomass at the start of each episode (before log-normal jitter) |
| `ecology_cfg.max_fish` | — | `5.0` | Fish carrying capacity (biomass ceiling). Fish is normalised by this value throughout the environment |
| `ecology_cfg.max_algae` | — | `5.0` | Algae carrying capacity. Algae is normalised by this value |

#### Environment runners

| Parameter | Default | Meaning |
|---|---|---|
| `env_runners.num_env_runners` | `1` | Number of parallel environment runner workers. Set to `0` in V1 to run environments in the main process (simpler, avoids some Ray overhead) |
| `env_runners.num_envs_per_env_runner` | `16` | Number of parallel environment copies per runner. Increases throughput |
| `env_runners.rollout_fragment_length` | `200` | Steps collected per environment before sending to the learner |
| `env_runners.batch_mode` | `complete_episodes` | Whether to wait for full episodes (`complete_episodes`) or cut at the fragment boundary (`truncate_episodes`) |

#### PPO / APPO training

| Parameter | Default | Meaning |
|---|---|---|
| `training.gamma` | `0.99` | Discount factor for future rewards. Values close to 1 make agents care about long-term consequences — important for sustainability |
| `training.lr` | `0.0003` | Learning rate for the policy network optimiser |
| `training.train_batch_size` | `3200` | Total steps in each training batch |
| `training.minibatch_size` | `512` | Steps per gradient update. Smaller values update more frequently but with noisier gradients |
| `training.vtrace` | `true` | (APPO only) Enables V-trace off-policy correction, which stabilises learning when data arrives asynchronously |
| `training.entropy_coeff` | `0.001` | (APPO only) Entropy bonus coefficient. Encourages exploration by rewarding diverse actions |
| `training.grad_clip` | `40.0` | (APPO only) Maximum gradient norm. Prevents catastrophic updates |

#### Agents

| Parameter | Default | Meaning |
|---|---|---|
| `agents.fisher.count` | `3` | Number of fishing agents. All agents share a single policy (`fisher_policy`) |
| `agents.fisher.observation_base_dim` | `4` | Number of base observation features (fish stock, algae stock, etc.) before the mechanism vector is appended. V1 uses 4; V0 uses 5 |
| `agents.fisher.action_space.low / high` | `0.0 / 1.0` | Agents output a harvest fraction in [0, 1]. The actual harvest scales with the current fish stock |

## What to expect as output

If W&B reporting is enabled (`reporting.reporter: wandb`), the following metrics are logged per ES iteration:

- `objective_score` — the fitness value for each mechanism candidate (higher is better)
- `mean_reward` — mean agent reward across all inner-loop steps for each mechanism
- `collapse_rate` — fraction of steps where fish stock fell below `sus_threshold`
- `mean_fish` — average normalised fish stock
- `min_fish` — minimum normalised fish stock observed
- `total_fines` — cumulative fines collected across all agents

Population-level ES statistics (mean mechanism parameters, sigma, generation) are logged alongside the per-mechanism metrics.

Trajectory plots (fish and algae timeseries per mechanism) are saved under `results/` when `output_dir` is passed to `BilevelConfigLoader.from_yaml`.

## How to modify the mechanism

### Changing default parameter values

Edit the `mechanism.default` section in your config file. For example, to set a stricter minimum stock threshold:

```yaml
mechanism:
  default:
    min_stock: 0.30   # was 0.10
```

### Switching between V0 and V1

- V0: set `mechanism.space: FisheryMechanismSpace` and use `inner.environment.env: FisheryRegulatedEnv`
- V1: set `mechanism.space: FisheryMechanismSpaceV1` and use `inner.environment.env: FisheryRegulatedEnvV1`

See the two config files for a complete working example of each.

### Changing which parameters ES optimises

By default in V0, `optimize_params` is an empty list (all parameters are fixed at their defaults — useful for single-mechanism evaluation). In V1, all seven parameters are optimised by default. To change this, modify `FisheryMechanismSpace` or `FisheryMechanismSpaceV1` in `mechanism.py` / `mechanism_v1.py` — look for the `optimize_params` argument in the constructor.

## V0 vs V1: key differences

| Aspect | V0 (`mechanism.py`) | V1 (`mechanism_v1.py`) |
|---|---|---|
| Enforcement style | Binary: violation detected with probability `catch_prob` → fine + fishing ban | Continuous: smooth risk penalty proportional to stock shortfall below `target_stock` |
| Ban mechanism | `ban_period` steps of forced inactivity after a detected violation | No ban — enforcement is entirely through the reward signal |
| Stochasticity | Enforcement is stochastic (`catch_prob`) | Deterministic — agents always feel the penalty |
| Parameters | 6 (includes `ban_period`, `catch_prob`) | 7 (adds `target_stock`, `risk_penalty_scale`, `risk_penalty_power`; removes ban/catch) |
| Mechanism vector dim | 6 | 7 |
| When to use | Simpler baseline; good for exploring discrete enforcement regimes | More realistic continuous deterrence; better for gradient-based analysis |

V1 is generally preferred for new experiments because the continuous penalty provides a smoother signal to both the RL agents and the ES optimiser.

## Ecology model notes

Each episode starts with fish and algae stocks drawn from a log-normal distribution centred on `fish_init` and `algae_init` (standard deviation 0.05 in log-space). This introduces realistic inter-episode variability.

When total desired harvest across all agents would exceed the available fish stock, all harvests are scaled down proportionally. This scarcity scaling is the main source of competition between agents: they are not playing a cooperative game.

Agent observations include the current fish stock, algae stock, a measure of their remaining ban period (V0), the effective quota, and all mechanism parameters. Agents therefore know the rules they are operating under, which is the assumption of mechanism-conditioned reinforcement learning.
