# Architecture

This document describes how a bilevel run is assembled and executed, with the
class names of `core/`. It is the reference for extending the framework; the
tutorials in `tutorials/` cover the same ground pedagogically.

## The two levels and the World

```
BilevelConfig.build_optimizer()
   |-- RayRuntime.ensure_initialized()                 start Ray (local mode)
   |-- World.options(name).remote()                    the shared blackboard actor
   |-- inner_cfg.build_optimizer(world=...)            RayOptimizer -> PolicyActor -> RLlib Algorithm
   |       env_creator -> MultiAgentRegulatedEnv x (num_envs_per_env_runner)
   |-- outer_cfg.build_optimizer(world=..., inner_opt) ESOptimizer -> RegulatorEnv(optimizer=inner)
   `-- outer.batch_capacity = inner.batch_capacity     ES population = inner envs / seeds

BilevelOptimizer.run()
   for generation in range(outer_iters):
       ESOptimizer.run()
           population = sample()                                   (n, d) in (0, 1)
           _, fitness, ... = RegulatorEnv.step(population)
               mechanisms = decode(population)                     one Mechanism per row
               publish MechanismContext(index, seed, published) per (candidate, seed)
               inner.reset(); for _ in train_iters: inner.run()    RLlib training iterations
               inner.evaluate()                                    if eval seeds are configured
               fitness = aggregate_rewards(...)                    one scalar per candidate
               flush consumed contexts
           update mean and sigma from (population, fitness)
```

The `World` (`core/world/base.py`) is a Ray actor holding contexts keyed by id,
an optimizer-to-contexts map and the mechanism registry. Two payload types
matter:

- `MechanismContext(index, seed, status, mechanism)` — a candidate. Its
  `status` follows `MechanismStatus`: `published -> train -> eval -> done`.
  `World.get_mechanism_by_id(mechanism_id, seed, mode)` returns the candidate
  whose `(index, seed)` matches and whose status allows the requested `mode`,
  and advances the status.
- `EnvStepContext(env_id, seed, policy_seed, status, mechanism, observation,
  reward, action, info)` — one record per environment step, published by the
  regulated environments and consumed by `aggregate_rewards`.

## The inner environment

`MultiAgentRegulatedEnv` (`core/envs/marl_regulated.py`) is an RLlib
`MultiAgentEnv`. At `reset` it fetches its candidate from the World; until a
candidate is published it steps inertly (zero rewards, no dynamics) so that
RLlib's environment checks can run. Each `step` normalizes the raw policy
actions (sigmoid squashing), applies the benchmark dynamics and the mechanism,
and publishes an `EnvStepContext`. On this branch `core/mechanism/` defines a
`MechanismSpace` protocol (`encode`, `decode`, `clip`, `sample`) and a
`VectorMechanism` value object, and the fishery environment applies the decoded
mechanism inside its `_step` override, which the `step` template method calls;
the hook-based step pipeline and the social-influence observation live on
`feature/social-influence-testing`.

Identity flows through the episode id: `core/callbacks.py::tag_episode_with_env_idx`
rewrites it to `env={i}|m={mechanism_id}|ps={policy_seed}|ss={seed}|raw=...`,
and `RayOptimizerConfig._apply_agents_to_rllib` builds one RLModule per
(candidate, seed) with a `policy_mapping_fn` that parses that id. This is the
only channel carrying mechanism and seed identity into RLlib.

## The outer environment and optimizer

`RegulatorEnv` (`core/envs/regulator.py`) is a single-agent gymnasium env whose
action is a population matrix and whose reward is the vector of fitness values.
Benchmarks subclass it and implement `aggregate_rewards`, which turns the
World's records into one fitness per candidate (the fishery averages a tail
window of rewards and biomass per candidate and seed, then combines harvest
and sustainability in `examples/bilevel_fishery/contexts.py::FitnessContext`).

`ESOptimizer` (`core/optimizers/es/optimizer.py`) keeps a mean in `(0, 1)^d`
handled in logit space and a scalar `sigma`; it samples antithetic populations,
estimates the natural-evolution-strategies gradient from the standardized
fitness, moves the mean and adapts `sigma` (expands after a worse generation,
contracts after a better one). `dimension == 0` is a *fixed mode* in which a
single fixed mechanism is evaluated and reported; `batch_capacity == 1` is a
sequential (1+1)-ES.

## Configuration

Every optimizer has an `OptimizerConfig` (`core/optimizers/config.py`) with
RLlib-style fluent builders (`.environment(...)`, `.training(...)`,
`.debugging(seed=, num_seeds=)`), `copy()` and `freeze()`, and a
`build_optimizer()` that instantiates the optimizer with a frozen copy of the
config and creates its environment. `RayOptimizerConfig` records every RLlib
builder call as a deferred operation applied to the `AlgorithmConfig` at build
time. `BilevelConfig` composes an inner and an outer config, the World name,
the Ray runtime options and the reporting backend.

`debugging(seed=s, num_seeds=n)` derives `n` policy seeds from `s`
(`numpy.random.SeedSequence`); the inner config's seeds are copied to the
outer env so that one `MechanismContext` is published per (candidate, seed).

## Extending

- **A new benchmark** is a `MultiAgentRegulatedEnv` subclass (the regulated
  system) plus a `RegulatorEnv` subclass (how outcomes become a fitness) and,
  usually, a `FitnessContext`-like schema. Declare the observation space so
  that it matches the benchmark features plus whatever the mechanism appends.
- **A new mechanism** must be a pure value object that can be encoded to and
  decoded from a normalized vector through a `MechanismSpace`; on this branch
  the regulated environment applies it in its `_step` override.
- **A new reporting backend** implements the reporter interface of
  `core/reporting/`; on the logging feature branch that means rendering
  labeled `Series` resolved from `Query` objects.

## Testing strategy

Unit tests never start Ray: `tests/conftest.py` provides `FakeWorld` (an
in-memory stand-in with the same `.remote()` call shape, `ray.get` patched to a
pass-through) and `FakeReporter`. Integration tests start a small local Ray
runtime (`ray_session` fixture), use the real `World` actor and run short
end-to-end configurations. Notebook tests execute `tutorials/*.ipynb` with
`nbconvert`.
