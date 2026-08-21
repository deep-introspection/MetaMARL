# Mechanism abstraction branch — research engineer handoff TODO

## Scope

This branch redesigns regulation as explicit mechanism algorithms that intervene
on the agent/environment loop through three channels:

\[
\mathcal M_\theta
=
\left(
\mathcal M_\theta^O,
\mathcal M_\theta^A,
\mathcal M_\theta^R
\right)
\]

with:

\[
o_t^\* = \mathcal M_\theta^O(s_t, o_t),
\qquad
a_t^\* = \mathcal M_\theta^A(s_t, a_t),
\qquad
r_t^\* = \mathcal M_\theta^R(r_t, s_t, a_t^\*, s_{t+1}).
\]

The implementation has been designed and mostly written, but **has not yet been
validated end-to-end against `dev`**. Treat the current code as an integration
branch, not as a validated replacement.

Primary concerned files:

```text
core/envs/hooks.py
core/envs/marl_regulated.py
core/envs/regulated.py
core/mechanism/algorithms/penalty.py
core/mechanism/algorithms/quota.py
core/mechanism/algorithms/social_influence.py
core/mechanism/algorithms/subsidy.py
core/mechanism/base.py
core/mechanism/composition/chained_mechanism.py
core/mechanism/composition/parallel_mechanism.py
core/mechanism/space.py
core/types.py
```

The first goal is **not new features**. The first goal is to make this complete
abstraction run end-to-end, cover the concerned modules with tests, and verify
that the new abstraction preserves benchmark behavior.

---

# 0. Definition of done

- [ ] The fishery benchmark builds and runs end-to-end with the new mechanism abstraction.
- [ ] A quota-only run completes training and evaluation.
- [ ] A quota + subsidy run completes.
- [ ] A quota + subsidy + social-observation run completes.
- [ ] Chained composition works for action, observation, and reward channels.
- [ ] Parallel composition has one coherent API and tests.
- [ ] All concrete `Mechanism` implementations satisfy the abstract base class.
- [ ] Mechanism optimizer vectors encode/decode correctly.
- [ ] Action and observation spaces agree with transformed values.
- [ ] Unit tests cover every concerned mechanism/env/composition module.
- [ ] Integration tests cover the benchmark + mechanism lifecycle.
- [ ] Reproducibility against `dev` is checked where practical.
- [ ] Quota behavior is numerically compared against the dev fishery benchmark if time permits.
- [ ] The tutorial notebooks run after the P0 integration fixes are merged.

---

# 1. P0 — make the current implementation internally consistent

There are several API mismatches in the supplied branch that should be fixed
before trying to compare results.

## 1.1 Reconcile `BilevelConfig.mechanism(...)`

The supplied config method currently has the shape:

```python
def mechanism(
    self,
    *,
    space: MechanismSpace,
    default: Mechanism = None,
    **kwargs,
) -> Self:
    ...
```

but the proposed benchmark config uses:

```python
.mechanism(
    mechanism=ChainedMechanism(...)
)
```

Choose and document one public API.

Recommended target:

```python
.mechanism(
    mechanism=mechanism,
    space=optional_space,
)
```

or, if `MechanismSpace` remains the owner of default construction:

```python
.mechanism(
    space=space,
    default=mechanism,
)
```

Acceptance:

- [ ] exactly one supported builder signature;
- [ ] examples and tutorials use that signature;
- [ ] `BilevelConfig.build_optimizer()` injects the same mechanism/space into inner and outer components;
- [ ] fixed mechanisms work without an unnecessary optimizer space;
- [ ] optimized mechanisms expose an optimizer dimension unambiguously.

---

## 1.2 Make every concrete mechanism instantiable

`Mechanism` currently declares these abstract methods/properties:

```text
dimension
encode
decode
clip
param_names
to_vector
```

plus identity implementations for:

```text
action
observation
reward
```

Audit every concrete class.

### `QuotaMechanism`

Currently shown:

```text
to_vector       YES
param_names     YES
action          YES
observation     YES

dimension       MISSING in supplied code
encode          MISSING
decode          MISSING
clip            MISSING
```

- [ ] implement missing abstract API or move common parameterized behavior into a reusable base.

### `SubsidyMechanism`

Currently shown:

```text
to_vector       YES
param_names     YES
reward          YES

dimension       MISSING
encode          MISSING
decode          MISSING
clip            MISSING
```

- [ ] implement missing abstract API.

### `SocialInfluenceMechanism`

Currently shown only with `observation(...)`.

- [ ] implement fixed/optimized parameter API;
- [ ] decide whether `influence_weight` is optimized or fixed;
- [ ] if fixed, `dimension == 0`;
- [ ] if optimized, define normalized encode/decode bounds.

### `ThresholdPenaltyMechanism`

Currently has `dimension`, `encode`, `decode`, `param_names`, `reward`.

- [ ] verify/implement `clip`;
- [ ] verify/implement `to_vector`;
- [ ] decide whether threshold/penalty are fixed or optimizer-controlled.

### `ChainedMechanism`

- [ ] verify/implement `clip`;
- [ ] define `to_vector` for the semantic vector exposed to agents;
- [ ] test concatenation/slicing of child optimizer vectors.

### `ParallelMechanism`

- [ ] same abstract-method audit;
- [ ] same vector semantics audit.

No concrete mechanism should remain abstract accidentally.

---

## 1.3 Remove stale constructor arguments from examples

The proposed example still uses old `MechanismSpace`-style arguments:

```python
QuotaMechanism(
    optimize_params=["fixed_quota"],
    default_fixed_quota=0.56224,
    default_max_demand_frac=1.0,
)
```

but the shown new dataclass is:

```python
QuotaMechanism(
    fixed_quota: float,
    bindings: ...,
    action_component: int = 0,
    ...
)
```

Likewise `SubsidyMechanism` currently declares:

```python
SubsidyMechanism(
    subsidy: float,
    cost: float,
    action_component: int = 1,
)
```

not old `optimize_params/default_*` arguments.

- [ ] update all examples to the new object model;
- [ ] keep optimization selection in one place only;
- [ ] do not duplicate defaults in both mechanism objects and spaces.

---

# 2. P0 — fix environment mechanism dispatch

The regulated env should have one explicit lifecycle:

```text
policy output
    ↓
normalize action
    ↓
benchmark action hook (optional)
    ↓
mechanism action transform
    ↓
benchmark intrinsic/base reward
    ↓
benchmark transition
    ↓
mechanism reward transform
    ↓
benchmark observation
    ↓
mechanism observation transform
    ↓
publish context
```

Audit `MultiAgentRegulatedEnv` against this.

## 2.1 Wrong mechanism method calls

The supplied `reward(...)` method returns:

```python
return self.mechanism.action(
    reward_dict,
    env=self,
)
```

- [ ] call `self.mechanism.reward(...)`.

The supplied `observation(...)` method returns:

```python
return self.mechanism.action(
    obs_with_theta,
    env=self,
)
```

- [ ] call `self.mechanism.observation(...)`.

---

## 2.2 Observation concatenation is currently dict-unsafe

The supplied code contains:

```python
theta = self.mechanism.to_vector()
obs_with_theta = np.concatenate(
    [observation_dict, theta],
    axis=0,
)
```

`observation_dict` is a multi-agent dictionary, not an array.

Target:

```python
obs_with_theta = {
    agent_id: np.concatenate(
        [
            np.asarray(observation, dtype=np.float32).reshape(-1),
            theta,
        ]
    ).astype(np.float32, copy=False)
    for agent_id, observation in observation_dict.items()
}
```

Then pass the dict through `mechanism.observation(...)`.

- [ ] add a regression test for this exact failure mode.

---

## 2.3 Avoid bypassing the public reward/observation pipeline

`step()` currently directly calls mechanism reward and public observation.

Decide whether public `reward(...)` is the pipeline method or remove it.
There should not be two overlapping reward paths.

Recommended:

```python
intrinsic_rewards = benchmark_reward(delivered_actions)
...
rewards = self.reward(
    intrinsic_rewards,
    action_after=delivered_actions,
)
obs = self.observation({})
```

- [ ] one reward path only;
- [ ] one observation path only;
- [ ] one action path only.

---

## 2.4 Fix "no published mechanism" fallback path

The supplied branch calls:

```python
self.observation(agent_id, self.S_t)
```

even though `observation(...)` accepts one `observation_dict`.

- [ ] make fallback reset/step behavior use the same observation pipeline;
- [ ] add a test where the world has not published a non-default mechanism.

---

# 3. P0 — fix the fishery benchmark under the new action semantics

The benchmark currently has a 2-component action:

```text
component 0 = harvest fraction
component 1 = restoration effort
```

but the transition currently treats the full vector as harvest:

```python
delivered_harvest = {
    agent_id: action * full_required_harvest
    for agent_id, action in A_t.items()
}
```

Target:

```python
harvest_fraction = float(action[0])
restoration_effort = float(action[1])
```

Then:

```python
requested_harvest_i = harvest_fraction_i * full_required_harvest
```

- [ ] extract action components deliberately;
- [ ] document the semantic component map;
- [ ] avoid implicit whole-vector arithmetic.

## 3.1 Restoration dynamics are currently disconnected

The supplied transition uses:

```python
growth = biological_growth + noise + kwargs["restoration"]
```

but the shown env step does not pass `restoration`.

Recommended:

```python
restoration = restoration_effectiveness * sum(
    action[1] for action in A_t.values()
)
```

and either pass it explicitly to the transition hook or derive it there.

The subsidy mechanism should modify reward; ecological restoration belongs in
the benchmark transition.

- [ ] connect restoration action to fish dynamics;
- [ ] keep ecology and incentive shaping separate.

## 3.2 Fix `K` reference

The transition contains:

```python
fish_next = float(np.clip(fish_next, 0.0, K))
```

- [ ] use `self.K` or deliberately remove the upper clipping;
- [ ] add boundary tests.

## 3.3 Define the base reward

The shown `FisheryRegulatedEnv` does not include a `@reward` hook.

- [ ] add or verify the benchmark base reward;
- [ ] test reward before any mechanism;
- [ ] test reward after subsidy/penalty.

---

# 4. P0 — fix `SubsidyMechanism`

Intended reward:

\[
r_{i,t}^{*}
=
r_{i,t}
-
c e_{i,t}^{2}
+
\sigma_\theta e_{i,t}.
\]

The supplied implementation contains:

```python
actions[agent_id[self.action_component]]
```

Fix to:

```python
actions[agent_id][self.action_component]
```

Recommended implementation:

```python
effort = float(
    actions[agent_id][self.action_component]
)

reward
+ self.subsidy * effort
- self.cost * effort**2
```

Acceptance:

- [ ] zero effort -> no subsidy/cost;
- [ ] positive effort -> exact analytical reward;
- [ ] component selection tested;
- [ ] reward type remains `float`;
- [ ] public bounds use `ValueError`, not only `assert`.

---

# 5. P0 — finish `SocialInfluenceMechanism`

Full social influence from the project slides:

\[
c_t^i
=
\sum_{j\neq i}
D_{KL}
\left[
\pi_j(a_t^j \mid a_t^i, s_t^j)
\|
\pi_j(a_t^j \mid s_t^j)
\right]
\]

and:

\[
r_t^i = r_{i,t} + \beta c_t^i.
\]

The supplied implementation currently implements only observation shaping:

\[
o_{i,t}^{*}
=
[
o_{i,t},
a_{1,t-1},
\dots,
a_{j,t-1},
\dots
].
\]

- [ ] document that this is observation augmentation, not the full Jacques et al. KL bonus;
- [ ] `influence_weight` is currently unused in the shown implementation;
- [ ] either implement the KL reward term or scope/rename the class;
- [ ] add `bindings` to the dataclass if constructor-injected bindings are intended;
- [ ] test peer-action ordering;
- [ ] test self-action exclusion;
- [ ] test observation dimensionality.

---

# 6. P0 — quota mechanism numerical tests

The quota computes:

\[
L = \sigma((0-q)/w_q),
\quad
U = \sigma((1-q)/w_q),
\quad
C_t = \sigma((b_t-q)/w_q)
\]

and:

\[
\alpha_t = \frac{C_t-L}{U-L}.
\]

Requested fraction \(u_{i,t}\) becomes:

\[
u_{i,t}^{*}
=
u_{i,t}
-
\operatorname{smooth}_{+}
\left(
u_{i,t}-\alpha_t;
w_u
\right).
\]

Tests:

- [ ] resource close to 0 -> allowed fraction near lower end;
- [ ] resource close to 1 -> allowed fraction near 1;
- [ ] resource near `fixed_quota` -> expected sigmoid transition;
- [ ] request below allowed fraction remains approximately unchanged;
- [ ] request above allowed fraction is smoothly capped;
- [ ] non-target action components are unchanged;
- [ ] input arrays are not mutated in place;
- [ ] per-agent mapping preserved;
- [ ] `allowed_frac` is available to the quota observation transform.

---

# 7. P0/P1 — optional quota reproducibility against `dev`

Preferred but not blocking if time is limited.

Create a deterministic quota-only fixture using the same:

```text
r
K
p
fish_init/B0
sigma
policy seed
environment seed
action trajectory
fixed quota
transition widths
```

Compare old dev and new quota transforms before involving RLlib.

At each step compare:

```text
resource_level
allowed_frac/effective quota
requested harvest fraction
delivered harvest fraction
fish stock
fish_norm
H_attempted
H_realized
reward if reward semantics are unchanged
```

Suggested tolerance:

```python
np.testing.assert_allclose(
    new,
    dev,
    rtol=1e-6,
    atol=1e-7,
)
```

If a difference is intentional, document whether it comes from normalization,
smoothing, reward semantics, or dynamics.

---

# 8. P1 — hook discovery tests

`core/envs/hooks.py` attaches markers and
`MultiAgentRegulatedEnv.__init_subclass__` discovers them.

Tests:

- [ ] `@reset` registers reset hook;
- [ ] `@action` registers action hook;
- [ ] `@reward` registers reward hook;
- [ ] `@observation` registers observation hook;
- [ ] `@transition` registers transition hook;
- [ ] inherited hooks behave intentionally;
- [ ] multiple hooks of one type either raise or have documented deterministic behavior.

Recommendation: fail fast rather than silently letting the last same-type hook
win.

---

# 9. P1 — mechanism binding tests

A binding is:

```python
binding: env -> runtime context value
```

Example:

```python
bindings={
    "resource_level": lambda env: (
        env.S_t["fish"] / max(env.K, EPS)
    ),
}
```

Tests:

- [ ] `resolve(env)` returns configured keys;
- [ ] missing required binding raises at construction;
- [ ] quota receives normalized resource level;
- [ ] social observation receives `previous_actions` and `agent_ids`;
- [ ] child bindings in compositions resolve against the correct env;
- [ ] bindings remain serializable in Ray/cloudpickle integration.

---

# 10. P1 — chained composition tests

For:

```python
ChainedMechanism(
    children=(m1, m2, m3)
)
```

contract:

\[
x^\*
=
M_3(M_2(M_1(x))).
\]

Tests:

- [ ] action order exactly follows child tuple order;
- [ ] reward order exactly follows child tuple order;
- [ ] observation order exactly follows child tuple order;
- [ ] each child receives previous child's transformed output;
- [ ] each child resolves its own env bindings;
- [ ] dimension is sum of child dimensions;
- [ ] encode is concatenation;
- [ ] decode slices correctly;
- [ ] parameter names preserve child identity/order;
- [ ] zero-dimension children do not break slicing.

Interaction test:

```text
child 1: multiply by 2
child 2: add 1

expected chain:
2x + 1
```

---

# 11. P1 — parallel composition API repair and tests

The supplied `ParallelMechanism` calls:

```python
child.apply_action(...)
child.apply_reward(...)
child.apply_observation(...)
```

while `Mechanism` exposes:

```python
action(...)
reward(...)
observation(...)
```

- [ ] reconcile this before use.

Recommended target:

```python
context = child.resolve(env)
child.action(copy, **context)
```

Parallel contract:

\[
M_{\parallel}^{A}(x)
=
\Gamma_A
\left(
x,
M_1^A(x),
\dots,
M_k^A(x)
\right).
\]

Each child receives the same original input.

Tests:

- [ ] every child sees the same original input;
- [ ] no child sees another child's output;
- [ ] merge receives original + tuple of outputs;
- [ ] merge ordering is documented;
- [ ] deep copies prevent cross-child mutation;
- [ ] action/reward/observation merge functions tested separately;
- [ ] dimensions/encode/decode tested.

---

# 12. P1 — action and observation spaces

The current example hard-codes:

```python
action_space = Box(shape=(2,))
```

and an observation shape based on an older mechanism-space abstraction.

Expected fishery features include:

```text
base observation:
    fish_norm
    total_usage_norm

quota augmentation:
    allowed_frac/effective quota

social augmentation:
    previous peer actions

optional mechanism parameter vector:
    theta
```

For 10 agents and 2-D actions, social influence adds:

```text
(10 - 1) * 2 = 18
```

features per agent.

- [ ] compute/validate final observation dimension;
- [ ] decide whether mechanisms expose `observation_dimension_delta`;
- [ ] decide whether `to_vector()` is always appended;
- [ ] remove dependencies on obsolete `FisheryMechanismSpace().full_dimension` where inappropriate;
- [ ] assert actual observation shape matches declared space;
- [ ] assert normalized action shape matches declared action space.

---

# 13. P1 — context publishing tests

Each step should preserve:

```text
env_id
seed
policy_seed
mode/status
mechanism_id
observation
reward
action
info
```

- [ ] values correspond to regulated action/reward/observation actually used;
- [ ] seeds remain immutable for an env instance;
- [ ] mechanism ID matches the published mechanism;
- [ ] publication does not modify behavior.

---

# 14. P1 — recommended test layout

```text
tests/envs/test_hooks.py
tests/envs/test_marl_regulated.py

tests/mechanism/test_base.py
tests/mechanism/algorithms/test_quota.py
tests/mechanism/algorithms/test_subsidy.py
tests/mechanism/algorithms/test_penalty.py
tests/mechanism/algorithms/test_social_influence.py

tests/mechanism/composition/test_chained_mechanism.py
tests/mechanism/composition/test_parallel_mechanism.py
tests/mechanism/test_space.py

tests/examples/test_fishery_regulated_env.py
tests/integration/test_fishery_mechanisms.py
```

Coverage target:

- [ ] meaningful branch coverage for all concerned files;
- [ ] aim for >=90% line coverage on pure mechanism/composition modules;
- [ ] every mechanism dispatch path covered even if distributed integration coverage is lower.

Suggested command:

```bash
pytest \
  tests/envs \
  tests/mechanism \
  tests/examples/test_fishery_regulated_env.py \
  tests/integration/test_fishery_mechanisms.py \
  --cov=core.envs \
  --cov=core.mechanism \
  --cov-report=term-missing
```

---

# 15. P1 — staged end-to-end smoke runs

Do not begin with 1000 outer iterations.

## Stage A — environment only

```text
2 agents
deterministic seed
fixed mechanism
horizon 5
```

## Stage B — inner optimizer only

```text
2 agents
1 mechanism
1 seed
horizon 10
1-2 train iterations
```

## Stage C — quota-only bilevel

```text
2 candidates
1 seed
2 outer generations
```

## Stage D — quota + subsidy

Verify restoration dynamics and reward incentive.

## Stage E — quota + subsidy + social observation

Verify observation-space growth.

## Stage F — evaluation

```text
2-3 explicit eval seeds
deterministic evaluation policy
```

Only then restore the larger benchmark configuration.

---

# 16. P2 — clarify stateful mechanism context

`QuotaMechanism` stores transient values in `_context`.

A frozen dataclass can still mutate a contained dict, but this makes the
mechanism stateful.

Decide:

- [ ] Is one mechanism object shared across multiple env instances?
- [ ] Could vectorized envs overwrite one another's `_context`?
- [ ] Should mechanism state reset per episode?
- [ ] Should step context live on the env instead?

Recommended default:

```text
Mechanism definition = immutable
Per-step mechanism context = environment-local
```

If mechanism-local context remains, tests must prove instances are not shared
across concurrently stepping envs.

---

# 17. P2 — public validation should not rely on `assert`

Examples use:

```python
assert 0.0 <= self.fixed_quota <= 1.0
```

- [ ] use explicit `ValueError` for public configuration;
- [ ] keep assertions for internal invariants only.

---

# 18. P2 — documentation acceptance

Ship:

```text
tutorials/mechanism_benchmarks_tutorial.py
tutorials/custom_benchmark_mechanism_tutorial.py
MECHANISM_ABSTRACTION_TODO.md
```

Tutorial examples must reflect the final merged public API.

---

# 19. Recommended implementation order

1. [ ] Reconcile `BilevelConfig.mechanism` public API.
2. [ ] Make all mechanism classes concretely instantiable.
3. [ ] Fix reward/observation dispatch in `MultiAgentRegulatedEnv`.
4. [ ] Fix per-agent observation concatenation.
5. [ ] Fix fishery 2-component action decomposition.
6. [ ] Connect restoration action to transition dynamics.
7. [ ] Fix subsidy indexing bug.
8. [ ] Scope/finish social influence behavior.
9. [ ] Repair `ParallelMechanism` method API.
10. [ ] Add unit tests for hooks and transforms.
11. [ ] Add composition tests.
12. [ ] Add deterministic fishery tests.
13. [ ] Run quota-only smoke benchmark.
14. [ ] Run quota + subsidy smoke benchmark.
15. [ ] Run social observation smoke benchmark.
16. [ ] Add evaluation smoke test.
17. [ ] Optional/preferred: numerical quota parity against `dev`.
18. [ ] Update tutorials to final API.
19. [ ] Run coverage and close remaining untested branches.
