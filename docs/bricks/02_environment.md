# Brique 2 — Environnement Gymnasium single-agent

> **Date** : 2026-05-25
> **Branche** : `rebuild/from-scratch`
> **Référence** : audit master `pre-rebuild-2026-05-25` —
> `core/envs/base.py`, `core/envs/regulated.py`,
> `examples/cartpole/regulated_env.py`.

## Pourquoi cette brique

On wrappe `ecology.step()` (Brique 1) dans une interface **Gymnasium standard**
single-agent : `reset()` / `step()` / `observation_space` / `action_space`.

Le résultat : un pêcheur simple qui peut pêcher dans un lac, avec :

- Aucune dépendance à Ray, World, mechanism ou multi-agent
- Une API conforme à Gymnasium 1.x
- Un cap physique sur la pêche (on ne peut pas pêcher ce qui n'existe pas)
- Une reward concave (`log1p`) — décision tracée en D-001

C'est l'unité de base sur laquelle les briques suivantes greffent :
- Brique 3 : un mécanisme de régulation (quota, amende)
- Brique 4 : plusieurs pêcheurs (multi-agent)
- Brique 5 : un `World` partagé pour les optimizers bilevel

## Reverse-prompts et corrections d'audit

| # | Prompt source | Correction |
|---|---|---|
| B2.1 | `BaseEnv` couplé au `World` Ray actor | **Suppression** du World — env autonome |
| B2.2 | `RegulatedEnv` fetch mechanism via Ray | **Suppression** — pas de mechanism en Brique 2 |
| B2.3 | `MultiAgentRegulatedEnv` via RLlib | **Suppression** — single-agent pur |
| B2.4 | `CartpoleRegulatedEnv` "single-agent" héritant du multi-agent | **Reproduit** comme `gymnasium.Env` direct, sans héritage exotique |

## Décisions design

| # | Décision | Choix |
|---|---|---|
| 1 | Action space | `Box(0, 1, shape=(1,), float32)` — intensité normalisée |
| 2 | Observation space | `Box(0, 1, shape=(2,), float32)` — `(fish/max_fish, algae/max_algae)` |
| 3 | Reward function | `log(1 + harvest_realized)` — concave (voir [D-001](../decisions/D-001-reward-function.md)) |
| 4 | Cap physique | `harvest_realized = min(action·max_rate,  0.99·fish/dt)` |
| 5 | Horizon | 200 steps par défaut, configurable |
| 6 | Terminated | Toujours `False` (le stock peut s'effondrer sans fin d'épisode) |
| 7 | Truncated | À `horizon` |
| 8 | Seeding | Standard `gymnasium` + `reset_state` de Brique 1 |

## Decision log — NOTE NADINE

[D-001 Reward function](../decisions/D-001-reward-function.md) — choix de
`log1p(harvest)` comme reward pour le pêcheur single-agent. Si on change la
reward plus tard, les agents pré-entraînés ne seront plus comparables.

## Concepts introduits

- **API Gymnasium 1.x** : `reset(seed) -> (obs, info)`, `step(action) ->
  (obs, reward, terminated, truncated, info)` (5-tuple, pas 4 comme dans
  l'ancien `gym`)
- **`Box` spaces** continus, normalisés `[0, 1]`
- **Cap physique** vs **clamp numérique** (sémantique différente)
- **Reward concave** (CRRA avec η=1, utility logarithmique)
- **Environnement autonome** sans état partagé externe

## Structure ajoutée

```
src/bilevel_fishery/envs/
├── __init__.py
└── fishery_env.py          FisheryEnv (gymnasium.Env)

tests/envs/
├── __init__.py
└── test_fishery_env.py     11 tests d'API + comportement

config/env_default.yaml     Paramètres par défaut
docs/decisions/D-001-...    ADR sur le choix de reward
notebooks/02_environment.ipynb  Exploration : random policy, harvest constant,
                                effet du max_harvest_rate
```

## Vérifications

```bash
make test           # 31 tests = 20 (Brique 1) + 11 (Brique 2), ≥ 92% coverage
make lint           # ruff strict clean
make typecheck      # mypy strict clean
make notebook-test  # exécute 00 + 01 + 02 end-to-end
```

## Ce qui ne fait PAS partie de la Brique 2

- Pas de mécanisme de régulation (quota, amende, ban) — Brique 3
- Pas de multi-agent — Brique 4
- Pas de `World` ni de Ray — Brique 5
- Pas d'apprentissage RL (PPO) — Brique 6

## Prochaine brique

**Brique 3** — Mechanism design : introduire la régulation (quota fixe, quota
proportionnel au stock, amende, seuil minimal). Le `FisheryEnv` sera étendu
pour intégrer ces contraintes dans le calcul de la reward (utility nette =
harvest - fine·violation).
