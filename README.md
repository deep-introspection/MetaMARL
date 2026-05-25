# bilevel-fishery

Framework de recherche pour l'**optimisation bilevel** appliquée à la gestion
durable des pêcheries.

- **Couche externe** : Evolution Strategies optimise les paramètres d'un
  mécanisme de régulation (quota, amende, seuil minimal de stock).
- **Couche interne** : multi-agent RL (PPO via Ray RLlib) entraîne les
  comportements des pêcheurs sous le mécanisme imposé.

> **Statut** : reconstruction pédagogique en cours sur la branche
> `rebuild/from-scratch`. Le code historique est archivé sous le tag
> `pre-rebuild-2026-05-25` (voir `docs/bricks/` pour la progression).

## Installation

Requiert Python 3.12 et [`uv`](https://docs.astral.sh/uv/).

```bash
make install
```

## Tests & qualité

```bash
make test       # pytest + coverage
make lint       # ruff check + format check
make format     # auto-fix
make typecheck  # mypy strict
make clean      # remove caches
```

## Structure

```
src/bilevel_fishery/   # code source (package)
tests/                 # pytest tests
docs/bricks/           # documentation pédagogique brique par brique
notebooks/             # notebooks pédagogiques pendants des briques
config/                # YAML d'expérimentation (alimenté au fil de l'eau)
```

## License

BSD-3-Clause — voir `LICENSE`.
