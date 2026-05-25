# Brique 0 — Squelette du projet

> **Date** : 2026-05-25
> **Branche** : `rebuild/from-scratch`
> **Archive de référence** : tag `pre-rebuild-2026-05-25`

## Pourquoi cette brique

Avant d'écrire la moindre ligne de code métier, on pose un **squelette d'outillage
strict** qui rendra tout le reste plus simple :

- Un seul fichier de config Python (`pyproject.toml`) — pas de `ruff.toml` ni
  `pytest.ini` séparés.
- Un layout `src/` standard, qui force le test contre le package installé
  (pas contre les fichiers du repo).
- Du linting **strict** dès le départ (ruff avec `E F I B UP RUF SIM N D`),
  parce qu'il est plus facile de garder une base propre que de la nettoyer
  plus tard.
- Des notebooks **versionnés**, mais avec `nbstripout` en pre-commit pour
  effacer les outputs automatiquement.
- Une CI minimale qui tourne ruff et pytest à chaque push.

## Reverse-prompts utilisés (et corrigés)

| # | Prompt source | Correction d'audit |
|---|---|---|
| B0.1 | Init projet Python 3.12 + BSD-3 + auteur | (aucune) |
| B0.2 | Deps runtime | **Retiré** 6 deps LLM mortes + retiré ruff/pytest du runtime |
| B0.3 | Deps dev | **Unifié** sur ruff (vs black+isort+flake8) ; ajouté nbstripout + pre-commit |
| B0.4 | Setuptools packages | **Layout `src/` explicite** au lieu de flat |
| B0.5 | `__init__.py` | **Déplacé** dans `src/bilevel_fishery/` |
| B0.6 | Ruff config | **Sélection durcie** : `E F I B UP RUF SIM N D` + convention NumPy |
| B0.7 | Pytest config | **Migré** dans `pyproject.toml`, `--cov=src/bilevel_fishery` |
| B0.8 | Gitignore | **`*.ipynb` GARDÉ versionné**, nettoyé via nbstripout |
| B0.9 | README | **Court et factuel** (~50 lignes) au lieu de 145 lignes hallucinées |

## Vérifications

```bash
make install   # uv sync + pre-commit install
make test      # pytest passe (2 tests)
make lint      # ruff check + format check passe
make typecheck # mypy strict passe
```

## Ce qui ne fait PAS partie de la Brique 0

- Pas de code métier (pêcheurs, poissons, mécanisme)
- Pas de dépendances scientifiques (numpy, scipy, torch, Ray) — ajoutées brique par brique
- Pas de config YAML d'expérience — `config/` est vide pour l'instant

## Prochaine brique

**Brique 1** — Modèle écologique pur : la dynamique poissons/algues (Lotka-Volterra)
sans RL, sans Ray. Du Python + numpy + scipy.
