# Brique 1 — Modèle écologique pur

> **Date** : 2026-05-25
> **Branche** : `rebuild/from-scratch`
> **Référence** : audit master `pre-rebuild-2026-05-25`,
> fichier `examples/bilevel_fishery/regulated_env.py` (méthode `transition_kernel`).

## Pourquoi cette brique

On extrait la **physique pure** du modèle de pêcherie : la dynamique poisson +
algue, sans aucune dépendance à Gym, Ray, agents, ou mécanisme de régulation.

Avantages :

- **Testable indépendamment** : 13 tests unitaires couvrent les équilibres,
  la positivité, la conservation et la parité entre intégrateurs.
- **Réutilisable** : on peut brancher Gym dessus (brique 2), un solveur
  alternatif (Schaefer), ou un benchmark Numba/JAX, sans toucher à la
  physique.
- **Lisible** : la dynamique tient en 4 lignes (vs 366 lignes dans master).

## Reverse-prompts et corrections d'audit

| # | Prompt source | Correction |
|---|---|---|
| B1.1 | Lotka-Volterra prédateur-proie | **Cite** Lotka (1925), Volterra (1926), Clark (1990) |
| B1.2 | Params dans `ecology_cfg: dict` | **Pydantic** `EcologyParams` frozen + validation |
| B1.3 | Bruit log-normal au reset | **`noise_std` paramétrable** (vs magic number 0.05) |
| B1.4 | Euler explicite | **RK45 par défaut** + Euler en option pédagogique |
| B1.5 | Clamp à `[0, max]` | **Suppression du clamp** + `EcologyInstabilityError` si dérive |
| B1.6 | Retour `dict[str, float]` | **`EcologicalState`** dataclass frozen+slots |
| B1.7 (ajout) | Séparer la physique du Gym | Module `ecology/dynamics.py` autonome |
| B1.8 (ajout) | Tests numériques de propriété | 13 tests dans `tests/ecology/` |
| B1.9 (ajout) | Citer les références | `ecology/references.md` + docstrings |

## Décisions de design (validées avec Rémy)

| Décision | Choix |
|---|---|
| Solveur ODE | RK45 par défaut + Euler en option |
| API harvest | `harvest: float` passé à `step()` |
| Représentation état | `@dataclass(frozen=True, slots=True)` |
| Stabilité | RK45 garantit positivité, sinon `EcologyInstabilityError` |

## Concepts introduits

- **Système d'ODE prédateur-proie** : F (fish) et A (algae) couplés
- **Intégration numérique** : Euler explicite vs RK45 adaptatif
- **Équilibre non-trivial** : `(F*, A*) = (α/β, γ/δ) = (10, 20)` aux défauts
- **Fonction pure** : `step(state, params, harvest) → state`, pas d'`self`
- **Immutabilité** : `@dataclass(frozen=True, slots=True)` + Pydantic `frozen=True`
- **Validation Pydantic** : champs `gt=0`, `model_validator` croisé
- **Tests de propriété** : équilibre stationnaire, décroissance exponentielle
  sans algue, croissance exponentielle sans poisson

## Vérifications

```bash
make test        # 13 + 2 = 15 tests passent, coverage 100% sur ecology/
make lint        # ruff clean
make typecheck   # mypy strict clean
```

## Structure finale

```
src/bilevel_fishery/ecology/
├── __init__.py       # re-export API publique
├── params.py         # EcologyParams (Pydantic, frozen)
├── state.py          # EcologicalState (dataclass frozen+slots)
├── dynamics.py       # step() + integrators + EcologyInstabilityError
└── references.md     # citations + justification

tests/ecology/
├── test_params.py        # 7 tests
├── test_dynamics.py      # 9 tests
└── test_solver_parity.py # 2 tests

config/ecology_default.yaml  # paramètres par défaut documentés
notebooks/01_ecology.ipynb   # exploration interactive
```

## Ce qui ne fait PAS partie de la Brique 1

- Pas d'API Gym (vient en Brique 2 : environnement single-agent)
- Pas de pêcheurs (Brique 2)
- Pas de mécanisme de régulation (Brique 3)
- Pas d'analyse de stabilité linéaire / Jacobien (peut-être Brique 1.5)

## Prochaine brique

**Brique 2** — Environnement Gymnasium single-agent : wrapper la dynamique
écologique dans l'API standard `reset() / step() / observation_space /
action_space`. Premier vrai pêcheur (un seul), comportement scripté pour
l'instant.
