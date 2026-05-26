# D-001 — Reward function du `FisheryEnv` (single-agent)

> **Date** : 2026-05-25
> **Statut** : Accepté
> **Brique** : 2 — Environnement Gymnasium single-agent
> **Décideur** : Rémy Ramadour (avec proposition Claude)
> **Note pour Nadine** : ce choix conditionne les valeurs de récompense de
> tout pêcheur entraîné dans cette brique. Si on change la reward plus
> tard, les agents pré-entraînés ne sont **plus comparables**.

## Contexte

Le pêcheur dans `FisheryEnv` doit recevoir une **reward** à chaque pas qui
représente la valeur économique de son action. Le choix de la fonction de
reward conditionne le comportement appris par l'agent RL.

## Décision

```python
reward = log(1 + harvest_realized)
```

Reward **concave** (utility logarithmique) sur la prise effective.

## Alternatives considérées

| Option | Formule | Avantage | Inconvénient |
|---|---|---|---|
| A — Brute | `harvest_realized` | Simple, interprétable | Pas d'aversion au risque, encourage prises massives |
| B — Utility scaling (master) | `harvest_realized * fish/max_fish` | Pénalise la raréfaction | Couplé au stock dispo, moins lisible |
| **C — Concave log (retenue)** | `log(1 + harvest_realized)` | Aversion au risque, prises régulières, économiquement standard | Légèrement plus complexe |

## Raison du choix

La concavité (rendement marginal décroissant) :

- Encourage l'agent à **étaler** ses prises plutôt qu'à faire de gros coups
  sporadiques.
- Est cohérente avec la théorie économique de l'utilité (aversion au risque).
- Donne un signal d'apprentissage **stable** : ne diverge pas pour de gros
  harvests, n'écrase pas pour des petits.

## Conséquences

- L'agent RL apprendra une stratégie de **lissage** (préférence pour régularité).
- Les valeurs de reward sont en `[0, log(1 + max_harvest_rate * dt)]` ≈ `[0, 0.7]`
  pour les paramètres par défaut.
- Si on compare 2 agents avec des fonctions de reward différentes, les courbes
  d'apprentissage ne sont **pas comparables** sans normalisation.

## Implémentation

`src/bilevel_fishery/envs/fishery_env.py:step()`, ligne avec `np.log1p`.

## Référence académique

- Pratt, J. W. (1964). Risk aversion in the small and in the large.
  *Econometrica*, 32(1/2), 122-136.
- Arrow, K. J. (1965). *Aspects of the Theory of Risk Bearing*.
  Yrjö Jahnssonin Säätiö, Helsinki.

(Utility logarithmique = constant relative risk aversion, CRRA, η=1.)
