#!/usr/bin/env bash
# Usage:
#   ./run.sh                                           # config par défaut
#   ./run.sh examples/bilevel_fishery/config_mps.yaml  # autre config
#   ./run.sh examples/bilevel_fishery/main_appo_one_mechanism_v1.yaml
#
# Note: on n'utilise pas `uv run` pour éviter un conflit entre les variables
# d'environnement injectées par uv et l'initialisation interne de Ray.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV" ]; then
    echo "Environnement virtuel introuvable. Lance d'abord : uv sync"
    exit 1
fi

source "$VENV/bin/activate"

CONFIG="${1:-examples/bilevel_fishery/config.yaml}"
shift 2>/dev/null || true

cd "$SCRIPT_DIR"
python -m examples.bilevel_fishery.main --config "$CONFIG" "$@"
