.PHONY: install test lint format typecheck precommit clean

install:
	uv sync --all-extras
	# `pre-commit install` is intentionally NOT called here: it refuses to run
	# when the user has a global `core.hooksPath` configured. Use `make precommit`
	# to run the hooks manually before pushing.

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy src/bilevel_fishery

precommit:
	uv run pre-commit run --all-files

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
