.PHONY: install test lint format typecheck precommit notebook-test clean

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

notebook-test:
	@echo "Executing every notebook in notebooks/ end-to-end..."
	@for nb in notebooks/*.ipynb; do \
		echo "→ $$nb"; \
		uv run jupyter nbconvert --to notebook --execute --inplace "$$nb" || exit 1; \
	done
	@echo "Stripping outputs (so the repo stays clean)..."
	@uv run nbstripout notebooks/*.ipynb
	@echo "All notebooks executed successfully."

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
