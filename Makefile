# Canonical task runner. Windows has no GNU make by default - use ./make.ps1
# with the same target names.
.PHONY: help setup test lint imports hooks eval clean

UV ?= uv

help:
	@echo "setup   - create the uv environment from uv.lock"
	@echo "test    - run the test suite"
	@echo "lint    - run all pre-commit hooks over every file"
	@echo "imports - check layer boundaries with import-linter"
	@echo "hooks   - install the git pre-commit hooks"
	@echo "eval    - produce reports/eval_<git-sha>.md (phase 2)"

setup:
	$(UV) sync --extra dev

test:
	$(UV) run pytest

imports:
	$(UV) run lint-imports

lint:
	$(UV) run pre-commit run --all-files

hooks:
	$(UV) run pre-commit install

eval:
	$(UV) run python -m reckon.eval.run

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
