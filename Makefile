# Canonical task runner. Windows has no GNU make by default - use ./make.ps1
# with the same target names.
.PHONY: help setup test lint imports hooks eval corpus contact-sheet clean

UV ?= uv

help:
	@echo "setup   - create the uv environment from uv.lock"
	@echo "test    - run the test suite"
	@echo "lint    - run all pre-commit hooks over every file"
	@echo "imports - check layer boundaries with import-linter"
	@echo "hooks   - install the git pre-commit hooks"
	@echo "eval    - produce reports/eval_<git-sha>.md"
	@echo "corpus  - build the synthetic corpus (long; use WORKERS=n)"
	@echo "contact-sheet - render 100 sampled pages for human review"

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

DOCUMENTS ?= 6300
WORKERS ?= 6

corpus:
	$(UV) run python -m reckon.data.build_corpus --documents $(DOCUMENTS) --workers $(WORKERS)

contact-sheet:
	$(UV) run python -m reckon.data.contact_sheet

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
