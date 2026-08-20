.PHONY: lint typecheck test smoke selector-ablations updater-ablations schemas

lint:
	python -m ruff check src tests

typecheck:
	python -m mypy src

test:
	python -m pytest

smoke:
	bash scripts/run_smoke.sh

selector-ablations:
	bash scripts/run_ablations.sh selector full

updater-ablations:
	bash scripts/run_ablations.sh updater full

schemas:
	python -m activemap export-schemas schemas
