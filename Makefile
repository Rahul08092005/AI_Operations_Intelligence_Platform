.PHONY: verify-sources inspect-data data-raw validate-raw db-init test lint format clean help

PYTHON ?= python

help:
	@echo "AI Operations Intelligence Platform - Available Targets:"
	@echo "  verify-sources : Verify downloaded raw source archives"
	@echo "  inspect-data   : Profile raw datasets and compute operating statistics"
	@echo "  data-raw       : Generate all unified raw datasets (prepare_data.py)"
	@echo "  validate-raw   : Run comprehensive raw data validation suite"
	@echo "  db-init        : Initialize database schema (PostgreSQL / SQLite)"
	@echo "  test           : Run pytest test suite"
	@echo "  lint           : Check code quality with ruff"
	@echo "  format         : Format code with ruff"
	@echo "  clean          : Remove temporary cache files"

verify-sources:
	$(PYTHON) scripts/verify_sources.py

inspect-data:
	$(PYTHON) scripts/inspect_raw_data.py

data-raw:
	$(PYTHON) scripts/prepare_data.py

validate-raw:
	$(PYTHON) scripts/validate_raw_outputs.py

db-init:
	$(PYTHON) scripts/init_db.py

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
