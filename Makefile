.PHONY: help install test eval-images run-snippets dashboard verify-submission slides clean

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python3
PIP ?= $(VENV)/bin/pip
STREAMLIT ?= $(VENV)/bin/streamlit
PYTEST ?= $(VENV)/bin/pytest

help:
	@echo "Available commands:"
	@echo "  make install            - Create virtualenv and install dependencies"
	@echo "  make test               - Run the complete pytest test suite"
	@echo "  make eval-images        - Run 100-image benchmark evaluation & failure taxonomy"
	@echo "  make run-snippets       - Process demo video clips through the safety pipeline"
	@echo "  make dashboard          - Launch Streamlit Human Reviewer Dashboard"
	@echo "  make slides             - Re-build visual HTML & PDF presentation slide deck"
	@echo "  make verify-submission  - Audit zero absolute paths, DB state, outputs, and tests"
	@echo "  make clean              - Remove transient caches"

$(VENV):
	python3 -m venv $(VENV)

install: $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test:
	$(PYTEST) -v

eval-images:
	$(PYTHON) scripts/evaluate_100_images.py

run-snippets:
	$(PYTHON) scripts/run_batch_snippets.py

dashboard:
	$(STREAMLIT) run app_dashboard.py

slides:
	$(PYTHON) scripts/build_slide_deck.py

verify-submission:
	$(PYTHON) scripts/verify_submission.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
