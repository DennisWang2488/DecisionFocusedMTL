.PHONY: install install-dev test test-fast test-grads lint typecheck quick full plot clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in editable mode
	pip install -e .

install-dev:  ## Install with dev dependencies
	pip install -e ".[dev]"

test:  ## Run all tests
	pytest tests/ -v

test-fast:  ## Run tests excluding slow ones
	pytest tests/ -v -m "not slow"

test-grads:  ## Run gradient correctness tests only
	pytest tests/test_losses.py tests/test_medical_gradients.py -v

lint:  ## Run ruff linter
	ruff check src/ tests/

typecheck:  ## Run mypy type checker
	mypy src/fair_dfl/

quick:  ## Small-sample smoke test (fast)
	python run_methods.py --methods FPLG FPTO --n-sample 500 --steps 20

full:  ## Full experiment run
	python run_methods.py --all --n-sample 0

clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info src/*.egg-info __pycache__ .pytest_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
