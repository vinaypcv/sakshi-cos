.PHONY: install test demo bench scenarios ablate all lint clean
PYTHONPATH := src
export PYTHONPATH

install:
	pip install -e ".[dev]" --break-system-packages

test:
	python -m pytest -q

demo:
	python -m sakshi --out runs demo

bench:
	python -m sakshi --out runs bench

scenarios:
	python -m sakshi --out runs scenarios

ablate:
	python -m sakshi --out runs ablate

all:
	python -m sakshi --out runs all

clean:
	rm -rf runs .pytest_cache **/__pycache__ src/**/__pycache__
