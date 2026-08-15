.PHONY: install demo test lint serve docker-build docker-run

install:
	pip install -e ".[dev]"

demo:
	python scripts/demo.py

test:
	pytest -v --cov=creditrisk --cov-report=term-missing

lint:
	ruff check src tests

serve:
	uvicorn creditrisk.api.main:app --reload --port 8002

docker-build:
	docker build -t creditrisk:local .

docker-run:
	docker compose up --build
