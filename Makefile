.PHONY: install lint format type test check up down logs ps migrate seed reconcile eval clean

install:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

type:
	uv run mypy packages services evaluation

test:
	uv run pytest

check: lint type test

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

migrate:
	docker compose run --rm migrate

# Ingest the bundled sample corpus through the worker container.
seed:
	docker compose run --rm -v "$(CURDIR)/sample_docs:/seed:ro" ingestion-worker \
		python -m ingestion_worker.seed /seed

# Drop vectors whose document record is gone (add ARGS=--dry-run to preview).
reconcile:
	docker compose run --rm ingestion-worker python -m ingestion_worker.reconcile $(ARGS)

# Evaluate the running stack against the bundled eval set (needs INAGECAS_API_KEYS).
eval:
	uv run python -m evaluation

clean:
	docker compose down -v
