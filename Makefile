.PHONY: lint-fix check

lint-fix:
	uv run ruff check --fix .

check:
	uv run mypy core
