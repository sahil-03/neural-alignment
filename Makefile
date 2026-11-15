.PHONY: lint-fix check modal-secret

# Modal commands here run in this project's environment, regardless of the profile default.
MODAL_ENV ?= neural-alignment
MODAL := MODAL_ENVIRONMENT=$(MODAL_ENV) uv run modal

lint-fix:
	uv run ruff check --fix .

check:
	uv run mypy core

# Create/replace the "r2" secret from the four R2 values in core/.env (CLOUDFLARE_API_TOKEN stays local).
modal-secret:
	@set -a; . core/.env; set +a; $(MODAL) secret create r2 --force \
		CLOUDFLARE_ACCOUNT_ID="$$CLOUDFLARE_ACCOUNT_ID" \
		CLOUDFLARE_R2_ACCESS_KEY_ID="$$CLOUDFLARE_R2_ACCESS_KEY_ID" \
		CLOUDFLARE_R2_SECRET_ACCESS_KEY="$$CLOUDFLARE_R2_SECRET_ACCESS_KEY" \
		CLOUDFLARE_S3_ENDPOINT="$$CLOUDFLARE_S3_ENDPOINT"

