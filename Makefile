SHELL := /bin/bash
.ONESHELL:

bootstrap-env:
	python -m venv .venv
	source .venv/bin/activate
	./scripts/install_requirements.sh

setup-git-auth:
	./scripts/setup_github_auth.sh
