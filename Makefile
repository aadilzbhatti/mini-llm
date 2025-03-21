SHELL := /bin/bash
.ONESHELL:

bootstrap-env:
	python -m venv .venv
	source .venv/bin/activate
	./scripts/install_requirements.sh
	pip install -e .

setup-git-auth:
	./scripts/setup_github_auth.sh

setup-hf:
	./scripts/setup_hf.sh