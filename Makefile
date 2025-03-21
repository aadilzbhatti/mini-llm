SHELL := /bin/bash
.ONESHELL:

bootstrap-env:
	python -m venv .venv
	source .venv/bin/activate
	./scripts/install_requirements.sh
	git submodule init
	git submodule update

setup-git-auth:
	./scripts/setup_github_auth.sh

setup-hf:
	./scripts/setup_hf.sh