SHELL := /bin/bash
.ONESHELL:

bootstrap-env:
	python -m venv .venv
	source .venv/bin/activate
	./scripts/install_requirements.sh
	git submodule init
	git submodule update

setup-hf:
	./scripts/setup_hf.sh

run-tensorboard:
	python scripts/run_tensorboard.py