SHELL := /bin/bash
.ONESHELL:

bootstrap-env:
	python -m venv .venv
	source .venv/bin/activate
	pip install -r requirements.txt
	pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
