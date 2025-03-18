#!/bin/bash

awk "{print \$1}" requirements.txt > packages.txt
for package in $(cat packages.txt); do
    pip install --no-dependencies --ignore-installed "$package" || true
done
rm packages.txt
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128