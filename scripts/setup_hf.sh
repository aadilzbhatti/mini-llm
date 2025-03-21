#!/bin/bash

pip install transformers huggingface_hub
git config --global credential.helper store
huggingface-cli login
git submodule init
git submodule update --remote