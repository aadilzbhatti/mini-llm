#!/bin/bash

sudo apt install python3.13-venv
python3.13 -m venv .venv
source .venv/bin/activate
pip install invoke