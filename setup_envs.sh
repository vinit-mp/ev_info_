#!/bin/bash

# Ensure pip is up to date first
python3 -m pip install --upgrade pip

# Create virtual environment
python3 -m venv llama_training_env

# Activate virtual environment
source llama_training_env/bin/activate

# Fix potential SSL issues
pip install --upgrade pip setuptools wheel
pip install certifi
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Install requirements with trusted host
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt