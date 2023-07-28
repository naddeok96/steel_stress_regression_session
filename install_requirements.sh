#!/bin/bash

# BE SURE TO CREATE VENV FIRST
# python3 -m venv your_virtual_env_name

# Check if the requirements.txt file exists
if [ -f "requirements.txt" ]; then
    # Install the packages using pip
    pip3 install -r requirements.txt
else
    echo "Error: requirements.txt file not found."
fi
