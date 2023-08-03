#!/bin/bash

# Activate the virtual environment
source ../envs/steelyDANN/bin/activate

# Loop through all .py files in the directory and execute them
for file in aux_methods/*.py; do
    if [ -f "$file" ]; then
        echo "$file"
        python "$file"
    fi
done