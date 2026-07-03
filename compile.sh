#!/bin/bash
# Ensure the target directory exists before starting the execution pipeline
mkdir -p "$2"

# Execute with the JAX CPU environment variable and log to the target directory
JAX_PLATFORMS=cpu alkaid convert "$@" 2>&1 | tee "$2/compile.log"