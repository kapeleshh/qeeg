#!/bin/bash

# Build the Docker image if it doesn't exist
docker-compose build

# Create output directory if it doesn't exist
mkdir -p examples/output

# Run the comprehensive example generator inside the Docker container
docker-compose run --rm qeeg python generate_examples.py

echo "Examples generated! Check examples/output directory for visualizations."
