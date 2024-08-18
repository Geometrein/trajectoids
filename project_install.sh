#!/bin/bash

# Check for requirements.txt file
if [ ! -f "requirements.txt" ]; then
  echo "requirements.txt not found. Please ensure it exists and try again."
  exit 1
fi

# Check if the venv directory exists
if [ -d ".venv" ]; then
  echo "Using existing virtual environment."
else
  # Create a Python virtual environment if it doesn't exist
  echo "Creating virtual environment and installing dependencies..."
  python3.10 -m venv .venv
  source .venv/bin/activate
  python3.10 -m pip install --upgrade pip
  pip3.10 install --no-cache-dir -r requirements.txt
  python -m ipykernel install --user --name="trajectoids"
  echo "Environment created & dependencies installed."
fi

echo "Setup completed. Virtual environment is ready to use."
