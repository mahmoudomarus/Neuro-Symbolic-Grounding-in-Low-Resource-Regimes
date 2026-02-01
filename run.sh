#!/usr/bin/env bash
# Run the agent with the project venv (not global Python).
set -e
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  echo "Creating venv..."
  python3 -m venv .venv
fi

echo "Installing dependencies in venv (first run may take a few minutes)..."
.venv/bin/pip install -q -r requirements.txt

echo "Running main.py with venv Python..."
.venv/bin/python src/main.py "$@"
