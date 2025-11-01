#!/bin/bash
# Development server script for Toronto AI Assistant

# Activate virtual environment
source .venv/bin/activate

# Run Quart app with auto-reload
export QUART_APP=app.main:app
hypercorn app.main:app --reload --bind 0.0.0.0:5000
