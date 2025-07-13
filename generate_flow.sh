#!/bin/bash
"""
Flow Generator Wrapper Script
"""

# Activate virtual environment
source venv/bin/activate

# Run the flow generator
python cli/flow_generator.py "$@"
