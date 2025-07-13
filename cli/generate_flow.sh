#!/bin/bash
#
# Flow Generator Wrapper Script
#
# Usage:
#     # From project root:
#     cli/generate_flow.sh [flow_name]
#     
#     # From CLI directory:
#     cd cli && ./generate_flow.sh [flow_name]
#

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root directory
cd "$PROJECT_ROOT"

# Activate virtual environment
source venv/bin/activate

# Run the flow generator
python cli/flow_generator.py "$@"
