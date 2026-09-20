#!/usr/bin/env bash
#
# run.sh - Launch the Ideafindr web UI
#
# Usage:
#   ./run.sh                    # Open the web UI (http://127.0.0.1:8000)
#   ./run.sh web --port 8080    # Web UI on another port
#   ./run.sh research "topic"   # Run the research command
#   ./run.sh doctor             # Check dependencies
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -d ".venv" ]]; then
    echo "Virtual environment not found. Run setup first:"
    echo "  uv sync --extra dev"
    exit 1
fi

source .venv/bin/activate

if [[ $# -eq 0 ]]; then
    ideafindr web
else
    ideafindr "$@"
fi
