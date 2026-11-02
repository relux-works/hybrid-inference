#!/bin/sh
set -eu
export PYTHONDONTWRITEBYTECODE=1
exec python3 "$(dirname "$0")/node_deploy.py" "$@"
