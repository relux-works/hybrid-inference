#!/bin/sh
set -eu
command -v python3 >/dev/null 2>&1 || { echo 'Python 3.10+ required; install: brew install python@3.12 (macOS), sudo apt-get install python3 (Linux)' >&2; exit 2; }
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 2)' || { echo 'Python 3.10+ required; install a supported Python 3 version' >&2; exit 2; }
exec python3 "$(dirname "$0")/manage.py" client "$@"
