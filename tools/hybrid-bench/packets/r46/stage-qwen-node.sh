#!/bin/bash
set -eu
exec bash "$(dirname "$0")/stage-node.sh" qwen "$@"
