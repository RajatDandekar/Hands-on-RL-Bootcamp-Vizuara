#!/usr/bin/env bash
set -euo pipefail
source "$HOME/venv/bin/activate"
if [ -f ".env" ]; then set -a; source .env; set +a; fi
export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES=1
# optionally tell server which adapter to load:
export LORA_ADAPTER_PATH="checkpoints/debug/2025-09-22/step-0010"
python service/chat_server.py
