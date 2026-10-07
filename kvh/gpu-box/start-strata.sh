#!/bin/sh
# Start Strata on the GPU box as the big-document prefill server (both GPUs, with the local patches).
# Run after a reboot of the GPU box: ssh gpu-box ~/kvh/start-strata.sh
# Strata needs the GPUs to itself: stop anything else using them first.
# Stop it:  pkill -f '[s]erve/server.py --engine strata'
# (the bracket keeps pkill/pgrep from matching their own command line, e.g. the shell of `ssh gpu-box '...'`)
set -eu
if pgrep -f '[s]erve/server.py --engine strata' >/dev/null; then echo "Strata already running"; exit 0; fi
cd "$HOME/strata"
setsid nohup .venv/bin/python serve/server.py --engine strata --config "$HOME/kvh/strata-peer.json" --port 8080 \
  > "$HOME/kvh/server-peer.out" 2>&1 < /dev/null &
for i in $(seq 1 60); do
  curl -sf -m3 http://127.0.0.1:8080/v1/models >/dev/null && { echo "Strata ready on 127.0.0.1:8080"; exit 0; }
  sleep 5
done
echo "Strata did not come up in 5 min; see ~/kvh/server-peer.out"; exit 1
