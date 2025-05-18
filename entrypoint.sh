#!/usr/bin/env sh
set -e

# --- Start Tailscale and join the network ---
tailscaled &
sleep 5                                   # give tailscaled time to come up
tailscale up --authkey="${TAILSCALE_AUTH_KEY}" --hostname vast-container

# (Optional) print the container’s Tailscale IPv4 for quick debugging
tailscale ip -4 || true

# --- Launch the application ---
exec python3 -u /app/main.py
