#!/usr/bin/env bash
set -euo pipefail

stage_dir="$(dirname "$(readlink -f "$0")")"
app_root="/opt/dfl-slm-dtn"
service_name="dfl-slm-memory-cleanup.service"

for required in \
  "$stage_dir/memory_cleanup_helper.py" \
  "$stage_dir/memory-cleanup.env" \
  "$stage_dir/$service_name"; do
  if [[ ! -f "$required" ]]; then
    printf 'Required staged file is missing: %s\n' "$required" >&2
    exit 1
  fi
done

token="$(sed -n 's/^DFL_MEMORY_CLEANUP_TOKEN=//p' "$stage_dir/memory-cleanup.env")"
if [[ ${#token} -lt 32 ]]; then
  printf 'Memory cleanup token is missing or too short\n' >&2
  exit 1
fi

install -d -o root -g root -m 0755 "$app_root/worker"
install -o root -g root -m 0644 \
  "$stage_dir/memory_cleanup_helper.py" \
  "$app_root/worker/memory_cleanup_helper.py"
install -o root -g root -m 0600 \
  "$stage_dir/memory-cleanup.env" \
  /etc/dfl-slm-memory-cleanup.env
install -o root -g root -m 0644 \
  "$stage_dir/$service_name" \
  "/etc/systemd/system/$service_name"

systemctl daemon-reload
systemctl enable --now "$service_name"
systemctl restart "$service_name"

for attempt in {1..10}; do
  if systemctl is-active --quiet "$service_name" &&
     ss -lnt "( sport = :5005 )" | grep -q LISTEN; then
    break
  fi
  sleep 1
done
if ! systemctl is-active --quiet "$service_name" ||
   ! ss -lnt "( sport = :5005 )" | grep -q LISTEN; then
  systemctl --no-pager --full status "$service_name" >&2 || true
  journalctl --no-pager -u "$service_name" -n 50 >&2 || true
  exit 1
fi

if ! /usr/bin/python3 -c '
import json
from urllib.request import Request, urlopen

with open("/etc/dfl-slm-memory-cleanup.env", encoding="ascii") as environment:
    token = next(
        line.rstrip("\r\n").partition("=")[2]
        for line in environment
        if line.startswith("DFL_MEMORY_CLEANUP_TOKEN=")
    )
request = Request(
    "http://127.0.0.1:5005/healthz",
    headers={"Authorization": "Bearer " + token},
)
with urlopen(request, timeout=3) as response:
    if response.status != 200 or json.load(response).get("status") != "ok":
        raise RuntimeError("Authenticated memory cleanup health check failed")
'; then
  printf 'Memory cleanup helper failed its authenticated health check\n' >&2
  journalctl --no-pager -u "$service_name" -n 50 >&2 || true
  exit 1
fi

printf 'Installed and authenticated %s on loopback TCP/5005\n' "$service_name"
