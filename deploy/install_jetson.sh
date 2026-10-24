#!/usr/bin/env bash
set -euo pipefail

node_id="${1:?node ID is required}"
role="${2:?node role is required}"
port="${3:?service port is required}"
stage_dir="$(dirname "$(readlink -f "$0")")"
app_dir="/opt/dfl-slm-dtn"
service_name="dfl-slm-status.service"

if [[ ! "$node_id" =~ ^[a-z0-9][a-z0-9_-]{0,31}$ ]]; then
  printf 'Invalid node ID: %s\n' "$node_id" >&2
  exit 2
fi
if [[ "$role" != "master" && "$role" != "worker" ]]; then
  printf 'Invalid node role: %s\n' "$role" >&2
  exit 2
fi
if [[ ! "$port" =~ ^[0-9]{1,5}$ ]] || (( port < 1 || port > 65535 )); then
  printf 'Invalid service port: %s\n' "$port" >&2
  exit 2
fi
if ! getent passwd hammad >/dev/null; then
  printf 'Required service account hammad was not found\n' >&2
  exit 1
fi
if [[ ! -f "$stage_dir/node.env" || ! -d "$stage_dir/common" || ! -f "$stage_dir/dfl-slm-status.service" ]]; then
  printf 'Staging directory is incomplete: %s\n' "$stage_dir" >&2
  exit 1
fi

grep -qxF "DFL_NODE_ID=$node_id" "$stage_dir/node.env" || {
  printf 'Staged node ID does not match install target\n' >&2
  exit 1
}
grep -qxF "DFL_NODE_ROLE=$role" "$stage_dir/node.env" || {
  printf 'Staged role does not match install target\n' >&2
  exit 1
}
grep -qxF "DFL_PORT=$port" "$stage_dir/node.env" || {
  printf 'Staged port does not match install target\n' >&2
  exit 1
}
token_line="$(grep '^DFL_API_TOKEN=' "$stage_dir/node.env" || true)"
if [[ "${#token_line}" -lt 46 ]]; then
  printf 'Staged API token is missing or too short\n' >&2
  exit 1
fi

install -d -o root -g root -m 0755 "$app_dir"
install -d -o root -g root -m 0755 "$app_dir/common"
cp -a "$stage_dir/common/." "$app_dir/common/"
find "$app_dir/common" -type d -name __pycache__ -prune -exec rm -rf -- {} +
chown -R root:root "$app_dir"
find "$app_dir" -type d -exec chmod 0755 {} +
find "$app_dir" -type f -exec chmod 0644 {} +
install -o hammad -g hammad -m 0600 "$stage_dir/node.env" /etc/dfl-slm-node.env
install -o root -g root -m 0644 "$stage_dir/$service_name" "/etc/systemd/system/$service_name"
systemctl daemon-reload
systemctl enable --now "$service_name"
systemctl restart "$service_name"

for attempt in {1..15}; do
  if systemctl is-active --quiet "$service_name"; then
    break
  fi
  sleep 1
done
if ! systemctl is-active --quiet "$service_name"; then
  systemctl --no-pager --full status "$service_name" >&2 || true
  journalctl --no-pager -u "$service_name" -n 50 >&2 || true
  exit 1
fi

printf 'Installed %s for %s on port %s\n' "$service_name" "$node_id" "$port"
systemctl --no-pager --full status "$service_name"
ss -lnt "( sport = :$port )"
