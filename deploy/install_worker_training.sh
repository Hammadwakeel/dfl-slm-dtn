#!/usr/bin/env bash
set -euo pipefail

stage_dir="$(dirname "$(readlink -f "$0")")"
app_root="/opt/dfl-slm-dtn"
service_name="dfl-slm-worker-training.service"
status_service="dfl-slm-status.service"
training_port=5004

for required in \
  "$stage_dir/worker/training_service.py" \
  "$stage_dir/worker/__init__.py" \
  "$stage_dir/common/__init__.py" \
  "$stage_dir/common/auth.py" \
  "$stage_dir/common/cifar_model.py" \
  "$stage_dir/train_shard.npz" \
  "$stage_dir/worker-training.env" \
  "$stage_dir/$service_name"; do
  if [[ ! -f "$required" ]]; then
    printf 'Required staged file is missing: %s\n' "$required" >&2
    exit 1
  fi
done

grep -qxF "DFL_TRAINING_PORT=$training_port" "$stage_dir/worker-training.env" || {
  printf 'Worker training service must be configured on TCP/%s\n' "$training_port" >&2
  exit 1
}
grep -q '^DFL_API_TOKEN=.\{32,\}$' "$stage_dir/worker-training.env" || {
  printf 'Worker training API token is missing or too short\n' >&2
  exit 1
}
grep -q '^DFL_PEERS_JSON_B64=.\{20,\}$' "$stage_dir/worker-training.env" || {
  printf 'Peer configuration is missing\n' >&2
  exit 1
}
if ! systemctl is-active --quiet "$status_service" ||
   ! ss -lnt "( sport = :5001 )" | grep -q LISTEN; then
  printf 'Existing status service is not healthy on TCP/5001; refusing install\n' >&2
  exit 1
fi

runuser -u hammad -- /usr/bin/python3 -c 'import numpy as np, tensorflow as tf; assert tf.__version__.startswith("2.15."); assert tf.config.list_physical_devices("GPU")' || {
  printf 'System Python must provide TensorFlow 2.15 with an enumerated GPU\n' >&2
  exit 1
}

install -d -o root -g root -m 0755 "$app_root/worker" "$app_root/worker/data" "$app_root/common"
install -o root -g root -m 0644 "$stage_dir/worker/training_service.py" "$app_root/worker/training_service.py"
install -o root -g root -m 0644 "$stage_dir/worker/__init__.py" "$app_root/worker/__init__.py"
install -o root -g root -m 0644 "$stage_dir/common/__init__.py" "$app_root/common/__init__.py"
install -o root -g root -m 0644 "$stage_dir/common/auth.py" "$app_root/common/auth.py"
install -o root -g root -m 0644 "$stage_dir/common/cifar_model.py" "$app_root/common/cifar_model.py"
install -o hammad -g hammad -m 0600 "$stage_dir/train_shard.npz" "$app_root/worker/data/train_shard.npz"
install -o hammad -g hammad -m 0600 "$stage_dir/worker-training.env" /etc/dfl-slm-worker-training.env
install -o root -g root -m 0644 "$stage_dir/$service_name" "/etc/systemd/system/$service_name"
install -d -o hammad -g hammad -m 0750 /var/log/dfl-slm-worker-training

systemctl daemon-reload
systemctl enable --now "$service_name"
systemctl restart "$service_name"

for attempt in {1..20}; do
  if systemctl is-active --quiet "$service_name" &&
     ss -lnt "( sport = :$training_port )" | grep -q LISTEN; then
    break
  fi
  sleep 1
done
if ! systemctl is-active --quiet "$service_name" ||
   ! ss -lnt "( sport = :$training_port )" | grep -q LISTEN; then
  systemctl --no-pager --full status "$service_name" >&2 || true
  journalctl --no-pager -u "$service_name" -n 50 >&2 || true
  exit 1
fi
if ! systemctl is-active --quiet "$status_service" ||
   ! ss -lnt "( sport = :5001 )" | grep -q LISTEN; then
  printf 'Existing status service failed after training service installation\n' >&2
  exit 1
fi

printf 'Installed %s on TCP/%s; status service remains active on TCP/5001\n' \
  "$service_name" "$training_port"
systemctl --no-pager --full status "$service_name"
