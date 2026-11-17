#!/usr/bin/env bash
set -euo pipefail

stage_dir="$(dirname "$(readlink -f "$0")")"
service_user="${DFL_SERVICE_USER:-hammad}"
app_root="/opt/dfl-slm-dtn"
service_name="dfl-slm-worker-training.service"
service_dropin_dir="/etc/systemd/system/${service_name}.d"
dataset_root="/var/lib/dfl-slm-dtn/datasets"
model_cache="/var/lib/dfl-slm-dtn/huggingface"
training_port=5004

status_api_healthy() {
  /usr/bin/python3 -c '
import json
from urllib.request import urlopen

with urlopen("http://127.0.0.1:5001/healthz", timeout=3) as response:
    if response.status != 200 or json.load(response).get("status") != "ok":
        raise RuntimeError("Worker status API health check failed")
'
}

for required in \
  "$stage_dir/worker/training_service.py" \
  "$stage_dir/worker/memory_cleanup_helper.py" \
  "$stage_dir/worker/model_factory.py" \
  "$stage_dir/worker/transformer-requirements.txt" \
  "$stage_dir/worker/legacy-keras-requirements.txt" \
  "$stage_dir/worker/nas_search.py" \
  "$stage_dir/worker/aggregation.py" \
  "$stage_dir/worker/__init__.py" \
  "$stage_dir/common/__init__.py" \
  "$stage_dir/common/auth.py" \
  "$stage_dir/common/cifar_model.py" \
  "$stage_dir/common/dataset_nas.py" \
  "$stage_dir/common/nas.py" \
  "$stage_dir/common/model_design.py" \
  "$stage_dir/train_shard.npz" \
  "$stage_dir/shards/dirichlet_a03.npz" \
  "$stage_dir/shards/dirichlet_a05.npz" \
  "$stage_dir/worker-training.env" \
  "$stage_dir/$service_name"; do
  if [[ ! -f "$required" ]]; then
    printf 'Required staged file is missing: %s\n' "$required" >&2
    exit 1
  fi
done
memory_cleanup_service="dfl-slm-memory-cleanup.service"
if [[ ! -f "$stage_dir/$memory_cleanup_service" ]]; then
  printf 'Required staged file is missing: %s\n' "$stage_dir/$memory_cleanup_service" >&2
  exit 1
fi

if [[ ! "$service_user" =~ ^[a-z_][a-z0-9_-]*\$?$ ]] || ! getent passwd "$service_user" >/dev/null; then
  printf 'Required service account was not found or is invalid: %s\n' "$service_user" >&2
  exit 1
fi

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
grep -q '^DFL_TRAINING_SHARDS_JSON_B64=.\{20,\}$' "$stage_dir/worker-training.env" || {
  printf 'Dirichlet training shard configuration is missing\n' >&2
  exit 1
}
grep -qxF "HF_HOME=$model_cache" "$stage_dir/worker-training.env" || {
  printf 'Hugging Face model cache must use the writable worker cache directory\n' >&2
  exit 1
}
if ! ss -lnt "( sport = :5001 )" | grep -q LISTEN ||
   ! status_api_healthy; then
  printf 'Existing status service is not healthy on TCP/5001; refusing install\n' >&2
  exit 1
fi

runuser -u "$service_user" -- /usr/bin/python3 -c 'import numpy as np, tensorflow as tf; assert tf.__version__.startswith("2.15."); assert tf.config.list_physical_devices("GPU"); tf.keras.utils.set_random_seed(0); model = tf.keras.Sequential([tf.keras.layers.Input(shape=(1,)), tf.keras.layers.Dense(1)]); assert model.count_params() == 2' || {
  printf 'System Python must provide TensorFlow 2.15 with an enumerated GPU\n' >&2
  exit 1
}
install -d -o root -g root -m 0755 "$app_root/worker" "$app_root/worker/data" "$app_root/worker/data/shards" "$app_root/common"
install -d -o root -g root -m 0755 /var/lib/dfl-slm-dtn
install -d -o "$service_user" -g "$service_user" -m 0750 "$dataset_root" "$model_cache"
install -d -o root -g root -m 0755 "$service_dropin_dir"
printf '[Service]\nReadWritePaths=%s %s\n' "$dataset_root" "$model_cache" \
  | install -o root -g root -m 0644 /dev/stdin "$service_dropin_dir/10-datasets.conf"
install -o root -g root -m 0644 "$stage_dir/worker/training_service.py" "$app_root/worker/training_service.py"
install -o root -g root -m 0644 "$stage_dir/worker/memory_cleanup_helper.py" "$app_root/worker/memory_cleanup_helper.py"
install -o root -g root -m 0644 "$stage_dir/worker/model_factory.py" "$app_root/worker/model_factory.py"
install -o root -g root -m 0644 "$stage_dir/worker/transformer-requirements.txt" "$app_root/worker/transformer-requirements.txt"
install -o root -g root -m 0644 "$stage_dir/worker/legacy-keras-requirements.txt" "$app_root/worker/legacy-keras-requirements.txt"
install -o root -g root -m 0644 "$stage_dir/worker/nas_search.py" "$app_root/worker/nas_search.py"
install -o root -g root -m 0644 "$stage_dir/worker/aggregation.py" "$app_root/worker/aggregation.py"
install -o root -g root -m 0644 "$stage_dir/worker/__init__.py" "$app_root/worker/__init__.py"
install -o root -g root -m 0644 "$stage_dir/common/__init__.py" "$app_root/common/__init__.py"
install -o root -g root -m 0644 "$stage_dir/common/auth.py" "$app_root/common/auth.py"
install -o root -g root -m 0644 "$stage_dir/common/cifar_model.py" "$app_root/common/cifar_model.py"
install -o root -g root -m 0644 "$stage_dir/common/dataset_nas.py" "$app_root/common/dataset_nas.py"
install -o root -g root -m 0644 "$stage_dir/common/nas.py" "$app_root/common/nas.py"
install -o root -g root -m 0644 "$stage_dir/common/model_design.py" "$app_root/common/model_design.py"
install -o "$service_user" -g "$service_user" -m 0600 "$stage_dir/train_shard.npz" "$app_root/worker/data/train_shard.npz"
install -o "$service_user" -g "$service_user" -m 0600 "$stage_dir/shards/dirichlet_a03.npz" "$app_root/worker/data/shards/dirichlet_a03.npz"
install -o "$service_user" -g "$service_user" -m 0600 "$stage_dir/shards/dirichlet_a05.npz" "$app_root/worker/data/shards/dirichlet_a05.npz"
install -o "$service_user" -g "$service_user" -m 0600 "$stage_dir/worker-training.env" /etc/dfl-slm-worker-training.env
memory_cleanup_token="$(sed -n 's/^DFL_API_TOKEN=//p' "$stage_dir/worker-training.env")"
if [[ ${#memory_cleanup_token} -lt 32 ]]; then
  printf 'Worker training API token is missing or too short\n' >&2
  exit 1
fi
printf 'DFL_MEMORY_CLEANUP_TOKEN=%s\n' "$memory_cleanup_token" \
  | install -o root -g root -m 0600 /dev/stdin /etc/dfl-slm-memory-cleanup.env
install -o root -g root -m 0644 "$stage_dir/$service_name" "/etc/systemd/system/$service_name"
install -o root -g root -m 0644 "$stage_dir/$memory_cleanup_service" "/etc/systemd/system/$memory_cleanup_service"
install -d -o "$service_user" -g "$service_user" -m 0750 /var/log/dfl-slm-worker-training
install -d -o "$service_user" -g "$service_user" -m 0750 "$app_root/worker/venv"
install -d -o root -g root -m 0755 "/etc/systemd/system/${service_name}.d"
printf '[Service]\nUser=%s\nGroup=%s\n' "$service_user" "$service_user" \
  | install -o root -g root -m 0644 /dev/stdin "$service_dropin_dir/20-service-user.conf"
if ! /usr/bin/python3 -c "import ensurepip" >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv >/dev/null
fi
if [[ -x "$app_root/worker/venv/bin/python" && ! -x "$app_root/worker/venv/bin/pip" ]]; then
  rm -rf "$app_root/worker/venv"
  install -d -o "$service_user" -g "$service_user" -m 0750 "$app_root/worker/venv"
fi
if [[ ! -x "$app_root/worker/venv/bin/python" ]]; then
  runuser -u "$service_user" -- /usr/bin/python3 -m venv \
    --system-site-packages \
    "$app_root/worker/venv"
fi
runuser -u "$service_user" -- "$app_root/worker/venv/bin/pip" install \
  --disable-pip-version-check \
  -r "$app_root/worker/transformer-requirements.txt"
runuser -u "$service_user" -- "$app_root/worker/venv/bin/pip" install \
  --disable-pip-version-check \
  --no-deps \
  -r "$app_root/worker/legacy-keras-requirements.txt"
runuser -u "$service_user" -- "$app_root/worker/venv/bin/python" -c \
  'from transformers import TFViTModel; import tensorflow as tf; assert tf.__version__.startswith("2.15."); assert tf.config.list_physical_devices("GPU"); tf.keras.utils.set_random_seed(0); model = tf.keras.Sequential([tf.keras.layers.Input(shape=(1,)), tf.keras.layers.Dense(1)]); assert model.count_params() == 2' || {
    printf 'Worker TensorFlow and Transformers dependencies are incompatible\n' >&2
    exit 1
  }

systemctl daemon-reload
systemctl enable --now "$memory_cleanup_service"
systemctl restart "$memory_cleanup_service"
for attempt in {1..10}; do
  if systemctl is-active --quiet "$memory_cleanup_service" &&
     ss -lnt "( sport = :5005 )" | grep -q LISTEN; then
    break
  fi
  sleep 1
done
if ! systemctl is-active --quiet "$memory_cleanup_service" ||
   ! ss -lnt "( sport = :5005 )" | grep -q LISTEN; then
  systemctl --no-pager --full status "$memory_cleanup_service" >&2 || true
  journalctl --no-pager -u "$memory_cleanup_service" -n 50 >&2 || true
  exit 1
fi
if ! /usr/bin/python3 -c '
import urllib.request

with open("/etc/dfl-slm-memory-cleanup.env", encoding="ascii") as environment:
    token = next(
        line.rstrip("\r\n").partition("=")[2]
        for line in environment
        if line.startswith("DFL_MEMORY_CLEANUP_TOKEN=")
    )
request = urllib.request.Request(
    "http://127.0.0.1:5005/healthz",
    headers={"Authorization": "Bearer " + token},
)
with urllib.request.urlopen(request, timeout=3) as response:
    if response.status != 200:
        raise RuntimeError("Authenticated cache cleanup health check failed")
'; then
  printf 'Privileged cache cleanup helper failed its authenticated health check\n' >&2
  journalctl --no-pager -u "$memory_cleanup_service" -n 50 >&2 || true
  exit 1
fi
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
if ! ss -lnt "( sport = :5001 )" | grep -q LISTEN ||
   ! status_api_healthy; then
  printf 'Existing status API failed after training service installation\n' >&2
  exit 1
fi

printf 'Installed %s on TCP/%s; existing status API remains healthy on TCP/5001\n' \
  "$service_name" "$training_port"
systemctl --no-pager --full status "$service_name"
