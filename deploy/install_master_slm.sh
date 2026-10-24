#!/usr/bin/env bash
set -euo pipefail

stage_dir="$(dirname "$(readlink -f "$0")")"
app_root="/opt/dfl-slm-dtn"
service_name="dfl-slm-master.service"
status_service="dfl-slm-status.service"

if ! getent passwd hammad >/dev/null; then
  printf 'Required service account hammad was not found\n' >&2
  exit 1
fi
for required in \
  "$stage_dir/master/app.py" \
  "$stage_dir/master/__init__.py" \
  "$stage_dir/master/wsgi.py" \
  "$stage_dir/master/requirements.txt" \
  "$stage_dir/master/training.py" \
  "$stage_dir/master/cifar_evaluator.py" \
  "$stage_dir/common/__init__.py" \
  "$stage_dir/common/cifar_model.py" \
  "$stage_dir/common/slm_action_schema.json" \
  "$stage_dir/data/cifar10_test.npz" \
  "$stage_dir/data/cifar10_train.npz" \
  "$stage_dir/master.env" \
  "$stage_dir/$service_name"; do
  if [[ ! -f "$required" ]]; then
    printf 'Required staged file is missing: %s\n' "$required" >&2
    exit 1
  fi
done

if ! systemctl is-active --quiet "$status_service"; then
  printf 'Refusing installation because the existing status service is not active\n' >&2
  exit 1
fi
if ss -lnt "( sport = :5002 )" | grep -q LISTEN; then
  printf 'Status service is listening on TCP/5002 and will not be modified\n'
else
  printf 'Expected status service listener TCP/5002 is absent; refusing deployment\n' >&2
  exit 1
fi

grep -q '^DFL_API_TOKEN=.\{32,\}$' "$stage_dir/master.env" || {
  printf 'SLM API token is missing or too short\n' >&2
  exit 1
}
grep -q '^DFL_STATUS_API_TOKEN=.\{32,\}$' "$stage_dir/master.env" || {
  printf 'Status API token is missing or too short\n' >&2
  exit 1
}
grep -qxF 'DFL_PORT=5003' "$stage_dir/master.env" || {
  printf 'SLM service must be configured for TCP/5003\n' >&2
  exit 1
}
grep -q '^DFL_WORKERS_JSON_B64=.\{20,\}$' "$stage_dir/master.env" || {
  printf 'Worker training API configuration is missing\n' >&2
  exit 1
}
runuser -u hammad -- /usr/bin/python3 -c 'import tensorflow as tf; assert tf.__version__.startswith("2.15."); assert tf.config.list_physical_devices("GPU")' || {
  printf 'System Python must provide TensorFlow 2.15 with an enumerated GPU for evaluation/baseline\n' >&2
  exit 1
}

install -d -o root -g root -m 0755 "$app_root/master"
install -d -o root -g root -m 0755 "$app_root/common" "$app_root/data"
install -o root -g root -m 0644 "$stage_dir/master/app.py" "$app_root/master/app.py"
install -o root -g root -m 0644 "$stage_dir/master/__init__.py" "$app_root/master/__init__.py"
install -o root -g root -m 0644 "$stage_dir/master/wsgi.py" "$app_root/master/wsgi.py"
install -o root -g root -m 0644 "$stage_dir/master/requirements.txt" "$app_root/master/requirements.txt"
install -o root -g root -m 0644 "$stage_dir/master/training.py" "$app_root/master/training.py"
install -o root -g root -m 0644 "$stage_dir/master/cifar_evaluator.py" "$app_root/master/cifar_evaluator.py"
install -o root -g root -m 0644 "$stage_dir/common/__init__.py" "$app_root/common/__init__.py"
install -o root -g root -m 0644 "$stage_dir/common/cifar_model.py" "$app_root/common/cifar_model.py"
install -o root -g root -m 0644 "$stage_dir/common/slm_action_schema.json" "$app_root/common/slm_action_schema.json"
install -o hammad -g hammad -m 0600 "$stage_dir/data/cifar10_test.npz" "$app_root/data/cifar10_test.npz"
install -o hammad -g hammad -m 0600 "$stage_dir/data/cifar10_train.npz" "$app_root/data/cifar10_train.npz"
install -d -o root -g root -m 0755 "$app_root/master/tests"
install -o root -g root -m 0644 "$stage_dir/master/tests/test_app.py" "$app_root/master/tests/test_app.py"

if [[ ! -x "$app_root/master/venv/bin/python" ]]; then
  python3 -m venv "$app_root/master/venv"
fi
"$app_root/master/venv/bin/pip" install --disable-pip-version-check -r "$app_root/master/requirements.txt"

install -d -o hammad -g hammad -m 0750 /var/log/dfl-slm-master
install -d -o hammad -g hammad -m 0750 /var/lib/dfl-slm-master
install -o hammad -g hammad -m 0600 "$stage_dir/master.env" /etc/dfl-slm-master.env
install -o root -g root -m 0644 "$stage_dir/$service_name" "/etc/systemd/system/$service_name"
systemctl daemon-reload
systemctl enable --now "$service_name"
systemctl restart "$service_name"

for attempt in {1..20}; do
  if systemctl is-active --quiet "$service_name" &&
     ss -lnt "( sport = :5003 )" | grep -q LISTEN; then
    break
  fi
  sleep 1
done
if ! systemctl is-active --quiet "$service_name"; then
  systemctl --no-pager --full status "$service_name" >&2 || true
  journalctl --no-pager -u "$service_name" -n 50 >&2 || true
  exit 1
fi
if ! ss -lnt "( sport = :5003 )" | grep -q LISTEN; then
  printf 'SLM service did not bind TCP/5003\n' >&2
  journalctl --no-pager -u "$service_name" -n 50 >&2 || true
  exit 1
fi
if ! systemctl is-active --quiet "$status_service" ||
   ! ss -lnt "( sport = :5002 )" | grep -q LISTEN; then
  printf 'Existing status service health check failed after SLM installation\n' >&2
  exit 1
fi
set -a
# Load the protected local environment only into this health-check process.
. /etc/dfl-slm-master.env
set +a
"$app_root/master/venv/bin/python" -c 'import json, os, urllib.request; r=urllib.request.Request("http://127.0.0.1:5003/status", headers={"Authorization":"Bearer "+os.environ["DFL_API_TOKEN"]}); d=json.load(urllib.request.urlopen(r, timeout=8)); assert d["node_id"] == "master" and d["ollama"]["model_available"]'
printf 'Installed %s on TCP/5003; %s remains active on TCP/5002\n' "$service_name" "$status_service"
systemctl --no-pager --full status "$service_name"
