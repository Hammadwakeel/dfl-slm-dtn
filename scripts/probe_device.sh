#!/usr/bin/env bash
set -u

printf '== host ==\n'
hostnamectl 2>&1 || uname -a
printf '\n== operating system and JetPack/L4T ==\n'
cat /etc/os-release 2>&1 || true
cat /etc/nv_tegra_release 2>&1 || printf 'JetPack/L4T marker not found\n'
printf 'Machine ID: '
cat /etc/machine-id 2>&1 || true
printf 'Boot ID: '
cat /proc/sys/kernel/random/boot_id 2>&1 || true
printf 'Hardware serial: '
if [ -r /proc/device-tree/serial-number ]; then
  tr -d '\000' < /proc/device-tree/serial-number
  printf '\n'
else
  printf 'not exposed by device tree\n'
fi

printf '\n== Python and toolchain ==\n'
python3 --version 2>&1 || python --version 2>&1 || true
if python3 -c 'import importlib.util; raise SystemExit(importlib.util.find_spec("tensorflow") is None)' 2>/dev/null; then
  python3 -c 'import tensorflow as tf; print("TensorFlow:", tf.__version__); print("TensorFlow GPUs:", tf.config.list_physical_devices("GPU"))' 2>&1 || true
else
  printf 'TensorFlow not installed\n'
fi
command -v pip3 >/dev/null && pip3 --version || true
command -v gcc >/dev/null && gcc --version | head -n 1 || true

printf '\n== Ollama ==\n'
command -v ollama >/dev/null 2>&1 && ollama --version || printf 'Ollama not installed\n'

printf '\n== memory and disk ==\n'
free -h 2>&1 || true
df -h / 2>&1 || true

printf '\n== CUDA and TensorRT ==\n'
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
elif [ -x /usr/local/cuda/bin/nvcc ]; then
  /usr/local/cuda/bin/nvcc --version
else
  printf 'nvcc not found\n'
fi
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || printf 'nvidia-smi not found\n'
if command -v dpkg-query >/dev/null 2>&1; then
  dpkg-query -W -f='${Package} ${Version} ${Architecture}\n' 'cuda-toolkit-*' 'libcudnn*' 'libnvinfer*' 2>/dev/null || true
fi

printf '\n== Jetson utilities and live readings ==\n'
command -v jtop >/dev/null 2>&1 && jtop --help 2>&1 | head -n 10 || printf 'jtop not installed\n'
if command -v tegrastats >/dev/null 2>&1; then
  timeout 3s tegrastats --interval 1000
  status=$?
  if [ "$status" -ne 0 ] && [ "$status" -ne 124 ]; then
    printf 'tegrastats exited with status %s\n' "$status"
  fi
else
  printf 'tegrastats not found\n'
fi

printf '\n== network interfaces and reachability ==\n'
ip -br address 2>&1 || true
ip -br link 2>&1 || true
for host in 192.168.11.12 192.168.11.15 192.168.11.123 192.168.11.14 192.168.11.13; do
  ping -c 1 -W 2 "$host" >/dev/null 2>&1 && printf 'PING_OK %s\n' "$host" || printf 'PING_FAIL %s\n' "$host"
done

printf '\n== SSH directory ==\n'
ls -la "$HOME/.ssh" 2>&1 || true
printf '\n== done ==\n'
