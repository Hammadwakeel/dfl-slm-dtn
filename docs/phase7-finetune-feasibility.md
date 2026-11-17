# B1 - Read-only report

Measured on `dfl-master` (`hammad@192.168.11.15`) over SSH on 2026-10-27. The inspection did not install packages or change service state. Both `ollama.service` and `dfl-slm-master.service` were active and running at the end of the checks.

## 1. Memory, swap, disk, and service memory

`free -h`:

| Resource | Total | Used | Free | Shared | Buff/cache | Available |
|---|---:|---:|---:|---:|---:|---:|
| RAM | 7.4 GiB | 5.8 GiB | 232 MiB | 151 MiB | 1.4 GiB | 1.3 GiB |
| Swap | 3.7 GiB | 1.6 GiB | 2.1 GiB | | | |

`swapon --show` reported six zram devices (`/dev/zram0` through `/dev/zram5`), each 635 MiB, with about 274–276 MiB used per device.

`df -h`:

| Filesystem | Size | Used | Available | Use | Mount |
|---|---:|---:|---:|---:|---|
| `/dev/mmcblk0p1` | 234G | 92G | 133G | 41% | `/` |
| `tmpfs` | 3.8G | 120K | 3.8G | 1% | `/dev/shm` |
| `tmpfs` | 1.5G | 152M | 1.4G | 10% | `/run` |
| `tmpfs` | 5.0M | 4.0K | 5.0M | 1% | `/run/lock` |
| `tmpfs` | 763M | 124K | 762M | 1% | `/run/user/1000` |

Service memory at inspection time:

| Service/process | systemd state | systemd cgroup `MemoryCurrent` | Main process RSS |
|---|---|---:|---:|
| `ollama.service` | active/running | 22,876,160 bytes (about 21.8 MiB) | 17,528 KiB (about 17.1 MiB) |
| `dfl-slm-master.service` | active/running | 787,402,752 bytes (about 750.8 MiB) | 27,580 KiB (about 26.9 MiB) |

The Ollama process listing contained only the `ollama` daemon at that moment; no separate runner process was listed.

## 2. JetPack/L4T, CUDA, Python, and power mode

| Item | Observed value |
|---|---|
| L4T release | R36, revision 3.0 |
| Installed L4T package | `nvidia-l4t-core 36.3.0-20240424200557` |
| JetPack | R36.3 corresponds to JetPack 6.0; the `nvidia-jetpack` meta-package itself was not installed |
| CUDA SDK version | 12.2.12 (`/usr/local/cuda/version.json`) |
| CUDA runtime package version | 12.2.140 |
| Python | Python 3.10.12 |
| `nvpmodel -q` | `NV Power Mode: 15W`, mode index `0` |

## 3. Existing Python packages

The requested package versions were checked in the system Python and in both discovered virtual environments.

| Package | System Python | `/home/hammad/Downloads/slm/.venv` | `/opt/dfl-slm-dtn/master/venv` |
|---|---|---|---|
| torch | Not installed | 2.4.0a0+3bcc3cddb5.nv24.7 | Not installed |
| torchvision | Not installed | Not installed | Not installed |
| transformers | Not installed | 4.46.3 | Not installed |
| peft | Not installed | 0.13.2 | Not installed |
| accelerate | Not installed | 1.1.1 | Not installed |
| bitsandbytes | Not installed | Not installed | Not installed |
| trl | Not installed | Not installed | Not installed |
| datasets | Not installed | 3.1.0 | Not installed |
| sentencepiece | Not installed | 0.2.2 | Not installed |

`torch.cuda.is_available()` in `/home/hammad/Downloads/slm/.venv` returned **True**. It was not queried in environments where torch is absent.

The system `pip list` completed successfully; none of the nine requested packages appeared in that list. Venv discovery found the two paths listed above.

## 4. PyTorch wheel for this Jetson

The measured combination is L4T R36.3, CUDA 12.2, Python 3.10, and `aarch64`. NVIDIA's [PyTorch for Jetson installation page](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) lists **PyTorch 2.3.0**, wheel `torch-2.3.0-cp310-cp310-linux_aarch64.whl`, for JetPack 6.0 (L4T R36.2/R36.3) with CUDA 12.2. This wheel is the verified matching option from that source.

The page also lists torchvision 0.18 for this JetPack/CUDA/Python combination. The installed `/home/hammad/Downloads/slm/.venv` instead has NVIDIA build `torch 2.4.0a0+3bcc3cddb5.nv24.7`, with CUDA availability observed as true.

## 5. Proposed new-venv install plan (not run)

Use a separate venv, leaving both running services and their environments untouched:

```bash
python3 -m venv /home/hammad/slm-finetune-b1/.venv
source /home/hammad/slm-finetune-b1/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl"
python -m pip install "transformers==4.46.3" "peft==0.13.2" "accelerate==1.1.1" "datasets==3.1.0" "sentencepiece==0.2.2"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The NVIDIA URL is the PyTorch 2.3.0 CUDA 12.2 / JetPack 6.0 R36.3 Python 3.10 aarch64 wheel listed by NVIDIA. This plan intentionally does not include bitsandbytes or TRL, which were not present in the existing environments. It has not been tested as an installation plan.

## 6. Llama 3.2 1B weights

**Yes.** A Hugging Face cache snapshot exists at `/home/hammad/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct`, with a `model.safetensors` snapshot symlink resolving to a local blob of **2,471,645,608 bytes**. This is a Hugging Face model weight file, not an Ollama blob.

## Collection errors and limitations

- `dpkg-query` reported: `no packages found matching nvidia-jetpack`. The installed `nvidia-l4t-core` version and `/etc/nv_tegra_release` were available and reported above.
- `nvcc --version` reported `nvcc not found on PATH`; CUDA version was read from `/usr/local/cuda/version.json`.
- Recursive environment discovery under `/opt` reported `find: ‘/opt/containerd’: Permission denied`; therefore the inventory is limited to visible venvs and cannot exclude a venv inside that inaccessible directory.
- Two collection script attempts ended with `bash: syntax error: unexpected end of file` while scanning additional model directories. A later direct HF snapshot listing successfully confirmed the snapshot symlinks and weight blob size. No host files were changed by these failed inspection commands.
- A follow-up config-inspection command had a PowerShell quoting error and did not return model architecture fields; the file identity and weight size were confirmed independently.

## B1b

Read-only follow-up on `dfl-master` over SSH on 2026-10-27. No packages, services, model files, or data were changed. No `.env` file or Hugging Face token file was opened or printed. Both `dfl-slm-master.service` and `ollama.service` were left running.

### 1. Processes, memory, and running services

`ps -eo pid,rss,comm,args --sort=-rss | head -16` returned the following 15 processes. RSS is in KiB. No credential-like values appeared in these command lines.

| PID | RSS KiB | Command | Arguments |
|---:|---:|---|---|
| 48755 | 494064 | python3 | `/usr/bin/python3 -u /opt/dfl-slm-dtn/master/cifar_evaluator.py /opt/dfl-slm-dtn/data/cifar10_test.npz /opt/dfl-slm-dtn/data/cifar10_train.npz` |
| 2027 | 281692 | gnome-software | `/usr/bin/gnome-software --gapplication-service` |
| 1963 | 81948 | packagekitd | `/usr/libexec/packagekitd` |
| 1857 | 81872 | gnome-shell | `/usr/bin/gnome-shell` |
| 225 | 52516 | systemd-journal | `/lib/systemd/systemd-journald` |
| 3455 | 45088 | update-manager | `/usr/bin/python3 /usr/bin/update-manager --no-update --no-focus-on-map` |
| 2188 | 27420 | tracker-miner-f | `/usr/libexec/tracker-miner-fs-3` |
| 1339 | 23580 | dockerd | `/usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock` |
| 48514 | 23088 | waitress-serve | `/opt/dfl-slm-dtn/master/venv/bin/python3 /opt/dfl-slm-dtn/master/venv/bin/waitress-serve --listen=0.0.0.0:5003 --threads=4 master.wsgi:app` |
| 836 | 17528 | ollama | `/usr/local/bin/ollama serve` |
| 1618 | 16132 | Xorg | `/usr/lib/xorg/Xorg vt2 -displayfd 3 -auth /run/user/1000/gdm/Xauthority -nolisten tcp -background none -noreset -keeptty -novtswitch -verbose 3` |
| 2042 | 15948 | python3 | `python3 /usr/share/nvpmodel_indicator/nvpmodel_indicator.py` |
| 3243 | 14760 | update-notifier | `update-notifier` |
| 2026 | 14304 | evolution-alarm | `/usr/libexec/evolution-data-server/evolution-alarm-notify` |
| 5605 | 14176 | python3 | `python3 -m common.status_agent` |

These top 15 RSS values sum to 1,204,116 KiB (about 1.15 GiB); RSS values can include shared pages and should not be treated as a precise additive breakdown of physical RAM.

`systemctl get-default` returned `graphical.target`. A graphical session is running: `gdm` and `gdm3` reported active, `lightdm` inactive; `gdm-session-worker`, `gdm-x-session`, Xorg, GNOME Shell, and related desktop processes were present.

The latest `free -h` snapshot during B1b reported 7.4 GiB total, 5.7 GiB used, 233 MiB free, 1.6 GiB buff/cache, and 1.4 GiB available. Swap was 3.7 GiB total, 2.0 GiB used, 1.7 GiB free. This differs slightly from the earlier B1 snapshot; these are live measurements at different times.

Selected `/proc/meminfo` values from B1b:

| Field | Value |
|---|---:|
| MemTotal | 7,803,264 kB |
| MemAvailable | 1,506,844 kB |
| Cached | 1,299,564 kB |
| Shmem | 141,672 kB |
| SwapTotal | 3,901,608 kB |
| SwapFree | 1,761,448 kB |
| AnonPages | 1,051,584 kB |
| Slab | 223,440 kB |
| SReclaimable | 112,280 kB |
| SUnreclaim | 111,160 kB |
| KernelStack | 10,544 kB |
| PageTables | 29,292 kB |

The reported `used` value is not the sum of the top process RSS values. The observed memory also includes file cache, shared memory, kernel allocations/slab, and memory used by processes outside the top 15; some cache/slab is reclaimable. These fields overlap in accounting and must not be added as independent buckets. The available-memory measure is 1,506,844 kB.

`zramctl` reported six active 635 MiB LZO-RLE swap devices:

| Device | Data | Compressed | Allocated | Streams |
|---|---:|---:|---:|---:|
| `/dev/zram0` | 348.0M | 181.1M | 184.2M | 6 |
| `/dev/zram1` | 346.7M | 182.6M | 185.7M | 6 |
| `/dev/zram2` | 348.6M | 183.2M | 186.3M | 6 |
| `/dev/zram3` | 349.6M | 183.2M | 186.3M | 6 |
| `/dev/zram4` | 348.6M | 181.2M | 184.2M | 6 |
| `/dev/zram5` | 347.8M | 181.5M | 184.6M | 6 |

Running services from `systemctl list-units --type=service --state=running`:

```text
accounts-daemon.service
avahi-daemon.service
bluetooth.service
colord.service
containerd.service
cron.service
dbus.service
dfl-slm-master.service
dfl-slm-status.service
docker.service
gdm.service
haveged.service
kerneloops.service
lpd.service
ModemManager.service
networkd-dispatcher.service
NetworkManager.service
nv-tee-supplicant.service
nvargus-daemon.service
nvfancontrol.service
nvmemwarning.service
nvphs.service
nvs-service.service
ollama.service
packagekit.service
polkit.service
power-profiles-daemon.service
rpcbind.service
rsyslog.service
rtkit-daemon.service
seatd.service
serial-getty@ttyGS0.service
serial-getty@ttyTCU0.service
ssh.service
switcheroo-control.service
systemd-journald.service
systemd-logind.service
systemd-resolved.service
systemd-timesyncd.service
systemd-udevd.service
udisks2.service
upower.service
user@1000.service
wpa_supplicant.service
```

### 2. Stop-candidate assessment (nothing stopped)

The observed reverse dependency listing showed `ollama.service` as a dependency of `dfl-slm-master.service`; both services also appear under `graphical.target`. Therefore neither Ollama nor the DFL master service, their Python process, or the active graphical target/session is a stop candidate for this task.

No process can be guaranteed safe to stop from a read-only snapshot alone. The following appear to be optional desktop or device functions rather than dependencies of the two target services, and are only **conditional candidates if their feature is not in use**: GNOME Software / `packagekit.service`, Update Manager / `update-notifier`, Tracker file indexing, Evolution alarm notification, Bluetooth, Avahi, printer service `lpd`, and ModemManager. Stopping them may affect the current desktop or other users and was not attempted. The `cifar_evaluator.py` process and `dfl-slm-status.service` were not suggested as candidates because they are project processes/services and could affect active project work or the status API.

### 3. `/home/hammad/Downloads/slm` files and model folders

Top-level `ls -la` listing (name and observed modification time):

```text
data/                                      2026-10-09 15:59
eval_driving_adapter.py                   2026-09-24 14:40
eval.log                                  2026-09-24 14:48
eval_mesh_full_test.py                    2026-09-24 20:24
launch_nodes.py                           2026-10-07 22:41
logs/                                     2026-10-09 21:06
mesh_dfl.log                              2026-09-24 18:19
mesh_full.log                             2026-09-24 20:35
mesh_full_test.log                        2026-09-24 20:25
mesh_full_test.pid                        2026-09-24 20:24
mesh_peers.json                           2026-10-05 13:12
mesh_r60.log                              2026-09-28 19:09
models/                                   2026-10-09 19:01
__pycache__/                              2026-10-05 13:12
_remote_jetson_setup.sh                   2026-09-24 16:20
resume_train.sh                           2026-09-23 18:52
run_eval.sh                               2026-09-24 14:40
run_mesh_dfl.sh                           2026-09-24 17:45
run_mesh_full_test.sh                     2026-09-24 20:24
run_node.sh                               2026-10-09 16:04
setup_and_train.sh                        2026-09-23 18:41
torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp310-cp310-linux_aarch64.whl  2024-07-20 08:15
train_llama32_driving_gpu.py              2026-09-23 18:41
train_llama_slm.py                        2026-10-05 13:12
train_mesh_dfl.py                         2026-10-09 15:59
.venv/                                    2026-09-23 18:44
```

Training/fine-tuning script filenames found outside `.venv`:

```text
resume_train.sh
setup_and_train.sh
train_llama32_driving_gpu.py
train_llama_slm.py
train_mesh_dfl.py
```

The `models/` listing contained 39 model directories:

```text
exp10-fedavg/                             2026-10-09 19:01
exp10-saaa/                               2026-10-09 21:06
exp10-smoke-lr1e-4/                       2026-10-09 17:29
exp10-smoke-r16/                          2026-10-09 17:07
exp10-smoke-r32/                          2026-10-09 16:48
llama3.2-1b-driving-balanced-fedavg/      2026-09-28 23:38
llama3.2-1b-driving-balanced-safety/      2026-09-29 04:26
llama3.2-1b-driving-lora/                 2026-09-24 14:40
llama3.2-1b-driving-mesh-dfl/             2026-09-24 18:19
llama3.2-1b-driving-mesh-full/            2026-09-24 20:19
llama3.2-1b-driving-mesh-r60/             2026-09-28 19:06
llama3.2-1b-driving-poison-fedavg/        2026-10-06 17:39
llama3.2-1b-driving-poison-nofs-fedavg/   2026-10-07 17:47
llama3.2-1b-driving-poison-nofs-saaa/     2026-10-07 21:16
llama3.2-1b-driving-poison-safety/        2026-10-06 20:12
llama3.2-1b-driving-poison-smoke/         2026-10-06 15:25
llama3.2-1b-driving-prop-median/          2026-10-07 06:05
llama3.2-1b-driving-prop-noise-fedavg/    2026-10-07 12:21
llama3.2-1b-driving-prop-noise-saaa/      2026-10-07 13:30
llama3.2-1b-driving-prop-poison-saaa/     2026-10-07 09:36
llama3.2-1b-driving-prop-saaa/            2026-10-07 03:04
llama3.2-1b-driving-prop-smoke/           2026-10-06 21:46
llama3.2-1b-driving-v2-fedavg/            2026-09-29 22:03
llama3.2-1b-driving-v2-fedavg-corrupt/    2026-09-30 06:10
llama3.2-1b-driving-v2-safety/            2026-09-30 01:58
llama3.2-1b-driving-v3-fedavg/            2026-09-30 20:26
llama3.2-1b-driving-v3-fedavg-refine/     2026-10-01 07:39
llama3.2-1b-driving-v3-safety/            2026-10-01 01:00
llama3.2-1b-driving-v3-safety-refine/     2026-10-01 04:44
llama3.2-1b-driving-v4-fedavg/            2026-10-01 19:39
llama3.2-1b-driving-v4-fedavg-w175/       2026-10-01 20:58
llama3.2-1b-driving-v5d-finetune/         2026-10-02 16:29
llama3.2-1b-driving-v5-fedavg/            2026-10-02 02:31
llama3.2-1b-driving-v5-fedavg-refine/     2026-10-02 15:09
llama3.2-1b-driving-v5-safety/            2026-10-02 13:15
llama3.2-1b-driving-wbs-nofs-fedavg/      2026-10-05 22:00
llama3.2-1b-driving-wbs-nofs-saaa/        2026-10-08 22:45
llama3.2-1b-driving-wbs-withfs-fedavg/    2026-10-05 17:15
llama3.2-1b-driving-wbs-withfs-safety/    2026-10-06 01:35
```

`du -sh` size and directory modification time for each model folder:

| Folder | Size | Modified |
|---|---:|---|
| exp10-fedavg | 599M | 2026-10-09 19:01 |
| exp10-saaa | 599M | 2026-10-09 21:06 |
| exp10-smoke-lr1e-4 | 154M | 2026-10-09 17:29 |
| exp10-smoke-r16 | 102M | 2026-10-09 17:07 |
| exp10-smoke-r32 | 154M | 2026-10-09 16:48 |
| llama3.2-1b-driving-balanced-fedavg | 932M | 2026-09-28 23:38 |
| llama3.2-1b-driving-balanced-safety | 1.5G | 2026-09-29 04:26 |
| llama3.2-1b-driving-lora | 96M | 2026-09-24 14:40 |
| llama3.2-1b-driving-mesh-dfl | 323M | 2026-09-24 18:19 |
| llama3.2-1b-driving-mesh-full | 116M | 2026-09-24 20:19 |
| llama3.2-1b-driving-mesh-r60 | 940M | 2026-09-28 19:06 |
| llama3.2-1b-driving-poison-fedavg | 1.1G | 2026-10-06 17:39 |
| llama3.2-1b-driving-poison-nofs-fedavg | 1.1G | 2026-10-07 17:47 |
| llama3.2-1b-driving-poison-nofs-saaa | 1.1G | 2026-10-07 21:16 |
| llama3.2-1b-driving-poison-safety | 1.1G | 2026-10-06 20:12 |
| llama3.2-1b-driving-poison-smoke | 141M | 2026-10-06 15:25 |
| llama3.2-1b-driving-prop-median | 43M | 2026-10-07 06:05 |
| llama3.2-1b-driving-prop-noise-fedavg | 1.1G | 2026-10-07 12:21 |
| llama3.2-1b-driving-prop-noise-saaa | 308M | 2026-10-07 13:30 |
| llama3.2-1b-driving-prop-poison-saaa | 1.1G | 2026-10-07 09:36 |
| llama3.2-1b-driving-prop-saaa | 1.7G | 2026-10-07 03:04 |
| llama3.2-1b-driving-prop-smoke | 197M | 2026-10-06 21:46 |
| llama3.2-1b-driving-v2-fedavg | 1.1G | 2026-09-29 22:03 |
| llama3.2-1b-driving-v2-fedavg-corrupt | 463M | 2026-09-30 06:10 |
| llama3.2-1b-driving-v2-safety | 1.1G | 2026-09-30 01:58 |
| llama3.2-1b-driving-v3-fedavg | 1.8G | 2026-09-30 20:26 |
| llama3.2-1b-driving-v3-fedavg-refine | 1.1G | 2026-10-01 07:39 |
| llama3.2-1b-driving-v3-safety | 1.8G | 2026-10-01 01:00 |
| llama3.2-1b-driving-v3-safety-refine | 1.2G | 2026-10-01 04:44 |
| llama3.2-1b-driving-v4-fedavg | 2.0G | 2026-10-01 19:39 |
| llama3.2-1b-driving-v4-fedavg-w175 | 20K | 2026-10-01 20:58 |
| llama3.2-1b-driving-v5d-finetune | 432M | 2026-10-02 16:29 |
| llama3.2-1b-driving-v5-fedavg | 2.5G | 2026-10-02 02:31 |
| llama3.2-1b-driving-v5-fedavg-refine | 197M | 2026-10-02 15:09 |
| llama3.2-1b-driving-v5-safety | 988M | 2026-10-02 13:15 |
| llama3.2-1b-driving-wbs-nofs-fedavg | 2.0G | 2026-10-05 22:00 |
| llama3.2-1b-driving-wbs-nofs-saaa | 1.7G | 2026-10-08 22:45 |
| llama3.2-1b-driving-wbs-withfs-fedavg | 2.0G | 2026-10-05 17:15 |
| llama3.2-1b-driving-wbs-withfs-safety | 1.4G | 2026-10-06 01:35 |

Filename-only inspection found adapter checkpoint files (`adapter_model.safetensors`) in 38 model folders. The inspected 39 model folders had no matching standalone merged/full model weight files named `model.safetensors`, `model-*.safetensors`, or `pytorch_model*.bin`. This filename-based check does not establish whether differently named weight files exist.

### 4. Full `.venv` package list and CUDA runtime

Output of `/home/hammad/Downloads/slm/.venv/bin/python -m pip list --format=freeze`:

```text
accelerate==1.1.1
aiohappyeyeballs==2.7.1
aiohttp==3.14.3
async-timeout==5.0.1
attrs==26.1.0
certifi==2026.7.22
charset-normalizer==3.5.1
cloudpickle==3.1.2
contourpy==1.3.2
cycler==0.12.1
datasets==3.1.0
dill==0.3.8
filelock==4.0.1
fonttools==4.65.0
frozenlist==1.8.0
fsspec==2024.9.0
hf-xet==1.6.0
huggingface_hub==0.36.2
idna==3.20
Jinja2==3.1.6
joblib==1.6.0
kiwisolver==1.5.1
MarkupSafe==3.0.3
matplotlib==3.10.9
mpmath==1.3.0
multidict==6.9.1
multiprocess==0.70.16
networkx==3.4.2
numpy==1.26.4
packaging==26.3
pandas==2.3.3
peft==0.13.2
pillow==12.3.0
pip==26.2.1
propcache==0.5.4
protobuf==7.36.2
psutil==7.2.2
pyarrow==25.0.1
pyparsing==3.3.3
python-dateutil==2.9.0.post0
pytz==2026.3.post1
PyYAML==6.0.3
regex==2026.9.10
requests==2.34.2
safetensors==0.8.0
scikit-learn==1.7.2
scipy==1.15.3
sentencepiece==0.2.2
setuptools==59.6.0
six==1.17.0
sympy==1.14.0
threadpoolctl==3.7.0
tokenizers==0.20.3
torch==2.4.0a0+3bcc3cddb5.nv24.7
tqdm==4.70.1
transformers==4.46.3
typing_extensions==4.16.0
tzdata==2026.4
urllib3==2.8.0
xxhash==4.0.1
yarl==1.25.1
```

CUDA details queried from that venv:

| Check | Result |
|---|---|
| `torch.__version__` | `2.4.0a0+3bcc3cddb5.nv24.07` |
| `torch.version.cuda` | `12.2` |
| `torch.cuda.get_device_name(0)` | `Orin` |
| `torch.cuda.mem_get_info()` | Succeeded: 1,269,014,528 free bytes of 7,990,542,336 total bytes |

### 5. Hugging Face snapshot names, sizes, and config fields

The snapshot directory is `.../models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6/`. Sizes below follow the snapshot symlinks to their local blobs. Only filenames and sizes were collected; no token files or credentials were read.

| File | Size |
|---|---:|
| `config.json` | 877 bytes |
| `tokenizer.json` | 9,085,657 bytes |
| `tokenizer_config.json` | 54,528 bytes |
| `generation_config.json` | 189 bytes |
| `special_tokens_map.json` | 296 bytes |
| `model.safetensors` | 2,471,645,608 bytes |

Selected fields from `config.json`:

```text
architectures=['LlamaForCausalLM']
hidden_size=2048
num_hidden_layers=16
vocab_size=128256
torch_dtype='bfloat16'
```

### B1b collection errors

No command errors were returned by the B1b collection. The multi-line SSH scripts were sent using `ssh ... "tr -d '\r' | bash -s"` as requested. No `.env` or Hugging Face token file was opened or printed.

## B1c - Earlier LoRA runs and memory attribution

Read-only inspection of `dfl-master` (`hammad@192.168.11.15`) on 2026-10-27. No training code was run, no services or processes were changed, and no files on the Jetson were modified. No `.env` file or Hugging Face token file was opened. Key-, token-, secret-, and password-like values were not included in this report.

### 1. Training script configuration

The scripts below describe defaults and code paths; their values are not proof that every historical run used those defaults. The recorded mesh-run configurations are taken from the run logs in section 3.

| Script | Base model and loading/dtype | LoRA | Sequence length | Batch / accumulation | Checkpointing | Optimizer / learning rate | Epochs / steps | Device |
|---|---|---|---:|---|---|---|---|---|
| `train_llama_slm.py` | `meta-llama/Llama-3.2-1B-Instruct`; CUDA uses FP16 and attempts 4-bit loading by default; CPU uses FP32 | defaults `r=8`, `alpha=16`, `dropout=0.05`, targets `q_proj,k_proj,v_proj,o_proj` | 256 default | per-device batch 1; gradient accumulation 32 | enabled in `TrainingArguments` | learning rate `2e-4`; optimizer left at the Transformers `TrainingArguments` default | 4 epochs; `max_steps=-1` unless overridden | `device_map="auto"` for 4-bit model loading; CPU path supported |
| `train_llama32_driving_gpu.py` | same base model; CUDA required; 4-bit NF4 with FP16 compute and double quantization when available; fallback loads FP16 on GPU | defaults `r=8`, `alpha=16`, `dropout=0.05`, targets `q_proj,k_proj,v_proj,o_proj` | requested 384, then selected dynamically from the training records up to the requested maximum | per-device batch 1; gradient accumulation 16 | enabled | `adamw_torch`; learning rate `2e-4` | 2 epochs; `max_steps=-1` unless overridden | GPU 0 (`device_map={"": 0}`) |

`train_llama32_driving_gpu.py` calls `torch.cuda.empty_cache()` only on its FP16 fallback path before loading the model. The inspected scripts did not contain a command to stop Ollama, `dfl-slm-master.service`, or another system service before training. `setup_and_train.sh` contains a `pip install bitsandbytes` attempt but was not run. `run_node.sh` is orchestration code; the inspected sections did not show a service-stop or cache-clearing step.

### 2. Adapter folders and weight files

The four adapter configurations read from `adapter_config.json` were:

| Folder | `r` | `lora_alpha` | `target_modules` | `base_model_name_or_path` |
|---|---:|---:|---|---|
| `models/exp10-smoke-r16` | 16 | 32 | `q_proj,k_proj,v_proj,o_proj` | `meta-llama/Llama-3.2-1B-Instruct` |
| `models/exp10-smoke-r32` | 32 | 64 | `q_proj,k_proj,v_proj,o_proj` | `meta-llama/Llama-3.2-1B-Instruct` |
| `models/llama3.2-1b-driving-lora` | 8 | 16 | `q_proj,k_proj,v_proj,o_proj` | `meta-llama/Llama-3.2-1B-Instruct` |
| `models/llama3.2-1b-driving-v5d-finetune` | 32 | 64 | `q_proj,k_proj,v_proj,o_proj` | `meta-llama/Llama-3.2-1B-Instruct` |

Recursive filename-and-size inventory (bytes; listed paths are relative to each folder):

<details>
<summary>exp10-smoke-r16</summary>

| File | Bytes |
|---|---:|
| `README.md` | 5,106 |
| `adapter_config.json` | 686 |
| `adapter_model.safetensors` | 13,648,488 |
| `latest.json` | 184 |
| `mesh_dfl_report.json` | 7,097 |
| `rounds.jsonl` | 2,064 |
| `special_tokens_map.json` | 325 |
| `test_metrics_8.json` | 1,487 |
| `tokenizer.json` | 17,210,018 |
| `tokenizer_config.json` | 54,557 |
| `plots/loss_vs_round.png` | 50,460 |
| `plots/peer_weights_heatmap.png` | 32,322 |
| `plots/probe_direction_acc_vs_round.png` | 45,092 |
| `plots/probe_steer_mae_vs_round.png` | 50,546 |
| `plots/val_loss_vs_round.png` | 52,409 |
| `published/exp10-smoke-r16/r1.pt` | 6,850,858 |
| `published/exp10-smoke-r16/r2.pt` | 6,850,858 |
| `round_001/README.md` | 5,106 |
| `round_001/adapter_config.json` | 686 |
| `round_001/adapter_model.safetensors` | 13,648,488 |
| `round_001/special_tokens_map.json` | 325 |
| `round_001/tokenizer.json` | 17,209,920 |
| `round_001/tokenizer_config.json` | 54,557 |
| `round_002/README.md` | 5,106 |
| `round_002/adapter_config.json` | 686 |
| `round_002/adapter_model.safetensors` | 13,648,488 |
| `round_002/special_tokens_map.json` | 325 |
| `round_002/tokenizer.json` | 17,210,018 |
| `round_002/tokenizer_config.json` | 54,557 |

</details>

<details>
<summary>exp10-smoke-r32</summary>

| File | Bytes |
|---|---:|
| `README.md` | 5,106 |
| `adapter_config.json` | 686 |
| `adapter_model.safetensors` | 27,280,080 |
| `latest.json` | 184 |
| `mesh_dfl_report.json` | 7,097 |
| `rounds.jsonl` | 2,064 |
| `special_tokens_map.json` | 325 |
| `test_metrics_8.json` | 1,486 |
| `tokenizer.json` | 17,210,018 |
| `tokenizer_config.json` | 54,557 |
| `plots/loss_vs_round.png` | 50,690 |
| `plots/peer_weights_heatmap.png` | 32,483 |
| `plots/probe_direction_acc_vs_round.png` | 61,638 |
| `plots/probe_steer_mae_vs_round.png` | 45,810 |
| `plots/val_loss_vs_round.png` | 65,471 |
| `published/exp10-smoke-r32/r1.pt` | 13,666,794 |
| `published/exp10-smoke-r32/r2.pt` | 13,666,794 |
| `round_001/README.md` | 5,106 |
| `round_001/adapter_config.json` | 686 |
| `round_001/adapter_model.safetensors` | 27,280,080 |
| `round_001/special_tokens_map.json` | 325 |
| `round_001/tokenizer.json` | 17,209,920 |
| `round_001/tokenizer_config.json` | 54,557 |
| `round_002/README.md` | 5,106 |
| `round_002/adapter_config.json` | 686 |
| `round_002/adapter_model.safetensors` | 27,280,080 |
| `round_002/special_tokens_map.json` | 325 |
| `round_002/tokenizer.json` | 17,210,018 |
| `round_002/tokenizer_config.json` | 54,557 |

</details>

<details>
<summary>llama3.2-1b-driving-lora</summary>

| File | Bytes |
|---|---:|
| `README.md` | 5,106 |
| `adapter_config.json` | 685 |
| `adapter_model.safetensors` | 6,832,520 |
| `special_tokens_map.json` | 325 |
| `test_predictions.jsonl` | 13,796 |
| `tokenizer.json` | 17,210,018 |
| `tokenizer_config.json` | 54,557 |
| `training_args.bin` | 5,368 |
| `checkpoint-1600/README.md` | 5,106 |
| `checkpoint-1600/adapter_config.json` | 685 |
| `checkpoint-1600/adapter_model.safetensors` | 6,832,520 |
| `checkpoint-1600/optimizer.pt` | 13,739,130 |
| `checkpoint-1600/rng_state.pth` | 14,244 |
| `checkpoint-1600/scheduler.pt` | 1,064 |
| `checkpoint-1600/special_tokens_map.json` | 325 |
| `checkpoint-1600/tokenizer.json` | 17,210,018 |
| `checkpoint-1600/tokenizer_config.json` | 54,557 |
| `checkpoint-1600/trainer_state.json` | 15,818 |
| `checkpoint-1600/training_args.bin` | 5,368 |
| `checkpoint-1908/README.md` | 5,106 |
| `checkpoint-1908/adapter_config.json` | 685 |
| `checkpoint-1908/adapter_model.safetensors` | 6,832,520 |
| `checkpoint-1908/optimizer.pt` | 13,739,130 |
| `checkpoint-1908/rng_state.pth` | 14,244 |
| `checkpoint-1908/scheduler.pt` | 1,064 |
| `checkpoint-1908/special_tokens_map.json` | 325 |
| `checkpoint-1908/tokenizer.json` | 17,210,018 |
| `checkpoint-1908/tokenizer_config.json` | 54,557 |
| `checkpoint-1908/trainer_state.json` | 18,423 |
| `checkpoint-1908/training_args.bin` | 5,368 |

</details>

<details>
<summary>llama3.2-1b-driving-v5d-finetune</summary>

| File | Bytes |
|---|---:|
| `README.md` | 5,106 |
| `adapter_config.json` | 686 |
| `adapter_model.safetensors` | 27,280,080 |
| `latest.json` | 198 |
| `mesh_dfl_report.json` | 13,562 |
| `rounds.jsonl` | 7,046 |
| `special_tokens_map.json` | 325 |
| `test_metrics_500.json` | 1,512 |
| `tokenizer.json` | 17,210,018 |
| `tokenizer_config.json` | 54,557 |
| `plots/loss_vs_round.png` | 51,405 |
| `plots/peer_weights_heatmap.png` | 33,311 |
| `plots/probe_direction_acc_vs_round.png` | 52,325 |
| `plots/probe_steer_mae_vs_round.png` | 52,325 |
| `plots/val_loss_vs_round.png` | 67,039 |
| `published/v5D-20261002/r1.pt` through `r7.pt` | 13,666,794 each |
| `round_001/README.md` through `round_007/README.md` | 5,106 each |
| `round_001/adapter_config.json` through `round_007/adapter_config.json` | 686 each |
| `round_001/adapter_model.safetensors` through `round_007/adapter_model.safetensors` | 27,280,080 each |
| `round_001/special_tokens_map.json` through `round_007/special_tokens_map.json` | 325 each |
| `round_001/tokenizer_config.json` through `round_007/tokenizer_config.json` | 54,557 each |
| `round_001/tokenizer.json` | 17,209,920 |
| `round_002/tokenizer.json` through `round_007/tokenizer.json` | 17,210,018 each |

</details>

The inspected weight files are PEFT adapters (`adapter_model.safetensors`) or round-level adapter weights (`r1.pt` etc.), not merged base-model weights. No merged/full model weight file was present in these four inventories.

### 3. Logs and recorded run measurements

All entries under `/home/hammad/Downloads/slm/logs/` were listed. The filenames were:

```text
balA-20260928.log, balA-20260928.plot.log, balA-20260928.test.log
balB-20260928.log, balB-20260928.plot.log, balB-20260928.test.log
c95-mkrum-20261007.log, c95-mkrum-20261007.test.log
c95-r64-20261007.log, c95-r64-20261007.plot.log, c95-r64-20261007.test.log
c95-rasa-20261007.log, c95-rasa-20261007.test.log
c95-saaa-20261007.log, c95-saaa-20261007.test.log
c95-smoke-20261007.log, c95-smoke-20261007.test.log
c95-trim-20261007.log, c95-trim-20261007.test.log
exp10-fedavg.log, exp10-fedavg.plot.log, exp10-fedavg.test.log
exp10-saaa.log, exp10-saaa.plot.log, exp10-saaa.test.log
exp10-smoke-lr1e-4.log, exp10-smoke-lr1e-4.plot.log, exp10-smoke-lr1e-4.test.log
exp10-smoke-r16.log, exp10-smoke-r16.plot.log, exp10-smoke-r16.test.log
exp10-smoke-r32.log, exp10-smoke-r32.plot.log, exp10-smoke-r32.test.log
mt20-fedavg-qwen25-15b-instruc.log, mt20-fedavg-qwen25-15b-instruc.plot.log, mt20-fedavg-qwen25-15b-instruc.test.log
mt20-improve-saaa-1.log
mt20-saaa-qwen25-15b-instruc-g1.log, mt20-saaa-qwen25-15b-instruc-g1.plot.log, mt20-saaa-qwen25-15b-instruc-g1.test.log
mt20-smoke-llama-32-3b-instru.log
mt20-smoke-qwen25-15b-instruc.log, mt20-smoke-qwen25-15b-instruc.plot.log, mt20-smoke-qwen25-15b-instruc.test.log
nofs-saaa-20261008.log, nofs-saaa-20261008.plot.log, nofs-saaa-20261008.test.log
poisonA-20261006.log, poisonA-20261006.test.log
poisonB-20261006.log, poisonB-20261006.plot.log, poisonB-20261006.test.log
poison-nofsA-20261007.log, poison-nofsA-20261007.test.log
poison-nofsS-20261007.log, poison-nofsS-20261007.test.log
poison-smoke-20261006.log
propA-20261006.log, propA-20261006.test.log
prop-med-20261007.log
prop-noiseF-20261007.log, prop-noiseF-20261007.test.log
prop-noiseS-20261007.log
prop-poison-20261007.log, prop-poison-20261007.test.log
prop-smoke-20261006.log, prop-smoke-20261006.test.log
run_node_exp10.log, run_node_exp10.nohup
run_node.log, run_node.nohup
run_node_mtown.log, run_node_mtown.nohup
run_node_poison.log, run_node_poison.nohup
run_node_prop.log, run_node_prop.nohup
run_node_v3.log, run_node_v3.nohup
run_node_v4.log, run_node_v4.nohup
run_node_v5.log, run_node_v5.nohup
run_node_vipC.log, run_node_vipC.nohup
run_node_vip.log, run_node_vip.nohup
run_node_wbs.log, run_node_wbs.nohup
twin-smoke-20260929-1723.log, twin-smoke-20260929-1723.nohup, twin-smoke-20260929-1723.plot.log, twin-smoke-20260929-1723.run_node.log, twin-smoke-20260929-1723.test.log
twin-smoke-ui-1747.log, twin-smoke-ui-1747.nohup, twin-smoke-ui-1747.run_node.log
v3A-20260930.log, v3A-20260930.plot.log, v3A-20260930.test.log
v3B-20260930.log, v3B-20260930.plot.log, v3B-20260930.test.log
v3C-20260930.log, v3C-20260930.plot.log, v3C-20260930.test.log
v3D-20260930.log, v3D-20260930.plot.log, v3D-20260930.test.log
v4A-20261001.log, v4A-20261001.test.log
v4B-20261001.log
v5A-20261001.log, v5A-20261001.test.log
v5B-20261001.log, v5B-20261001.plot.log, v5B-20261001.test.log
v5C-20261001.log
v5D-20261002.log, v5D-20261002.plot.log, v5D-20261002.test.log
vipA-20260929.log, vipA-20260929.plot.log, vipA-20260929.test.log
vipB-20260929.log, vipB-20260929.plot.log, vipB-20260929.test.log
vipC-20260929.log, vipC-20260929.plot.log, vipC-20260929.test.log
wbsA-20261005.log, wbsA-20261005.test.log
wbsB-20261005.log, wbsB-20261005.log.bak_dupes, wbsB-20261005.plot.log, wbsB-20261005.test.log
wbsC-20261005.log, wbsC-20261005.plot.log, wbsC-20261005.test.log
```

The project-root `*.log` files were `eval.log`, `mesh_dfl.log`, `mesh_full.log`, `mesh_full_test.log`, `mesh_r60.log`, and `train.log`.

Representative relevant log lines (file names and line numbers are from the files on the Jetson):

| File and line | Recorded line / result |
|---|---|
| `logs/balA-20260928.log:3` | `config: local_steps=40 bs=1 accum=8 lr=0.0002 max_length=416 lora r=16 alpha=32 dropout=0.05 targets=['q_proj', 'k_proj', 'v_proj', 'o_proj']` |
| `logs/balA-20260928.log:24` | `local train done ... elapsed=308s (7.70s/step) peak_alloc=3052MiB` |
| `logs/balB-20260928.log:3` | Same recorded local training configuration as balA; `rounds_done=40` appears at line 1189. |
| `logs/exp10-smoke-r16.log:3` | `local_steps=40 bs=1 accum=8 lr=0.0002 max_length=416 lora r=16 alpha=32 dropout=0.05` |
| `logs/exp10-smoke-r16.log:26` | `elapsed=321s (8.04s/step) peak_alloc=3065MiB` |
| `logs/exp10-smoke-r32.log:3` | `local_steps=40 bs=1 accum=8 lr=0.0002 max_length=416 lora r=32 alpha=64 dropout=0.05` |
| `logs/exp10-smoke-r32.log:26` | `elapsed=325s (8.13s/step) peak_alloc=3117MiB` |
| `logs/v5D-20261002.log:3` | `local_steps=40 bs=1 accum=8 lr=0.0002 max_length=416 lora r=32 alpha=64 dropout=0.05` |
| `logs/v5D-20261002.log:27` | `elapsed=323s (8.06s/step) peak_alloc=3117MiB` |
| `train.log:33466` | `train_runtime=33192.3593`, `train_samples_per_second=0.92`, `train_steps_per_second=0.057`, `epoch=2.0`; no explicit sample count or total step count on that line or elsewhere in this log. |

Across the LoRA mesh logs, recorded local training steps were typically 40; most measured step durations were around 7.4–8.8 seconds and peak allocated GPU memory ranged from 3,052 to 3,221 MiB. Those are logged ranges, not a reconstruction of one particular run. Exact sample counts were **not logged** in the inspected training output. The root `train.log` recorded 33,192.3593 seconds of runtime, but did not explicitly record a total step count.

The all-log scan found these exact termination lines: `mt20-smoke-llama-32-3b-instru.log:252`, `poison-smoke-20261006.log:43`, `prop-med-20261007.log:97` and `:136`, `prop-noiseS-20261007.log:147`, `prop-smoke-20261006.log:63`, `propA-20261006.log:11`, `twin-smoke-ui-1747.log:42`, and `v5C-20261001.log:229`, each containing `FATAL Killed: signal 15`. No CUDA out-of-memory/OOM message was found. Signal 15 is reported as logged; the evidence does not establish that these were OOM terminations.

### 4. Current evaluator process and memory

At collection, `cifar_evaluator.py` was PID `48755`, child of PID `48514` (`waitress-serve`), with start time `Sat Oct 24 23:51:56 2026`. Its cgroup was `/system.slice/dfl-slm-master.service`.

| Measurement | Observed |
|---|---:|
| `/proc/48755/status` `VmRSS` | 489,940 kB |
| `/proc/48755/status` `VmSwap` | 1,475,288 kB |
| `smaps_rollup` RSS | 491,212 kB |
| `smaps_rollup` PSS | 476,378 kB |
| PSS anonymous / file / shared | 369,664 / 106,714 / 0 kB |
| `smaps_rollup` Swap | 1,475,280 kB |

The process-related `ps` query found only this `cifar_evaluator.py` process; no training or launch process matched the query. The nvmap client report attributed `3,385,608K` to PID 48755, compared with about 3.2 GiB peak allocation in prior training logs. This points to the evaluator as the main observed GPU-memory consumer at the time. The nvmap report also listed `xdg-desktop-por` (PID 2179, 426K), `gnome-shell` (PID 1857, 5,682K), and Xorg (PID 1618, 25,350K).

### 5. tegrastats and nvmap

`timeout 6s tegrastats --interval 1000` produced five samples and then exited with status `124` as expected from the timeout. The samples showed RAM `5940/7620MB`, swap `2110/3810MB`, `GR3D_FREQ 0%`, and GPU temperature around 45 C. No training load was observed during that short sample.

`sudo -n cat /sys/kernel/debug/nvmap/iovmm/clients` succeeded and returned the client allocations listed above. The alternate path `/sys/kernel/debug/nvmap/clients` returned exactly `cat: /sys/kernel/debug/nvmap/clients: No such file or directory` (exit status 1). The `iovmm/clients` file was readable; no fallback discovery was needed.

### 6. `dfl-slm-master.service` selected unit settings

Only the requested unit fields are shown:

```text
After=network-online.target ollama.service dfl-slm-status.service
Wants=network-online.target ollama.service
User=hammad
WorkingDirectory=/opt/dfl-slm-dtn
ExecStart=/opt/dfl-slm-dtn/master/venv/bin/waitress-serve --listen=0.0.0.0:5003 --threads=4 master.wsgi:app
Restart=on-failure
WantedBy=multi-user.target
```

No `Requires=` line was present in the selected output. `ollama.service` is explicitly listed in `After=` and `Wants=`.

## B2/B3 - LoRA readiness benchmark (partial; stopped on compatibility failure)

Benchmark date: 2026-10-27. No real training data or accuracy evaluation was used. The base Llama 3.2 1B Hugging Face weight file was present locally at `/home/hammad/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6/model.safetensors` (2,471,645,608 bytes). The Ollama model `llama3.2:1b` was also listed, but the benchmark used the Hugging Face snapshot.

### Pre-check and environment

The CIFAR evaluator process was present as PID `48755`, in `/system.slice/dfl-slm-master.service`, blocked on its evaluator pipe (`pipe_read`). No CIFAR job/lock files or active training/launch process were found. Its open descriptors included TensorRT/GPU device handles; it was treated as the persistent idle evaluator, not an active CIFAR run.

| State | RAM used / free / available | Swap used / free |
|---|---:|---:|
| Before stopping services | 5,831 / 281 / 1,429 MB | 2,118 / 1,691 MB |
| After stopping services | 2,096 / 4,016 / 5,163 MB | 623 / 3,187 MB |
| Final, after restoration | 3,396 / 1,501 / 3,878 MB | 832 / 2,977 MB |

The root filesystem (`/dev/mmcblk0p1`, also containing `/home`) had 133 GB available before the benchmark.

Both `ollama` and `dfl-slm-master.service` were active before the benchmark, stopped successfully, and were active again afterward. Three post-stop tegrastats samples showed 2,205 MB RAM used and 623 MB swap used. No service state was changed beyond the requested stop/start.

Pre-existing Python package checks:

| Environment | torch | transformers | peft | bitsandbytes | CUDA available |
|---|---|---|---|---|---|
| `/usr/bin/python3` | Not installed | Not installed | Not installed | Not installed | Not applicable |
| `/home/hammad/Downloads/slm/.venv` | 2.4.0a0+3bcc3cddb5.nv24.7 | 4.46.3 | 0.13.2 | Not installed | True |
| `/opt/dfl-slm-dtn/master/venv` | Not installed | Not installed | Not installed | Not installed | Not applicable |

A new venv was created at `/home/hammad/slm-ft-venv`. It used the existing local Jetson wheel `torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp310-cp310-linux_aarch64.whl` and installed:

| Package | Installed version |
|---|---|
| torch | 2.4.0a0+3bcc3cddb5.nv24.7 |
| transformers | 4.46.3 |
| peft | 0.13.2 |
| accelerate | 1.1.1 |
| datasets | 3.1.0 |
| sentencepiece | 0.2.2 |
| numpy | 2.2.6 |

`torch.cuda.is_available()` and `torch.cuda.is_bf16_supported()` both returned `True`; a CUDA FP16 256x256 matrix multiplication completed and returned finite values. NVIDIA's R36.3 / JetPack 6.0 / CUDA 12.2 wheel match documented in B1 is PyTorch 2.3.0; this benchmark instead used the already-local NVIDIA Jetson `2.4.0a0+3bcc3cddb5.nv24.07` wheel, whose CUDA operation was measured working.

### Benchmark measurements

The synthetic JSONL at `/home/hammad/slm-ft-bench/fake_chat_200.jsonl` contains 200 short placeholder chat rows (48,400 bytes). The benchmark used batch 1, gradient accumulation 8, max length 256 with fixed padding, BF16, gradient checkpointing, AdamW, 30 optimizer steps, and the seven discovered Llama linear projection suffixes `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` (excluding `lm_head`). No real-data accuracy claim is made.

| Rank | Peak torch CUDA allocated | Peak system RAM + swap used* | Seconds / step | Padded tokens / second | Status / exit |
|---:|---:|---:|---:|---:|---|
| 8 | 2,852.9 MiB | 5,964 MB (RAM peak 5,211 MB; swap peak 756 MB) | 5.8423 | 350.54 | 30 steps completed; exit 0 |
| 16 | Not measured | 6,064 MB (RAM peak 5,231 MB; swap peak 833 MB) | Not measured | Not measured | Intentionally terminated; exit 143 |
| 32 | Not run | Not measured | Not measured | Not measured | Skipped after the compatibility failure |

\*Combined peak is the maximum per-sample sum of RAM-used and swap-used from tegrastats; separate RAM and swap maxima may occur in different samples.

Before both rank 8 and rank 16, PyTorch emitted this compatibility warning and failed NumPy initialization:

```text
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash.
...
UserWarning: Failed to initialize NumPy: _ARRAY_API not found
```

The rank-8 CUDA training loop nevertheless completed its 30 steps. Because this is a compatibility failure under the stop-on-failure rule, rank 16 was terminated and no retry, package downgrade, rank-32 run, or alternative benchmark was attempted. Rank 16 was intentionally sent SIGTERM (signal 15) and exited 143; this was not an OOM. Rank 8 had no OOM/Killed event.

### Dataset-size estimate and decision

For 1,500 rows x 3 epochs at effective batch 8, using a per-epoch partial final batch: `ceil(1500 / 8) x 3 = 564` optimizer steps. At the measured rank-8 rate of 5.8423 seconds/step, the provisional estimate is **54.9 minutes**. This extrapolation is based only on the synthetic rank-8 run and is not validated for real data or a clean dependency environment.

**Decision: DOES NOT FIT as a completed readiness validation.** This is due to the NumPy ABI/init failure and incomplete rank sweep, not a measured CUDA OOM. The model's r=16/r=32 memory fit remains undetermined. Alternatives such as shorter sequence length, a verified 8-bit optimizer, or PC training followed by GGUF export were not tested or installed.

### Restoration and live verification

The initial immediate readiness curls after systemd reported active returned exact transient errors `curl: (7) Failed to connect to 127.0.0.1 port 11434 after 0 ms: Connection refused` and the same error for port 5003. A subsequent readiness check returned Ollama HTTP 200; authenticated master calls then succeeded.

The authenticated `/chat` prompt `Please retrieve the current master status only; do not change settings or start any work.` returned reply `The current master status is available.` with action `get_status`. The explicit `set_topology` command returned `topology: bidirectional_ring`. A following `get_status` succeeded but reported device health only and did not include topology, so topology was confirmed by the `set_topology` response, not by the `get_status` payload.

Final `systemctl is-active` reported both `ollama` and `dfl-slm-master.service` as `active`. Final `free -m`: RAM total 7,620 MB, used 3,396 MB, free 1,501 MB, available 3,878 MB; swap total 3,810 MB, used 832 MB, free 2,977 MB.

## Real LoRA training attempt - stopped on runtime error

The real run was started on 2026-10-27 using the local Hugging Face snapshot only. It used the validator-accepted subset after the original dataset validation gate did not pass; rejected rows were not used.

| Split | Accepted rows | Unsafe share |
|---|---:|---:|
| Train | 1,254 | 12.60% |
| Validation | 169 | 8.88% |
| Test | 198 | 9.60% |

The accepted test subset had 15 wording/template families not present in the accepted training subset. The accepted data did not meet the earlier 15-20% unsafe-share target. These caveats were known when the user explicitly directed the run to proceed.

| Setting | Value |
|---|---|
| Base | `meta-llama/Llama-3.2-1B-Instruct`, cached snapshot, `local_files_only=True` |
| LoRA | r=8, alpha=16, dropout=0.05; `down_proj`, `gate_proj`, `k_proj`, `o_proj`, `q_proj`, `up_proj`, `v_proj` |
| Sequence length | 384; no truncation (train maximum 306, validation maximum 285) |
| Batch / accumulation / effective batch | 1 / 8 / 8 |
| Epochs / expected optimizer steps | 3 / 471 |
| Precision / optimizer | BF16 / AdamW |
| Gradient checkpointing | Enabled |
| Loss mask | System and user tokens masked with -100; assistant JSON target tokens trained |

**Outcome: training failed before the first optimizer step.** The exact exception was:

```text
ValueError: Asking to pad but the tokenizer does not have a padding token. Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`.
```

| Measurement | Result |
|---|---:|
| Completed optimizer steps | 0 |
| Training duration / seconds per step / tokens per second | Not measured |
| Peak CUDA allocated memory | Not measured |
| Peak RAM used | 4,019 MB |
| Peak swap used | 841 MB |
| Peak RAM + swap used | 4,860 MB |
| Run report process exit status | 0 (the wrapper recorded the caught training failure) |
| SSH command exit status | 1 |
| Adapter / trained weights | Not produced |

No training-time estimate is reported because no optimizer step completed. Validation/test loss and accuracy were not measured; no accuracy claim is made. The remote run report and logs are under `/home/hammad/slm-ft-outputs/llama32-1b-slm-20261027T195000/`.

The restoration trap ran: `ollama` and `dfl-slm-master.service` both reported `active`. Ollama readiness returned HTTP 200; an unauthenticated master readiness request returned HTTP 401, after which the authenticated status request succeeded with action `get_status`. The topology-setting response confirmed `bidirectional_ring`. Final `free -m`: RAM 7,620 MB total, 3,336 MB used, 644 MB free, 3,944 MB available; swap 3,810 MB total, 840 MB used, 2,969 MB free.

### Pad-fix retry - training completed; test evaluation stopped on schema error

The first retry attempt stopped before training because service control required `sudo`; the exact error was `Failed to stop dfl-slm-master.service: Interactive authentication required.` No services were stopped in that attempt. The training retry then used `sudo -n` for service control and fixed the padding failure by setting a missing tokenizer pad token to its EOS token and setting the model config's pad-token ID accordingly.

The training phase completed using the same local-only rank-8 configuration and accepted train/validation rows described above. The trainer's last saved checkpoint was `checkpoint-468` (global step 468, epoch 2.9856459); the training log reported 3,421.8721 seconds runtime, 0.137 steps/second, and train loss 0.0327417. The derived runtime per step is approximately 7.31 seconds. Validation evaluation losses were 0.0101505, 0.00240084, and 0.00209377 at the three epoch evaluations.

The adapter was saved before test evaluation failed. Adapter path on the master: `/home/hammad/slm-ft-outputs/llama32-1b-slm-20261027T201700-padfix-sudo/adapter/`; `adapter_model.safetensors` is 22,573,704 bytes and the directory is 39 MB. The test split has 198 rows, but those JSONL objects contain `label` and `params`, not `target`. The existing test-evaluation code indexes `row['target']` and raised the exact error `KeyError: 'target'`. Per the stop-on-error instruction, no test-data conversion or evaluation retry was performed. Test loss/metrics and accuracy are therefore unavailable; no accuracy claim is made. The run's `training_report.json` records this post-training error, so it does not contain the completed training metrics.

Measured tegrastats peaks for the run: RAM used 6,834 MB; swap used 1,161 MB; maximum per-sample RAM-plus-swap used 7,931 MB. Peak CUDA allocated memory was not persisted after the later test-schema exception. Both `ollama` and `dfl-slm-master.service` were restored and active; readiness returned Ollama HTTP 200 and master HTTP 401 without auth, then authenticated `set_topology` and `get_status` commands both succeeded. Final `free -m`: RAM 7,620 MB total, used 1,928 MB, free 4,755 MB, available 5,406 MB; swap 3,810 MB total, used 1,034 MB, free 2,776 MB.

### Reply-style LoRA run - completed on 2026-11-02

The supplied train/validation/test splits contained 980/209/211 conversations. The run fine-tuned only natural-language reply style with LoRA rank 8 for three epochs, using the locally cached Llama 3.2 1B Instruct model. Training took 2,262.14 seconds and reported a training loss of 0.10970. The adapter was saved at `/home/hammad/slm-ft-outputs/reply-style-20261102T2158JST/adapter/` (about 39 MB); the complete report is alongside it in `training_report.json`.

| Metric | Base model | Fine-tuned |
|---|---:|---:|
| Validation loss | 3.87175 | 0.0001135 |
| Held-out test loss | 3.87391 | 0.0001149 |

A greedy generation-format probe on 12 held-out prompts produced no empty or JSON-shaped responses (mean 292.8 characters). The first probe attempt exposed a Jetson `torch.distributed.Logger` compatibility error; the adapter had already been saved and both held-out evaluations had completed. The probe was rerun with `synced_gpus=False`, and the trainer now sets an explicit attention mask for this check.

These extremely low held-out losses are a signal to inspect examples and label repetition before making a quality claim. The format probe checks only emptiness and JSON shape, not whether recommendations are factually correct or sufficiently natural. The adapter is for prose replies only; the active SLM action contract remains unchanged. Final `free -m`: RAM 7,620 MB total, used 1,928 MB, free 4,755 MB, available 5,406 MB; swap 3,810 MB total, used 1,034 MB, free 2,776 MB.

The full machine-readable training and evaluation report is saved in [reply-style-lora-evaluation.json](./reply-style-lora-evaluation.json). It includes base/tuned validation and test loss, generation-format probe, training throughput and duration, step count, and adapter sizes. Accuracy, precision, recall and F1 are not included because this is next-token language generation, not a classification model. Peak CUDA allocation was not recorded because the original run stopped during its post-training probe.

### DTN integration

The LoRA is integrated as a separate, authenticated, loopback-only reply service on the master. The existing Ollama model continues to produce the schema-validated action; the adapter cannot choose or execute actions. The DTN backend supplies the validated action context and stored MNIST summary. Its generated text is shown separately as an experimental draft; the original action-model response remains primary.

A live probe supplied completed MNIST results with 81% validation accuracy, but the LoRA draft claimed “low-quality results with high variance.” That claim was unsupported by the supplied context. Therefore the draft is explicitly labeled for verification and is not used as the authoritative assistant answer. The low loss/perplexity and format probe do not demonstrate factuality; a broader grounded-answer evaluation is needed before promoting the LoRA output.
