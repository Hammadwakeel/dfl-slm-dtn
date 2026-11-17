# MNIST Model Studio

Model Studio builds and distributes a validated architecture to all three workers before training. The UI supports a configurable Keras layer stack (convolutions, pooling, normalization, dropout, dense layers, patch embeddings, and transformer encoder blocks) and the curated `facebook/deit-tiny-patch16-224` pretrained vision transformer. Custom models must end with a linear Dense classifier sized to the dataset classes. Designs are limited to 32 layers and 20 million parameters.

The selected optimizer, learning rate, and batch size are stored with the installed architecture. Round count and local epoch count are selected when training starts. The coordinator and workers independently validate the model and dataset metadata; workers must report identical initialized-weight hashes before distributed training can begin.

## Removing saved results and installed models

The Experiments page can permanently remove a completed experiment group, including NAS runs; the Architecture Search page also offers deletion for its latest NAS result. Active groups cannot be removed. Model Studio can remove the installed MNIST model from the coordinator and all three workers while training is idle. Worker-side checks prevent deletion from racing an active or scheduled training run. If a worker cannot be reached during model deletion, the UI reports any workers already cleared and keeps the coordinator copy so deletion can be retried.

The worker deployment installs TensorFlow 2.15-compatible `transformers==4.44.2` in a virtual environment that shares the system TensorFlow runtime, without the TensorFlow extra. DeiT-Tiny's first initialization downloads its public TensorFlow checkpoint from Hugging Face into `/var/lib/dfl-slm-dtn/huggingface`; each worker therefore needs outbound access to `huggingface.co` at initialization. The regular custom Keras model path does not require that download.

## MNIST architecture search

The Architecture Search page runs the bounded CNN candidate search on all three MNIST worker shards. Per-worker candidate scores appear as each worker finishes; combined validation scores and the selected architecture appear when all workers finish. Candidate selection aggregates worker validation accuracy and parameter count; the official MNIST test split is reserved for final evaluation. After the search completes, **Use selected architecture in Model Studio** loads the winning configuration into the normal model installation and DFL training flow.

## DFL topology optimization

The Topology page's **Run 5-topology sweep** currently evaluates five worker communication graphs using Dirichlet `alpha=0.3` and three seeds (15 runs). The optimizer scores worker validation accuracy, convergence, peer communication, and measured peer-wait time; runtime is reported separately and the official test split is not used for selection. Older topology results keep their original runtime-based reward and are labeled as legacy.

The separate APG workbench is a PC-only simulation. It is not a real-worker topology-learning run and is labeled accordingly in the UI.
