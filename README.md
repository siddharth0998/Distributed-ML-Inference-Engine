# Distributed-ML-Inference-Engine

This project serves a DistilBERT sentiment model from a C++ gRPC server using LibTorch.

## What is implemented

- C++ gRPC inference server with dynamic batching.
- C++ tokenizer that reads `models/vocab.txt`.
- Python model export script (`export_model.py`) that creates TorchScript `model.pt`.
- Python client (`client.py`) and load test (`load_test.py`).

## Project structure

- `src/main.cpp`: gRPC server, worker pool, dynamic batching, tokenizer integration.
- `src/SafeQueue.hpp`: thread-safe queue used by worker threads.
- `src/Tokenizer.hpp`: simple BERT-style tokenizer (lowercase, punctuation strip, vocab lookup).
- `src/inference.proto`: gRPC request/response contract.
- `CMakeLists.txt`: C++ build config and proto code generation.
- `Dockerfile`: containerized build and runtime command.
- `export_model.py`: exports Hugging Face model to TorchScript.
- `client.py`: quick functional client.
- `load_test.py`: concurrent load test script.
- `models/`: expected runtime model assets (`model.pt`, `vocab.txt`).

## Prerequisites

1. Docker Desktop running (recommended path).
2. Python 3.10+ for exporter/client/load test.
3. Python Virtual Environment (Highly Recommended):
   To avoid version conflicts with other projects, it is highly recommended to use an isolated environment.

```bash
# 1. Create the virtual environment
python -m venv venv

# 2. Activate it
source venv/bin/activate

# 3. Install all exact required packages
pip install -r requirements.txt
```

## Quick start (Docker)

Run from repo root:

```bash
cd /Users/siddharthpatel/Distributed-ML-Inference-Engine
```

### 1) Export model (if you do not already have `models/model.pt`)

```bash
python export_model.py
mv model.pt models/model.pt
```

`models/vocab.txt` must also exist (already present in this repo).

### 2) Build image

```bash
docker build -t ml-engine .
```

### 3) Run server container

Model assets are already bundled into the image during build.

```bash
docker run -p 50051:50051 ml-engine
```

### 4) Test with simple client

In a second terminal, generate the Python gRPC stubs first, then run the client:

```bash
cd /Users/siddharthpatel/Distributed-ML-Inference-Engine
sh generate_python_protos.sh
python client.py
```

### 5) Run load test

```bash
cd /Users/siddharthpatel/Distributed-ML-Inference-Engine
python load_test.py
```

Note: `load_test.py` downloads SST-2 from Hugging Face, so internet is required.

## Optional local (non-Docker) run

You can build locally, but you need local installs of:

- CMake
- gRPC C++ libs
- Protobuf compiler + grpc plugin
- LibTorch

Build:

```bash
mkdir -p build
cd build
cmake ..
make -j
```

Run:

```bash
./Distributed-ML-Inference-Engine /absolute/path/to/model.pt /absolute/path/to/vocab.txt
```

## Common issues

### Docker daemon/socket errors

If you see errors about `docker.sock`, Docker Desktop is not running, wrong context is active, or socket permissions are broken.

Try:

```bash
open -a Docker
unset DOCKER_HOST
docker context use desktop-linux
docker info
```

### `ModuleNotFoundError: inference_pb2`

Run Python scripts from repo root:

```bash
cd /Users/siddharthpatel/Distributed-ML-Inference-Engine
python load_test.py
```

### gRPC connection errors in client/load test

Make sure server container is running and listening on `localhost:50051` before starting client/load test.

### Sharing with Windows users

This image is self-contained (binary + model + vocab), so your friend only needs:

```bash
docker build -t ml-engine .
docker run -p 50051:50051 ml-engine
```
