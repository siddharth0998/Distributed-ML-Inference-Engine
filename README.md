# Distributed-ML-Inference-Engine

Distributed-ML-Inference-Engine is a high-throughput sentiment analysis inference service. It serves a TorchScript-exported DistilBERT model from a native C++ gRPC server using LibTorch, accepts text over a `Predict` RPC, tokenizes the text in C++, batches concurrent requests, runs model inference, and returns negative/positive confidence scores.

The goal of the project is to show how a Python-trained transformer model can be exported and served from a production-style C++ backend with lower runtime overhead than a typical Python web service.

## What this project does

1. Exports Hugging Face `distilbert-base-uncased-finetuned-sst-2-english` to TorchScript with `export_model.py`.
2. Starts a C++ gRPC inference server on `0.0.0.0:50051`.
3. Loads the TorchScript model (`models/model.pt`) and vocabulary (`models/vocab.txt`) at startup.
4. Tokenizes incoming text with a lightweight C++ BERT-style tokenizer.
5. Queues incoming requests and dynamically batches them for better throughput.
6. Runs batched inference with LibTorch and applies softmax to produce class probabilities.
7. Returns two scores in the gRPC response: negative probability and positive probability.

At a high level:

```text
Python/Hugging Face model
        |
        v
TorchScript model.pt
        |
        v
C++ gRPC server + LibTorch
        |
        v
Dynamic batching worker pool
        |
        v
Sentiment scores returned to Python or any gRPC client
```

## What is implemented

- C++ gRPC inference server listening on port `50051`.
- Dynamic batching worker pool for grouping concurrent requests.
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

## Quick start (Native Apple Silicon)

Run from repo root:

```bash
cd /Users/siddharthpatel/Distributed-ML-Inference-Engine
```

### 1) Export model (if you do not already have `models/model.pt`)

```bash
source venv/bin/activate
python export_model.py
mv model.pt models/model.pt
```

`models/vocab.txt` must also exist (already present in this repo).

### 2) Install System Dependencies and LibTorch

```bash
# Install cmake, grpc, and protobuf via Homebrew
brew install cmake grpc protobuf pkg-config

# Download and unzip the macOS ARM64 LibTorch build
sh setup_mac_native.sh
```

### 3) Build Engine Natively

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_PREFIX_PATH=$PWD/../libtorch
make -j
```

### 4) Run server natively

```bash
./Distributed-ML-Inference-Engine ../models/model.pt ../models/vocab.txt
```

### 5) Test with simple client

In a second terminal, activate your virtual environment, generate the Python gRPC stubs first, then run the client:

```bash
cd /Users/siddharthpatel/Distributed-ML-Inference-Engine
source venv/bin/activate
sh generate_python_protos.sh
python client.py
```

### 6) Run load test

```bash
python load_test.py
```

Note: `load_test.py` downloads SST-2 from Hugging Face, so internet is required.

## Optional Docker Run

If you wish to run the engine inside a Linux container (note: this uses Rosetta emulation on Apple Silicon and is much slower):

```bash
docker build -t ml-engine .
docker run -p 50051:50051 ml-engine
```

## Benchmark results

The following load test was run with `load_test.py` using the SST-2 validation split, 872 total requests, and 50 concurrent users.

| Runtime | Total time | Throughput | Average latency | P50 latency | P90 latency | P99 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Docker on Apple Silicon (`linux/amd64`) | 5.65 sec | 154.24 RPS | 317.85 ms | 256.14 ms | 494.08 ms | 1364.03 ms |
| Native macOS ARM64 build with CMake | 1.45 sec | 602.60 RPS | 81.28 ms | 76.10 ms | 139.62 ms | 176.29 ms |

### Performance note

The Docker benchmark above was run on Apple Silicon using a `linux/amd64` container image. Because this container does not run as a native ARM64 binary on the Mac, Docker executes the x86_64 Linux build through emulation, which adds overhead and makes inference significantly slower.

For best performance on Apple Silicon, build and run the engine natively with CMake and the macOS ARM64 LibTorch build. In native mode, the server uses the Mac's ARM64 CPU architecture directly, which produced about 3.9x higher throughput and much lower latency in this test.

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
