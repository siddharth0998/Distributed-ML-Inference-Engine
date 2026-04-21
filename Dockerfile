FROM --platform=linux/amd64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install C++, LibTorch dependencies, AND gRPC/Protobuf compilers
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    unzip \
    pkg-config \
    libgrpc++-dev \
    protobuf-compiler-grpc \
    protobuf-compiler \
    libprotobuf-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download LibTorch CPU build (x86_64)
WORKDIR /opt
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.2%2Bcpu.zip -O libtorch.zip && \
    unzip libtorch.zip && \
    rm libtorch.zip

ENV CMAKE_PREFIX_PATH=/opt/libtorch

WORKDIR /app
COPY CMakeLists.txt .
COPY src/ src/
COPY models/ models/

# Build the project
RUN mkdir build && cd build && \
    cmake .. && \
    make

# Validate required model assets at build time.
RUN test -f /app/models/model.pt && test -f /app/models/vocab.txt

# Expose standard gRPC port 50051
EXPOSE 50051

CMD ["./build/Distributed-ML-Inference-Engine", "/app/models/model.pt", "/app/models/vocab.txt"]
