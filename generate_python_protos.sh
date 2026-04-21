#!/bin/bash
# Generate gRPC Python stubs from the proto file
# This avoids checking generated files into version control

echo "Generating Python gRPC stubs from src/inference.proto..."

python -m grpc_tools.protoc -I ./src --python_out=. --grpc_python_out=. ./src/inference.proto

if [ $? -eq 0 ]; then
    echo "Successfully generated: inference_pb2.py and inference_pb2_grpc.py"
else
    echo "Error: Failed to generate Python stubs. Make sure grpcio-tools is installed (pip install grpcio-tools)."
    exit 1
fi
