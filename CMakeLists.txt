cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Distributed-ML-Inference-Engine)

# 1. Find LibTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 2. Find gRPC and Protobuf Compilers
find_program(PROTOC protoc)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)

# 3. Setup folders for the generated C++ code
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(PROTO_SRC ${CMAKE_CURRENT_BINARY_DIR}/generated/inference.pb.cc)
set(PROTO_HDR ${CMAKE_CURRENT_BINARY_DIR}/generated/inference.pb.h)
set(GRPC_SRC ${CMAKE_CURRENT_BINARY_DIR}/generated/inference.grpc.pb.cc)
set(GRPC_HDR ${CMAKE_CURRENT_BINARY_DIR}/generated/inference.grpc.pb.h)

# 4. Command to compile the .proto file into C++ code
add_custom_command(
    OUTPUT ${PROTO_SRC} ${PROTO_HDR} ${GRPC_SRC} ${GRPC_HDR}
    COMMAND ${PROTOC}
    ARGS --grpc_out=${CMAKE_CURRENT_BINARY_DIR}/generated
         --cpp_out=${CMAKE_CURRENT_BINARY_DIR}/generated
         --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
         -I ${CMAKE_CURRENT_SOURCE_DIR}/src
         ${CMAKE_CURRENT_SOURCE_DIR}/src/inference.proto
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/inference.proto
)

# Tell CMake to include the generated files
include_directories(${CMAKE_CURRENT_BINARY_DIR}/generated)

# 5. Build our executable using main.cpp AND the generated gRPC files
add_executable(Distributed-ML-Inference-Engine 
    src/main.cpp 
    ${PROTO_SRC} 
    ${GRPC_SRC}
)

# 6. Link PyTorch, gRPC, and Protobuf libraries
target_link_libraries(Distributed-ML-Inference-Engine 
    "${TORCH_LIBRARIES}" 
    grpc++ 
    grpc 
    protobuf
)
set_property(TARGET Distributed-ML-Inference-Engine PROPERTY CXX_STANDARD 17)