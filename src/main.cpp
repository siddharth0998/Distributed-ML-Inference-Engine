#include <torch/script.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <future>
#include <chrono>

#include <grpcpp/grpcpp.h>
#include "inference.grpc.pb.h"
#include "SafeQueue.hpp"
#include "Tokenizer.hpp"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using inference::InferenceEngine;
using inference::PredictRequest;
using inference::PredictResponse;

struct InferenceTask {
    std::vector<torch::jit::IValue> inputs;
    std::promise<std::vector<float>> result_promise;
};

class InferenceWorkerPool {
private:
    SafeQueue<std::shared_ptr<InferenceTask>> queue_;
    std::vector<std::thread> workers_;
    torch::jit::script::Module module_;

    void WorkerLoop(int worker_id) {
        std::cout << "Worker Thread [" << worker_id << "] started." << std::endl;
        
        // Define our Dynamic Batching parameters
        const int MAX_BATCH_SIZE = 20;
        const auto MAX_WAIT_TIME = std::chrono::milliseconds(10);

        while (true) {
            std::vector<std::shared_ptr<InferenceTask>> batch;
            std::shared_ptr<InferenceTask> first_task;

            // 1. Wait indefinitely for the VERY FIRST request
            if (!queue_.pop(first_task)) break; // Queue shut down
            batch.push_back(first_task);

            // 2. Start the timer and gather more requests
            auto start_time = std::chrono::steady_clock::now();
            
            while (batch.size() < MAX_BATCH_SIZE) {
                auto elapsed = std::chrono::steady_clock::now() - start_time;
                if (elapsed >= MAX_WAIT_TIME) break; // Time's up!

                std::shared_ptr<InferenceTask> next_task;
                if (queue_.pop_for(next_task, MAX_WAIT_TIME - elapsed)) {
                    batch.push_back(next_task);
                } else {
                    break; // Queue was empty for the remaining time
                }
            }

            std::cout << "[Worker " << worker_id << "] Stacked " << batch.size() << " requests into a single matrix. Running inference..." << std::endl;


            // 3. Stack the Tensors using torch::cat
            std::vector<torch::Tensor> input_ids_list;
            std::vector<torch::Tensor> attention_mask_list;
            
            for (auto& task : batch) {
                input_ids_list.push_back(task->inputs[0].toTensor());
                attention_mask_list.push_back(task->inputs[1].toTensor());
            }

            // Concatenate along dimension 0 (the batch dimension)
            torch::Tensor batched_input_ids = torch::cat(input_ids_list, 0);
            torch::Tensor batched_attention_mask = torch::cat(attention_mask_list, 0);

            std::vector<torch::jit::IValue> batched_inputs;
            batched_inputs.push_back(batched_input_ids);
            batched_inputs.push_back(batched_attention_mask);

            // 4. Run the Neural Network ONCE for the entire batch
            try {
                auto output_tuple = module_.forward(batched_inputs).toTuple();
                at::Tensor raw_logits = output_tuple->elements()[0].toTensor();

                // --- THE 1 LINE OF SOFTMAX MATH ---
                // "dim=1" means apply the formula horizontally across the two classes
                at::Tensor probabilities = torch::softmax(raw_logits, /*dim=*/1);

                // 5. Scatter the results back to the correct users
                for (size_t i = 0; i < batch.size(); ++i) {
                    std::vector<float> final_scores = {
                        probabilities[i][0].item<float>(), // Negative %
                        probabilities[i][1].item<float>()  // Positive %
                    };
                    batch[i]->result_promise.set_value(final_scores);
                }
            } catch (const std::exception& e) {
                std::cerr << "Batch inference failed: " << e.what() << std::endl;
                for (auto& task : batch) {
                    task->result_promise.set_exception(std::current_exception());
                }
            }
        }
    }

public:
    InferenceWorkerPool(const std::string& model_path, int num_threads) {
        module_ = torch::jit::load(model_path);
        module_.eval();
        for (int i = 0; i < num_threads; ++i) {
            workers_.emplace_back(&InferenceWorkerPool::WorkerLoop, this, i);
        }
    }

    ~InferenceWorkerPool() {
        queue_.shutdown();
        for (auto& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }

    void SubmitTask(std::shared_ptr<InferenceTask> task) {
        queue_.push(task);
    }
};

class InferenceServiceImpl final : public InferenceEngine::Service {
private:
    InferenceWorkerPool& pool_;
    BertTokenizer tokenizer_;

public:
    // Initialize the tokenizer with our vocab file
    InferenceServiceImpl(InferenceWorkerPool& pool, const std::string& vocab_path) 
        : pool_(pool), tokenizer_(vocab_path) {}

    Status Predict(ServerContext* context, const PredictRequest* request, PredictResponse* reply) override {
        auto task = std::make_shared<InferenceTask>();
        auto future = task->result_promise.get_future();

        // 1. USE THE C++ TOKENIZER!
        std::vector<int64_t> input_ids_vec;
        std::vector<int64_t> attention_mask_vec;
        tokenizer_.encode(request->text(), input_ids_vec, attention_mask_vec);

        // 2. Convert standard C++ vectors into PyTorch Tensors
        auto opts = torch::TensorOptions().dtype(torch::kLong);
        torch::Tensor input_ids = torch::from_blob(input_ids_vec.data(), {1, 12}, opts).clone();
        torch::Tensor attention_mask = torch::from_blob(attention_mask_vec.data(), {1, 12}, opts).clone();

        task->inputs.push_back(input_ids);
        task->inputs.push_back(attention_mask);

        // 3. Submit to the Dynamic Batching queue
        pool_.SubmitTask(task);

        try {
            std::vector<float> results = future.get();
            reply->add_logits(results[0]);
            reply->add_logits(results[1]);
            reply->set_status("Success");
            return Status::OK;
        } catch (const std::exception& e) {
            reply->set_status("Error during inference");
            return Status::CANCELLED;
        }
    }
};

void RunServer(const char* model_path, const char* vocab_path) {
    std::string server_address("0.0.0.0:50051");
    
    InferenceWorkerPool pool(model_path, 4);
    InferenceServiceImpl service(pool, vocab_path);

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Dynamic Batching Engine with Tokenizer listening on " << server_address << std::endl;
    server->Wait();
}

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./Distributed-ML-Inference-Engine <path-to-model> <path-to-vocab>\n";
        return -1;
    }
    RunServer(argv[1], argv[2]);
    return 0;
}