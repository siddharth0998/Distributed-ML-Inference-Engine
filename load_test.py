import grpc
import inference_pb2
import inference_pb2_grpc
import concurrent.futures
import time
import numpy as np
from datasets import load_dataset

def send_request(stub, text_input):
    request = inference_pb2.PredictRequest(text=text_input)
    start_time = time.time()
    try:
        response = stub.Predict(request)
        end_time = time.time()
        
        # Calculate percentages
        neg_pct = response.logits[0] * 100
        pos_pct = response.logits[1] * 100
        prediction = "Positive" if pos_pct > neg_pct else "Negative"
        
        return (end_time - start_time), prediction, text_input
    except grpc.RpcError as e:
        return None, "Error", text_input

def run_load_test():
    print("Fetching SST-2 Dataset from Hugging Face...")
    # Load the validation split of SST-2
    dataset = load_dataset("sst2", split="validation")
    
    # Grab the first 1,000 sentences for our load test
    test_sentences = dataset['sentence'][:1000]
    total_requests = len(test_sentences)
    
    print("Connecting to the C++ Engine...")
    channel = grpc.insecure_channel('localhost:50051')
    stub = inference_pb2_grpc.InferenceEngineStub(channel)

    # CONCURRENCY LEVEL: How many users are hitting the server at the exact same millisecond?
    # We set this to 50 to heavily stress test your Dynamic Batching SafeQueue.
    concurrent_users = 50 
    
    print(f"\n🚀 FIRING {total_requests} REQUESTS ({concurrent_users} concurrent users)...\n")
    
    latencies = []
    
    start_total = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        # Map the sentences to the executor
        futures = [executor.submit(send_request, stub, sentence) for sentence in test_sentences]
        
        for future in concurrent.futures.as_completed(futures):
            latency, pred, text = future.result()
            if latency is not None:
                latencies.append(latency)

    end_total = time.time()
    total_time = end_total - start_total
    rps = total_requests / total_time

    # --- Print Professional Backend Metrics ---
    print("---------------------------------------------------")
    print("📊 LOAD TEST RESULTS")
    print("---------------------------------------------------")
    print(f"Total Requests Processed : {len(latencies)}")
    print(f"Total Time Elapsed       : {total_time:.2f} seconds")
    print(f"Throughput (RPS)         : {rps:.2f} Requests Per Second\n")
    
    # Convert latencies to milliseconds
    latencies_ms = np.array(latencies) * 1000
    print(f"Average Latency          : {np.mean(latencies_ms):.2f} ms")
    print(f"P50 Latency (Median)     : {np.percentile(latencies_ms, 50):.2f} ms")
    print(f"P90 Latency              : {np.percentile(latencies_ms, 90):.2f} ms")
    print(f"P99 Latency (Tail)       : {np.percentile(latencies_ms, 99):.2f} ms")
    print("---------------------------------------------------")

if __name__ == '__main__':
    run_load_test()