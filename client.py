import grpc
import inference_pb2
import inference_pb2_grpc
import concurrent.futures
import time

def send_request(stub, text_input):
    request = inference_pb2.PredictRequest(text=text_input)
    try:
        start_time = time.time()
        response = stub.Predict(request)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        
        # Extract the percentages from the C++ response
        negative_pct = response.logits[0] * 100
        positive_pct = response.logits[1] * 100
        
        print(f"Text: '{text_input}'")
        print(f"  -> Negative: {negative_pct:.2f}% | Positive: {positive_pct:.2f}%")
        print(f"  -> Latency:  {latency_ms:.2f} ms\n")
        
    except grpc.RpcError as e:
        print(f"gRPC Error: {e.code()} - {e.details()}")

def run():
    print("Connecting to the C++ Engine...")
    channel = grpc.insecure_channel('localhost:50051')
    stub = inference_pb2_grpc.InferenceEngineStub(channel)

    # A mix of positive and negative reviews
    test_sentences = [
        "I absolutely love this product, it is amazing!",
        "This is terrible, I hate it so much.",
        "Best purchase I have ever made. Highly recommend.",
        "Worst experience ever, it crashed immediately.",
        "The movie was incredibly boring and a waste of time.",
        "Fast shipping, great customer service, five stars!"
    ]

    print(f"\n--- Firing {len(test_sentences)} real reviews at the engine ---\n")
    
    # Fire all of them concurrently to trigger the Dynamic Batching
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(test_sentences)) as executor:
        for sentence in test_sentences:
            executor.submit(send_request, stub, sentence)

if __name__ == '__main__':
    run()