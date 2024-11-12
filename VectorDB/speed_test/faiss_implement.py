import faiss
import numpy as np
import time
import psutil
import os

def create_faiss_index(dimensions, euclidean = True): # Dot Product if false
    if euclidean:
        return faiss.IndexFlatL2(dimensions)
    return faiss.IndexFlatIP(dimensions)

def insert_vectors_into_faiss(index, dataset):
    index.add(dataset)
    print(f"Inserted {index.ntotal} vectors into the FAISS index.")

def search_faiss_index(index, query_vector, top_n=15):
    distances, indices = index.search(query_vector.reshape(1, -1), top_n)
    
    """print(f"Top {top_n} closest vectors:")
    for i in range(top_n):
        print(f"Index: {indices[0][i]}, Distance: {distances[0][i]}")"""
    
    return distances, indices

def profile_faiss_search(index, num_runs, dimensions):
    process = psutil.Process(os.getpid())
    
    times = []
    cpu_percentages = []
    memory_usages = []

    for i in range(num_runs):
        print(i)

        # Start time
        start_time = time.time()

        # Create a new random vector and perform the search
        new_vector = np.random.randn(dimensions).astype(np.float32)
        search_faiss_index(index, new_vector)

        # End time and profiling
        end_time = time.time()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)

        # Record CPU and memory usage
        cpu_percent = process.cpu_percent(interval=None)
        memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        cpu_percentages.append(cpu_percent)
        memory_usages.append(memory_usage)

    # Summary
    print(f"Average time per run: {np.mean(times):.4f} seconds")
    print(f"Average CPU usage per run: {np.mean(cpu_percentages):.2f}%")
    print(f"Average memory usage per run: {np.mean(memory_usages):.2f} MB")

