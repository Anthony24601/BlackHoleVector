import numpy as np
import time
import psutil
import os

def generate_high_dimensional_vectors(num_vectors, dimensions):
    return np.random.randn(num_vectors, dimensions)

def find_top_closest_vectors(dataset, new_vector, top_n=15):
    dot_products = np.dot(dataset, new_vector)
    closest_indices = np.argsort(dot_products)[-top_n:][::-1]
    return closest_indices

def profile_execution(dataset, num_runs, dimensions):
    process = psutil.Process(os.getpid())
    
    # Profiling variables
    times = []
    cpu_percentages = []
    memory_usages = []

    for i in range(num_runs):
        print(i)
        # Start time
        start_time = time.time()

        # Create a new vector and find closest vectors
        new_vector = np.random.randn(dimensions)
        closest_vectors_indices = find_top_closest_vectors(dataset, new_vector)

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