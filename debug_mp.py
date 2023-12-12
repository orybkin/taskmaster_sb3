import multiprocessing
import time
import random

# Task for each process: Calculate the sum of a large number of random numbers
def compute_sum(n):
    total_sum = sum(random.random() for _ in range(n))
    return total_sum

def run_parallel(n, num_processes):
    tasks = [n // num_processes] * num_processes
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(compute_sum, tasks)
    return results

def main():
    n = 20000000

    # Running tasks in parallel
    start_time = time.time()
    run_parallel(n, 50)
    print(f"Parallel execution time @ 50: {time.time() - start_time:.4f} seconds")

    # Running tasks in parallel
    start_time = time.time()
    run_parallel(n, 20)
    print(f"Parallel execution time @ 20: {time.time() - start_time:.4f} seconds")

    # Running tasks in parallel
    start_time = time.time()
    run_parallel(n, 5)
    print(f"Parallel execution time @ 5: {time.time() - start_time:.4f} seconds")

    # Running tasks sequentially
    start_time = time.time()
    compute_sum(n)
    print(f"Sequential execution time: {time.time() - start_time:.4f} seconds")

if __name__ == "__main__":
    main()
