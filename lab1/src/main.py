import time
from itertools import cycle
from typing import Literal, Callable

from matplotlib import pyplot as plt

from src.bubblesort import bubble_sort
from src.generator import generate_array
from src.quicksort import quick_sort
from src.shellsort import shell_sort


def run_experiment(
        algorithms: list[Callable],
        sizes: list[int],
        array_type: Literal['sorted', 'reverse', 'random']) -> dict[str, list[float]]:
    results = {alg.__name__: [] for alg in algorithms}

    print(f"--- Running experiment for '{array_type}' arrays ---")

    for size in sizes:
        print(f"Benchmarking size: {size}...")
        base_data = generate_array(size, array_type)

        for alg in algorithms:
            data_copy = base_data.copy()

            start_time = time.perf_counter()
            alg(data_copy)
            end_time = time.perf_counter()

            elapsed_time = end_time - start_time
            results[alg.__name__].append(elapsed_time)

    return results


def plot_results(results: dict[str, list[float]], sizes: list[int], title: str):
    plt.figure(figsize=(10, 6))

    marker_types = ['o', 's', '^', 'd', 'v', '<', '>', 'p', '*', 'h']
    marker_cycle = cycle(marker_types)

    for (alg_name, times) in results.items():
        plt.plot(sizes, times, marker=next(marker_cycle), linewidth=2, label=alg_name)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Array Size (Number of Elements)", fontsize=12)
    plt.ylabel("Execution Time (Seconds)", fontsize=12)

    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.show()


def main():
    algorithms_to_test = [bubble_sort, shell_sort, quick_sort]
    data_sizes = [100, 500, 1000, 2000, 3000, 5000, 10000]
    for data_type in ['sorted', 'reverse', 'random']:
        experiment_results = run_experiment(
            algorithms=algorithms_to_test,
            sizes=data_sizes,
            array_type=data_type
        )

        plot_results(
            results=experiment_results,
            sizes=data_sizes,
            title=f"Sorting Algorithm Performance ({data_type.capitalize()} Data)"
        )


if __name__ == "__main__":
    main()
