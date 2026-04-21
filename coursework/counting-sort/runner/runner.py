import os
import json
import subprocess
import matplotlib.pyplot as plt

# --- Experiment Configurations ---
TRIES = 5
METRICS_FILE = "data/metrics.json"
MASTER_METRICS_FILE = "data/master_metrics.json"

# Baseline defaults for keeping variables constant while scaling another
DEFAULT_P = 4
DEFAULT_ELEMENTS = 1_000_000
DEFAULT_MAX_VAL = 100

# Variables to iterate through
P_VALUES = [1, 2, 4, 6, 8, 10]
ELEMENTS_VALUES = [1_000_000, 2_500_000, 5_000_000, 7_500_000, 10_000_000]
MAX_VAL_VALUES = [10, 100, 1000, 10_000, 100_000, 1_000_000, 10_000_000]


def run_single_experiment(p, elements, max_val):
    """Runs a single MPI configuration and returns the generated metrics."""
    if os.path.exists(METRICS_FILE):
        os.remove(METRICS_FILE)

    print(f"Running p={p}, elements={elements}, max_val={max_val}...")
    env = os.environ.copy()

    cmd = [
        "mpiexec", "-np", str(p),
        "../target/debug/counting-sort",
        "--elements", str(elements),
        "--max-val", str(max_val),
        "--tries", str(TRIES)
    ]
    subprocess.run(cmd, check=True, env=env)

    if not os.path.exists(METRICS_FILE):
        print(f"Warning: Metrics file not found after running p={p}.")
        return []

    with open(METRICS_FILE, 'r') as f:
        return json.load(f)


def run_all_experiments():
    """Generates the required configurations and runs them."""
    configs = set()

    for p in P_VALUES:
        configs.add((p, DEFAULT_ELEMENTS, DEFAULT_MAX_VAL))
    for el in ELEMENTS_VALUES:
        configs.add((DEFAULT_P, el, DEFAULT_MAX_VAL))
    for mv in MAX_VAL_VALUES:
        configs.add((DEFAULT_P, DEFAULT_ELEMENTS, mv))

    master_data = []
    for p, el, mv in configs:
        master_data.extend(run_single_experiment(p, el, mv))

    with open(MASTER_METRICS_FILE, 'w') as f:
        json.dump(master_data, f, indent=4)

    return master_data


def plot_results(data):
    """Plots a 2x2 grid: Speedup, Efficiency, Time vs Elements, Time vs Max Value."""
    if not data:
        print("No data available to plot.")
        return

    tasks = list(set(item['task_name'] for item in data))
    plt.figure(figsize=(16, 10))

    # --- Plot 1: Speedup vs Processes ---
    plt.subplot(2, 2, 1)
    plt.title(f"Speedup vs Processes\n(Elements={DEFAULT_ELEMENTS}, MaxVal={DEFAULT_MAX_VAL})")
    plt.xlabel("Number of Processes (p)")
    plt.ylabel("Speedup (S)")
    plt.xticks(P_VALUES)
    plt.grid(True)

    # --- Plot 2: Efficiency vs Processes ---
    plt.subplot(2, 2, 2)
    plt.title(f"Efficiency vs Processes\n(Elements={DEFAULT_ELEMENTS}, MaxVal={DEFAULT_MAX_VAL})")
    plt.xlabel("Number of Processes (p)")
    plt.ylabel("Efficiency (E)")
    plt.xticks(P_VALUES)
    plt.grid(True)

    for task in tasks:
        task_data = [m for m in data if
                     m['task_name'] == task and m['elements'] == DEFAULT_ELEMENTS and m['max_val'] == DEFAULT_MAX_VAL]
        task_data.sort(key=lambda x: x['p'])

        t_1 = next((m['avg_time'] for m in task_data if m['p'] == 1), None)
        if not t_1: continue

        ps, speedups, efficiencies = [], [], []
        for m in task_data:
            p = m['p']
            speedup = t_1 / m['avg_time']
            ps.append(p)
            speedups.append(speedup)
            efficiencies.append(speedup / p)

        plt.subplot(2, 2, 1)
        plt.plot(ps, speedups, marker='o', linestyle='-', linewidth=2, label=task, color='blue')
        plt.subplot(2, 2, 2)
        plt.plot(ps, efficiencies, marker='s', linestyle='-', linewidth=2, label=task, color='black')

    plt.subplot(2, 2, 1).legend()
    plt.subplot(2, 2, 2).legend()

    # --- Plot 3: Time vs Elements (RAW TIME) ---
    plt.subplot(2, 2, 3)
    plt.title(f"Execution Time vs Elements\n(p={DEFAULT_P}, MaxVal={DEFAULT_MAX_VAL})")
    plt.xlabel("Number of Elements")
    plt.ylabel("Time (seconds)")
    plt.grid(True)

    for task in tasks:
        task_data = [m for m in data if
                     m['task_name'] == task and m['p'] == DEFAULT_P and m['max_val'] == DEFAULT_MAX_VAL]
        task_data.sort(key=lambda x: x['elements'])

        if task_data:
            elements = [m['elements'] for m in task_data]
            times = [m['avg_time'] for m in task_data]  # Using raw time
            plt.plot(elements, times, marker='^', linestyle='-', linewidth=2, label=task, color='green')
    plt.legend()

    # --- Plot 4: Time vs Max Value (RAW TIME) ---
    plt.subplot(2, 2, 4)
    plt.title(f"Execution Time vs Max Value\n(p={DEFAULT_P}, Elements={DEFAULT_ELEMENTS})")
    plt.xlabel("Maximum Value (Log Scale)")
    plt.ylabel("Time (seconds)")
    plt.xscale('log')
    plt.grid(True)

    for task in tasks:
        task_data = [m for m in data if
                     m['task_name'] == task and m['p'] == DEFAULT_P and m['elements'] == DEFAULT_ELEMENTS]
        task_data.sort(key=lambda x: x['max_val'])

        if task_data:
            max_vals = [m['max_val'] for m in task_data]
            times = [m['avg_time'] for m in task_data]  # Using raw time
            plt.plot(max_vals, times, marker='d', linestyle='-', linewidth=2, label=task, color='orange')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("Starting experiments...")
    master_data = run_all_experiments()

    print("Generating plots...")
    plot_results(master_data)
