import subprocess
import json
import os
import matplotlib.pyplot as plt

P_VALUES = [1, 2, 4, 8]
ELEMENTS = 5_000_000
TRIES = 5
MAX_VAL = 100
METRICS_FILE = "data/metrics.json"


def run_mpi_experiments():
    if os.path.exists(METRICS_FILE):
        os.remove(METRICS_FILE)

    for p in P_VALUES:
        print(f"Starting execution for p={p}...")
        env = os.environ.copy()
        cmd = [
            "mpiexec", "-np", str(p),
            "../target/debug/counting-sort",
            "--elements", str(ELEMENTS),
            "--max-val", str(MAX_VAL),
            "--tries", str(TRIES)
        ]
        subprocess.run(cmd, check=True, env=env)


def plot_results():
    if not os.path.exists(METRICS_FILE):
        print("Metrics file not found. Did the experiments run?")
        return

    with open(METRICS_FILE, 'r') as f:
        data = json.load(f)

    tasks = list(set(item['task_name'] for item in data))

    plt.figure(figsize=(12, 5))

    task_metrics = {}
    for task in tasks:
        task_data = sorted([m for m in data if m['task_name'] == task], key=lambda x: x['p'])

        t_1 = next((m['avg_time'] for m in task_data if m['p'] == 1), None)

        if t_1 is None:
            print(f"Warning: No baseline p=1 data found for '{task}'. Skipping plot.")
            continue

        ps = []
        speedups = []
        efficiencies = []

        for m in task_data:
            p = m['p']
            t_p = m['avg_time']

            speedup = t_1 / t_p
            efficiency = speedup / p

            ps.append(p)
            speedups.append(speedup)
            efficiencies.append(efficiency)

        task_metrics[task] = {'ps': ps, 'speedups': speedups, 'efficiencies': efficiencies}

    plt.subplot(1, 2, 1)
    for task, metrics in task_metrics.items():
        plt.plot(metrics['ps'], metrics['speedups'], marker='o', label=task)

    plt.title(f"Speedup (Averaged over {TRIES} runs)")
    plt.xlabel("Number of Processes (p)")
    plt.ylabel("Speedup (S)")
    plt.xticks(P_VALUES)
    # Add a dashed ideal speedup line for reference (S = p)
    # plt.plot(P_VALUES, P_VALUES, 'k--', alpha=0.5, label='Ideal Speedup')
    plt.legend()
    plt.grid(True)

    # Plot 2: Efficiency
    plt.subplot(1, 2, 2)
    for task, metrics in task_metrics.items():
        plt.plot(metrics['ps'], metrics['efficiencies'], marker='s', label=task)

    plt.title(f"Efficiency (Averaged over {TRIES} runs)")
    plt.xlabel("Number of Processes (p)")
    plt.ylabel("Efficiency (E)")
    plt.xticks(P_VALUES)
    # Add a dashed ideal efficiency line for reference (E = 1.0)
    # plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Ideal Efficiency')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_mpi_experiments()
    plot_results()
