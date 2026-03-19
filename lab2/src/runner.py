import subprocess
import json
import os
import matplotlib.pyplot as plt

P_VALUES = [1, 2, 4, 8]
ARRAY_SIZE = 50_000_000
TRIES_PER_P = 10
METRICS_FILE = "metrics.json"


def run_mpi_experiments():
    if os.path.exists(METRICS_FILE):
        os.remove(METRICS_FILE)

    for p in P_VALUES:
        print(f"Starting execution for p={p}...")

        env = os.environ.copy()
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"

        cmd = [
            "mpiexec", "-n", str(p),
            "python", "-m", "src.main",
            "--size", str(ARRAY_SIZE),
            "--tries", str(TRIES_PER_P)
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

    # --- Speedup ---
    plt.subplot(1, 2, 1)
    for task in tasks:
        task_data = sorted([m for m in data if m['task_name'] == task], key=lambda x: x['p'])
        ps = [m['p'] for m in task_data]
        speedups = [m['speedup'] for m in task_data]
        plt.plot(ps, speedups, marker='o', label=task)

    # # S = p
    # plt.plot(P_VALUES, P_VALUES, '--', color='gray', label='Ideal Speedup')
    plt.title(f"Speedup (Averaged over {TRIES_PER_P} runs)")
    plt.xlabel("Number of Processes (p)")
    plt.ylabel("Speedup (S)")
    plt.xticks(P_VALUES)
    plt.legend()
    plt.grid(True)

    # --- Efficiency ---
    plt.subplot(1, 2, 2)
    for task in tasks:
        task_data = sorted([m for m in data if m['task_name'] == task], key=lambda x: x['p'])
        ps = [m['p'] for m in task_data]
        efficiencies = [m['efficiency'] for m in task_data]
        plt.plot(ps, efficiencies, marker='s', label=task)

    # # E = 1
    # plt.axhline(y=1, color='gray', linestyle='--', label='Ideal Efficiency')
    plt.title(f"Efficiency (Averaged over {TRIES_PER_P} runs)")
    plt.xlabel("Number of Processes (p)")
    plt.ylabel("Efficiency (E)")
    plt.xticks(P_VALUES)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_mpi_experiments()
    plot_results()
