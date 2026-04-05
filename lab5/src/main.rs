mod algs;

use algs::models::Matrix;

use clap::Parser;
use clap::ValueEnum;
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::time::Instant;

#[derive(ValueEnum, Clone, Debug)]
enum Algorithm {
    Sequential,
    MpiParallel,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value_t = 500, help = "Size of the N x N matrix")]
    n: usize,
    #[arg(long, default_value_t = 5, help = "Number of tries to average")]
    tries: usize,
    #[arg(long, value_enum, default_value_t = Algorithm::Sequential, help = "Algorithm to use")]
    algo: Algorithm,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ExperimentMetrics {
    task_name: String,
    p: i32,
    avg_time: f64,
}

fn save_metrics_to_json(metrics_list: &[ExperimentMetrics], filename: &str) {
    let mut existing_data: Vec<ExperimentMetrics> = Vec::new();

    if let Ok(mut file) = File::open(filename) {
        let mut content = String::new();
        if file.read_to_string(&mut content).is_ok() {
            if let Ok(data) = serde_json::from_str(&content) {
                existing_data = data;
            }
        }
    }

    existing_data.extend_from_slice(metrics_list);

    let json_string =
        serde_json::to_string_pretty(&existing_data).expect("Failed to serialize metrics");
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(filename)
        .expect("Failed to open metrics file for writing");

    file.write_all(json_string.as_bytes())
        .expect("Failed to write to metrics file");
}

fn verify_result(expected_x: &[f64], computed_x: &[f64]) {
    let epsilon = 1e-6;
    for i in 0..expected_x.len() {
        if (expected_x[i] - computed_x[i]).abs() > epsilon {
            panic!(
                "Verification failed at index {}: computed {}, expected {}",
                i, computed_x[i], expected_x[i]
            );
        }
    }
}

fn run_experiment<F, E>(
    world: &SimpleCommunicator,
    task_name: &str,
    a: &Matrix,
    b: &[f64],
    expected_x: &[f64],
    solve_func: F,
    tries: usize,
) -> Option<ExperimentMetrics>
where
    F: Fn(&SimpleCommunicator, &Matrix, &[f64]) -> Result<Option<Vec<f64>>, E>,
    E: std::fmt::Display,
{
    let rank = world.rank();
    let size = world.size();
    let mut total_time = 0.0;

    for _ in 0..tries {
        world.barrier();
        let start_time = Instant::now();

        let result = solve_func(world, a, b).unwrap_or_else(|e| {
            panic!("Gaussian Elimination failed: {}", e);
        });

        let elapsed = start_time.elapsed().as_secs_f64();

        if rank == 0 {
            if let Some(ref res) = result {
                verify_result(expected_x, res);
                total_time += elapsed;
            } else {
                panic!("Rank 0 expected a Some(Vec<f64>) result, but got None.");
            }
        }
    }

    if rank == 0 {
        Some(ExperimentMetrics {
            task_name: task_name.to_string(),
            p: size,
            avg_time: total_time / (tries as f64),
        })
    } else {
        None
    }
}

fn main() {
    let args = Args::parse();

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    // Create a diagonally dominant matrix to guarantee it can be solved safely without pivoting
    let a = Matrix::random_diagonally_dominant(args.n, args.n);

    // Generate an expected solution `x` and calculate `b` = `A` * `expected_x`
    let mut expected_x = vec![0.0; args.n];
    let mut b = vec![0.0; args.n];

    if rank == 0 {
        for i in 0..args.n {
            expected_x[i] = rand::random::<f64>();
        }
        for i in 0..args.n {
            let mut sum = 0.0;
            for j in 0..args.n {
                sum += a[(i, j)] * expected_x[j];
            }
            b[i] = sum;
        }
    }

    let (algo_name, solve_func) = match args.algo {
        Algorithm::Sequential => (
            "Sequential Gauss",
            algs::gauss_seq::solve_mpi_mock
                as fn(&SimpleCommunicator, &Matrix, &[f64]) -> Result<Option<Vec<f64>>, String>,
        ),
        Algorithm::MpiParallel => (
            "MPI Cyclic Gauss",
            algs::gauss_mpi::solve_mpi
                as fn(&SimpleCommunicator, &Matrix, &[f64]) -> Result<Option<Vec<f64>>, String>,
        ),
    };

    let task_name = format!("Algorithm: {} (N={})", algo_name, args.n);

    let metrics_opt = run_experiment(
        &world,
        &task_name,
        &a,
        &b,
        &expected_x,
        solve_func,
        args.tries,
    );

    if rank == 0 {
        if let Some(metrics) = metrics_opt {
            save_metrics_to_json(&[metrics], "data/metrics.json");
            println!(
                "--> Done p={} | Averaged over {} tries. Verification successful for N={}",
                size, args.tries, args.n
            );
        }
    }
}
