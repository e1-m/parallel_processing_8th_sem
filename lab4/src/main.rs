mod algs;

use algs::models::Matrix;

use clap::Parser;
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::time::Instant;

#[derive(clap::ValueEnum, Clone, Debug)]
enum Algorithm {
    Naive,
    Fox,
    Cannon,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value_t = 400)]
    m: usize,
    #[arg(long, default_value_t = 400)]
    k: usize,
    #[arg(long, default_value_t = 400)]
    n: usize,
    #[arg(long, default_value_t = 5, help = "Number of tries to average")]
    tries: usize,
    #[arg(long, value_enum, default_value_t = Algorithm::Naive, help = "Algorithm to use")]
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

fn verify_result(a: &Matrix, b: &Matrix, result: &Matrix) {
    let nd_a = Array2::from_shape_vec((a.rows(), a.cols()), a.as_slice().to_vec()).unwrap();
    let nd_b = Array2::from_shape_vec((b.rows(), b.cols()), b.as_slice().to_vec()).unwrap();
    let expected = nd_a.dot(&nd_b);

    let epsilon = 1e-9;

    for i in 0..result.rows() {
        for j in 0..result.cols() {
            let actual = result[(i, j)];
            let exp = expected[[i, j]];

            if (actual - exp).abs() > epsilon {
                panic!(
                    "Verification failed at ({}, {}): result got {}, ndarray got {}",
                    i, j, actual, exp
                );
            }
        }
    }
}

fn run_experiment<F, E>(
    world: &SimpleCommunicator,
    task_name: &str,
    a: &Matrix,
    b: &Matrix,
    multiply_func: F,
    tries: usize,
) -> Option<ExperimentMetrics>
where
    F: Fn(&SimpleCommunicator, &Matrix, &Matrix) -> Result<Option<Matrix>, E>,
    E: std::fmt::Display,
{
    let rank = world.rank();
    let size = world.size();
    let mut total_time = 0.0;

    for _ in 0..tries {
        world.barrier();
        let start_time = Instant::now();

        let result = multiply_func(world, a, b).unwrap_or_else(|e| {
            panic!("Matrix Multiplication failed: {}", e);
        });

        let elapsed = start_time.elapsed().as_secs_f64();

        if rank == 0 {
            if let Some(res) = result {
                verify_result(a, b, &res);
                total_time += elapsed;
            } else {
                panic!("Rank 0 expected a Some(Matrix) result, but got None.");
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

    let a = Matrix::random(args.m, args.k);
    let b = Matrix::random(args.k, args.n);

    let (algo_name, multiply_func) = match args.algo {
        Algorithm::Naive => (
            "Naive",
            algs::naive::multiply_mpi
                as fn(&SimpleCommunicator, &Matrix, &Matrix) -> Result<Option<Matrix>, String>,
        ),
        Algorithm::Fox => (
            "Fox",
            algs::fox::multiply_mpi
                as fn(&SimpleCommunicator, &Matrix, &Matrix) -> Result<Option<Matrix>, String>,
        ),
        Algorithm::Cannon => (
            "Cannon",
            algs::cannon::multiply_mpi
                as fn(&SimpleCommunicator, &Matrix, &Matrix) -> Result<Option<Matrix>, String>,
        ),
    };

    let task_name = format!(
        "Algorithm: {} ({}x{} * {}x{})",
        algo_name, args.m, args.k, args.k, args.n
    );

    let metrics_opt = run_experiment(&world, &task_name, &a, &b, multiply_func, args.tries);

    if rank == 0 {
        if let Some(metrics) = metrics_opt {
            save_metrics_to_json(&[metrics], "data/metrics.json");
            println!(
                "--> Done p={} | Averaged over {} tries. Verification successful!",
                size, args.tries
            );
        }
    }
}
