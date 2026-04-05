mod algs;

use clap::Parser;
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::time::Instant;

#[derive(clap::ValueEnum, Clone, Debug, PartialEq)]
enum Algorithm {
    Trapezoidal,
    Simpson,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(
        long,
        default_value_t = 1_000_000,
        help = "Number of integration intervals"
    )]
    intervals: usize,
    #[arg(long, default_value_t = 5, help = "Number of tries to average")]
    tries: usize,
    #[arg(long, value_enum, default_value_t = Algorithm::Simpson, help = "Algorithm to use")]
    algo: Algorithm,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ExperimentMetrics {
    task_name: String,
    p: i32,
    avg_time: f64,
    result: f64,
}

fn target_function(x: f64) -> f64 {
    (x * x - 5.0 * x + 6.0) * (3.0 * x).cos()
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

    if let Some(parent) = std::path::Path::new(filename).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(filename)
        .expect("Failed to open metrics file for writing");

    file.write_all(json_string.as_bytes())
        .expect("Failed to write to metrics file");
}

fn verify_result(actual: f64, expected: f64) {
    let epsilon = 1e-8;

    if (actual - expected).abs() > epsilon {
        panic!(
            "Verification failed: MPI result {}, Sequential result {}",
            actual, expected
        );
    }
}

fn run_experiment(
    world: &SimpleCommunicator,
    task_name: &str,
    a: f64,
    b: f64,
    n: usize,
    f: fn(f64) -> f64,
    seq_func: fn(f64, f64, usize, fn(f64) -> f64) -> f64,
    tries: usize,
) -> Option<ExperimentMetrics> {
    let rank = world.rank();
    let size = world.size();
    let mut total_time = 0.0;

    // Calculate expected sequentially on Rank 0 for verification
    let expected_result = if rank == 0 {
        Some(seq_func(a, b, n, f))
    } else {
        None
    };

    let mut last_result = 0.0;

    for _ in 0..tries {
        world.barrier();
        let start_time = Instant::now();

        let result =
            algs::parallel::integrate_mpi(world, a, b, n, f, seq_func).unwrap_or_else(|e| {
                panic!("MPI Integration failed: {}", e);
            });

        let elapsed = start_time.elapsed().as_secs_f64();

        if rank == 0 {
            if let Some(res) = result {
                verify_result(res, expected_result.unwrap());
                total_time += elapsed;
                last_result = res;
            } else {
                panic!("Rank 0 expected a Some(f64) result, but got None.");
            }
        }
    }

    if rank == 0 {
        Some(ExperimentMetrics {
            task_name: task_name.to_string(),
            p: size,
            avg_time: total_time / (tries as f64),
            result: last_result,
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

    // Integration limits
    let a = 0.0;
    let b = PI / 2.0;

    let (algo_name, seq_func): (&str, fn(f64, f64, usize, fn(f64) -> f64) -> f64) = match args.algo
    {
        Algorithm::Trapezoidal => ("Trapezoidal", algs::sequential::trapezoidal),
        Algorithm::Simpson => ("Simpson", algs::sequential::simpson),
    };

    let task_name = format!("Algorithm: {} (intervals={})", algo_name, args.intervals);

    let metrics_opt = run_experiment(
        &world,
        &task_name,
        a,
        b,
        args.intervals,
        target_function,
        seq_func,
        args.tries,
    );

    if rank == 0 {
        if let Some(metrics) = metrics_opt {
            save_metrics_to_json(&[metrics.clone()], "data/metrics.json");
            println!(
                "--> Done p={} | Algorithm: {} | Averaged over {} tries. \nResult: {:.8} \nVerification successful!",
                size, algo_name, args.tries, metrics.result
            );
        }
    }
}
