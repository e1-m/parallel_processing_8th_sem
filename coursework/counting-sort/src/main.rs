mod counting_sort;

use clap::Parser;
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value_t = 1_000_000, help = "Number of elements to sort")]
    elements: usize,
    #[arg(long, default_value_t = 5, help = "Number of tries to average")]
    tries: usize,
    #[arg(long, default_value_t = 1000, help = "Maximum value for Counting Sort")]
    max_val: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ExperimentMetrics {
    task_name: String,
    p: i32,
    avg_time: f64,
}

fn generate_data(n: usize, max_val: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 137) % (max_val + 1)) as u32).collect()
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

fn verify_result(actual: &[u32], expected: &[u32]) {
    if actual != expected {
        panic!("Verification failed: MPI result does not match Sequential result");
    }
}

fn run_experiment(
    world: &SimpleCommunicator,
    task_name: &str,
    global_data_opt: Option<&[u32]>,
    total_elements: usize,
    max_val: usize,
    tries: usize,
) -> Option<ExperimentMetrics> {
    let rank = world.rank() as usize;
    let size = world.size() as usize;
    let mut total_time = 0.0;

    let expected_result = if rank == 0 {
        Some(counting_sort::counting_sort_sequential(global_data_opt.unwrap(), max_val))
    } else {
        None
    };

    for _ in 0..tries {
        world.barrier();
        let start_time = Instant::now();


        let result = counting_sort::counting_sort_mpi(world, global_data_opt, total_elements, max_val)
            .unwrap_or_else(|e| {
                panic!("MPI Sorting failed: {}", e);
            });

        let elapsed = start_time.elapsed().as_secs_f64();

        if rank == 0 {
            if let Some(res) = result {
                verify_result(&res, expected_result.as_ref().unwrap());
                total_time += elapsed;
            } else {
                panic!("Rank 0 expected a Some(Vec<u32>) result, but got None.");
            }
        }
    }

    if rank == 0 {
        Some(ExperimentMetrics {
            task_name: task_name.to_string(),
            p: size as i32,
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

    let data = if rank == 0 {
        Some(generate_data(args.elements, args.max_val))
    } else {
        None
    };

    let task_name = format!(
        "Algorithm: Parallel Counting Sort (elements={}, max_val={})",
        args.elements, args.max_val
    );

    let metrics_opt = run_experiment(
        &world,
        &task_name,
        data.as_deref(),
        args.elements,
        args.max_val,
        args.tries,
    );

    if rank == 0 {
        if let Some(metrics) = metrics_opt {
            save_metrics_to_json(&[metrics.clone()], "data/metrics.json");
            println!(
                "--> Done p={} | Algorithm: Counting Sort | Averaged over {} tries. \nVerification successful! \nAvg Time: {:.6}s",
                size, args.tries, metrics.avg_time
            );
        }
    }
}