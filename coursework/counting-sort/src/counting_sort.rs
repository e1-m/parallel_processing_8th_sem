use mpi::topology::SimpleCommunicator;
use mpi::traits::*;

pub fn counting_sort_sequential(arr: &[u32], max_val: usize) -> Vec<u32> {
    if arr.is_empty() {
        return Vec::new();
    }

    let mut counts = vec![0u32; max_val + 1];

    // Count occurrences
    for &val in arr {
        counts[val as usize] += 1;
    }

    // Reconstruct the sorted array
    let mut sorted = Vec::with_capacity(arr.len());
    for (val, &count) in counts.iter().enumerate() {
        for _ in 0..count {
            sorted.push(val as u32);
        }
    }

    sorted
}

fn calculate_local_size(rank: usize, size: usize, total_elements: usize) -> usize {
    let base_n = total_elements / size;
    let remainder = total_elements % size;
    if rank < remainder { base_n + 1 } else { base_n }
}

fn distribute_data(
    world: &SimpleCommunicator,
    data: Option<&[u32]>,
    total_elements: usize,
) -> Vec<u32> {
    let rank = world.rank() as usize;
    let size = world.size() as usize;

    let local_n = calculate_local_size(rank, size, total_elements);
    let mut local_data = vec![0u32; local_n];

    let root_process = world.process_at_rank(0);

    if rank == 0 {
        let global_data = data.expect("Rank 0 must provide global data");

        let mut counts: Vec<mpi::Count> = Vec::with_capacity(size);
        let mut displs: Vec<mpi::Count> = Vec::with_capacity(size);
        let mut current_offset = 0;

        for target_rank in 0..size {
            let target_n = calculate_local_size(target_rank, size, total_elements) as mpi::Count;
            counts.push(target_n);
            displs.push(current_offset);
            current_offset += target_n;
        }

        let partition = mpi::datatype::Partition::new(global_data, counts, displs);
        root_process.scatter_varcount_into_root(&partition, &mut local_data[..]);
    } else {
        root_process.scatter_varcount_into(&mut local_data[..]);
    }

    local_data
}
fn count_local_frequencies(data: &[u32], max_val: usize) -> Vec<u32> {
    let mut counts = vec![0u32; max_val + 1];
    for &val in data {
        counts[val as usize] += 1;
    }
    counts
}

fn reconstruct_sorted_array(counts: &[u32], total_elements: usize) -> Vec<u32> {
    let mut sorted = Vec::with_capacity(total_elements);
    for (val, &count) in counts.iter().enumerate() {
        for _ in 0..count {
            sorted.push(val as u32);
        }
    }
    sorted
}

fn reduce_and_reconstruct(
    world: &SimpleCommunicator,
    local_counts: &[u32],
    max_val: usize,
    total_elements: usize,
) -> Result<Option<Vec<u32>>, String> {
    let rank = world.rank() as usize;
    let root_process = world.process_at_rank(0);

    let mut global_counts = vec![0u32; max_val + 1];

    if rank == 0 {
        root_process.reduce_into_root(
            local_counts,
            &mut global_counts[..],
            mpi::collective::SystemOperation::sum(),
        );

        let sorted = reconstruct_sorted_array(&global_counts, total_elements);
        Ok(Some(sorted))
    } else {
        root_process.reduce_into(local_counts, mpi::collective::SystemOperation::sum());
        Ok(None)
    }
}

pub fn counting_sort_mpi(
    world: &SimpleCommunicator,
    data: Option<&[u32]>,
    total_elements: usize,
    max_val: usize,
) -> Result<Option<Vec<u32>>, String> {
    let local_data = distribute_data(world, data, total_elements);
    let local_counts = count_local_frequencies(&local_data, max_val);
    reduce_and_reconstruct(world, &local_counts, max_val, total_elements)
}
