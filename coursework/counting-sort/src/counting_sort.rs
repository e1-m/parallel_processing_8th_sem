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

fn calculate_local_offset(rank: usize, size: usize, total_elements: usize) -> usize {
    let mut offset = 0;
    for r in 0..rank {
        offset += calculate_local_size(r, size, total_elements);
    }
    offset
}

fn reduce_and_reconstruct(
    world: &SimpleCommunicator,
    local_counts: &[u32],
    max_val: usize,
    total_elements: usize,
) -> Result<Option<Vec<u32>>, String> {
    let rank = world.rank() as usize;
    let size = world.size() as usize;

    // All-Reduce: Every process needs the full frequency array
    let mut global_counts = vec![0u32; max_val + 1];
    world.all_reduce_into(
        local_counts,
        &mut global_counts[..],
        mpi::collective::SystemOperation::sum(),
    );

    // Prefix Sum
    // This maps out where each number's sequence starts in the final sorted array
    let mut prefix_sum = vec![0usize; max_val + 1];
    let mut current_sum = 0;
    for i in 0..=max_val {
        prefix_sum[i] = current_sum;
        current_sum += global_counts[i] as usize;
    }

    // Determine this rank's specific chunk of the final array
    let my_start_idx = calculate_local_offset(rank, size, total_elements);
    let my_chunk_size = calculate_local_size(rank, size, total_elements);
    let my_end_idx = my_start_idx + my_chunk_size;

    // Generate the local chunk in parallel
    let mut local_sorted_chunk = Vec::with_capacity(my_chunk_size);

    for val in 0..=max_val {
        let val_start = prefix_sum[val];
        let val_count = global_counts[val] as usize;
        let val_end = val_start + val_count;

        // Find the overlap between this rank's assigned indices and this value's global indices
        let overlap_start = std::cmp::max(my_start_idx, val_start);
        let overlap_end = std::cmp::min(my_end_idx, val_end);

        // If there is an overlap, write that number 'overlap_count' times
        if overlap_start < overlap_end {
            let overlap_count = overlap_end - overlap_start;
            for _ in 0..overlap_count {
                local_sorted_chunk.push(val as u32);
            }
        }
    }

    // Gather the chunks back to Rank 0 using Gatherv
    let root_process = world.process_at_rank(0);

    if rank == 0 {
        let mut final_array = vec![0u32; total_elements];

        let mut counts: Vec<mpi::Count> = Vec::with_capacity(size);
        let mut displs: Vec<mpi::Count> = Vec::with_capacity(size);
        let mut current_offset = 0;

        for target_rank in 0..size {
            let target_n = calculate_local_size(target_rank, size, total_elements) as mpi::Count;
            counts.push(target_n);
            displs.push(current_offset);
            current_offset += target_n;
        }

        let mut partition = mpi::datatype::PartitionMut::new(&mut final_array[..], counts, displs);
        root_process.gather_varcount_into_root(&local_sorted_chunk[..], &mut partition);

        Ok(Some(final_array))
    } else {
        root_process.gather_varcount_into(&local_sorted_chunk[..]);
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
