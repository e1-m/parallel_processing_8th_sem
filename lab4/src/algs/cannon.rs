use crate::algs::models::Matrix;
use mpi::topology::SimpleCommunicator;

pub fn multiply_mpi(
    world: &SimpleCommunicator,
    a: &Matrix,
    b: &Matrix,
) -> Result<Option<Matrix>, String> {
    unimplemented!()
}
