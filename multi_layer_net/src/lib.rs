use std::fmt::Debug;

use na::Scalar;
use rand::Rng;

extern crate nalgebra as na;

pub mod grads;
pub mod grads_exteded;
pub mod layers;
pub mod multi_layer_net;
pub mod multi_layer_net_extended;
pub mod params;
pub mod params_extended;

pub(crate) fn init_matrix_with_standard_normal(row: usize, column: usize) -> na::DMatrix<f64> {
    let mut matrix = na::DMatrix::<f64>::zeros(row, column);
    matrix
        .iter_mut()
        .for_each(|t| *t = rand::thread_rng().sample(rand_distr::StandardNormal));
    matrix
}

pub(crate) fn broadcast_vector_rowwise<T: Scalar + Copy + Debug + rand_distr::num_traits::Zero>(
    vec: &na::DVector<T>,
    nrows: usize,
) -> na::DMatrix<T> {
    dbg!(vec.len());
    dbg!(vec.as_slice().repeat(nrows).as_slice().len());
    na::DMatrix::<T>::from_row_slice(nrows, vec.nrows(), vec.as_slice().repeat(nrows).as_slice())
}

#[test]
fn test_broadcast_vector_rowwise() {
    let x = na::dmatrix![1,2,3;1,2,3;1,2,3];
    let v = na::dvector![1, 2, 3];
    assert_eq!(x, broadcast_vector_rowwise(&v, 3));
    x.column_iter().for_each(|x| -> () {
        panic!("{}", x);
    });
}
