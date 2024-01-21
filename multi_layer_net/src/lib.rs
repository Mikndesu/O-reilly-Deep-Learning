use rand::Rng;

extern crate nalgebra as na;

pub mod grads;
pub mod layers;
pub mod multi_layer_net;
pub mod params;

pub(crate) fn init_matrix_with_standard_normal(column: usize, row: usize) -> na::DMatrix<f64> {
    let mut matrix = na::DMatrix::<f64>::zeros(column, row);
    matrix
        .iter_mut()
        .for_each(|t| *t = rand::thread_rng().sample(rand_distr::StandardNormal));
    matrix
}