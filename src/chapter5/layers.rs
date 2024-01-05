use self::{affine_layer::Affine, relu_layer::Relu, sigmoid_layer::Sigmoid};

pub mod affine_layer;
pub mod relu_layer;
pub mod sigmoid_layer;
pub mod softmax_with_loss_layer;

pub trait Layer {
    fn forwards(&mut self, x: &na::DMatrix<f64>) -> na::DMatrix<f64>;
    fn backwards(&mut self, x: &na::DMatrix<f64>) -> na::DMatrix<f64>;
}
