use mopa::mopafy;

pub mod affine_layer;
pub mod relu_layer;
pub mod sigmoid_layer;
pub mod softmax_with_loss_layer;

pub trait Layer: mopa::Any {
    fn forwards(&mut self, x: &na::DMatrix<f64>) -> na::DMatrix<f64>;
    fn backwards(&mut self, x: &na::DMatrix<f64>) -> na::DMatrix<f64>;
}

mopafy!(Layer);
