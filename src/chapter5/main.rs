mod layers;
mod train_neural_net;
mod two_layer_net;
extern crate nalgebra as na;

fn main() {
    train_neural_net::train_neural_net();
}
