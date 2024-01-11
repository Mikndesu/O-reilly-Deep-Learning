use std::{
    cell::{Ref, RefCell},
    ops::Deref,
    rc::Rc,
};

use mylib::mnist::{self, load_label, load_normalised_image, DatasetType};
use na::{dmatrix, DMatrix, Scalar};
use nalgebra as na;
use rand::Rng;

use crate::layers::{
    affine_layer::Affine, relu_layer::Relu, softmax_with_loss_layer::SoftmaxWithLoss, Layer,
};

macro_rules! numerical_gradient_loss {
    ($instance:ident, $x:expr, $t:expr, $elm:ident) => {{
        let h = 1e-4;
        let mut grad = $instance.params.borrow_mut().$elm.borrow_mut().clone();
        for i in 0..grad.shape().0 {
            for j in 0..grad.shape().1 {
                let index = (i, j);
                let tmp_val = $instance.params.borrow_mut().$elm.borrow_mut()[index];
                $instance.params.borrow_mut().$elm.borrow_mut()[index] = tmp_val + h;
                let fxh1 = $instance.loss($x, $t);
                $instance.params.borrow_mut().$elm.borrow_mut()[index] = tmp_val - h;
                let fxh2 = $instance.loss($x, $t);
                grad[index] = (fxh1 - fxh2) / (2.0 * h);
                $instance.params.borrow_mut().$elm.borrow_mut()[index] = tmp_val;
            }
        }
        grad
    }};
}

pub struct TwoLayerNet {
    pub params: Rc<RefCell<Params>>,
    pub grads: Grads,
    pub layers: Layers,
    pub last_layer: SoftmaxWithLoss,
}

pub struct Layers {
    pub affine1: Rc<RefCell<Affine>>,
    relu1: Rc<RefCell<Relu>>,
    affine2: Rc<RefCell<Affine>>,
}

#[derive(Clone, Debug)]
pub struct Grads {
    pub d_w1: na::DMatrix<f64>,
    pub d_b1: na::DVector<f64>,
    pub d_w2: na::DMatrix<f64>,
    pub d_b2: na::DVector<f64>,
}

pub struct Params {
    pub w1: Rc<RefCell<na::DMatrix<f64>>>,
    pub b1: Rc<RefCell<na::DVector<f64>>>,
    pub w2: Rc<RefCell<na::DMatrix<f64>>>,
    pub b2: Rc<RefCell<na::DVector<f64>>>,
}

impl Params {
    pub fn new(
        weight_init_std: f64,
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
    ) -> Self {
        let w1: na::DMatrix<f64> =
            init_matrix_with_standard_normal(input_size, hidden_size) * weight_init_std;
        let b1: na::DVector<f64> = na::DVector::<f64>::zeros(hidden_size);
        let w2: na::DMatrix<f64> =
            init_matrix_with_standard_normal(hidden_size, output_size) * weight_init_std;
        let b2: na::DVector<f64> = na::DVector::<f64>::zeros(output_size);
        Self {
            w1: Rc::new(RefCell::new(w1)),
            b1: Rc::new(RefCell::new(b1)),
            w2: Rc::new(RefCell::new(w2)),
            b2: Rc::new(RefCell::new(b2)),
        }
    }
}

impl Grads {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let d_w1: na::DMatrix<f64> = init_matrix_with_standard_normal(input_size, hidden_size);
        let d_b1: na::DVector<f64> = na::DVector::<f64>::zeros(hidden_size);
        let d_w2: na::DMatrix<f64> = init_matrix_with_standard_normal(hidden_size, output_size);
        let d_b2: na::DVector<f64> = na::DVector::<f64>::zeros(output_size);
        Self {
            d_w1,
            d_b1,
            d_w2,
            d_b2,
        }
    }
}

impl Layers {
    pub fn new(params: Rc<RefCell<Params>>) -> Self {
        let clone = Rc::clone(&params);
        let w1 = clone.borrow().w1.clone();
        let b1 = clone.borrow().b1.clone();
        let w2 = clone.borrow().w2.clone();
        let b2 = clone.borrow().b2.clone();
        Self {
            affine1: Rc::new(RefCell::new(Affine::new(w1, b1))),
            relu1: Rc::new(RefCell::new(Relu::new())),
            affine2: Rc::new(RefCell::new(Affine::new(w2, b2))),
        }
    }
}

impl TwoLayerNet {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let weight_init_std = 0.01;
        let params = Rc::new(RefCell::new(Params::new(
            weight_init_std,
            input_size,
            hidden_size,
            output_size,
        )));
        Self {
            grads: Grads::new(input_size, hidden_size, output_size),
            params: params.clone(),
            last_layer: SoftmaxWithLoss::new(),
            layers: Layers::new(params),
        }
    }

    pub fn predict(&self, x: &na::DMatrix<f64>) -> na::DMatrix<f64> {
        let mut x = x.clone();
        let layers: Vec<Rc<RefCell<dyn Layer>>> = vec![
            self.layers.affine1.clone(),
            self.layers.relu1.clone(),
            self.layers.affine2.clone(),
        ];
        for layer in layers {
            x = layer.borrow_mut().forwards(&x);
        }
        x
    }

    pub fn loss(&mut self, x: &na::DMatrix<f64>, t: &na::DMatrix<u8>) -> f64 {
        let y = self.predict(x);
        self.last_layer.forwards(&y, t)
    }

    pub fn accuracy(&self, x: &na::DMatrix<f64>, t: &na::DMatrix<u8>) -> f64 {
        let y = self.predict(x);
        fn max_index_by_column<T>(matrix: &na::DMatrix<T>) -> Vec<(usize, usize)>
        where
            T: Scalar + PartialOrd,
        {
            matrix
                .row_iter()
                .enumerate()
                .map(|(i, column)| -> (usize, usize) {
                    let tmp = column.transpose().argmax().0;
                    (i, tmp)
                })
                .collect()
        }
        let y_max: Vec<(usize, usize)> = max_index_by_column(&y);
        let t_max: Vec<(usize, usize)> = max_index_by_column(t);
        let result = y_max
            .iter()
            .zip(t_max.iter())
            .map(|(_y, _t)| -> i32 {
                if _y == _t {
                    1
                } else {
                    0
                }
            })
            .collect::<Vec<i32>>()
            .iter()
            .sum::<i32>();
        (result as f64 / y.shape().0 as f64).into()
    }

    // x.shape should be (n, 784) and as well t.shape (n, 10)
    pub fn numerical_gradient(&mut self, x: &na::DMatrix<f64>, t: &na::DMatrix<u8>) {
        self.grads.d_w1 = numerical_gradient_loss!(self, x, t, w1);
        self.grads.d_b1 = numerical_gradient_loss!(self, x, t, b1);
        self.grads.d_w2 = numerical_gradient_loss!(self, x, t, w2);
        self.grads.d_b2 = numerical_gradient_loss!(self, x, t, b2);
    }

    pub fn gradient(&mut self, x: &na::DMatrix<f64>, t: &na::DMatrix<u8>) {
        self.loss(x, t);
        let layers: Vec<Rc<RefCell<dyn Layer>>> = vec![
            self.layers.affine2.clone(),
            self.layers.relu1.clone(),
            self.layers.affine1.clone(),
        ];
        let mut dout = self.last_layer.backwards(1.0);
        for layer in layers {
            dout = layer.borrow_mut().backwards(&dout);
        }
        self.grads.d_w1 = self.layers.affine1.deref().borrow().dw.clone();
        self.grads.d_b1 = self.layers.affine1.deref().borrow().db.clone();
        self.grads.d_w2 = self.layers.affine2.deref().borrow().dw.clone();
        self.grads.d_b2 = self.layers.affine2.deref().borrow().db.clone();
    }
}

fn init_matrix_with_standard_normal(column: usize, row: usize) -> DMatrix<f64> {
    let mut matrix = na::DMatrix::<f64>::zeros(column, row);
    matrix
        .iter_mut()
        .for_each(|t| *t = rand::thread_rng().sample(rand_distr::StandardNormal));
    matrix
}

#[test]
fn gradient_check() {
    let mut network = TwoLayerNet::new(784, 50, 10);
    let dataset_dir = std::env::current_dir().unwrap().join("dataset");
    mnist::init_mnist();
    let train_img = load_normalised_image(DatasetType::TrainImg, &dataset_dir).flatten();
    let train_label = load_label(DatasetType::TrainLabel, &dataset_dir).as_one_hot();
    let test_img = load_normalised_image(DatasetType::TestImg, &dataset_dir).flatten();
    let test_label = load_label(DatasetType::TestLabel, &dataset_dir).as_one_hot();
    let mut img_batch = na::DMatrix::<f64>::zeros(3, 784);
    let mut label_batch = na::DMatrix::<u8>::zeros(3, 10);
    img_batch
        .row_iter_mut()
        .zip((0..3).into_iter())
        .for_each(|(mut column, n)| column.copy_from_slice(&train_img.column(n).as_slice()));
    label_batch
        .row_iter_mut()
        .zip((0..3).into_iter())
        .for_each(|(mut column, n)| column.copy_from(&train_label.row(n)));
    network.numerical_gradient(&img_batch, &label_batch);
    let grad_numerical = network.grads.clone();
    network.gradient(&img_batch, &label_batch);
    let backprop = network.grads.clone();
    println!(
        "dw1 {}",
        (&grad_numerical.d_w1 - &backprop.d_w1).sum() / grad_numerical.d_w1.nrows() as f64
    );
    println!(
        "db1 {}",
        (&grad_numerical.d_b1 - &backprop.d_b1).sum() / grad_numerical.d_b1.nrows() as f64
    );
    println!(
        "dw2 {}",
        (&grad_numerical.d_w2 - &backprop.d_w2).sum() / grad_numerical.d_w2.nrows() as f64
    );
    println!(
        "db2 {}",
        (&grad_numerical.d_b2 - &backprop.d_b2).sum() / grad_numerical.d_b2.nrows() as f64
    );
    panic!("panic");
}

#[test]
fn test_accuracy() {
    let t: na::DMatrix<u8> = dmatrix![0,0,1,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,1,0];
    let x: na::DMatrix<f64> = dmatrix![-0.0031, -0.0076,  0.0073, -0.0048, -0.0027, -0.0019, -0.0059,  0.0017,  0.0058,  0.0034;
    0.0002, -0.0136,  0.0085, -0.0022, -0.0032, -0.0048, -0.0029,  0.0063,  0.0097,  0.0042];
    fn max_index_by_column<T>(matrix: &na::DMatrix<T>) -> Vec<(usize, usize)>
    where
        T: Scalar + PartialOrd,
    {
        matrix
            .row_iter()
            .enumerate()
            .map(|(i, column)| -> (usize, usize) {
                let tmp = column.transpose().argmax().0;
                dbg!((i, tmp));
                (i, tmp)
            })
            .collect()
    }
    let x_max: Vec<(usize, usize)> = max_index_by_column(&x);
    let t_max: Vec<(usize, usize)> = max_index_by_column(&t);
    let result = x_max
        .iter()
        .zip(t_max.iter())
        .map(|(_y, _t)| -> i32 {
            if _y == _t {
                1
            } else {
                0
            }
        })
        .collect::<Vec<i32>>()
        .iter()
        .sum::<i32>();
    let a = result as f64 / x.shape().0 as f64;
    dbg!(a);
}

#[test]
fn test_predict() {
    let dataset_dir = std::env::current_dir().unwrap().join("dataset");
    mnist::init_mnist();
    let mut network = TwoLayerNet::new(784, 50, 10);
    let _train_img = mnist::load_normalised_image(DatasetType::TrainImg, &dataset_dir).flatten();
    let train_label = load_label(DatasetType::TrainLabel, &dataset_dir).as_one_hot();
    let mut img_batch = na::DMatrix::<f64>::zeros(4, 784);
    let mut label_batch = na::DMatrix::<u8>::zeros(4, 10);
    img_batch
        .row_iter_mut()
        .zip((0..4).into_iter())
        .for_each(|(mut column, n)| column.copy_from_slice(&_train_img.column(n).as_slice()));
    label_batch
        .row_iter_mut()
        .zip((0..4).into_iter())
        .for_each(|(mut column, n)| column.copy_from(&train_label.row(n)));
    let mut result = network.predict(&img_batch);
    result.apply(|a| *a = (*a * 10000.0).round() / 10000.0);
    println!("{}", result);
    println!("{}", label_batch);
    println!("{}", network.accuracy(&img_batch, &label_batch));
    panic!();
}
