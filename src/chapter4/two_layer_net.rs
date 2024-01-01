use na::{DMatrix, Scalar};
use nalgebra as na;
use rand::Rng;

macro_rules! numerical_gradient_loss {
    ($instance:ident, $x:expr, $t:expr, $elm:ident) => {{
        let h = 1e-4;
        let mut grad = $instance.params.$elm.clone();
        for i in 0..grad.shape().0 {
            for j in 0..grad.shape().1 {
                let index = (i, j);
                let tmp_val = $instance.params.$elm[index];
                $instance.params.$elm[index] = tmp_val + h;
                let fxh1 = $instance.loss(&$x, &$t);
                $instance.params.$elm[index] = tmp_val - h;
                let fxh2 = $instance.loss(&$x, &$t);
                grad[index] = (fxh1 - fxh2) / (2.0 * h);
                $instance.params.$elm[index] = tmp_val;
            }
        }
        grad
    }};
}

pub struct TwoLayerNet {
    pub params: Params,
    pub grads: Grads,
}

pub struct Grads {
    pub d_w1: na::DMatrix<f64>,
    pub d_b1: na::DVector<f64>,
    pub d_w2: na::DMatrix<f64>,
    pub d_b2: na::DVector<f64>,
}

pub struct Params {
    pub w1: na::DMatrix<f64>,
    pub b1: na::DVector<f64>,
    pub w2: na::DMatrix<f64>,
    pub b2: na::DVector<f64>,
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
        Self { w1, b1, w2, b2 }
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

impl TwoLayerNet {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let weight_init_std = 0.01;
        Self {
            params: Params::new(weight_init_std, input_size, hidden_size, output_size),
            grads: Grads::new(input_size, hidden_size, output_size),
        }
    }

    fn predict(&self, x: &na::DMatrix<f64>) -> na::DMatrix<f64> {
        let (w1, w2) = (&self.params.w1, &self.params.w2);
        let (b1, b2) = (&self.params.b1, &self.params.b2);
        let mut a1 = x * w1;
        a1.row_iter_mut().for_each(|mut t| t += b1.transpose());
        let z1 = sigmoid(&a1);
        let mut a2 = z1 * w2;
        a2.row_iter_mut().for_each(|mut t| t += b2.transpose());
        let y = softmax(&a2);
        y
    }

    pub fn loss(&self, x: &na::DMatrix<f64>, t: &na::DMatrix<u8>) -> f64 {
        let y = self.predict(x);
        cross_entropy_error(&y, t)
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
        self.grads.d_w1 = numerical_gradient_loss!(self, &x, &t, w1);
        self.grads.d_b1 = numerical_gradient_loss!(self, &x, &t, b1);
        self.grads.d_w2 = numerical_gradient_loss!(self, &x, &t, w2);
        self.grads.d_b2 = numerical_gradient_loss!(self, &x, &t, b2);
    }
}

fn init_matrix_with_standard_normal(column: usize, row: usize) -> DMatrix<f64> {
    let mut matrix = na::DMatrix::<f64>::zeros(column, row);
    matrix
        .iter_mut()
        .for_each(|t| *t = rand::thread_rng().sample(rand_distr::StandardNormal));
    matrix
}

fn sigmoid(x: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    let mut matrix = x.clone();
    matrix
        .iter_mut()
        .zip(x.iter())
        .for_each(|(_matrix, &_x)| *_matrix = 1.0 / (1.0 + (-_x).exp()));
    matrix
}

fn softmax(x: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    let mut matrix = x.clone();
    matrix.row_iter_mut().for_each(|mut column| -> () {
        let c = column.max();
        let exp_x = column.iter().map(|&t| (t - c).exp()).collect::<Vec<f64>>();
        let sum = exp_x.iter().sum::<f64>();
        column
            .iter_mut()
            .zip(exp_x.iter())
            .for_each(|(column, &_exp_x)| *column = _exp_x / sum);
    });
    matrix
}

fn cross_entropy_error(y: &na::DMatrix<f64>, t: &na::DMatrix<u8>) -> f64 {
    let delta = 1e-7;
    let batch_size = y.shape().0 as f64;
    -t.iter()
        .zip(y.iter())
        .map(|(a, b)| (b + delta).ln() * (*a as f64))
        .collect::<Vec<f64>>()
        .iter()
        .sum::<f64>()
        / batch_size
}
