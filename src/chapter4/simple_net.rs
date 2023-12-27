use std::ops::Deref;

use nalgebra::{DMatrix, DVector};
use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

struct SimpleNet {
    w: na::Matrix2x3<f64>,
}

impl SimpleNet {
    fn new() -> Self {
        let mut w = na::Matrix2x3::<f64>::zeros();
        w.iter_mut()
            .for_each(|t| *t = thread_rng().sample(StandardNormal));
        Self { w }
    }

    fn predict(&self, x: na::Vector2<f64>) -> na::Vector3<f64> {
        (x.transpose() * self.w).transpose()
    }

    fn softmax(&self, x: na::Vector3<f64>) -> na::DVector<f64> {
        let c = x.max();
        let exp_x = x.iter().map(|&t| (t - c).exp()).collect::<Vec<f64>>();
        let sum = exp_x.iter().sum::<f64>();
        exp_x.iter().map(|t| t / sum).collect::<Vec<f64>>().into()
    }

    fn cross_entropy_error(&self, y: na::DVector<f64>, t: &na::DVector<u8>) -> f64 {
        let delta = 1e-7;
        -t.clone()
            .cast::<f64>()
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (b + delta).ln() * a)
            .collect::<Vec<f64>>()
            .iter()
            .sum::<f64>()
    }

    fn loss(&self, x: na::Vector2<f64>, t: &na::DVector<u8>) -> f64 {
        let z = self.predict(x);
        let y = self.softmax(z);
        self.cross_entropy_error(y, t)
    }

    fn numerical_gradient(
        &mut self,
        x: na::Vector2<f64>,
        t: &na::DVector<u8>,
    ) -> na::Matrix2x3<f64> {
        let h = 1e-4;
        let mut grad = na::Matrix2x3::<f64>::zeros();
        for i in 0..self.w.shape().0 {
            for j in 0..self.w.shape().1 {
                let index = (i, j);
                let tmp_val = self.w[index];
                self.w[index] = tmp_val + h;
                let fxh1 = self.loss(x, t);
                self.w[index] = tmp_val - h;
                let fxh2 = self.loss(x, t);
                grad[index] = (fxh1 - fxh2) / (2.0 * h);
                self.w[index] = tmp_val;
            }
        }
        grad
    }

    fn numerical_gradient_vec(
        &self,
        f: fn(&na::DVector<f64>) -> f64,
        x: &mut na::DVector<f64>,
    ) -> na::DVector<f64> {
        let h = 1e-4;
        let length = x.shape().0;
        let mut grad = DVector::<f64>::zeros(length);
        for idx in 0..length {
            let tmp_val = x[idx];
            x[idx] = tmp_val + h;
            let fxh1 = f(&x);
            x[idx] = tmp_val - h;
            let fxh2 = f(&x);
            grad[idx] = (fxh1 - fxh2) / (2.0 * h);
            x[idx] = tmp_val;
        }
        grad
    }
}

#[test]
fn test_loss() {
    let mut network = SimpleNet::new();
    network.w = na::Matrix2x3::<f64>::from_vec(vec![
        0.47355232, 0.85557411, 0.9977393, 0.03563661, 0.84668094, 0.69422093,
    ]);
    let x = na::Vector2::<f64>::from_vec(vec![0.6, 0.9]);
    let p = network.predict(x);
    dbg!(p);
    // this prints the index that indicates the maximum value
    dbg!(p.argmax().0);
    let t = na::DVector::<u8>::from_vec(vec![0, 0, 1]);
    // dbg!(network.loss(x, t));
    println!("{:?}", network.numerical_gradient(x, &t));
}

#[test]
fn test_cross_entropy_error() {
    let t = na::DVector::<u8>::from_vec(vec![0, 0, 1, 0, 0, 0, 0, 0, 0, 0]);
    let y = na::DVector::<f64>::from_vec(vec![0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]);
    fn cross_entropy_error(y: na::DVector<f64>, t: na::DVector<u8>) -> f64 {
        let delta = 1e-7;
        -t.cast::<f64>()
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (b + delta).ln() * a)
            .collect::<Vec<f64>>()
            .iter()
            .sum::<f64>()
    }
    dbg!(cross_entropy_error(y, t));
}

#[test]
fn test_numerical_gradient() {
    fn funtion_2(x: &na::DVector<f64>) -> f64 {
        x[0].powi(2) + x[1].powi(2)
    }
    let net = SimpleNet::new();
    // dbg!(net.numerical_gradient_1d(
    //     funtion_2,
    //     &mut DMatrix::<f64>::from_columns(&[DVector::from_vec(vec![3.0, 4.0])])
    // ));
}
