mod simple_net;

extern crate nalgebra as na;
use mylib::mnist::{self, load_normalised_image, Label, NormalisedImageVec};
use na::{DVector, Dyn};
use rand::seq::IteratorRandom;

fn gradient_descent(f: fn(&na::DVector<f64>) -> f64, init_x: na::DVector<f64>) -> na::DVector<f64> {
    let mut x = init_x.clone();
    let lr = 0.1;
    for _ in 0..100 {
        let mut grad = numerical_gradient(f, &mut x);
        grad.apply(|t| *t *= lr);
        x = x - grad;
    }
    x
}

fn numerical_diff(f: fn(f64) -> f64, x: f64) -> f64 {
    let h = 1e-4;
    (f(x + h) - f(x - h)) / (2 as f64 * h)
}

fn numerical_gradient(
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

fn sum_squared_error(y: na::DVector<f64>, t: na::DVector<u8>) -> f64 {
    (y - t.cast::<f64>())
        .iter()
        .map(|x| x.powi(2))
        .collect::<Vec<f64>>()
        .iter()
        .sum::<f64>()
        * 0.5
}

fn cross_entropy_error(
    y: na::OMatrix<f64, Dyn, na::Const<10>>,
    t: na::OMatrix<u8, Dyn, na::Const<10>>,
) -> f64 {
    let delta = 1e-7;
    let batch_size = y.shape().0 as f64;
    -t.cast::<f64>()
        .iter()
        .zip(y.iter())
        .map(|(a, b)| (b + delta).ln() * a)
        .collect::<Vec<f64>>()
        .iter()
        .sum::<f64>()
        / batch_size
}

fn cross_entropy_error_from_label(y: na::OMatrix<f64, Dyn, na::Const<10>>, t: Label) -> f64 {
    cross_entropy_error(y, t.as_one_hot())
}

fn mini_batch() -> Vec<usize> {
    let dataset_dir = mnist::init_mnist();
    let train_img: NormalisedImageVec =
        load_normalised_image(&mnist::DatasetType::TrainImg.file_name(), &dataset_dir);
    let batch_size = 10;
    let batch_mask =
        (0..train_img.as_ref().len()).choose_multiple(&mut rand::thread_rng(), batch_size);
    batch_mask
}

#[test]
fn test_mini_batch() {
    dbg!(mini_batch());
}

#[test]
fn test_cross_entropy_error() {
    let y = na::OMatrix::<f64, Dyn, na::Const<10>>::from_vec(vec![
        0.1, 0.1, 0.1, 0.05, 0.1, 0.05, 0.6, 0.1, 0.1, 0.0, 0.1, 0.0, 0.05, 0.1, 0.05, 0.1, 0.1,
        0.1, 0.0, 0.1, 0.0, 0.1, 0.1, 0.6, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0,
    ]);
    let t: na::OMatrix<u8, Dyn, na::Const<10>> =
        na::OMatrix::<u8, Dyn, na::Const<10>>::from_vec(vec![
            0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
            0,
        ]);
    assert_eq!(format!("{:.4}", cross_entropy_error(y, t)), "1.7053");
}

#[test]
fn test_cross_entropy_error_from_label() {
    let y = na::OMatrix::<f64, Dyn, na::Const<10>>::from_vec(vec![
        0.1, 0.1, 0.1, 0.05, 0.1, 0.05, 0.6, 0.1, 0.1, 0.0, 0.1, 0.0, 0.05, 0.1, 0.05, 0.1, 0.1,
        0.1, 0.0, 0.1, 0.0, 0.1, 0.1, 0.6, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0,
    ]);
    let t = Label::from(vec![2, 8, 2]);
    assert_eq!(
        format!("{:.4}", cross_entropy_error_from_label(y, t)),
        "1.7053"
    );
}

#[test]
fn test_numerical_diff() {
    fn function_1(x: f64) -> f64 {
        0.01 * x.powi(2) + 0.1 * x
    }
    dbg!(numerical_diff(function_1, 5.0));
}

#[test]
fn test_numerical_gradient() {
    fn funtion_2(x: &na::DVector<f64>) -> f64 {
        x[0].powi(2) + x[1].powi(2)
    }
    dbg!(numerical_gradient(
        funtion_2,
        &mut DVector::from_vec(vec![3.0, 4.0])
    ));
}

#[test]
fn test_gradient_descent() {
    fn funtion_2(x: &na::DVector<f64>) -> f64 {
        x[0].powi(2) + x[1].powi(2)
    }
    let init_x = DVector::from_vec(vec![-3.0, 4.0]);
    dbg!(gradient_descent(funtion_2, init_x));
}

fn main() {}
