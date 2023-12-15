use crate::mnist::{self, load_normalised_image, Label, NormalisedImageVec};
use na::Dyn;
use nalgebra as na;
use rand::seq::IteratorRandom;

fn sum_squared_error(y: na::DVector<f32>, t: na::DVector<u8>) -> f32 {
    (y - t.cast::<f32>())
        .iter()
        .map(|x| x.powi(2))
        .collect::<Vec<f32>>()
        .iter()
        .sum::<f32>()
        * 0.5
}

fn cross_entropy_error(
    y: na::OMatrix<f32, Dyn, na::Const<10>>,
    t: na::OMatrix<u8, Dyn, na::Const<10>>,
) -> f32 {
    let delta = 1e-7;
    let batch_size = y.shape().0 as f32;
    -t.cast::<f32>()
        .iter()
        .zip(y.iter())
        .map(|(a, b)| (b + delta).ln() * a)
        .collect::<Vec<f32>>()
        .iter()
        .sum::<f32>()
        / batch_size
}

fn cross_entropy_error_from_label(y: na::OMatrix<f32, Dyn, na::Const<10>>, t: Label) -> f32 {
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
    let y = na::OMatrix::<f32, Dyn, na::Const<10>>::from_vec(vec![
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
    let y = na::OMatrix::<f32, Dyn, na::Const<10>>::from_vec(vec![
        0.1, 0.1, 0.1, 0.05, 0.1, 0.05, 0.6, 0.1, 0.1, 0.0, 0.1, 0.0, 0.05, 0.1, 0.05, 0.1, 0.1,
        0.1, 0.0, 0.1, 0.0, 0.1, 0.1, 0.6, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0,
    ]);
    let t = Label::from(vec![2, 8, 2]);
    assert_eq!(
        format!("{:.4}", cross_entropy_error_from_label(y, t)),
        "1.7053"
    );
}
