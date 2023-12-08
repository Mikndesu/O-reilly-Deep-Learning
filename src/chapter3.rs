use nalgebra::{RowDVector, Storage};
use plotters::{
    backend::BitMapBackend,
    chart::ChartBuilder,
    drawing::IntoDrawingArea,
    series::LineSeries,
    style::{IntoFont, RED, WHITE},
};

fn step_function(x: na::RowDVector<f64>) -> na::RowDVector<i32> {
    x.iter()
        .map(|&t| if t.is_sign_positive() { 1 } else { 0 })
        .collect::<Vec<i32>>()
        .into()
}

fn sigmoid(x: na::RowDVector<f64>) -> na::RowDVector<f64> {
    x.iter()
        .map(|&t| 1.0 / (1.0 + (-t).exp()))
        .collect::<Vec<f64>>()
        .into()
}

fn relu(x: na::RowDVector<f64>) -> na::RowDVector<f64> {
    x.iter().map(|&t| t.max(0.0)).collect::<Vec<f64>>().into()
}

fn identity_fucntion(x: na::RowDVector<f64>) -> na::RowDVector<f64> {
    x
}

fn first_layer(x: na::RowVector2<f64>) -> na::RowDVector<f64> {
    let w1 = na::Matrix2x3::<f64>::new(0.1, 0.3, 0.5, 0.2, 0.4, 0.6);
    let b1 = na::Vector3::<f64>::new(0.1, 0.2, 0.3);
    let a1 = (&x * &w1).transpose() + b1;
    sigmoid(RowDVector::from_row_slice(a1.as_slice()))
}

fn second_layer(z1: na::RowVector3<f64>) -> na::RowDVector<f64> {
    let w2 = na::Matrix3x2::<f64>::new(0.1, 0.4, 0.2, 0.5, 0.3, 0.6);
    let b2 = na::Vector2::<f64>::new(0.1, 0.2);
    let a2 = (&z1 * &w2).transpose() + b2;
    sigmoid(RowDVector::from_row_slice(a2.as_slice()))
}

fn output_layer(z2: na::RowVector2<f64>) -> na::RowDVector<f64> {
    let w3 = na::Matrix2::<f64>::new(0.1, 0.3, 0.2, 0.4);
    let b3 = na::Vector2::<f64>::new(0.1, 0.2);
    let a3 = (&z2 * &w3).transpose() + b3;
    identity_fucntion(RowDVector::from_row_slice(a3.as_slice()))
}

#[test]
fn test_step_function() {
    let x: Vec<f64> = (-500..=500)
        .into_iter()
        .map(|t| (t as f64) * 0.01)
        .collect();
    let y = step_function(x.clone().into());

    let root = BitMapBackend::new("step_function.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut plot = ChartBuilder::on(&root)
        .caption("Step Function", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-5.0f64..5.5f64, 0.0..1.3)
        .unwrap();
    plot.configure_mesh().draw().unwrap();
    plot.draw_series(LineSeries::new(
        x.iter().zip(y.iter()).map(|(x, y)| (*x, (*y).into())),
        &RED,
    ))
    .unwrap();
}

#[test]
fn test_sigmoid() {
    let x: Vec<f64> = (-500..=500)
        .into_iter()
        .map(|t| (t as f64) * 0.01)
        .collect();
    let y = sigmoid(x.clone().into());

    let root = BitMapBackend::new("sigmoid.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut plot = ChartBuilder::on(&root)
        .caption("Sigmoid", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-5.0f64..5.5f64, 0.0..1.3)
        .unwrap();
    plot.configure_mesh().draw().unwrap();
    plot.draw_series(LineSeries::new(
        x.iter().zip(y.iter()).map(|(x, y)| (*x, (*y).into())),
        &RED,
    ))
    .unwrap();
}

#[test]
fn test_relu() {
    let x: Vec<f64> = (-500..=500)
        .into_iter()
        .map(|t| (t as f64) * 0.01)
        .collect();
    let y = relu(x.clone().into());

    let root = BitMapBackend::new("relu.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut plot = ChartBuilder::on(&root)
        .caption("ReLU", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-5.0f64..5.5f64, 0.0..6.0)
        .unwrap();
    plot.configure_mesh().draw().unwrap();
    plot.draw_series(LineSeries::new(
        x.iter().zip(y.iter()).map(|(x, y)| (*x, (*y).into())),
        &RED,
    ))
    .unwrap();
}
