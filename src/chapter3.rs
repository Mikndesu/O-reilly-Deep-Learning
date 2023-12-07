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
