use overfit_weight_decay::overfit_weight_decay_train;
use plotters::{
    backend::BitMapBackend,
    chart::ChartBuilder,
    drawing::IntoDrawingArea,
    element::PathElement,
    series::LineSeries,
    style::{Color, IntoFont, BLACK, BLUE, RED, WHITE},
};

extern crate nalgebra as na;

pub mod optimiser;
pub mod overfit_weight_decay;

fn main() {
    overfit_weight_decay_train();
}

fn plot_loss(train_loss_list: &Vec<f64>, plot_name: &str) {
    let iters_num = train_loss_list.len();
    let name = format!("{}.png", plot_name).to_string();
    let root = BitMapBackend::new(&name, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut plot = ChartBuilder::on(&root)
        .caption("Loss", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..(iters_num as f64), 0.0..2.5)
        .unwrap();
    plot.configure_mesh().x_desc("Iterations").draw().unwrap();
    plot.draw_series(LineSeries::new(
        (0..iters_num)
            .into_iter()
            .zip(train_loss_list.iter())
            .map(|(x, y)| (x as f64, *y)),
        &RED,
    ))
    .unwrap();
}

fn plot_accuracy(train_accuracy_list: &Vec<f64>, test_accuracy_list: &Vec<f64>, plot_name: &str) {
    let epocn_num = train_accuracy_list.len();
    let name = format!("{}.png", plot_name).to_string();
    let root = BitMapBackend::new(&name, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut plot = ChartBuilder::on(&root)
        .caption("Accuracy", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0..(epocn_num as f64), 0.0..100.0)
        .unwrap();
    plot.configure_mesh()
        .x_desc("Epochs")
        .y_desc("% Accuracy")
        .draw()
        .unwrap();
    plot.draw_series(LineSeries::new(
        train_accuracy_list
            .iter()
            .enumerate()
            .map(|(x, y)| (x as f64, *y * 100.0)),
        &RED,
    ))
    .unwrap()
    .label("Train Dataset Accuracy")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    plot.draw_series(LineSeries::new(
        test_accuracy_list
            .iter()
            .enumerate()
            .map(|(x, y)| (x as f64, *y * 100.0)),
        &BLUE,
    ))
    .unwrap()
    .label("Test Dataset Accuracy")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
    plot.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();
    root.present().unwrap();
}
