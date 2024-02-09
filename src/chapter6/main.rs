use na::dmatrix;
use over_fit_decay_batch_norm::overfit_weight_decay_batch_norm_train;
// use overfit_weight_decay::overfit_weight_decay_train;
use plotters::{
    backend::BitMapBackend,
    chart::ChartBuilder,
    drawing::IntoDrawingArea,
    element::PathElement,
    series::LineSeries,
    style::{Color, IntoFont, BLACK, BLUE, RED, WHITE},
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, ParallelBridge, ParallelIterator,
};

extern crate nalgebra as na;

pub mod optimiser;
mod over_fit_decay_batch_norm;
mod overfit_weight_decay;

fn main() {
    // overfit_weight_decay_train();
    overfit_weight_decay_batch_norm_train();
}

fn plot_loss(train_loss_list: &Vec<f64>, plot_name: &str) {
    let iters_num = train_loss_list.len();
    let y_max = train_loss_list
        .iter()
        .max_by(|a, b| a.total_cmp(b))
        .unwrap();
    let name = format!("{}.png", plot_name).to_string();
    let root = BitMapBackend::new(&name, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut plot = ChartBuilder::on(&root)
        .caption("Loss", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..(iters_num as f64), 0.0..(y_max * 1.1))
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

#[test]
fn aaa() {
    // let mut x = na::dmatrix![1,2,3;4,5,6;7,8,9;10,11,12;].cast::<f64>();
    let mut x = init_matrix_with_standard_normal(10000, 100);
    let mut y = x.clone();
    let mu = x.row_mean_tr();
    let start1 = std::time::Instant::now();
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            let index = (i, j);
            x[index] = x[index] - mu[j];
        }
    }
    let end1 = start1.elapsed();
    fn broadcast_vector_rowwise<T: na::Scalar + Copy + rand_distr::num_traits::Zero>(
        vec: &na::DVector<T>,
        nrows: usize,
    ) -> na::DMatrix<T> {
        na::DMatrix::<T>::from_row_slice(
            nrows,
            vec.nrows(),
            vec.as_slice().repeat(nrows).as_slice(),
        )
    }
    fn init_matrix_with_standard_normal(row: usize, column: usize) -> na::DMatrix<f64> {
        let mut matrix = na::DMatrix::<f64>::zeros(row, column);
        matrix.iter_mut().for_each(|t| {
            *t = rand::Rng::sample(&mut rand::thread_rng(), rand_distr::StandardNormal)
        });
        matrix
    }
    let start2 = std::time::Instant::now();
    y.par_column_iter_mut()
        .zip(mu.as_slice().par_iter())
        .for_each(|(mut col, a)| col.add_scalar_mut(-*a));
    let end2 = start2.elapsed();
    println!(
        "end1 {}.{:03}s end2 {}.{:03}s",
        end1.as_secs(),
        end1.subsec_millis(),
        end2.as_secs(),
        end2.subsec_millis(),
    );
    assert_eq!(x, y);
    panic!("");
}
