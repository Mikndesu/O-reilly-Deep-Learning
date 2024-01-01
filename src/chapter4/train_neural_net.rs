use mylib::mnist::{self, load_label, load_normalised_image, DatasetType};
use plotters::{
    backend::BitMapBackend,
    chart::ChartBuilder,
    drawing::IntoDrawingArea,
    series::LineSeries,
    style::{IntoFont, RED, WHITE},
};
use rand::seq::IteratorRandom;

use crate::two_layer_net;

pub fn train_neural_net() {
    let dataset_dir = std::env::current_dir().unwrap().join("dataset");
    mnist::init_mnist();
    let train_img = load_normalised_image(DatasetType::TrainImg, &dataset_dir).flatten();
    let train_label = load_label(DatasetType::TrainLabel, &dataset_dir).as_one_hot();
    let test_img = load_normalised_image(DatasetType::TestImg, &dataset_dir).flatten();
    let test_label = load_label(DatasetType::TestLabel, &dataset_dir).as_one_hot();
    let mut network = two_layer_net::TwoLayerNet::new(784, 50, 10);
    let mut rng = rand::thread_rng();
    let iters_num = 1800;
    let train_size = train_img.shape().0;
    let batch_size = 600;
    let learning_rate = 0.1;
    let mut train_loss_list = vec![];
    let mut train_accuracy_list = vec![];
    let mut test_accuracy_list = vec![];
    let iter_per_epoch = 1.max(train_size / batch_size);
    let mut img_batch = na::DMatrix::<f64>::zeros(batch_size, 784);
    let mut label_batch = na::DMatrix::<u8>::zeros(batch_size, 10);
    for i in 0..iters_num {
        println!("Now {} times iteration has finished", i);
        let batch_mask = (0..train_size).choose_multiple(&mut rng, batch_size);
        img_batch
            .row_iter_mut()
            .zip(batch_mask.iter())
            .for_each(|(mut column, n)| column.copy_from(&train_img.row(*n)));
        label_batch
            .row_iter_mut()
            .zip(batch_mask.iter())
            .for_each(|(mut column, n)| column.copy_from(&train_label.row(*n)));
        network.numerical_gradient(&img_batch, &label_batch);
        network.params.w1 -= learning_rate * &network.grads.d_w1;
        network.params.b1 -= learning_rate * &network.grads.d_b1;
        network.params.w2 -= learning_rate * &network.grads.d_w2;
        network.params.b2 -= learning_rate * &network.grads.d_b2;
        let loss = network.loss(&img_batch, &label_batch);
        train_loss_list.push(loss);
        if (i + 1) % iter_per_epoch == 0 {
            let train_acc = network.accuracy(
                &train_img
                    .clone()
                    .resize(train_img.shape().0, train_img.shape().1, 0.0),
                &train_label
                    .clone()
                    .resize(train_label.shape().0, train_label.shape().1, 0),
            );
            let test_acc = network.accuracy(
                &test_img
                    .clone()
                    .resize(test_img.shape().0, test_img.shape().1, 0.0),
                &test_label
                    .clone()
                    .resize(test_label.shape().0, test_label.shape().1, 0),
            );
            train_accuracy_list.push(train_acc);
            test_accuracy_list.push(test_acc);
            println!("Train Acc. {} Test Acc. {}", train_acc, test_acc);
        }
    }
    println!("Training has finished! Now starting to plot.");
    let root = BitMapBackend::new("Iteration.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut plot = ChartBuilder::on(&root)
        .caption("Loss", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..(iters_num as f64), 0.0..3.0)
        .unwrap();
    plot.configure_mesh().draw().unwrap();
    plot.draw_series(LineSeries::new(
        (0..iters_num)
            .into_iter()
            .zip(train_loss_list.iter())
            .map(|(x, y)| (x as f64, *y)),
        &RED,
    ))
    .unwrap();

    let root = BitMapBackend::new("Accuracy.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut plot = ChartBuilder::on(&root)
        .caption("Accuracy", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..1.0, 0.0..4.0)
        .unwrap();
    plot.configure_mesh().draw().unwrap();
    plot.draw_series(LineSeries::new(
        (0..iters_num)
            .into_iter()
            .zip(train_accuracy_list.iter())
            .map(|(x, y)| (x as f64, *y)),
        &RED,
    ))
    .unwrap();
    plot.draw_series(LineSeries::new(
        (0..iters_num)
            .into_iter()
            .zip(test_accuracy_list.iter())
            .map(|(x, y)| (x as f64, *y)),
        &RED,
    ))
    .unwrap();
}

#[test]
fn test() {
    train_neural_net();
}
