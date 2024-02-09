use std::time::Instant;

use ::multi_layer_net::multi_layer_net_extended;
use multi_layer_net::multi_layer_net;
use mylib::mnist::{self, load_label, load_normalised_image, DatasetType};
use rand::seq::IteratorRandom;

use crate::{
    optimiser::{sgd, sgd_ext},
    plot_accuracy, plot_loss,
};

fn train() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let dataset_dir = std::env::current_dir().unwrap().join("dataset");
    mnist::init_mnist();
    let train_img = load_normalised_image(DatasetType::TrainImg, &dataset_dir)
        .flatten()
        .columns(0, 300)
        * 1.0;
    let train_label = load_label(DatasetType::TrainLabel, &dataset_dir)
        .as_one_hot()
        .rows(0, 300)
        * 1;
    let test_img = load_normalised_image(DatasetType::TestImg, &dataset_dir).flatten();
    let test_label = load_label(DatasetType::TestLabel, &dataset_dir).as_one_hot();
    let mut network = multi_layer_net_extended::MultiLayerNetExtended::new(
        784,
        [100; 6].to_vec(),
        10,
        0.1,
        "relu",
        "relu",
    );
    let mut rng = rand::thread_rng();
    let optimiser = sgd_ext::SGDExt::new(0.01);
    let max_epochs = 201;
    let train_size = train_img.shape().1;
    let batch_size = 100;
    let mut train_loss_list = vec![];
    let mut train_accuracy_list = vec![];
    let mut test_accuracy_list = vec![];
    let iter_per_epoch = 1.max(train_size / batch_size);
    let mut epoch_count = 0;
    for i in 0..10i32.pow(9) {
        let mut img_batch = na::DMatrix::<f64>::zeros(batch_size, 784);
        let mut label_batch = na::DMatrix::<u8>::zeros(batch_size, 10);
        let batch_mask = (0..train_size).choose_multiple(&mut rng, batch_size);
        img_batch
            .row_iter_mut()
            .zip(batch_mask.iter())
            .for_each(|(mut column, n)| column.copy_from_slice(&train_img.column(*n).as_slice()));
        label_batch
            .row_iter_mut()
            .zip(batch_mask.iter())
            .for_each(|(mut column, n)| column.copy_from(&train_label.row(*n)));
        network.gradient(&img_batch, &label_batch);
        optimiser.update(&network.params, &network.grads);
        let loss = network.loss(&img_batch, &label_batch, false);
        train_loss_list.push(loss);
        if i == 0 || (i + 1) % (iter_per_epoch as i32) == 0 {
            let mut img_matrix = na::DMatrix::<f64>::from_element(train_img.ncols(), 784, 0.0);
            img_matrix
                .row_iter_mut()
                .zip((0..train_img.ncols()).into_iter())
                .for_each(|(mut column, n)| {
                    column.copy_from_slice(&train_img.column(n).as_slice())
                });
            let train_acc = network.accuracy(
                &img_matrix,
                &train_label
                    .clone()
                    .resize(train_label.shape().0, train_label.shape().1, 0),
            );
            let mut img_matrix = na::DMatrix::<f64>::from_element(test_img.ncols(), 784, 0.0);
            img_matrix
                .row_iter_mut()
                .zip((0..test_img.ncols()).into_iter())
                .for_each(|(mut column, n)| column.copy_from_slice(&test_img.column(n).as_slice()));
            let test_acc = network.accuracy(
                &img_matrix,
                &test_label
                    .clone()
                    .resize(test_label.shape().0, test_label.shape().1, 0),
            );
            train_accuracy_list.push(train_acc);
            test_accuracy_list.push(test_acc);
            print!("Training has done {} times! ", i + 1);
            println!(
                "Train Acc. {:.1}% Test Acc. {:.1}%",
                train_acc * 100.0,
                test_acc * 100.0
            );
            epoch_count += 1;
            if epoch_count >= max_epochs {
                break;
            }
        }
    }
    (train_loss_list, train_accuracy_list, test_accuracy_list)
}

pub fn overfit_weight_decay_batch_norm_train() {
    let start = Instant::now();
    let (train_loss_list, train_accuracy_list, test_accuracy_list) = train();
    let end = start.elapsed();
    println!("Training has finished! Now starting to plot.");
    plot_loss(&train_loss_list, "Iteration Overfit");
    plot_accuracy(
        &train_accuracy_list,
        &test_accuracy_list,
        "Accuracy Overfit",
    );
    println!(
        "Training takes {}.{:03}s",
        end.as_secs(),
        end.subsec_millis()
    );
    println!(
        "Testdata accuracy is {:.1}%",
        test_accuracy_list.last().unwrap() * 100.0
    );
}
