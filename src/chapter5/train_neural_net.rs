use mylib::mnist::{self, load_label, load_normalised_image, DatasetType};
use rand::seq::IteratorRandom;

use crate::two_layer_net;

pub fn train_neural_net() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let dataset_dir = std::env::current_dir().unwrap().join("dataset");
    mnist::init_mnist();
    let train_img = load_normalised_image(DatasetType::TrainImg, &dataset_dir).flatten();
    let train_label = load_label(DatasetType::TrainLabel, &dataset_dir).as_one_hot();
    let test_img = load_normalised_image(DatasetType::TestImg, &dataset_dir).flatten();
    let test_label = load_label(DatasetType::TestLabel, &dataset_dir).as_one_hot();
    let mut network = two_layer_net::TwoLayerNet::new(784, 50, 10);
    let mut rng = rand::thread_rng();
    let iters_num = 10000;
    let train_size = train_img.shape().1;
    let batch_size = 100;
    let learning_rate = 0.1;
    let mut train_loss_list = vec![];
    let mut train_accuracy_list = vec![];
    let mut test_accuracy_list = vec![];
    let iter_per_epoch = 1.max(train_size / batch_size);
    let mut img_batch = na::DMatrix::<f64>::zeros(batch_size, 784);
    let mut label_batch = na::DMatrix::<u8>::zeros(batch_size, 10);
    for i in 0..iters_num {
        if i % 100 == 0 {
            println!("Now {} times iteration has finished", i);
        }
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
        *network.params.borrow_mut().w1.borrow_mut() -= learning_rate * &network.grads.d_w1;
        *network.params.borrow_mut().b1.borrow_mut() -= learning_rate * &network.grads.d_b1;
        *network.params.borrow_mut().w2.borrow_mut() -= learning_rate * &network.grads.d_w2;
        *network.params.borrow_mut().b2.borrow_mut() -= learning_rate * &network.grads.d_b2;
        let loss = network.loss(&img_batch, &label_batch);
        train_loss_list.push(loss);
        if i == 0 || (i + 1) % iter_per_epoch == 0 {
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
            println!("Train Acc. {} Test Acc. {}", train_acc, test_acc);
        }
    }
    (train_loss_list, train_accuracy_list, test_accuracy_list)
}
