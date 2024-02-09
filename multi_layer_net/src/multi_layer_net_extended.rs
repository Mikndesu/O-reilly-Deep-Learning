use std::{cell::RefCell, rc::Rc};

use nalgebra::Scalar;
use rand_distr::num_traits::Pow;

use crate::{
    grads_exteded::GradsExt,
    init_matrix_with_standard_normal,
    layers::{
        affine_layer::Affine, batch_normalisation_layer::BatchNormalisationLayer, relu_layer::Relu,
        sigmoid_layer::Sigmoid, softmax_with_loss_layer::SoftmaxWithLoss, Layer,
    },
    params_extended::ParamsExt,
};

pub struct MultiLayerNetExtended {
    input_size: usize,
    hidden_size_list: Vec<usize>,
    output_size: usize,
    hidden_layer_num: usize,
    pub params: Rc<RefCell<ParamsExt>>,
    pub grads: GradsExt,
    layers: Vec<Rc<RefCell<dyn Layer>>>,
    last_layer: SoftmaxWithLoss,
    weight_decay_lambda: f64,
}

impl MultiLayerNetExtended {
    pub fn new(
        input_size: usize,
        hidden_size_list: Vec<usize>,
        output_size: usize,
        weight_decay_lambda: f64,
        weight_init_std: &str,
        activation: &str,
    ) -> Self {
        let mut layers: Vec<Rc<RefCell<dyn Layer>>> = vec![];
        let params = Rc::new(RefCell::new(init_weight(
            input_size,
            &hidden_size_list,
            output_size,
            weight_init_std,
        )));
        for idx in 0..hidden_size_list.len() {
            layers.push(Rc::new(RefCell::new(Affine::new(
                params.borrow().weight_list[idx].clone(),
                params.borrow().bias_list[idx].clone(),
            ))));
            layers.push(Rc::new(RefCell::new(BatchNormalisationLayer::new(
                params.borrow().gamma_list[idx].clone(),
                params.borrow().beta_list[idx].clone(),
                0.9,
            ))));
            layers.push(activation_layer(activation));
        }
        layers.push(Rc::new(RefCell::new(Affine::new(
            params.borrow().weight_list[hidden_size_list.len()].clone(),
            params.borrow().bias_list[hidden_size_list.len()].clone(),
        ))));
        let grads = GradsExt::new(params.borrow().weight_list.len());
        Self {
            input_size,
            output_size,
            hidden_layer_num: hidden_size_list.len(),
            hidden_size_list,
            weight_decay_lambda,
            last_layer: SoftmaxWithLoss::new(),
            layers,
            params,
            grads,
        }
    }

    pub fn predict(&self, x: &na::DMatrix<f64>, train_flg: bool) -> na::DMatrix<f64> {
        let mut x = x.clone();
        for layer in self.layers.iter() {
            x = layer.borrow_mut().forwards(&x, train_flg);
        }
        x
    }

    pub fn loss(&mut self, x: &na::DMatrix<f64>, t: &na::DMatrix<u8>, train_flg: bool) -> f64 {
        let y = self.predict(x, train_flg);
        let mut weight_decay = 0.0;
        for idx in 0..=self.hidden_layer_num {
            weight_decay += 0.5
                * self.weight_decay_lambda
                * self.params.borrow().weight_list[idx].borrow().norm().pow(2);
        }
        self.last_layer.forwards(&y, t) + weight_decay
    }

    pub fn accuracy(&self, x: &na::DMatrix<f64>, t: &na::DMatrix<u8>) -> f64 {
        let y = self.predict(x, false);
        fn max_index_by_column<T>(matrix: &na::DMatrix<T>) -> Vec<(usize, usize)>
        where
            T: Scalar + PartialOrd,
        {
            matrix
                .row_iter()
                .enumerate()
                .map(|(i, column)| -> (usize, usize) {
                    let tmp = column.transpose().argmax().0;
                    (i, tmp)
                })
                .collect()
        }
        let y_max: Vec<(usize, usize)> = max_index_by_column(&y);
        let t_max: Vec<(usize, usize)> = max_index_by_column(t);
        let result = y_max
            .iter()
            .zip(t_max.iter())
            .map(|(_y, _t)| -> i32 {
                if _y == _t {
                    1
                } else {
                    0
                }
            })
            .collect::<Vec<i32>>()
            .iter()
            .sum::<i32>();
        (result as f64 / y.shape().0 as f64).into()
    }

    pub fn gradient(&mut self, x: &na::DMatrix<f64>, t: &na::DMatrix<u8>) {
        self.loss(x, t, true);
        let mut dout = self.last_layer.backwards(1.0);
        for layer in self.layers.iter().rev() {
            dout = layer.borrow_mut().backwards(&dout);
        }
        for idx in 0..=self.hidden_layer_num {
            match self.layers[idx * 3].borrow().downcast_ref::<Affine>() {
                Some(affine) => {
                    self.grads.d_weight_list[idx] = Rc::new(RefCell::new(
                        &affine.dw + &*affine.w.borrow() * self.weight_decay_lambda,
                    ));
                    self.grads.d_bias_list[idx] = Rc::new(RefCell::new(affine.db.clone()));
                }
                None => panic!("downcasting could not be performed."),
            }
            if idx != self.hidden_layer_num {
                match self.layers[(idx * 3) + 1]
                    .borrow()
                    .downcast_ref::<BatchNormalisationLayer>()
                {
                    Some(batch_layer) => {
                        self.grads.d_gamma_list[idx] =
                            Rc::new(RefCell::new(batch_layer.dgamma.clone()));
                        self.grads.d_beta_list[idx] =
                            Rc::new(RefCell::new(batch_layer.dbeta.clone()));
                    }
                    None => panic!("downcasting could not be performed."),
                }
            }
        }
    }
}

fn activation_layer(activation_layer: &str) -> Rc<RefCell<dyn Layer>> {
    if activation_layer == "relu" {
        Rc::new(RefCell::new(Relu::new()))
    } else if activation_layer == "sigmoid" {
        Rc::new(RefCell::new(Sigmoid::new()))
    } else {
        panic!("Unknown layer name was given.");
    }
}

fn init_weight(
    input_size: usize,
    hidden_size_list: &Vec<usize>,
    output_size: usize,
    weight_init_std: &str,
) -> ParamsExt {
    let mut all_size_list: Vec<usize> = vec![];
    all_size_list.push(input_size);
    all_size_list.extend(hidden_size_list);
    all_size_list.push(output_size);
    let mut params = ParamsExt::new(all_size_list.len());
    for idx in 0..all_size_list.len() - 1 {
        let mut scale = 0.05;
        if weight_init_std == "relu" || weight_init_std == "he" {
            scale = (2.0 / all_size_list[idx] as f64).sqrt();
        } else if weight_init_std == "sigmoid" || weight_init_std == "xavier" {
            scale = (1.0 / all_size_list[idx] as f64).sqrt();
        }
        params.weight_list[idx] = Rc::new(RefCell::new(
            scale * init_matrix_with_standard_normal(all_size_list[idx], all_size_list[idx + 1]),
        ));
        params.bias_list[idx] = Rc::new(RefCell::new(na::DVector::<f64>::zeros(
            all_size_list[idx + 1],
        )));
        params.gamma_list[idx] = Rc::new(RefCell::new(na::DVector::<f64>::from_element(
            all_size_list[idx + 1],
            1.0,
        )));
        params.beta_list[idx] = Rc::new(RefCell::new(na::DVector::<f64>::zeros(
            all_size_list[idx + 1],
        )));
    }
    params
}
