use std::{cell::RefCell, rc::Rc};

use multi_layer_net::{grads_exteded::GradsExt, params_extended::ParamsExt};

pub struct SGDExt {
    lr: f64,
}

impl SGDExt {
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }

    pub fn update(&self, params: &Rc<RefCell<ParamsExt>>, grads: &GradsExt) {
        for (i, m) in params.borrow_mut().weight_list.iter_mut().enumerate() {
            *m.borrow_mut() -= self.lr * &*grads.d_weight_list[i].borrow();
        }
        for (i, m) in params.borrow_mut().bias_list.iter_mut().enumerate() {
            *m.borrow_mut() -= self.lr * &*grads.d_bias_list[i].borrow();
        }
        for (i, m) in params.borrow_mut().gamma_list.iter_mut().enumerate() {
            if !grads.d_gamma_list[i].borrow().is_empty() {
                *m.borrow_mut() -= self.lr * &*grads.d_gamma_list[i].borrow();
            }
        }
        for (i, m) in params.borrow_mut().beta_list.iter_mut().enumerate() {
            if !grads.d_gamma_list[i].borrow().is_empty() {
                *m.borrow_mut() -= self.lr * &*grads.d_beta_list[i].borrow();
            }
        }
    }
}
