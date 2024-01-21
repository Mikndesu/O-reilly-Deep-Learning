use std::{cell::RefCell, rc::Rc};

use multi_layer_net::{grads::Grads, params::Params};

struct SGD {
    lr: f64,
}

impl SGD {
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }

    pub fn update(&self, params: &Rc<RefCell<Params>>, grads: &Grads) {
        for (i, m) in params.borrow_mut().weight_list.iter_mut().enumerate() {
            *m.borrow_mut() -= self.lr * &*grads.d_weight_list[i].borrow();
        }
        for (i, m) in params.borrow_mut().bias_list.iter_mut().enumerate() {
            *m.borrow_mut() -= self.lr * &*grads.d_bias_list[i].borrow();
        }
    }
}
