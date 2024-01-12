use std::{cell::RefCell, rc::Rc};

use two_layer_net::two_layer_net::{Grads, Params};

struct SGD {
    lr: f64,
}

impl SGD {
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }

    pub fn update(&self, params: &Rc<RefCell<Params>>, grads: &Grads) {
        *params.borrow_mut().w1.borrow_mut() -= self.lr * &grads.d_w1;
        *params.borrow_mut().b1.borrow_mut() -= self.lr * &grads.d_b1;
        *params.borrow_mut().w2.borrow_mut() -= self.lr * &grads.d_w2;
        *params.borrow_mut().b2.borrow_mut() -= self.lr * &grads.d_b2;
    }
}
