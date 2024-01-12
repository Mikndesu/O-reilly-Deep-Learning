use paste::paste;
use std::{cell::RefCell, rc::Rc};
use two_layer_net::two_layer_net::{Grads, Params};

struct Momentum {
    lr: f64,
    momentum: f64,
    v: V,
}

struct V {
    v_w1: na::DMatrix<f64>,
    v_b1: na::DVector<f64>,
    v_w2: na::DMatrix<f64>,
    v_b2: na::DVector<f64>,
}

impl Momentum {
    pub fn new(lr: f64, momentum: f64) -> Self {
        Self {
            lr,
            momentum,
            v: V::new(),
        }
    }

    pub fn update(&mut self, params: &Rc<RefCell<Params>>, grads: &Grads) {
        if self.v.v_w1.nrows() == 0 {
            self.v.v_w1 = na::DMatrix::<f64>::zeros(grads.d_w1.nrows(), grads.d_w1.ncols());
            self.v.v_w2 = na::DMatrix::<f64>::zeros(grads.d_w2.nrows(), grads.d_w2.ncols());
            self.v.v_b1 = na::DVector::<f64>::zeros(grads.d_b1.nrows());
            self.v.v_b2 = na::DVector::<f64>::zeros(grads.d_b2.nrows());
        }
        macro_rules! momentum_update {
            ($elm:ident) => {
                paste!{
                    self.v.[<v_ $elm>] = self.momentum * &self.v.[<v_ $elm>] - self.lr * &grads.[<d_ $elm>];
                    *params.borrow_mut().$elm.borrow_mut() += &self.v.[<v_ $elm>];
                }
            };
        }
        momentum_update!(w1);
        momentum_update!(b1);
        momentum_update!(w2);
        momentum_update!(b2);
    }
}

impl V {
    fn new() -> Self {
        Self {
            v_w1: na::DMatrix::<f64>::zeros(0, 0),
            v_b1: na::DVector::<f64>::zeros(0),
            v_w2: na::DMatrix::<f64>::zeros(0, 0),
            v_b2: na::DVector::<f64>::zeros(0),
        }
    }
}
