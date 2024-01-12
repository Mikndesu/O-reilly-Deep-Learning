use paste::paste;
use std::{cell::RefCell, rc::Rc};
use two_layer_net::two_layer_net::{Grads, Params};

struct AdaGrad {
    lr: f64,
    h: H,
}

struct H {
    w1: na::DMatrix<f64>,
    b1: na::DVector<f64>,
    w2: na::DMatrix<f64>,
    b2: na::DVector<f64>,
}

impl H {
    fn new() -> Self {
        Self {
            w1: na::DMatrix::<f64>::zeros(0, 0),
            b1: na::DVector::<f64>::zeros(0),
            w2: na::DMatrix::<f64>::zeros(0, 0),
            b2: na::DVector::<f64>::zeros(0),
        }
    }
}

impl AdaGrad {
    pub fn new(lr: f64) -> Self {
        Self { lr, h: H::new() }
    }

    pub fn update(&mut self, params: &Rc<RefCell<Params>>, grads: &Grads) {
        if self.h.w1.nrows() == 0 {
            self.h.w1 = na::DMatrix::<f64>::zeros(grads.d_w1.nrows(), grads.d_w1.ncols());
            self.h.w2 = na::DMatrix::<f64>::zeros(grads.d_w2.nrows(), grads.d_w2.ncols());
            self.h.b1 = na::DVector::<f64>::zeros(grads.d_b1.nrows());
            self.h.b2 = na::DVector::<f64>::zeros(grads.d_b2.nrows());
        }
        macro_rules! ada_grad_update {
            ($elm:ident) => {
                paste! {
                    self.h.$elm += &grads.[<d_ $elm>].component_mul(&grads.[<d_ $elm>]);
                    let mut tmp = self.h.$elm.clone();
                    tmp.apply(|a| *a = a.sqrt() + 1e7);
                    *params.borrow_mut().$elm.borrow_mut() -= self.lr * &grads.[<d_ $elm>].component_div(&tmp);
                }
            };
        }
        ada_grad_update!(w1);
        ada_grad_update!(b1);
        ada_grad_update!(w2);
        ada_grad_update!(b2);
    }
}
