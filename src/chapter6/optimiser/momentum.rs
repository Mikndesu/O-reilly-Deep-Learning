use multi_layer_net::{grads::Grads, params::Params};
use std::{cell::RefCell, rc::Rc};

struct Momentum {
    lr: f64,
    momentum: f64,
    v: V,
}

struct V {
    v_weight_list: Vec<na::DMatrix<f64>>,
    v_bias_list: Vec<na::DVector<f64>>,
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
        if self.v.v_weight_list[0].nrows() == 0 {
            self.v
                .v_weight_list
                .iter_mut()
                .enumerate()
                .for_each(|(i, m)| m.copy_from(&grads.d_bias_list[i].borrow()));
            self.v
                .v_bias_list
                .iter_mut()
                .enumerate()
                .for_each(|(i, v)| v.copy_from(&grads.d_bias_list[i].borrow()));
        }
        self.v
            .v_weight_list
            .iter_mut()
            .enumerate()
            .for_each(|(i, m)| {
                *m = self.momentum * &*m - self.lr * &*grads.d_weight_list[i].borrow()
            });
        for (i, _) in self.v.v_weight_list.iter().enumerate() {
            *params.borrow_mut().weight_list[i].borrow_mut() += &self.v.v_weight_list[i];
        }
        self.v
            .v_bias_list
            .iter_mut()
            .enumerate()
            .for_each(|(i, m)| {
                *m = self.momentum * &*m - self.lr * &*grads.d_bias_list[i].borrow()
            });
        for (i, _) in self.v.v_bias_list.iter().enumerate() {
            *params.borrow_mut().bias_list[i].borrow_mut() += &self.v.v_bias_list[i];
        }
    }
}

impl V {
    fn new() -> Self {
        let v_weight_list = vec![
            na::DMatrix::<f64>::zeros(0, 0),
            na::DMatrix::<f64>::zeros(0, 0),
        ];
        let v_bias_list = vec![na::DVector::<f64>::zeros(0), na::DVector::<f64>::zeros(0)];
        Self {
            v_weight_list,
            v_bias_list,
        }
    }
}
