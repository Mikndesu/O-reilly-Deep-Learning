use multi_layer_net::{grads::Grads, params::Params};
use std::{cell::RefCell, rc::Rc};

struct AdaGrad {
    lr: f64,
    h: H,
}

struct H {
    weight_list: Vec<na::DMatrix<f64>>,
    bias_list: Vec<na::DVector<f64>>,
}

impl H {
    fn new() -> Self {
        let weight_list = vec![
            na::DMatrix::<f64>::zeros(0, 0),
            na::DMatrix::<f64>::zeros(0, 0),
        ];
        let bias_list = vec![na::DVector::<f64>::zeros(0), na::DVector::<f64>::zeros(0)];
        Self {
            weight_list,
            bias_list,
        }
    }
}

impl AdaGrad {
    pub fn new(lr: f64) -> Self {
        Self { lr, h: H::new() }
    }

    pub fn update(&mut self, params: &Rc<RefCell<Params>>, grads: &Grads) {
        if self.h.weight_list[0].nrows() == 0 {
            self.h
                .weight_list
                .iter_mut()
                .enumerate()
                .for_each(|(i, m)| m.copy_from(&*grads.d_weight_list[i].borrow()));
            self.h
                .bias_list
                .iter_mut()
                .enumerate()
                .for_each(|(i, v)| v.copy_from(&*grads.d_bias_list[i].borrow()));
        }
        self.h
            .weight_list
            .iter_mut()
            .enumerate()
            .for_each(|(i, m)| {
                *m = grads.d_weight_list[i]
                    .borrow()
                    .component_mul(&*grads.d_weight_list[i].borrow());
            });
        for (i, m) in self.h.weight_list.iter().enumerate() {
            let mut tmp = m.clone();
            tmp.apply(|a| *a = a.sqrt() + 1e7);
            *params.borrow_mut().weight_list[i].borrow_mut() -=
                self.lr * &grads.d_weight_list[i].borrow_mut().component_div(&tmp);
        }
        self.h.bias_list.iter_mut().enumerate().for_each(|(i, m)| {
            *m = grads.d_bias_list[i]
                .borrow()
                .component_mul(&*grads.d_bias_list[i].borrow())
        });
        for (i, m) in self.h.bias_list.iter().enumerate() {
            let mut tmp = m.clone();
            tmp.apply(|a| *a = a.sqrt() + 1e7);
            *params.borrow_mut().bias_list[i].borrow_mut() -=
                self.lr * &grads.d_bias_list[i].borrow_mut().component_div(&tmp);
        }
    }
}
