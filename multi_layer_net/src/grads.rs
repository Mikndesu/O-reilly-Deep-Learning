use std::{cell::RefCell, rc::Rc};

#[derive(Clone, Debug)]
pub struct Grads {
    pub d_weight_list: Vec<Rc<RefCell<na::DMatrix<f64>>>>,
    pub d_bias_list: Vec<Rc<RefCell<na::DVector<f64>>>>,
}

impl Grads {
    pub fn new() -> Self {
        Self {
            d_weight_list: vec![],
            d_bias_list: vec![],
        }
    }
}
