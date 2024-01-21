use std::{cell::RefCell, rc::Rc};

pub struct Params {
    pub weight_list: Vec<Rc<RefCell<na::DMatrix<f64>>>>,
    pub bias_list: Vec<Rc<RefCell<na::DVector<f64>>>>,
}

impl Params {
    pub fn new() -> Self {
        Self {
            weight_list: vec![],
            bias_list: vec![],
        }
    }
}
