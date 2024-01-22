use std::{cell::RefCell, rc::Rc};

pub struct Params {
    pub weight_list: Vec<Rc<RefCell<na::DMatrix<f64>>>>,
    pub bias_list: Vec<Rc<RefCell<na::DVector<f64>>>>,
}

impl Params {
    pub fn new(size: usize) -> Self {
        Self {
            weight_list: vec![Rc::new(RefCell::new(na::DMatrix::<f64>::zeros(0, 0))); size],
            bias_list: vec![Rc::new(RefCell::new(na::DVector::<f64>::zeros(0))); size],
        }
    }
}
