use std::{cell::RefCell, rc::Rc};

pub struct ParamsExt {
    pub weight_list: Vec<Rc<RefCell<na::DMatrix<f64>>>>,
    pub bias_list: Vec<Rc<RefCell<na::DVector<f64>>>>,
    pub gamma_list: Vec<Rc<RefCell<na::DVector<f64>>>>,
    pub beta_list: Vec<Rc<RefCell<na::DVector<f64>>>>,
}

impl ParamsExt {
    pub fn new(size: usize) -> Self {
        Self {
            weight_list: vec![Rc::new(RefCell::new(na::DMatrix::<f64>::zeros(0, 0))); size],
            bias_list: vec![Rc::new(RefCell::new(na::DVector::<f64>::zeros(0))); size],
            gamma_list: vec![Rc::new(RefCell::new(na::DVector::<f64>::zeros(0))); size],
            beta_list: vec![Rc::new(RefCell::new(na::DVector::<f64>::zeros(0))); size],
        }
    }
}
