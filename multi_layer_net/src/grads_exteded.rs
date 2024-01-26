use std::{cell::RefCell, rc::Rc};

#[derive(Clone, Debug)]
pub struct GradsExt {
    pub d_weight_list: Vec<Rc<RefCell<na::DMatrix<f64>>>>,
    pub d_bias_list: Vec<Rc<RefCell<na::DVector<f64>>>>,
    pub d_gamma_list: Vec<Rc<RefCell<na::DVector<f64>>>>,
    pub d_beta_list: Vec<Rc<RefCell<na::DVector<f64>>>>,
}

impl GradsExt {
    pub fn new(size: usize) -> Self {
        Self {
            d_weight_list: vec![Rc::new(RefCell::new(na::DMatrix::<f64>::zeros(0, 0))); size],
            d_bias_list: vec![Rc::new(RefCell::new(na::DVector::<f64>::zeros(0))); size],
            d_gamma_list: vec![Rc::new(RefCell::new(na::DVector::<f64>::zeros(0))); size],
            d_beta_list: vec![Rc::new(RefCell::new(na::DVector::<f64>::zeros(0))); size],
        }
    }
}
