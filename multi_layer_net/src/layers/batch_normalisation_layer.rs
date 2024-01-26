use core::panic;
use std::{cell::RefCell, rc::Rc};

use nalgebra::dmatrix;
use rand_distr::num_traits::Pow;

use super::Layer;

pub struct BatchNormalisationLayer {
    gamma: Rc<RefCell<na::DVector<f64>>>,
    beta: Rc<RefCell<na::DVector<f64>>>,
    momentum: f64,
    running_mean: na::DVector<f64>,
    running_var: na::DVector<f64>,
    batch_size: usize,
    xc: na::DMatrix<f64>,
    xn: na::DMatrix<f64>,
    std: na::DVector<f64>,
    pub dgamma: na::DVector<f64>,
    pub dbeta: na::DVector<f64>,
}

impl BatchNormalisationLayer {
    pub fn new(
        gamma: Rc<RefCell<na::DVector<f64>>>,
        beta: Rc<RefCell<na::DVector<f64>>>,
        momentum: f64,
    ) -> Self {
        Self {
            gamma,
            beta,
            momentum,
            running_mean: na::DVector::<f64>::zeros(0),
            running_var: na::DVector::<f64>::zeros(0),
            batch_size: 0,
            xc: na::DMatrix::<f64>::zeros(0, 0),
            xn: na::DMatrix::<f64>::zeros(0, 0),
            std: na::DVector::<f64>::zeros(0),
            dgamma: na::DVector::<f64>::zeros(0),
            dbeta: na::DVector::<f64>::zeros(0),
        }
    }
}

impl Layer for BatchNormalisationLayer {
    fn forwards(&mut self, x: &na::DMatrix<f64>, train_flg: bool) -> na::DMatrix<f64> {
        if self.running_mean.is_empty() {
            let d = x.ncols();
            self.running_mean = na::DVector::<f64>::zeros(d);
            self.running_var = na::DVector::<f64>::zeros(d);
        }
        let mut xc = x.clone();
        let mut xn;
        if train_flg {
            let mu = x.row_mean_tr();
            for i in 0..xc.nrows() {
                for j in 0..xc.ncols() {
                    let index = (i, j);
                    xc[index] = x[index] - mu[j];
                }
            }
            let var = x.row_variance_tr();
            let mut std = var.clone();
            std.apply(|a| *a = (*a + 10e-7).sqrt());
            xn = xc.clone();
            for i in 0..xn.nrows() {
                for j in 0..xn.ncols() {
                    let index = (i, j);
                    xn[index] = xc[index] / std[j];
                }
            }
            self.batch_size = x.nrows();
            self.xc = xc;
            self.xn = xn.clone();
            self.std = std;
            self.running_mean = self.momentum * &self.running_mean + (1.0 - self.momentum) * mu;
            self.running_var = self.momentum * &self.running_var + (1.0 - self.momentum) * var;
        } else {
            for i in 0..xc.nrows() {
                for j in 0..xc.ncols() {
                    let index = (i, j);
                    xc[index] = x[index] - self.running_mean[j];
                }
            }
            let mut std = self.running_var.clone();
            std.apply(|a| *a = (*a + 10e-7).sqrt());
            xn = na::DMatrix::zeros(x.nrows(), x.ncols());
            for i in 0..xn.nrows() {
                for j in 0..xn.ncols() {
                    let index = (i, j);
                    xn[index] = xc[index] / std[j];
                }
            }
        }
        let mut out = na::DMatrix::zeros(x.nrows(), x.ncols());
        for i in 0..out.nrows() {
            for j in 0..out.ncols() {
                let index = (i, j);
                out[index] = xn[index] * self.gamma.borrow()[j] + self.beta.borrow()[j];
            }
        }
        out
    }

    fn backwards(&mut self, dout: &na::DMatrix<f64>) -> na::DMatrix<f64> {
        let dbeta = dout.row_sum_tr();
        let dgamma = (&self.xn.component_mul(dout)).row_sum_tr();
        let mut dxn = dout.clone();
        for i in 0..dxn.nrows() {
            for j in 0..dxn.ncols() {
                let index = (i, j);
                dxn[index] = dout[index] * self.gamma.borrow()[j];
            }
        }
        let mut dxc = dxn.clone();
        for i in 0..dxn.nrows() {
            for j in 0..dxn.ncols() {
                let index = (i, j);
                dxc[index] = dxn[index] / self.std[j];
            }
        }
        let mut tmp = dxn.component_mul(&self.xc);
        for i in 0..tmp.nrows() {
            for j in 0..tmp.ncols() {
                let index = (i, j);
                tmp[index] = tmp[index] / (self.std[j] * self.std[j]);
            }
        }
        let dstd = tmp.row_sum_tr() * (-1.0);
        let mut dvar = dstd.clone();
        for i in 0..dvar.nrows() {
            for j in 0..dvar.ncols() {
                let index = (i, j);
                dvar[index] = 0.5 * dstd[index] / self.std[j];
            }
        }
        for i in 0..dxc.nrows() {
            for j in 0..dxc.ncols() {
                let index = (i, j);
                dxc[index] += (2.0 / (self.batch_size as f64)) * self.xc[index] * dvar[j];
            }
        }
        let dmu = dxc.row_sum_tr();
        let mut dx = dxc.clone();
        for i in 0..dxc.nrows() {
            for j in 0..dxc.ncols() {
                let index = (i, j);
                dx[index] = dxc[index] - dmu[j] / (self.batch_size as f64);
            }
        }
        self.dgamma = dgamma;
        self.dbeta = dbeta;
        dx
    }
}

#[test]
fn aaa() {
    let a = dmatrix![1,2,3;4,5,6].cast::<f64>();
    let mu = na::DVector::<f64>::from_fn(a.ncols(), |i, _| a.column(i).mean());
    assert_eq!(a.row_mean().transpose(), mu);
}
