use super::Layer;

pub struct Sigmoid {
    out: na::DMatrix<f64>,
}

impl Layer for Sigmoid {
    fn forwards(&mut self, x: &na::DMatrix<f64>, train_flg: bool) -> na::DMatrix<f64> {
        self.out = x.clone();
        self.out.apply(|a| *a = 1.0 / (1.0 + (-*a).exp()));
        self.out.clone()
    }

    fn backwards(&mut self, dout: &na::DMatrix<f64>) -> na::DMatrix<f64> {
        let mut tmp = self.out.clone();
        tmp.apply(|a| *a = 1.0 - *a);
        dout.component_mul(&self.out).component_mul(&tmp)
    }
}

impl Sigmoid {
    pub fn new() -> Self {
        Self {
            out: na::DMatrix::<f64>::from_element(0, 0, 0.0),
        }
    }
}

#[test]
fn test_sigmoid() {
    let mut sigmoid_layer = Sigmoid::new();
    let mut out = sigmoid_layer.forwards(
        &na::DMatrix::<f64>::from_vec(2, 2, vec![1.0, 0.0, -0.5, 3.0]),
        false,
    );
    out.apply(|x| *x = (*x * 1000.0).round() / 1000.0);
    assert_eq!(
        out,
        na::DMatrix::<f64>::from_vec(2, 2, vec![0.731, 0.5, 0.378, 0.953])
    );
    let dy = na::DMatrix::<f64>::from_element(2, 2, 1.0);
    let mut out = sigmoid_layer.backwards(&dy);
    out.apply(|x| *x = (*x * 1000.0).round() / 1000.0);
    assert_eq!(
        out,
        na::DMatrix::<f64>::from_vec(2, 2, vec![0.197, 0.250, 0.235, 0.045])
    );
}
