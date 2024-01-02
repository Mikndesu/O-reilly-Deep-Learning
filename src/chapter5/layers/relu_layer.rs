struct Relu {
    mask: na::DMatrix<bool>,
}

impl Relu {
    pub fn new() -> Self {
        Self {
            mask: na::DMatrix::<bool>::from_element(0, 0, false),
        }
    }

    pub fn forwards(&mut self, x: &na::DMatrix<f64>) -> na::DMatrix<f64> {
        self.mask = na::DMatrix::<bool>::from_element(x.shape().0, x.shape().1, false);
        let mut output = x.clone();
        for i in 0..x.shape().0 {
            for j in 0..x.shape().1 {
                let index = (i, j);
                if x[index] <= 0.0 {
                    self.mask[index] = true;
                    output[index] = 0.0;
                }
            }
        }
        output
    }

    pub fn backwards<'a>(&self, dout: &'a mut na::DMatrix<f64>) -> &'a na::DMatrix<f64> {
        for i in 0..dout.shape().0 {
            for j in 0..dout.shape().1 {
                let index = (i, j);
                if self.mask[index] == true {
                    dout[index] = 0.0;
                }
            }
        }
        dout
    }
}

#[test]
fn test_relu() {
    let mut relu_layer = Relu::new();
    assert_eq!(
        relu_layer.forwards(&na::DMatrix::<f64>::from_vec(
            2,
            2,
            vec![1.0, 0.0, -0.5, 3.0]
        )),
        na::DMatrix::<f64>::from_vec(2, 2, vec![1.0, 0.0, 0.0, 3.0])
    );
    assert_eq!(
        relu_layer.mask,
        na::DMatrix::<bool>::from_vec(2, 2, vec![false, true, true, false])
    );
    let mut dy = na::DMatrix::<f64>::from_element(2, 2, 1.0);
    assert_eq!(
        *relu_layer.backwards(&mut dy),
        na::DMatrix::<f64>::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0])
    );
}
