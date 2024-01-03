pub struct SoftmaxWithLoss {
    y: na::DMatrix<f64>,
    t: na::DMatrix<u8>,
    loss: f64,
}

impl SoftmaxWithLoss {
    pub fn new() -> Self {
        Self {
            y: na::DMatrix::<f64>::from_element(0, 0, 0.0),
            t: na::DMatrix::<u8>::from_element(0, 0, 0),
            loss: 0.0f64,
        }
    }

    pub fn forwards(&mut self, x: na::DMatrix<f64>, t: na::DMatrix<u8>) -> f64 {
        self.t = t;
        self.y = Self::softmax(&x);
        self.loss = Self::cross_entropy_error(&self.y, &self.t);
        self.loss
    }

    pub fn backwards(&self, dout: f64) -> na::DMatrix<f64> {
        let batch_size = self.t.shape().0;
        let mut tmp = &self.y - &self.t.clone().cast::<f64>();
        tmp.apply(|a| *a = *a / batch_size as f64);
        tmp
    }

    fn softmax(x: &na::DMatrix<f64>) -> na::DMatrix<f64> {
        let mut matrix = x.clone();
        matrix.row_iter_mut().for_each(|mut column| -> () {
            let c = column.max();
            let exp_x = column.iter().map(|&t| (t - c).exp()).collect::<Vec<f64>>();
            let sum = exp_x.iter().sum::<f64>();
            column
                .iter_mut()
                .zip(exp_x.iter())
                .for_each(|(column, &_exp_x)| *column = _exp_x / sum);
        });
        matrix
    }

    fn cross_entropy_error(y: &na::DMatrix<f64>, t: &na::DMatrix<u8>) -> f64 {
        let delta = 1e-7;
        let batch_size = y.shape().0 as f64;
        -t.iter()
            .zip(y.iter())
            .map(|(a, b)| (b + delta).ln() * (*a as f64))
            .collect::<Vec<f64>>()
            .iter()
            .sum::<f64>()
            / batch_size
    }
}
