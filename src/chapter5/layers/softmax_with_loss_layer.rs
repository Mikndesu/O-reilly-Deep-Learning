use nalgebra::DMatrix;

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

    pub fn forwards(&mut self, x: &na::DMatrix<f64>, t: &na::DMatrix<u8>) -> f64 {
        self.t = t.clone();
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

#[test]
fn test_softmax_with_loss() {
    let x = na::DMatrix::<f64>::from_row_slice(
        3,
        10,
        &vec![
            1.0, 3.0, 5.0, 7.0, 9.0, 1.5, 3.5, 5.5, 7.5, 9.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0,
        ],
    );
    let t = na::DMatrix::<u8>::from_row_slice(
        3,
        10,
        &vec![
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1,
        ],
    );
    let mut net = SoftmaxWithLoss::new();
    dbg!(net.forwards(&x, &t));
    dbg!(net.backwards(1.0f64));
}
