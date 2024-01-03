struct Affine {
    w: na::DMatrix<f64>,
    b: na::DVector<f64>,
    x: na::DMatrix<f64>,
    dw: na::DMatrix<f64>,
    db: na::DVector<f64>,
}

impl Affine {
    pub fn new(w: na::DMatrix<f64>, b: na::DVector<f64>) -> Self {
        Self {
            w,
            b,
            x: na::DMatrix::<f64>::from_element(0, 0, 0.0),
            dw: na::DMatrix::<f64>::from_element(0, 0, 0.0),
            db: na::DVector::<f64>::from_element(0, 0.0),
        }
    }

    pub fn forwards(&mut self, x: na::DMatrix<f64>) -> na::DMatrix<f64> {
        self.x = x;
        let B = na::DMatrix::<f64>::from_row_slice(
            self.x.nrows(),
            self.b.nrows(),
            self.b.as_slice().repeat(self.x.nrows()).as_slice(),
        );
        &self.x * &self.w + B
    }

    pub fn backwards(&mut self, dout: na::DMatrix<f64>) -> na::DMatrix<f64> {
        let dx = &dout * &self.w.transpose();
        self.dw = &self.x.transpose() * &dout;
        self.db = na::DVector::<f64>::from_fn(dout.ncols(), |i, _| dout.column(i).sum());
        dx
    }
}

#[test]
fn test_affine() {
    let x = na::DMatrix::<f64>::from_element(20, 50, 0.0);
    let w = na::DMatrix::<f64>::from_element(50, 10, 0.0);
    let b = na::DVector::<f64>::from_element(10, 0.0);
    let mut affine = Affine::new(w, b);
    dbg!(affine.forwards(x).shape());
    let dy = na::DMatrix::<f64>::from_element(20, 10, 0.0);
    dbg!(affine.backwards(dy).shape());
    dbg!(affine.dw.shape());
    dbg!(affine.db.shape());
}
