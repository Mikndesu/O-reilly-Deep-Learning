use nalgebra::Const;

use crate::mnist;

fn img_show() {
    let dataset_dir = std::env::current_dir().unwrap().join("dataset");
    mnist::init_mnist();
    let train_img_flattened = mnist::load_image("train-images-idx3-ubyte", &dataset_dir).flatten();
    let train_img_label = mnist::load_label("train-labels-idx1-ubyte", &dataset_dir);
    // println!("{:?}", train_img_label.shape());
    save_img_from_matrix(
        train_img_flattened
            .column(0)
            .reshape_generic(Const::<28>, Const::<28>)
            .into(),
    );
}

fn save_img_from_matrix(matrix: na::OMatrix<u8, Const<28>, Const<28>>) {
    // let matrix = na::OMatrix::<u8, Const<28>, Const<28>>::from_row_slice(slice);
    let mut img = image::GrayImage::new(28, 28);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        *pixel = image::Luma([matrix[(x as usize, y as usize)]; 1]);
    }
    img.save("result.png").unwrap();
}

#[test]
fn test_img_show() {
    img_show();
}
