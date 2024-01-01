extern crate nalgebra as na;
use flate2::bufread::GzDecoder;
use nalgebra::{Const, Dyn, Scalar};
use reqwest::header::USER_AGENT;
use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

const URL_BASE: &str = "http://yann.lecun.com/exdb/mnist/";
pub enum DatasetType {
    TrainImg,
    TrainLabel,
    TestImg,
    TestLabel,
}

impl DatasetType {
    pub fn file_name(&self) -> String {
        match self {
            DatasetType::TrainImg => "train-images-idx3-ubyte",
            DatasetType::TrainLabel => "train-labels-idx1-ubyte",
            DatasetType::TestImg => "t10k-images-idx3-ubyte",
            DatasetType::TestLabel => "t10k-labels-idx1-ubyte",
        }
        .to_string()
    }
    fn values() -> Vec<DatasetType> {
        vec![
            DatasetType::TrainImg,
            DatasetType::TrainLabel,
            DatasetType::TestImg,
            DatasetType::TestLabel,
        ]
    }
}

pub type ImageVec = ImageBase<u8>;
pub type NormalisedImageVec = ImageBase<f64>;

pub struct ImageBase<T: Clone + Scalar> {
    image_vec: Vec<na::OMatrix<T, Const<28>, Const<28>>>,
}

pub struct Label {
    pub label: Vec<u8>,
}

impl<T: Clone + Copy + Scalar> ImageBase<T> {
    pub fn flatten(&self) -> na::OMatrix<T, Dyn, Const<784>> {
        let mut vec: Vec<T> = vec![];
        self.image_vec
            .iter()
            .for_each(|x| x.iter().for_each(|y| vec.push(*y)));
        na::OMatrix::<T, Dyn, Const<784>>::from_vec(vec)
    }
}

impl Label {
    pub fn as_one_hot(&self) -> na::OMatrix<u8, Dyn, na::Const<10>> {
        let mut a = na::OMatrix::<u8, Dyn, Const<10>>::zeros(self.label.len());
        a.row_iter_mut()
            .enumerate()
            .for_each(|(i, mut row)| row[self.label[i] as usize] = 1u8);
        a
    }
}

impl<T: Clone + Copy + Scalar> From<Vec<na::OMatrix<T, Const<28>, Const<28>>>> for ImageBase<T> {
    fn from(value: Vec<na::OMatrix<T, Const<28>, Const<28>>>) -> Self {
        ImageBase::<T> { image_vec: value }
    }
}

impl<T: Clone + Copy + Scalar> AsRef<Vec<na::OMatrix<T, Const<28>, Const<28>>>> for ImageBase<T> {
    fn as_ref(&self) -> &Vec<na::OMatrix<T, Const<28>, Const<28>>> {
        &self.image_vec
    }
}

impl From<Vec<u8>> for Label {
    fn from(value: Vec<u8>) -> Self {
        Label { label: value }
    }
}

impl AsRef<Vec<u8>> for Label {
    fn as_ref(&self) -> &Vec<u8> {
        &self.label
    }
}

pub fn load_label(_type: DatasetType, dataset_dir: &Path) -> Label {
    let mut buf_reader = load(&DatasetType::file_name(&_type), dataset_dir, 8);
    let mut buf: Vec<u8> = vec![];
    let _ = buf_reader.read_to_end(&mut buf);
    buf.into()
}

pub fn load_image(_type: DatasetType, dataset_dir: &Path) -> ImageVec {
    let mut buf_reader = load(&DatasetType::file_name(&_type), dataset_dir, 16);
    let mut vec: Vec<na::OMatrix<u8, Const<28>, Const<28>>> = vec![];
    while let Ok(buf) = read_exact_bytes(&mut buf_reader, 784) {
        let matrix = na::OMatrix::<u8, Const<28>, Const<28>>::from_vec(buf);
        vec.push(matrix);
    }
    vec.into()
}

pub fn load_normalised_image(_type: DatasetType, dataset_dir: &Path) -> NormalisedImageVec {
    let x = load_image(_type, dataset_dir);
    let mut vec: Vec<na::OMatrix<f64, Const<28>, Const<28>>> = vec![];
    x.as_ref().iter().for_each(|t| -> () {
        let mut matrix = t.cast::<f64>();
        matrix.apply(|t| *t /= 255.0);
        vec.push(matrix);
    });
    vec.into()
}

pub fn init_mnist() -> PathBuf {
    let dataset_dir = std::env::current_dir().unwrap().join("dataset");
    DatasetType::values()
        .iter()
        .for_each(|v| download(&format!("{}.gz", v.file_name()), &dataset_dir));
    decode_gzip_files(&dataset_dir);
    dataset_dir
}

fn download(file_name: &str, dataset_dir: &Path) {
    let _ = fs::create_dir(dataset_dir);
    let file_path = dataset_dir.join(file_name);
    if file_path.exists() {
        return;
    }
    println!("downloading {} now in progress...", file_name);
    let client = reqwest::blocking::Client::new();
    let bytes = client
        .get(URL_BASE.to_string() + file_name)
        .header(
            USER_AGENT,
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0",
        )
        .send()
        .unwrap()
        .bytes()
        .unwrap();
    let mut out = File::create(file_path).unwrap();
    std::io::copy(&mut bytes.as_ref(), &mut out).unwrap();
}

fn read_exact_bytes(buf_reader: &mut BufReader<File>, bytes: usize) -> Result<Vec<u8>, ()> {
    let mut buffer = vec![0u8; bytes];
    let result = buf_reader.read_exact(&mut buffer);
    match result {
        Ok(_) => return Ok(buffer),
        Err(_) => return Err(()),
    }
}

fn load(file_name: &str, dataset_dir: &Path, offsets: usize) -> BufReader<File> {
    let file_path = dataset_dir.join(file_name);
    let file = match File::open(file_path) {
        Err(why) => panic!("{}", why),
        Ok(file) => file,
    };
    let mut buf_reader = BufReader::new(file);
    let _ = read_exact_bytes(&mut buf_reader, offsets);
    buf_reader
}

fn decode_gzip_files(dataset_dir: &Path) {
    let dir = fs::read_dir(dataset_dir).unwrap();
    for item in dir.into_iter() {
        let path = &item.as_ref().unwrap().path();
        if path.file_name().unwrap() == ".DS_Store" || path.extension().is_none() {
            continue;
        }
        let file = File::open(path).unwrap();
        let file = BufReader::new(file);
        let mut file = GzDecoder::new(file);
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        let mut buf_writer = BufWriter::new(
            File::create(
                &item
                    .as_ref()
                    .unwrap()
                    .path()
                    .parent()
                    .unwrap()
                    .join(&item.as_ref().unwrap().path().file_stem().unwrap()),
            )
            .unwrap(),
        );
        buf_writer.write(&bytes.as_slice()).unwrap();
    }
}

#[test]
fn test() {
    let file_path = std::env::current_dir()
        .unwrap()
        .join("dataset")
        .join("train-images-idx3-ubyte");
    let file = match File::open(file_path) {
        Err(why) => panic!("{}", why),
        Ok(file) => file,
    };
    let mut buf_reader = BufReader::new(file);
    #[inline]
    fn read_exact_bytes(buf_reader: &mut BufReader<File>, bytes: usize) -> Vec<u8> {
        let mut buffer = vec![0u8; bytes];
        buf_reader.read_exact(&mut buffer).unwrap();
        buffer
    }
    read_exact_bytes(&mut buf_reader, 16);
    let matrix =
        na::OMatrix::<u8, Const<28>, Const<28>>::from_vec(read_exact_bytes(&mut buf_reader, 784));
    let mut img = image::GrayImage::new(28, 28);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        *pixel = image::Luma([matrix[(x as usize, y as usize)]; 1]);
    }
    img.save("result.png").unwrap();
}

#[test]
fn aaa() {
    let dvec = vec![
        na::OMatrix::<i32, Const<3>, Const<3>>::new(1, 1, 1, 1, 2, 2, 2, 2, 3),
        na::OMatrix::<i32, Const<3>, Const<3>>::new(3, 3, 3, 4, 4, 4, 4, 5, 5),
    ];
    let v = vec![1, 1, 2, 1, 2, 2, 1, 2, 3, 3, 4, 4, 3, 4, 5, 3, 4, 5];
    let mut vec: Vec<i32> = vec![];
    dvec.iter()
        .for_each(|x| x.iter().for_each(|y| vec.push(*y)));
    let a = na::OMatrix::<i32, Const<9>, Dyn>::from_vec(vec);
    assert_eq!(a, na::OMatrix::<i32, Const<9>, Dyn>::from_vec(v))
}

#[test]
fn test_init_mnist() {
    init_mnist();
}

#[test]
fn test_as_one_hot() {
    let label = Label::from(vec![2, 8, 2]);
    let one_hot = label.as_one_hot();
}
