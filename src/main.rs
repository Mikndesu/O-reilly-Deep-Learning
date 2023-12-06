extern crate nalgebra as na;

fn main() {
    println!("Hello, world!");
}

fn and(x1: f64, x2: f64) -> i32 {
    let x = na::Vector2::new(x1, x2);
    let w = na::Vector2::new(0.5, 0.5);
    let b = -0.7;
    let tmp = x.dot(&w) + b;
    return if tmp > 0.0 { 1 } else { 0 };
}

#[test]
fn test_and() {
    assert_eq!(and(1.into(), 1.into()), 1);
    assert_eq!(and(1.into(), 0.into()), 0);
    assert_eq!(and(0.into(), 1.into()), 0);
    assert_eq!(and(0.into(), 0.into()), 0);
}
