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

fn nand(x1: f64, x2: f64) -> i32 {
    let x = na::Vector2::new(x1, x2);
    let w = na::Vector2::new(-0.5, -0.5);
    let b = 0.7;
    let tmp = x.dot(&w) + b;
    return if tmp > 0.0 { 1 } else { 0 };
}

fn or(x1: f64, x2: f64) -> i32 {
    let x = na::Vector2::new(x1, x2);
    let w = na::Vector2::new(0.5, 0.5);
    let b = -0.2;
    let tmp = x.dot(&w) + b;
    return if tmp > 0.0 { 1 } else { 0 };
}

fn xor(x1: f64, x2: f64) -> i32 {
    and(nand(x1, x2).into(), or(x1, x2).into())
}

#[test]
fn test_and() {
    assert_eq!(and(1.into(), 1.into()), 1);
    assert_eq!(and(1.into(), 0.into()), 0);
    assert_eq!(and(0.into(), 1.into()), 0);
    assert_eq!(and(0.into(), 0.into()), 0);
}

#[test]
fn test_nand() {
    assert_eq!(nand(1.into(), 1.into()), 0);
    assert_eq!(nand(1.into(), 0.into()), 1);
    assert_eq!(nand(0.into(), 1.into()), 1);
    assert_eq!(nand(0.into(), 0.into()), 1);
}

#[test]
fn test_or() {
    assert_eq!(or(1.into(), 1.into()), 1);
    assert_eq!(or(1.into(), 0.into()), 1);
    assert_eq!(or(0.into(), 1.into()), 1);
    assert_eq!(or(0.into(), 0.into()), 0);
}

#[test]
fn test_xor() {
    assert_eq!(xor(1.into(), 1.into()), 0);
    assert_eq!(xor(1.into(), 0.into()), 1);
    assert_eq!(xor(0.into(), 1.into()), 1);
    assert_eq!(xor(0.into(), 0.into()), 0);
}
