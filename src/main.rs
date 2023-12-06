fn main() {
    println!("Hello, world!");
}

fn and(x1: f64, x2: f64) -> i32 {
    let w1 = 0.5;
    let w2 = 0.5;
    let theta = 0.7;
    let tmp = x1 * w1 + x2 * w2;
    return if tmp > theta { 1 } else { 0 };
}

#[test]
fn test_and() {
    assert_eq!(and(1.into(), 1.into()), 1);
    assert_eq!(and(1.into(), 0.into()), 0);
    assert_eq!(and(0.into(), 1.into()), 0);
    assert_eq!(and(0.into(), 0.into()), 0);
}
