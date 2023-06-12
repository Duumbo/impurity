use std::path::Path;

use impurity::parse::orbitale::parse_orbitale_def;

#[test]
fn test_parse_2x2() {
    let size = 2;
    let fp = Path::new("tests/orbitales_def/test_parse_2x2.csv");
    let fij = parse_orbitale_def(&fp.to_path_buf(), size).unwrap();
    assert_eq!(fij.len(), 4);
    assert_eq!(fij[1], -fij[2]);
    assert!(fij[1] <= 0.0);
}

#[test]
fn test_parse_3x3() {
    let size = 3;
    let fp = Path::new("tests/orbitales_def/test_parse_3x3.csv");
    let fij = parse_orbitale_def(&fp.to_path_buf(), size).unwrap();
    assert_eq!(fij.len(), 9);
    assert_eq!(fij[1], -fij[3]);
    assert_eq!(fij[6], -fij[2]);
    assert_eq!(fij[5], -fij[7]);
    assert!(fij[1] <= 0.0);
    assert!(fij[2] <= 0.0);
    assert!(fij[5] <= 0.0);
}

#[test]
fn test_int_parse_error() {
    let size = 2;
    let fp = Path::new("tests/orbitales_def/test_int_parse_error.csv");
    let fij = parse_orbitale_def(&fp.to_path_buf(), size);
    match fij {
        Ok(_) => panic!("Should have errored."),
        Err(_) => println!("Error as expected."),
    }
}

#[test]
fn test_csv_parse_error() {
    let size = 2;
    let fp = Path::new("tests/orbitales_def/test_csv_parse_error.csv");
    let fij = parse_orbitale_def(&fp.to_path_buf(), size);
    match fij {
        Ok(_) => panic!("Should have errored."),
        Err(_) => println!("Error as expected."),
    }
}
