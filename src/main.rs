use std::path::Path;

use impurity::parse::orbitale::parse_orbitale_def;
use impurity::pfaffian::compute_pfaffian_wq;
use impurity::SIZE;

fn main() {
    let orbitale_fp = Path::new("data/orbitale.csv");
    let mut fij = parse_orbitale_def(&orbitale_fp.to_path_buf(), SIZE).unwrap();
    let pfaffian = compute_pfaffian_wq(&mut fij);
    println!("{}", pfaffian);
}

