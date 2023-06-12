use impurity::parse::orbitale::parse_orbitale_def;
use impurity::pfaffian::compute_pfaffian_wq;

fn main() {
    let mut fij = parse_orbitale_def().unwrap();
    let pfaffian = compute_pfaffian_wq(&mut fij);
    println!("{}", pfaffian);
}

