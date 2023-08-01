use csv::{Reader, StringRecord};
use derive_more::{Constructor, Error};
use rand::Rng;
use std::fmt;
use std::fs::File;
use std::path::PathBuf;

/// Parse the orbital variationnal definition
/// # Arguments
/// * __`fp`__ - File path to the definition file, in csv format.
/// * __`s`__ - Size of the system.
pub fn parse_orbitale_def(fp: &PathBuf, s: usize) -> Result<Vec<f64>> {
    // Parameter reference vector. column packed.
    let mut fij: Vec<f64> = vec![0.0; s * s];
    // Parameter vector. Ordered by "n".
    let mut orbitale_params: Vec<f64> = Vec::new();
    let mut n_orbitale_params: usize = 0;
    // RNG thread
    let mut rng = rand::thread_rng();

    // Parse input file.
    let file = File::open(fp)?;
    let mut reader = Reader::from_reader(file);
    for (k, result) in reader.records().enumerate() {
        let rec = result?;
        // Should have 3 column.
        if rec.len() != 3 {
            println!(
                "Error at line {} in orbitale.csv, invalid number of elements.",
                k
            );
            return Err(OrbitaleParseError::new(
                "Invalid number of argument on a line.".to_owned(),
            ));
        }

        // Parse element coordinates.
        let i = parse_single_elem(&rec, 0, k)?;
        let j = parse_single_elem(&rec, 1, k)?;

        let n = rec.get(2).unwrap().parse::<usize>();
        match n {
            Ok(param) => {
                if n_orbitale_params < param {
                    n_orbitale_params += 1;
                    orbitale_params.push(rng.gen());
                }
                let var_param: f64 = orbitale_params[param - 1];
                fij[i + j * s] = var_param;
                fij[j + i * s] = -var_param;
            }
            Err(error) => {
                println!("Error at line {}, invalid parameter identifier.", k);
                return Err(OrbitaleParseError::from(error));
            }
        };
    }
    Ok(fij)
}

fn parse_single_elem(line: &StringRecord, col: usize, l: usize) -> Result<usize> {
    match line.get(col).unwrap().parse::<usize>() {
        Ok(v) => Ok(v),
        Err(error) => {
            println!(
                "Expected valid coordinates in orbitale.csv at line {}, col {}",
                l, col
            );
            Err(OrbitaleParseError::from(error))
        }
    }
}

type Result<T> = std::result::Result<T, OrbitaleParseError>;

/// Error in the orbital params definition.
#[derive(Debug, Clone, Error, Constructor)]
pub struct OrbitaleParseError {
    pub details: String,
}

impl fmt::Display for OrbitaleParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Parsing error encountered in orbitale.csv.")
    }
}

impl From<std::num::ParseIntError> for OrbitaleParseError {
    fn from(err: std::num::ParseIntError) -> Self {
        println!("{}", err);
        OrbitaleParseError::new("Expected to parse an integer.".to_owned())
    }
}

impl From<csv::Error> for OrbitaleParseError {
    fn from(err: csv::Error) -> Self {
        OrbitaleParseError::new(err.to_string())
    }
}

impl From<std::io::Error> for OrbitaleParseError {
    fn from(err: std::io::Error) -> Self {
        OrbitaleParseError::new(err.to_string())
    }
}
