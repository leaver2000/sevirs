#![allow(unused_imports)]
use pyo3::prelude::pyfunction;
use pyo3::prelude::*;
use rand::Rng;
use std::cmp::Ordering;
use std::io;

// Feature = tuple[
//     tuple[int, int, int],
//     tuple[int, int, int],
//     tuple[int, int, int],
// ]

// def engineer_features(arr: np.ndarray) -> list[Feature]:
//     x, y = arr.shape  # (PATCH_SIZE, PATCH_SIZE)
//     return [tuple(map(tuple, arr[i : i + 3, j : j + 3])) for i in range(x) for j in range(y)]

// data = engineer_features(arr[:, :, 0])
// # np.array([[list(d) for d in x] for x in data[:5]])

#[pyfunction]
fn guess() {
    println!("Guess the number!");

    let secret_number = rand::thread_rng().gen_range(1..101);

    loop {
        println!("Please input your guess.");

        let mut guess = String::new();

        io::stdin()
            .read_line(&mut guess)
            .expect("Failed to read line");

        let guess: u32 = match guess.trim().parse() {
            Ok(num) => num,
            Err(_) => continue,
        };

        println!("You guessed: {}", guess);

        match guess.cmp(&secret_number) {
            Ordering::Less => println!("Too small!"),
            Ordering::Greater => println!("Too big!"),
            Ordering::Equal => {
                println!("You win!");
                break;
            }
        }
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _lib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(guess, m)?)?;

    Ok(())
}
