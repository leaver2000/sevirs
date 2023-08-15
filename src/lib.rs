#![allow(unused_imports)]
use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, Dim};
use numpy::{IntoPyArray, PyArray, PyArray3, PyArray4, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::pyfunction;
use pyo3::prelude::*;
use rand::Rng;
use std::cmp::Ordering;
use std::io;

// fn to_numpy(&self) -> PyResult<Py<PyArray<i32, Dim<[usize; 3]>>>> {
//     let gil = Python::acquire_gil();
//     let py = gil.python();
//     let arr = self.arr.clone().unwrap();
//     let arr = PyArray::from_owned_array(py, arr);
//     Ok(arr)
// }

type Matrix<T> = Vec<Vec<T>>;

fn flat_extend(data: &Matrix<i32>, n: usize) -> Vec<i32> {
    if data.len() != data[0].len() {
        panic!("data size is not equal to size");
    }
    let size = data.len();
    let mut result: Vec<i32> = vec![];
    for i in 0..size {
        for j in 0..size {
            for k in 0..n {
                let x = i + k;
                for l in 0..n {
                    let y = j + l;
                    if x >= size || y >= size {
                        result.push(0);
                        continue;
                    }
                    result.push(data[x][y]);
                }
            }
        }
    }
    result
}

#[pyclass]
struct GridEncoder {
    // - properties
    #[pyo3(get)]
    shape: (usize, usize, usize, usize),

    #[pyo3(get)]
    values: Vec<i32>,
}

#[pymethods]
impl GridEncoder {
    #[new]
    fn new(data: Matrix<i32>, n: usize) -> Self {
        if data.len() != data[0].len() {
            panic!("data size is not equal to size");
        }
        let size = data.len();
        let shape: (usize, usize, usize, usize) = (size, size, n, n);
        let values: Vec<i32> = flat_extend(&data, n);

        Self { values, shape }
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray<i32, Dim<[usize; 4]>>>> {
        Ok(self
            .values
            .clone()
            .into_pyarray(py)
            .reshape(self.shape)
            .unwrap()
            .to_owned())
    }
}

// #[pyfunction]
// fn guess() {
//     println!("Guess the number!");

//     let secret_number = rand::thread_rng().gen_range(1..101);

//     loop {
//         println!("Please input your guess.");

//         let mut guess = String::new();

//         io::stdin()
//             .read_line(&mut guess)
//             .expect("Failed to read line");

//         let guess: u32 = match guess.trim().parse() {
//             Ok(num) => num,
//             Err(_) => continue,
//         };

//         println!("You guessed: {}", guess);

//         match guess.cmp(&secret_number) {
//             Ordering::Less => println!("Too small!"),
//             Ordering::Greater => println!("Too big!"),
//             Ordering::Equal => {
//                 println!("You win!");
//                 break;
//             }
//         }
//     }
// }

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _lib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<GridEncoder>()?;

    Ok(())
}
