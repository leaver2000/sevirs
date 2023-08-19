#![allow(unused_imports)]
// use numpy::ndarray::Dim;
use numpy::ndarray::*;
use numpy::{IntoPyArray, PyArray, ToPyArray};
use pyo3::prelude::*;

type Matrix<T> = Vec<Vec<T>>;
// struct ArrayInerface<T> {
//     data: T,
// }
// impl ArrayInerface {
//     fn new(data: Vec<T>) -> Self {
//         Self { data }
//     }
//     fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray<i32, IxDyn>>>> {
//         Ok(self
//             .values
//             .clone()
//             .into_pyarray(py)
//             .reshape(self.shape)
//             .unwrap()
//             .to_owned())
//     }

// }

fn flat_decompose(data: &Matrix<i32>, n: usize) -> Vec<i32> {
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

pub fn decompose2d(data: Matrix<i32>, n: usize) -> Matrix<Matrix<i32>> {
    let n_rows = data.len();
    let n_columns = data[0].len();
    let mut result: Matrix<Matrix<i32>> = vec![];

    for i in 0..n_rows {
        let mut row: Matrix<Vec<i32>> = vec![];
        for j in 0..n_columns {
            let mut column: Matrix<i32> = vec![];
            for k in 0..n {
                let mut vector: Vec<i32> = vec![];
                let xi = i + k;
                for l in 0..n {
                    let yi = j + l;
                    if xi >= n_rows || yi >= n_columns {
                        vector.push(0);
                        continue;
                    }

                    vector.push(data[xi][yi]);
                }
                column.push(vector);
            }
            row.push(column);
        }
        result.push(row);
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
        let values: Vec<i32> = flat_decompose(&data, n);
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

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _lib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<GridEncoder>()?;

    Ok(())
}
