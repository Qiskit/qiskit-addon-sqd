use hashbrown::HashMap;
use num_complex::Complex;
use numpy::ndarray::{Array2, Array3, Axis};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;
use pyo3::Bound;
use rayon::prelude::*;

#[pyfunction]
fn generate_sparse_elements(
    py: Python<'_>,
    keys: PyReadonlyArray2<u64>,
    connected_bs_batch: PyReadonlyArray3<bool>,
    amplitudes: PyReadonlyArray2<Complex<f64>>,
    coeffs: PyReadonlyArray1<Complex<f64>>,
) -> PyResult<(
    Py<PyArray1<Complex<f64>>>,
    Py<PyArray1<usize>>,
    Py<PyArray1<usize>>,
)> {
    // Immutable array view of interior of Numpy array
    let bs_batch = connected_bs_batch.as_array();
    let amps_batch = amplitudes.as_array();
    let coeffs_arr = coeffs.as_array();
    let keys_arr = keys.as_array();

    let mut idx_map: HashMap<(u64, u64), usize> = HashMap::with_capacity(keys_arr.shape()[0]);
    for (i, row) in keys_arr.outer_iter().enumerate() {
        idx_map.insert((row[0], row[1]), i);
    }

    // Collect each (batch_id, (bs_batch, amps_batch)) to a Vec so we can use Rayon
    // parallel iterator
    let batch_items: Vec<_> = bs_batch
        .outer_iter()
        .zip(amps_batch.outer_iter())
        .enumerate()
        .collect();

    // In parallel, create a dict mapping (batch_id, (conn_bs, amps)) to a hash map
    // defining a sparse matrix (i.e. {(batch_id, (conn_bs, amps)): {(row, col): amplitude}})
    let partial_accumulators: Vec<HashMap<(usize, usize), Complex<f64>>> = batch_items
        .par_iter()
        .map(|(batch_id, (bs, amps))| {
            let coeff = coeffs_arr[*batch_id];
            let mut local_acc = HashMap::new();

            for (i, row) in bs.outer_iter().enumerate() {
                let mut key_bits = (0u64, 0u64);
                let n = row.len();
                for (idx, &b) in row.iter().enumerate() {
                    let rev_idx = n - 1 - idx;
                    if rev_idx < 64 {
                        key_bits.0 <<= 1;
                        if b {
                            key_bits.0 |= 1;
                        }
                    } else {
                        key_bits.1 <<= 1;
                        if b {
                            key_bits.1 |= 1;
                        }
                    }
                }
                if let Some(&col) = idx_map.get(&key_bits) {
                    let val = coeff * amps[i];
                    *local_acc.entry((i, col)).or_insert(Complex::new(0.0, 0.0)) += val;
                }
            }
            local_acc
        })
        .collect();

    // Merge all partial accumulators
    let mut accumulator: HashMap<(usize, usize), Complex<f64>> = HashMap::new();
    for local_acc in partial_accumulators {
        for (key, val) in local_acc {
            *accumulator.entry(key).or_insert(Complex::new(0.0, 0.0)) += val;
        }
    }

    // Convert to final vectors and return
    let mut final_rows = Vec::with_capacity(accumulator.len());
    let mut final_cols = Vec::with_capacity(accumulator.len());
    let mut final_vals = Vec::with_capacity(accumulator.len());

    for ((row, col), val) in accumulator {
        final_rows.push(row);
        final_cols.push(col);
        final_vals.push(val);
    }

    let py_vals = PyArray1::from_vec(py, final_vals);
    let py_rows = PyArray1::from_vec(py, final_rows);
    let py_cols = PyArray1::from_vec(py, final_cols);

    Ok((py_vals.into(), py_rows.into(), py_cols.into()))
}

/// Project each of the `L` terms in a Pauli operator, specified by `diag`, `sign`, and
/// `imag` vectors, onto the subspace specified by an `MxN`-shaped `bitstring_matrix`.
/// This results in `L` `MxN` bitstring matrices, each representing one Pauli term and
/// associated with a unit amplitude in `{1.0, -1.0, 1.0j, -1.0j}`.
///
/// The columns of all input data structures represent qubits `(N, 0]` respectively
/// (i.e. index `0` represents qubit `N` and index `N-1` represents qubit `0`).
///
/// Each row in `diag`, `sign`, and `imag` represents one Pauli term in an operator.
#[pyfunction]
fn connected_elements_and_amplitudes(
    py: Python<'_>,
    bitstring_matrix: PyReadonlyArray2<bool>,
    diag: PyReadonlyArray2<bool>,
    sign: PyReadonlyArray2<bool>,
    imag: PyReadonlyArray2<bool>,
) -> PyResult<(Py<PyArray3<bool>>, Py<PyArray2<Complex<f64>>>)> {
    // Immutable array view of interior of Numpy array
    let bitstring_matrix = bitstring_matrix.as_array();
    let diag = diag.as_array();
    let sign = sign.as_array();
    let imag = imag.as_array();

    let (m, n) = bitstring_matrix.dim();
    let (b, _n_diag) = diag.dim();

    // Output arrays
    let mut conn_ele = Array3::<bool>::from_elem((b, m, n), false);
    let mut amplitudes = Array2::<Complex<f64>>::zeros((b, m));

    // Collect to Vecs for parallel iteration
    let conn_batches: Vec<_> = conn_ele.axis_iter_mut(Axis(0)).collect();
    let amp_batches: Vec<_> = amplitudes.axis_iter_mut(Axis(0)).collect();
    let diag_batches: Vec<_> = diag.axis_iter(Axis(0)).collect();
    let sign_batches: Vec<_> = sign.axis_iter(Axis(0)).collect();
    let imag_batches: Vec<_> = imag.axis_iter(Axis(0)).collect();

    // Run through each ((bs_batch, amp_batch, (diag, sign, imag))) combo in parallel
    // Calculate the connected elements and their amplitudes
    conn_batches
        .into_par_iter()
        .zip(amp_batches.into_par_iter())
        .zip(
            diag_batches.into_par_iter().zip(
                sign_batches
                    .into_par_iter()
                    .zip(imag_batches.into_par_iter()),
            ),
        )
        .for_each(
            |((mut conn_batch, mut amp_batch), (diag_row, (sign_row, imag_row)))| {
                for (i, row) in bitstring_matrix.outer_iter().enumerate() {
                    let mut total = Complex::new(1.0, 0.0);
                    for (j, &b_val) in row.iter().enumerate() {
                        if b_val == diag_row[j] {
                            conn_batch[[i, j]] = true;
                        }
                        if b_val && sign_row[j] {
                            total *= -1.0;
                        }
                        if imag_row[j] {
                            total *= Complex::new(0.0, 1.0);
                        }
                    }
                    amp_batch[i] = total;
                }
            },
        );

    Ok((
        conn_ele.into_pyarray(py).into(),
        amplitudes.into_pyarray(py).into(),
    ))
}

#[pymodule]
fn _accelerate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_sparse_elements, m)?)?;
    m.add_function(wrap_pyfunction!(connected_elements_and_amplitudes, m)?)?;
    Ok(())
}
