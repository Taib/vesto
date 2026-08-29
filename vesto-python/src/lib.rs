use ndarray::Array1;
use pyo3::prelude::*;
use vesto_core::Vesto;

#[pyclass(name = "Vesto")]
struct PyVesto {
    inner: Vesto,
}

#[pymethods]
impl PyVesto {
    #[new]
    fn new() -> Self {
        Self {
            inner: Vesto::new(),
        }
    }

    fn insert(&mut self, vectors: Vec<Vec<f32>>) -> PyResult<()> {
        let vectors = vectors.into_iter().map(Array1::from).collect();
        self.inner
            .insert(vectors)
            .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))
    }

    fn search(&self, query: Vec<f32>, top_k: usize) -> PyResult<Vec<(f32, usize)>> {
        let query = Array1::from(query);

        self.inner
            .search(&query, top_k)
            .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))
    }

    fn get(&self, index: usize) -> PyResult<Vec<f32>> {
        self.inner
            .get(index)
            .map(|vec| vec.to_vec())
            .map_err(|err| pyo3::exceptions::PyIndexError::new_err(err.to_string()))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("Vesto(len={})", self.inner.len())
    }
}

#[pymodule]
fn vesto(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVesto>()?;

    Ok(())
}
