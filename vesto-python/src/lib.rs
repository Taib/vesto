use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use ndarray::Array1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vesto_core::Vesto;
use vesto_core::collection::{Collection, Schema};

fn to_py<E: Debug>(e: E) -> PyErr {
    PyValueError::new_err(format!("{e:?}"))
}
#[pyclass(name = "VestoCollection")]
struct PyVestoCollection {
    inner: Arc<Mutex<Collection>>,
}

#[pymethods]
impl PyVestoCollection {
    fn add_index(&mut self, name: String, index_type: String) -> PyResult<()> {
        let mut collection = self.inner.lock().map_err(to_py)?;
        collection
            .create_index_by_type(name, index_type)
            .map_err(to_py)?;

        Ok(())
    }

    fn insert(&self, vectors: Vec<Vec<f32>>) -> PyResult<Vec<u64>> {
        let vectors = vectors.into_iter().map(Array1::from).collect();
        let ids = {
            let mut collection = self.inner.lock().map_err(to_py)?;
            collection.insert(vectors).map_err(to_py)?
        };
        Ok(ids.into_iter().map(|id| id.0).collect())
    }

    fn search(
        &self,
        index_name: &str,
        query: Vec<f32>,
        top_k: usize,
    ) -> PyResult<Vec<(f32, Vec<f32>)>> {
        let query = Array1::from(query);
        let results = {
            let collection = self.inner.lock().map_err(to_py)?;
            collection
                .search(index_name, &query, top_k)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e:?}")))?
        };
        Ok(results
            .into_iter()
            .map(|(score, vector)| (score, vector.to_vec()))
            .collect())
    }
}

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

    pub fn add_collection(
        &mut self,
        name: String,
        vfield_name: String,
        metric_name: String,
        dim: usize,
    ) -> PyResult<PyVestoCollection> {
        let collection = Collection::new(Schema {
            name,
            vfield_name,
            dim,
            metric_name,
        });
        let shared = self
            .inner
            .add_collection(collection)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{err:?}")))?;
        Ok(PyVestoCollection { inner: shared })
    }
    fn get_collection(&self, name: &str) -> PyResult<PyVestoCollection> {
        self.inner
            .get_collection(name)
            .map(|inner| PyVestoCollection { inner })
            .ok_or_else(|| PyValueError::new_err(format!("No collection {name:?}")))
    }
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("Vesto(collections={})", self.inner.len())
    }
}

#[pymodule]
fn vesto(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVesto>()?;
    m.add_class::<PyVestoCollection>()?;

    Ok(())
}
