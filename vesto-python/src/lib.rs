use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use ndarray::Array1;
use pyo3::exceptions::PyValueError;
use pyo3::{prelude::*, types::PyDict};
use pyo3_stub_gen::define_stub_info_gatherer;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use vesto_core::Vesto;
use vesto_core::collection::Collection;

fn to_py<E: Debug>(e: E) -> PyErr {
    PyValueError::new_err(format!("{e:?}"))
}

fn python_dict_to_metadata(dict: &Bound<'_, PyDict>) -> PyResult<vesto_core::types::Metadata> {
    let mut fields = std::collections::HashMap::new();
    for (key, value) in dict.iter() {
        let key: String = key.extract().map_err(to_py)?;
        let value: vesto_core::types::MetaValue = if let Ok(f) = value.extract::<f32>() {
            vesto_core::types::MetaValue::Float(f)
        } else if let Ok(b) = value.extract::<bool>() {
            vesto_core::types::MetaValue::Bool(b)
        } else if let Ok(s) = value.extract::<String>() {
            vesto_core::types::MetaValue::Text(s)
        } else {
            return Err(to_py(format!("Unsupported metadata type for key {key:?}")));
        };
        fields.insert(key, value);
    }

    Ok(vesto_core::types::Metadata { fields })
}

#[gen_stub_pyclass]
#[pyclass(name = "VestoCollection")]
struct PyVestoCollection {
    inner: Arc<Mutex<Collection>>,
}
#[gen_stub_pymethods]
#[pymethods]
impl PyVestoCollection {
    /// Create an index in a vector field.
    ///
    /// Args:
    ///     vector_field: name of the vector field
    ///     name: index name
    ///     index_type: type of index (FLAT, HSNW)
    ///     metric_name: metric name (L2, COSINE)
    ///     dim: dimension of the vectors
    fn add_index(
        &mut self,
        vector_field: &str,
        name: &str,
        index_type: &str,
        metric_name: &str,
        dim: usize,
    ) -> PyResult<()> {
        let mut collection = self.inner.lock().map_err(to_py)?;
        collection
            .create_index_by_type(vector_field, name, index_type, metric_name, dim)
            .map_err(to_py)?;

        Ok(())
    }

    /// Insert vectors in the collection. This will update all the indexes attached.
    /// Return the entity_ids of the vectors inserted.
    ///
    /// Args:
    ///     vector_field: name of the vector field
    ///     vectors: list of vectors
    ///     metadata: list of metadata for each vector
    fn insert(
        &self,
        vector_field: &str,
        vectors: Vec<Vec<f32>>,
        metadata: Vec<Bound<'_, PyDict>>,
    ) -> PyResult<Vec<u64>> {
        let metadata = metadata
            .into_iter()
            .map(|dict| python_dict_to_metadata(&dict))
            .collect::<Result<Vec<_>, _>>()?;
        let vectors = vectors.into_iter().map(Array1::from).collect();
        let ids = {
            let mut collection = self.inner.lock().map_err(to_py)?;
            collection
                .insert(
                    vector_field,
                    vectors,
                    if metadata.is_empty() {
                        None
                    } else {
                        Some(metadata)
                    },
                )
                .map_err(to_py)?
        };
        Ok(ids.into_iter().map(|id| id.0).collect())
    }

    /// Search a vector using a given index.
    ///
    /// Args:
    ///     index_name: index name
    ///     query: query vector
    ///     top_k: top k matches
    fn search(
        &self,
        vector_field: &str,
        index_name: &str,
        query: Vec<f32>,
        top_k: usize,
        with_metadata: bool,
    ) -> PyResult<Vec<(f32, Vec<f32>)>> {
        let query = Array1::from(query);
        let results = {
            let collection = self.inner.lock().map_err(to_py)?;
            collection
                .search(vector_field, index_name, &query, top_k, with_metadata)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e:?}")))?
        };
        Ok(results
            .into_iter()
            .map(|(score, vector, _)| (score, vector.to_vec()))
            .collect())
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "Vesto")]
struct PyVesto {
    inner: Vesto,
}
#[gen_stub_pymethods]
#[pymethods]
impl PyVesto {
    #[new]
    fn new() -> Self {
        Self {
            inner: Vesto::new(),
        }
    }

    /// Create a collection and return a handle to it.
    ///
    /// Args:
    ///     name: collection name
    pub fn add_collection(&mut self, name: &str) -> PyResult<PyVestoCollection> {
        let collection = Collection::new(name);
        let shared = self
            .inner
            .add_collection(collection)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{err:?}")))?;
        Ok(PyVestoCollection { inner: shared })
    }

    /// Return a handle to a collection.
    ///
    /// Args:
    ///     name: collection name
    fn get_collection(&self, name: &str) -> PyResult<PyVestoCollection> {
        self.inner
            .get_collection(name)
            .map(|inner| PyVestoCollection { inner })
            .ok_or_else(|| PyValueError::new_err(format!("No collection {name:?}")))
    }

    /// Return the number of collections.
    ///
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// For printing purposes.
    ///
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

define_stub_info_gatherer!(stub_info);
