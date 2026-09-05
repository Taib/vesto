use std::collections::HashMap;

use crate::{
    error::VestoError,
    index::VestoIndex,
    metrics::parse_metric_from_str,
    types::{EntityId, Metadata, Vector},
    vector_field::VectorField,
};

pub type CollectionID = String;

pub struct Collection {
    pub name: String,
    vector_fields: HashMap<String, VectorField>,
    metadata: HashMap<EntityId, Metadata>,
}

impl Collection {
    pub fn new(name: &str) -> Self {
        Self {
            metadata: HashMap::new(),
            name: name.to_string(),
            vector_fields: HashMap::new(),
        }
    }
    pub fn insert(
        &mut self,
        vector_field: &str,
        vectors: Vec<Vector>,
        metadata: Option<Vec<Metadata>>,
    ) -> Result<Vec<EntityId>, VestoError> {
        if metadata.is_some() && metadata.as_ref().unwrap().len() != vectors.len() {
            return Err(VestoError::MetadataLengthMismatch);
        }
        if let Some(field) = self.vector_fields.get_mut(vector_field) {
            let ids = field.insert(vectors)?;
            if let Some(meta) = metadata {
                for (id, m) in ids.iter().zip(meta.into_iter()) {
                    self.metadata.insert(*id, m);
                }
            }
            Ok(ids)
        } else {
            Err(VestoError::KeyNotFound)
        }
    }
    pub fn create_index_by_type(
        &mut self,
        vector_field: &str,
        name: &str,
        index_type: &str,
        metric_name: &str,
        dim: usize,
    ) -> Result<(), VestoError> {
        if !self.vector_fields.contains_key(vector_field) {
            self.add_vector_field(vector_field, dim)?;
        }
        self.vector_fields
            .get_mut(vector_field)
            .unwrap()
            .create_index_by_type(
                name.to_string(),
                index_type.to_string(),
                parse_metric_from_str(metric_name),
            )
    }
    pub fn add_vector_field(&mut self, name: &str, dim: usize) -> Result<(), VestoError> {
        if self.vector_fields.contains_key(name) {
            return Err(VestoError::DuplicateVectorField);
        }
        let field = VectorField::new(name, dim);
        self.vector_fields.insert(name.to_string(), field);
        Ok(())
    }
    pub fn add_index<I>(
        &mut self,
        vector_field: &str,
        index: I,
        dim: usize,
    ) -> Result<(), VestoError>
    where
        I: VestoIndex + Send + 'static,
    {
        if !self.vector_fields.contains_key(vector_field) {
            self.add_vector_field(vector_field, dim)?;
        }
        self.vector_fields
            .get_mut(vector_field)
            .unwrap()
            .add_index(index)
    }
    pub fn search(
        &self,
        vector_field: &str,
        index_name: &str,
        query: &Vector,
        top_k: usize,
        with_metadata: bool,
    ) -> Result<Vec<(f32, Vector, Option<Metadata>)>, VestoError> {
        if let Some(field) = self.vector_fields.get(vector_field) {
            let result = field.search(index_name, query, top_k)?;
            Ok(result
                .into_iter()
                .map(|(score, id, vector)| {
                    if with_metadata {
                        let metadata = self.metadata.get(&id).cloned();
                        (score, vector, metadata)
                    } else {
                        (score, vector, None)
                    }
                })
                .collect())
        } else {
            Err(VestoError::KeyNotFound)
        }
    }
}
