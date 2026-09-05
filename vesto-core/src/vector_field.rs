use std::collections::HashMap;

use crate::{
    error::VestoError,
    flat::VestoFlatIndex,
    hnsw::VestoHNSWIndex,
    index::VestoIndex,
    metrics::MetricsName,
    store::{VestoStore, VestoStoreTrait},
    types::{EntityId, Vector},
};

pub struct VectorField {
    name: String,
    store: Box<dyn VestoStoreTrait + 'static + Send>,
    indexes: HashMap<String, Box<dyn VestoIndex + Send>>,
}

impl VectorField {
    pub fn new(name: &str, dim: usize) -> Self {
        Self {
            store: Box::new(VestoStore::new(dim)),
            name: name.to_string(),
            indexes: HashMap::new(),
        }
    }
    pub fn insert(&mut self, vectors: Vec<Vector>) -> Result<Vec<EntityId>, VestoError> {
        let ids = self.store.insert(vectors)?;
        for index in self.indexes.values_mut() {
            index.insert(
                ids.clone(),
                Some(self.store.as_ref()), /* vector refs */
            )?;
        }
        Ok(ids)
    }
    pub fn create_index_by_type(
        &mut self,
        name: String,
        index_type: String,
        metric_name: MetricsName,
    ) -> Result<(), VestoError> {
        if self.indexes.contains_key(&name) {
            return Err(VestoError::DuplicateIndex);
        }
        let index: Box<dyn VestoIndex + Send> = match index_type.to_lowercase().as_str() {
            "flat" => Box::new(VestoFlatIndex::new(&name, metric_name)),
            "hnsw" => Box::new(VestoHNSWIndex::new(&name, metric_name, None)),
            _ => return Err(VestoError::UnknownIndexType),
        };
        self.indexes.insert(name, index);
        Ok(())
    }
    pub fn add_index<I>(&mut self, index: I) -> Result<(), VestoError>
    where
        I: VestoIndex + Send + 'static,
    {
        let name = index.name();
        if self.indexes.contains_key(&name) {
            return Err(VestoError::DuplicateIndex);
        }
        self.indexes
            .insert(name, Box::new(index))
            .ok_or_else(|| VestoError::BadHeader)?;
        Ok(())
    }
    pub fn search(
        &self,
        index_name: &str,
        query: &Vector,
        top_k: usize,
    ) -> Result<Vec<(f32, EntityId, Vector)>, VestoError> {
        if let Some(index) = self.indexes.get(index_name) {
            let positions = index.search(self.store.as_ref(), query, top_k)?;
            let mut ans = Vec::new();
            for pos in positions {
                let vector = self.store.get(&pos.1);
                if vector.is_some() {
                    ans.push((pos.0, pos.1, vector.unwrap()));
                }
            }
            return Ok(ans);
        }
        return Err(VestoError::KeyNotFound);
    }
}
