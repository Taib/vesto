use std::collections::HashMap;

use crate::{
    error::VestoError,
    flat::VestoFlatIndex,
    hnsw::VestoHNSWIndex,
    index::VestoIndex,
    metrics::parse_metric_from_str,
    store::{VestoStore, VestoStoreTrait},
    types::{EntityId, Vector},
};

// TODO
// pub struct VectorField {
//     name: String,
//     store: Box<dyn VectorStore>,
//     indexes: HashMap<String, Box<dyn VectorIndex>>,
// }
// pub enum Value {
//     Text(String),
//     Int(i64),
//     Float(f64),
//     Bool(bool),
// }
// pub struct Metadata {
//     fields: HashMap<String, Value>,
// }
// pub struct Collection {
//     name: String,
//     vector_fields: HashMap<String, VectorField>,
//     metadata: HashMap<EntityId, Metadata>,
// }
pub type CollectionID = String;

pub struct Schema {
    pub name: String,
    pub vfield_name: String,
    pub metric_name: String,
    pub dim: usize,
}

pub struct Collection {
    pub schema: Schema,    // dim, metric, vfield name
    pub store: VestoStore, // id -> vector + payload
    pub indexes: HashMap<String, Box<dyn VestoIndex + Send>>,
}

impl Collection {
    pub fn new(schema: Schema) -> Self {
        Self {
            indexes: HashMap::new(),
            store: VestoStore::new(schema.dim as usize),
            schema: schema,
        }
    }
    pub fn insert(&mut self, vectors: Vec<Vector>) -> Result<Vec<EntityId>, VestoError> {
        let ids = self.store.insert(vectors)?;
        for index in self.indexes.values_mut() {
            index.insert(ids.clone(), Some(&self.store) /* vector refs */)?;
        }
        Ok(ids)
    }
    pub fn create_index_by_type(
        &mut self,
        name: String,
        index_type: String,
    ) -> Result<(), VestoError> {
        if self.indexes.contains_key(&name) {
            return Err(VestoError::DuplicateIndex);
        }
        let metric_name = parse_metric_from_str(&self.schema.metric_name);
        let index: Box<dyn VestoIndex + Send> = match index_type.to_lowercase().as_str() {
            "flat" => Box::new(VestoFlatIndex::new(
                &name,
                &self.schema.vfield_name,
                metric_name,
            )),
            "hnsw" => Box::new(VestoHNSWIndex::new(
                &name,
                &self.schema.vfield_name,
                metric_name,
                None,
            )),
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
    ) -> Result<Vec<(f32, Vector)>, VestoError> {
        if let Some(index) = self.indexes.get(index_name) {
            let positions = index.search(&self.store, query, top_k)?;
            let mut ans = Vec::new();
            for pos in positions {
                let vector = self.store.get(&pos.1);
                if vector.is_some() {
                    ans.push((pos.0, vector.unwrap()));
                }
            }
            return Ok(ans);
        }
        return Err(VestoError::KeyNotFound);
    }
}
