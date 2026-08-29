use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

mod error;
mod types;
mod store;
mod index;
mod metrics;
pub mod collection;
mod flat;
mod hnsw;

use error::VestoError;
use collection::{Collection, CollectionID};
pub struct Vesto {
    collections: HashMap<CollectionID, Arc<Mutex<Collection>>>,
}

impl Vesto {
    pub fn new() -> Self {
        Self {
            collections: HashMap::new(),
        }
    }

    pub fn add_collection(
        &mut self,
        collection: Collection,
    ) -> Result<Arc<Mutex<Collection>>, VestoError> {
        let name = collection.schema.name.clone();
        if self.collections.contains_key(&name) {
            return Err(VestoError::DuplicateCollection);
        }
        let shared = Arc::new(Mutex::new(collection));

        // Registry keeps one clone, caller gets another
        self.collections.insert(name, Arc::clone(&shared));
        Ok(shared)
    }

    pub fn get_collection(&self, collection_name: &str) -> Option<Arc<Mutex<Collection>>> {
        self.collections.get(collection_name).map(Arc::clone)
    }

    pub fn len(&self) -> usize {
        self.collections.len()
    }
}
