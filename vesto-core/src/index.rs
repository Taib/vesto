use crate::{
    error::VestoError,
    store::VestoStoreTrait,
    types::{EntityId, Score, Vector},
};

pub trait VestoIndex {
    fn name(&self) -> String;
    fn vfield_name(&self) -> String;
    fn insert(
        &mut self,
        data: Vec<EntityId>,
        store_get: Option<&dyn crate::store::VestoStoreTrait>,
    ) -> Result<(), VestoError>;
    fn search(
        &self,
        store_get: &dyn VestoStoreTrait,
        query: &Vector,
        top_k: usize,
    ) -> Result<Vec<(Score, EntityId)>, VestoError>;
}
