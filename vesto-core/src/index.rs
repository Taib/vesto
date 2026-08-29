use crate::{
    error::VestoError,
    metrics::MetricsName,
    store::VestoStoreTrait,
    types::{EntityId, Score, Vector},
};

pub trait VestoIndex {
    fn new(name: &str, vfield_name: &str, metric_name: MetricsName) -> Self
    where
        Self: Sized;
    fn name(&self) -> String;
    fn vfield_name(&self) -> String;
    fn insert(&mut self, data: Vec<EntityId>) -> Result<(), VestoError>;
    fn search(
        &self,
        store_get: &dyn VestoStoreTrait,
        query: &Vector,
        top_k: usize,
    ) -> Result<Vec<(Score, EntityId)>, VestoError>;
}
