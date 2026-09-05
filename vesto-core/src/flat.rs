use crate::error::VestoError;
use crate::index::VestoIndex;
use crate::metrics::Metric;
use crate::store::VestoStoreTrait;
use crate::types::{EntityId, Score, Vector};

pub struct VestoFlatIndex {
    name: String,
    data: Vec<EntityId>,
    metric: Metric,
}

impl VestoFlatIndex {
    pub fn new(name: &str, metric_name: crate::metrics::MetricsName) -> Self
    where
        Self: Sized,
    {
        Self {
            data: Vec::new(),
            metric: Metric::new(metric_name),
            name: String::from(name),
        }
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl VestoIndex for VestoFlatIndex {
    fn insert(
        &mut self,
        data: Vec<EntityId>,
        _: Option<&dyn crate::store::VestoStoreTrait>,
    ) -> Result<(), VestoError> {
        self.data.extend(data);
        Ok(())
    }
    fn search(
        &self,
        store_get: &dyn VestoStoreTrait,
        query: &Vector,
        top_k: usize,
    ) -> Result<Vec<(Score, EntityId)>, VestoError> {
        let mut scores = self
            .data
            .iter()
            .filter_map(|&entity_id| {
                let vector = store_get.get(&entity_id)?;
                match self.metric.distance(&vector, query) {
                    Ok(score) => Some(Ok((score, entity_id))),
                    Err(e) => Some(Err(e)),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        scores.sort_by(|a, b| a.0.total_cmp(&b.0));
        scores.truncate(top_k.min(scores.len()));
        Ok(scores)
    }
    fn name(&self) -> String {
        self.name.clone()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::store::VestoStore;
    use ndarray::array;

    #[test]
    fn search_returns_best_match() {
        println!("Vector data: Vesto");
        let mut index = VestoFlatIndex::new("test", crate::metrics::MetricsName::Cosine);
        let mut store = VestoStore::new(3);
        let ids = store
            .insert(vec![
                array![1.0, 0.0, 0.0],
                array![0.9, 0.1, 0.0],
                array![0.0, 1.0, 0.0],
                array![0.0, 0.0, 1.0],
            ])
            .unwrap();
        index.insert(ids, None).unwrap();
        let results = index.search(&store, &array![1.0, 0.1, 0.0], 2).unwrap();
        assert_eq!(results[0].1, EntityId(1));
    }
}
