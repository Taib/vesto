/*
An implementation of the HNSW (Hierarchical Navigable Small World)
algorithm for approximate nearest neighbor search.
Reference:
 - https://arxiv.org/abs/1603.09320
*/
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::{
    cmp::min,
    collections::{HashMap, HashSet},
};

use crate::{
    error::VestoError,
    index::VestoIndex,
    metrics::Metric,
    store::VestoStoreTrait,
    types::{EntityId, Score, Vector},
};

#[derive(Clone, Copy, PartialEq)]
struct Candidate {
    dist: f32,
    id: EntityId,
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.total_cmp(&other.dist) // larger dist = "greater"
    }
}
#[derive(Default)]
struct Layer {
    // node -> its neighbors on the layer
    adjacency: HashMap<EntityId, Vec<EntityId>>,
}

struct HNSWGraph {
    layers: Vec<Layer>,
    metric: Metric,
    entry_point: Option<EntityId>,
}

impl HNSWGraph {
    fn dist(&self, store_get: &dyn VestoStoreTrait, id: &EntityId, query: &Vector) -> f32 {
        let v = store_get.get(id).unwrap();
        self.metric.distance(&v, query).unwrap()
    }
    fn knn_search(
        &self,                           //
        store_get: &dyn VestoStoreTrait, //
        query: &Vector,                  // query element
        k: usize,                        // number of nearest neighbors to return
        ef: usize,                       // size of the dynamic candidate list
    ) -> Result<Vec<(f32, EntityId)>, VestoError> {
        let mut W; // set for the current nearest elements
        let mut ep = self.entry_point.ok_or(VestoError::EmptyIndex)?; // get enter point for hnsw
        let L = self.layers.len().saturating_sub(1); // level of ep 

        for lc in (1..=L).rev() {
            W = self.search_layer(store_get, &query, ep, 1, lc);
            (_, ep) = self.nearest_element(store_get, &W, &query).unwrap();
        }
        W = self.search_layer(store_get, &query, ep, ef, 0);

        Ok(self.select_neighbors(store_get, &query, &W, k)?)
    }
    fn insert(
        &mut self,
        store_get: &dyn VestoStoreTrait, //
        item: EntityId,
        M: usize,               // number of established connections
        M_max: usize,           //maximum number of connections for each element per layer
        ef_construction: usize, // size of the dynamic candidate list
        m_l: f32,               // normalization factor for level generation
    ) -> Result<(), VestoError> {
        let mut W;
        let mut ep = self.entry_point.unwrap_or(item).clone(); // get enter point for hnsw
        let L = self.layers.len().saturating_sub(1); // top layer fo hnsw
        let l: usize = (-(1.0 - rand::random::<f32>()).ln() * m_l).floor() as usize; // new element's level 
        if self.entry_point.is_none() {
            return self.fresh_start_insert(item, l);
        }
        let query = store_get.get(&item).unwrap();

        for lc in (l + 1..=L).rev() {
            W = self.search_layer(store_get, &query, ep, 1, lc);
            (_, ep) = self.nearest_element(store_get, &W, &query).unwrap();
        }
        for lc in (0..min(L, l) + 1).rev() {
            W = self.search_layer(store_get, &query, ep, ef_construction, lc);
            let neighbors = self
                .select_neighbors(store_get, &query, &W, M)
                .unwrap();
            // add bidirectionall connectionts from neighbors to q at layer lc
            self.layers[lc]
                .adjacency
                .insert(item, neighbors.iter().map(|(_, el)| el.clone()).collect());

            for (_, e) in neighbors {
                // add bidirectionall connectionts from neighbors to q at layer lc
                self.layers[lc].adjacency.entry(e).or_default().push(item);
                // shrink connections if needed
                let e_conn = self.neighbourhood(&e, lc);
                if e_conn.len() > M_max {
                    // shrink connections of e
                    let e_vector = store_get.get(&e).unwrap();
                    let e_new_conn = self
                        .select_neighbors(store_get, &e_vector, &e_conn, M_max)
                        .unwrap();
                    // update neighbourhood of e at layer lc to e_new_conn;
                    self.layers[lc].adjacency.remove(&e);
                    self.layers[lc]
                        .adjacency
                        .insert(e, e_new_conn.iter().map(|(_, el)| el.clone()).collect());
                }
            }
        }
        if l > L {
            // set enter point for hnsw to q
            let prev = self.entry_point;
            while self.layers.len() <= l {
                self.layers.push(Layer::default());
            }
            if let Some(p) = prev {
                for lc in (L + 1)..=l {
                    self.layers[lc].adjacency.insert(item, vec![p]);
                }
            }
            self.entry_point = Some(item);
        }
        Ok(())
    }
    fn fresh_start_insert(&mut self, item: EntityId, l: usize) -> Result<(), VestoError> {
        for lc in 0..=l {
            self.layers.push(Layer::default());
            self.layers[lc].adjacency.insert(item, Vec::new());
        }
        self.entry_point = Some(item);
        Ok(())
    }
    fn select_neighbors(
        &self,
        store_get: &dyn VestoStoreTrait, //
        query: &Vector,
        W: &Vec<EntityId>,
        M: usize,
    ) -> Result<Vec<(f32, EntityId)>, VestoError> {
        let mut scores = W
            .iter()
            .filter_map(|el| {
                let vector = store_get.get(el)?;
                match self.metric.distance(&vector, query) {
                    Ok(score) => Some(Ok((score, el.clone()))),
                    Err(e) => Some(Err(e)),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        scores.sort_by(|a, b| a.0.total_cmp(&b.0));
        scores.truncate(M.min(scores.len()));
        Ok(scores)
    }
    fn search_layer(
        &self,
        store_get: &dyn VestoStoreTrait,
        query: &Vector,
        ep: EntityId,
        ef: usize,
        l: usize,
    ) -> Vec<EntityId> {
        let ep_dist = self.dist(store_get, &ep, query);

        let mut visited: HashSet<EntityId> = HashSet::new();
        visited.insert(ep);

        // C: min-heap (nearest on top) - Reverse flips the max-heap
        let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
        candidates.push(Reverse(Candidate {
            dist: ep_dist,
            id: ep,
        }));

        // W: max-heap (furthest on top) - result set
        let mut result: BinaryHeap<Candidate> = BinaryHeap::new();
        result.push(Candidate {
            dist: ep_dist,
            id: ep,
        });

        while let Some(Reverse(c)) = candidates.pop() {
            // furthest currently in the result set
            let furthest = result.peek().map(|f| f.dist).unwrap_or(f32::INFINITY);
            if c.dist > furthest {
                // all elements in W (result set) are evaluated
                // nearest remaining candidate is worse than our worst kept result
                break; 
            }

            for e in self.neighbourhood(&c.id, l) {
                if visited.insert(e) {
                    // true only if "e" was newly inserted
                    let e_dist = self.dist(store_get, &e, query);
                    let furthest = result.peek().map(|f| f.dist).unwrap_or(f32::INFINITY);

                    if e_dist < furthest || result.len() < ef {
                        candidates.push(Reverse(Candidate {
                            dist: e_dist,
                            id: e,
                        }));
                        result.push(Candidate {
                            dist: e_dist,
                            id: e,
                        });
                        if result.len() > ef {
                            result.pop(); // evict furthest - O(log n), always the right one
                        }
                    }
                }
            }
        }

        result.into_iter().map(|c| c.id).collect()
    }
    fn neighbourhood(&self, id: &EntityId, l: usize) -> Vec<EntityId> {
        return self.layers[l]
            .adjacency
            .get(&id)
            .cloned()
            .unwrap_or_default();
    }

    fn nearest_element(
        &self,
        store_get: &dyn VestoStoreTrait, //
        set: &Vec<EntityId>,
        query: &Vector, // query element
    ) -> Option<(usize, EntityId)> {
        if set.is_empty() {
            return None;
        }
        let mut min_score = f32::INFINITY;
        let mut response = set[0];
        let mut min_pos: usize = 0;
        for (pos, el) in set.iter().enumerate() {
            if let Some(vector) = store_get.get(el) {
                let score = self.metric.distance(query, &vector).unwrap();
                if score < min_score {
                    response = el.clone();
                    min_score = score;
                    min_pos = pos;
                }
            }
        }
        return Some((min_pos, response));
    }
}

pub struct VestoHSNWIndex {
    name: String,
    vfield_name: String,
    data: HNSWGraph,

    max_connections: usize,
    max_connections_per_layer: usize,
    ef_construction: usize,
    m_l: f32,
}
pub struct VestoHSNWIndexExtraParams {
    max_connections: usize,           // number of established connections
    max_connections_per_layer: usize, //maximum number of connections for each element per layer
    ef_construction: usize,           // size of the dynamic candidate list
    m_l: f32,                         // normalization factor for level generation
}

impl VestoHSNWIndex {
    pub fn new(
        name: &str,
        vfield_name: &str,
        metric_name: crate::metrics::MetricsName,
        extra: Option<VestoHSNWIndexExtraParams>,
    ) -> Self
    where
        Self: Sized,
    {
        let extra = extra.unwrap_or(VestoHSNWIndexExtraParams {
            max_connections: 16,
            max_connections_per_layer: 16,
            ef_construction: 100,
            m_l: 1.0 / (16f32).ln(),
        });
        Self {
            name: String::from(name),
            vfield_name: String::from(vfield_name),
            data: HNSWGraph {
                layers: Vec::new(),
                metric: Metric::new(metric_name),
                entry_point: None,
            },
            ef_construction: extra.ef_construction,
            m_l: extra.m_l,
            max_connections: extra.max_connections,
            max_connections_per_layer: extra.max_connections_per_layer,
        }
    }
    pub fn len(&self) -> usize {
        self.data.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.layers.is_empty()
    }
}

impl VestoIndex for VestoHSNWIndex {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn vfield_name(&self) -> String {
        self.vfield_name.clone()
    }

    fn insert(
        &mut self,
        data: Vec<EntityId>,
        store_get: Option<&dyn VestoStoreTrait>,
    ) -> Result<(), VestoError> {
        if store_get.is_none() {
            return Err(VestoError::RequiredParameterMissing {
                param: "store".to_string(),
            });
        }
        for id in &data {
            self.data.insert(
                store_get.unwrap(),
                id.clone(),
                self.max_connections,
                self.max_connections_per_layer,
                self.ef_construction,
                self.m_l,
            )?;
        }
        Ok(())
    }

    fn search(
        &self,
        store_get: &dyn VestoStoreTrait,
        query: &Vector,
        top_k: usize,
    ) -> Result<Vec<(Score, EntityId)>, VestoError> {
        let results = self
            .data
            .knn_search(store_get, &query, top_k, self.ef_construction)?;
        return Ok(results);
    }
}

#[cfg(test)]
mod recall_test {
    use super::*;
    use crate::flat::VestoFlatIndex;
    use crate::metrics::MetricsName;
    use crate::store::VestoStore;
    use ndarray::array;
    use std::collections::HashSet;

    #[test]
    fn finds_obvious_nearest() {
        let mut store = VestoStore::new(2);
        let ids = store
            .insert(vec![
                array![1.0, 0.0], // id 0
                array![0.9, 0.1], // id 1  <- closest to query
                array![0.0, 1.0], // id 2
                array![0.0, 0.9], // id 3
            ])
            .unwrap();

        let mut hnsw = VestoHSNWIndex::new("h", "v", crate::metrics::MetricsName::L2, None);
        hnsw.insert(ids.clone(), Some(&store)).unwrap();

        // let query = array![1.0, 0.1];
        // let results = hnsw.search(&store, &query, 2).unwrap();
        let query = array![0.95, 0.08]; // clearly closest to id 1 [0.9, 0.1]
        let results = hnsw.search(&store, &query, 2).unwrap();
        assert_eq!(results[0].1, ids[1]);
        // println!("results: {:?}", results);
        // assert!(!results.is_empty(), "search returned nothing");
        // assert_eq!(results[0].1, ids[1], "nearest should be id 1 (0.9, 0.1)");
    }

    #[test]
    fn hnsw_recall_vs_bruteforce() {
        let dim = 10;
        let n = 100;
        let mut store = VestoStore::new(dim);

        // random vectors
        let vecs: Vec<_> = (0..n)
            .map(|_| ndarray::Array1::from_shape_fn(dim, |_| rand::random::<f32>()))
            .collect();
        let ids = store.insert(vecs).unwrap();

        // build both indexes
        let mut flat = VestoFlatIndex::new("flat", "v", MetricsName::L2);
        flat.insert(ids.clone(), Some(&store)).unwrap();
        let mut hnsw = VestoHSNWIndex::new("hnsw", "v", MetricsName::L2, None);
        hnsw.insert(ids.clone(), Some(&store)).unwrap();

        // query with several stored vectors, compare top-10
        let k = 10;
        let mut hits = 0;
        let mut total = 0;
        for &qid in ids.iter().take(30) {
            let q = store.get(&qid).unwrap();
            let truth: HashSet<_> = flat
                .search(&store, &q, k)
                .unwrap()
                .into_iter()
                .map(|(_, id)| id)
                .collect();
            let got: HashSet<_> = hnsw
                .search(&store, &q, k)
                .unwrap()
                .into_iter()
                .map(|(_, id)| id)
                .collect();
            hits += truth.intersection(&got).count();
            total += truth.len();
        }
        let recall = hits as f32 / total as f32;
        println!("recall@{k} = {recall:.3}");
        assert!(recall > 0.8, "recall too low: {recall}");
    }
}
