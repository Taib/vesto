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
    use_heuristic_selection: bool,
    heuristic_keep_pruned_connections: bool, // flag indicating whether or not to add discarded elements
    heuristic_extend_candidates: bool, // flag indicating whether or not to extend candidate list
}

impl HNSWGraph {
    fn dist(&self, store_get: &dyn VestoStoreTrait, id: &EntityId, query: &Vector) -> f32 {
        let v = store_get.get_view(id).unwrap();
        self.metric.distance(&v, query).unwrap()
    }
    fn knn_search(
        &self,                                 //
        store_get: &dyn VestoStoreTrait,       //
        query: &Vector,                        // query element
        k: usize,                              // number of nearest neighbors to return
        ef: usize,                             // size of the dynamic candidate list
        extend_candidates: Option<bool>, // flag indicating whether or not to extend candidate list
        keep_pruned_connections: Option<bool>, // flag indicating whether or not to add
    ) -> Result<Vec<(f32, EntityId)>, VestoError> {
        let mut W; // set for the current nearest elements
        let mut ep = vec![self.entry_point.ok_or(VestoError::EmptyIndex)?]; // get enter point for hnsw
        let L = self.layers.len().saturating_sub(1); // level of ep 

        for lc in (1..=L).rev() {
            W = self.search_layer(store_get, &query, &ep, 1, lc);
            let (_, n_el) = self.nearest_element(store_get, &W, &query).unwrap();
            ep = vec![n_el];
        }
        W = self.search_layer(store_get, &query, &ep, ef, 0);

        Ok(self.select_neighbors(
            store_get,
            &query,
            &W,
            k,
            extend_candidates,
            keep_pruned_connections,
        )?)
    }
    fn insert(
        &mut self,
        store_get: &dyn VestoStoreTrait, //
        item: EntityId,
        M: usize,                              // number of established connections
        M_max: usize, //maximum number of connections for each element per layer
        ef_construction: usize, // size of the dynamic candidate list
        m_l: f32,     // normalization factor for level generation
        extend_candidates: Option<bool>, // flag indicating whether or not to extend candidate list
        keep_pruned_connections: Option<bool>, // flag indicating whether or not to add discarded elements
    ) -> Result<(), VestoError> {
        let mut W;
        let mut ep = vec![self.entry_point.unwrap_or(item)]; // get entry points for hnsw
        let L = self.layers.len().saturating_sub(1); // top layer fo hnsw
        let l: usize = (-(1.0 - rand::random::<f32>()).ln() * m_l).floor() as usize; // new element's level 
        if self.entry_point.is_none() {
            return self.fresh_start_insert(item, l);
        }
        let query = store_get.get(&item).unwrap();

        for lc in (l + 1..=L).rev() {
            W = self.search_layer(store_get, &query, &ep, 1, lc);
            let (_, n_el) = self.nearest_element(store_get, &W, &query).unwrap();
            ep = vec![n_el];
        }
        for lc in (0..min(L, l) + 1).rev() {
            let Mmax = if lc == 0 { M * 2 } else { M_max };
            W = self.search_layer(store_get, &query, &ep, ef_construction, lc);
            let neighbors = self
                .select_neighbors(
                    store_get,
                    &query,
                    &W,
                    M,
                    extend_candidates,
                    keep_pruned_connections,
                )
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
                if e_conn.len() > Mmax {
                    // shrink connections of e
                    let e_vector = store_get.get(&e).unwrap();
                    let e_new_conn = self
                        .select_neighbors(store_get, &e_vector, e_conn, Mmax, None, None)
                        .unwrap();
                    // update neighbourhood of e at layer lc to e_new_conn;
                    self.layers[lc].adjacency.remove(&e);
                    self.layers[lc]
                        .adjacency
                        .insert(e, e_new_conn.iter().map(|(_, el)| el.clone()).collect());
                }
            }
            ep = W.clone();
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
        W: &[EntityId],
        M: usize,
        extend_candidates: Option<bool>,
        keep_pruned_connections: Option<bool>,
    ) -> Result<Vec<(f32, EntityId)>, VestoError> {
        if self.use_heuristic_selection {
            return self.select_neighbors_heuristic(
                store_get,
                query,
                W.to_vec(),
                M,
                0,
                extend_candidates.unwrap_or(self.heuristic_extend_candidates),
                keep_pruned_connections.unwrap_or(self.heuristic_keep_pruned_connections),
            );
        }
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
        ep: &Vec<EntityId>,
        ef: usize,
        l: usize,
    ) -> Vec<EntityId> {
        let mut visited: HashSet<EntityId> = HashSet::new();

        // C: min-heap (nearest on top) - Reverse flips the max-heap
        let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();

        // W: max-heap (furthest on top) - result set
        let mut result: BinaryHeap<Candidate> = BinaryHeap::new();

        for &e in ep {
            let ep_dist = self.dist(store_get, &e, query);
            candidates.push(Reverse(Candidate {
                dist: ep_dist,
                id: e,
            }));
            result.push(Candidate {
                dist: ep_dist,
                id: e,
            });
        }

        while let Some(Reverse(c)) = candidates.pop() {
            // furthest currently in the result set
            let furthest = result.peek().map(|f| f.dist).unwrap_or(f32::INFINITY);
            if c.dist > furthest {
                // all elements in W (result set) are evaluated
                // nearest remaining candidate is worse than our worst kept result
                break;
            }

            for e in self.neighbourhood(&c.id, l) {
                if visited.insert(*e) {
                    // true only if "e" was newly inserted
                    let e_dist = self.dist(store_get, e, query);
                    let furthest = result.peek().map(|f| f.dist).unwrap_or(f32::INFINITY);

                    if e_dist < furthest || result.len() < ef {
                        candidates.push(Reverse(Candidate {
                            dist: e_dist,
                            id: *e,
                        }));
                        result.push(Candidate {
                            dist: e_dist,
                            id: *e,
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
    fn neighbourhood(&self, id: &EntityId, l: usize) -> &[EntityId] {
        return self.layers[l]
            .adjacency
            .get(&id)
            .map(Vec::as_slice)
            .unwrap_or_default();
    }

    fn nearest_element(
        &self,
        store_get: &dyn VestoStoreTrait, //
        ids_list: &Vec<EntityId>,
        query: &Vector, // query element
    ) -> Option<(usize, EntityId)> {
        if ids_list.is_empty() {
            return None;
        }
        let mut min_score = f32::INFINITY;
        let mut response = ids_list[0];
        let mut min_pos: usize = 0;
        for (pos, el) in ids_list.iter().enumerate() {
            if let Some(vector) = store_get.get(el) {
                let score = self.metric.distance(query, &vector).unwrap();
                if score < min_score {
                    response = *el;
                    min_score = score;
                    min_pos = pos;
                }
            }
        }
        return Some((min_pos, response));
    }

    fn select_neighbors_heuristic(
        &self,                           //
        store_get: &dyn VestoStoreTrait, //
        query: &Vector,
        candidates_list: Vec<EntityId>, // candidate elements
        max_neighbors: usize,           // number of neighbors to return
        lc: usize,                      // layer number
        extend_candidates: bool,        // flag indicating whether or not to extend candidate list
        keep_pruned_connections: bool,  // flag indicating whether or not to add discarded elements
    ) -> Result<Vec<(f32, EntityId)>, VestoError> {
        let mut result: Vec<(f32, EntityId)> = Vec::new();
        let mut work_set: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();

        for e in &candidates_list {
            let ep_dist = self.dist(store_get, e, query);
            work_set.push(Reverse(Candidate {
                dist: ep_dist,
                id: *e,
            }));
        }

        if extend_candidates {
            // Implementation for extending candidates
            for e in &candidates_list {
                for e_adj in self.neighbourhood(e, lc) {
                    work_set.push(Reverse(Candidate {
                        dist: self.dist(store_get, e_adj, query),
                        id: *e_adj,
                    }));
                }
            }
        }
        let mut discard_set: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
        while let Some(Reverse(w)) = work_set.pop()
            && result.len() < max_neighbors
        {
            let w_vec = store_get.get(&w.id).unwrap();
            let closest_in_result = result
                .iter()
                .map(|(_, e)| self.dist(store_get, e, &w_vec))
                .min_by(|a, b| a.partial_cmp(b).unwrap());
            if result.len() == 0 {
                result.push((w.dist, w.id));
            } else {
                if closest_in_result.unwrap() > w.dist {
                    result.push((w.dist, w.id));
                } else {
                    discard_set.push(Reverse(Candidate {
                        dist: w.dist,
                        id: w.id,
                    }));
                }
            }
        }
        if keep_pruned_connections {
            while let Some(Reverse(w_d)) = discard_set.pop()
                && result.len() < max_neighbors
            {
                result.push((w_d.dist, w_d.id));
            }
        }
        Ok(result)
    }
}

pub struct VestoHNSWIndex {
    name: String,
    data: HNSWGraph,

    max_connections: usize,
    max_connections_per_layer: usize,
    ef_construction: usize,
    m_l: f32,
}
pub struct VestoHNSWIndexExtraParams {
    max_connections: usize,                // number of established connections
    max_connections_per_layer: usize, //maximum number of connections for each element per layer
    ef_construction: usize,           // size of the dynamic candidate list
    m_l: f32,                         // normalization factor for level generation
    use_heuristic_selection: bool,    // flag indicating whether or not to use heuristic selection
    extend_candidates: Option<bool>,  // flag indicating whether or not to extend candidate list
    keep_pruned_connections: Option<bool>, // flag indicating whether or not to add discarded elements
}

impl VestoHNSWIndex {
    pub fn new(
        name: &str,
        metric_name: crate::metrics::MetricsName,
        extra: Option<VestoHNSWIndexExtraParams>,
    ) -> Self
    where
        Self: Sized,
    {
        let extra = extra.unwrap_or(VestoHNSWIndexExtraParams {
            max_connections: 16,
            max_connections_per_layer: 16,
            ef_construction: 100,
            m_l: 1.0 / (16f32).ln(),
            use_heuristic_selection: true,
            extend_candidates: None,
            keep_pruned_connections: None,
        });
        Self {
            name: String::from(name),
            data: HNSWGraph {
                layers: Vec::new(),
                metric: Metric::new(metric_name),
                entry_point: None,
                use_heuristic_selection: extra.use_heuristic_selection,
                heuristic_extend_candidates: extra.extend_candidates.unwrap_or(false),
                heuristic_keep_pruned_connections: extra.keep_pruned_connections.unwrap_or(true),
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

    fn extract_from_json_params(
        &self,
        params: Option<&serde_json::Value>,
    ) -> VestoHNSWIndexExtraParams {
        if params.is_none() {
            return VestoHNSWIndexExtraParams {
                max_connections: self.max_connections,
                max_connections_per_layer: self.max_connections_per_layer,
                ef_construction: self.ef_construction,
                m_l: self.m_l,
                use_heuristic_selection: self.data.use_heuristic_selection,
                extend_candidates: None,
                keep_pruned_connections: None,
            };
        }
        let params = params.unwrap();
        let max_connections = params
            .get("max_connections")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(self.max_connections);
        let max_connections_per_layer = params
            .get("max_connections_per_layer")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(self.max_connections_per_layer);
        let ef_construction = params
            .get("ef_construction")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(self.ef_construction);
        let m_l = params
            .get("m_l")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(self.m_l);
        let use_heuristic_selection = params
            .get("use_heuristic_selection")
            .and_then(|v| v.as_bool())
            .unwrap_or(self.data.use_heuristic_selection);
        let extend_candidates = params.get("extend_candidates").and_then(|v| v.as_bool());
        let keep_pruned_connections = params
            .get("keep_pruned_connections")
            .and_then(|v| v.as_bool());
        VestoHNSWIndexExtraParams {
            max_connections,
            max_connections_per_layer,
            ef_construction,
            m_l,
            use_heuristic_selection,
            extend_candidates,
            keep_pruned_connections,
        }
    }
}

impl VestoIndex for VestoHNSWIndex {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn insert(
        &mut self,
        data: Vec<EntityId>,
        store_get: Option<&dyn VestoStoreTrait>,
        params: Option<&serde_json::Value>,
    ) -> Result<(), VestoError> {
        if store_get.is_none() {
            return Err(VestoError::RequiredParameterMissing {
                param: "store".to_string(),
            });
        }
        let params = self.extract_from_json_params(params);
        for id in &data {
            self.data.insert(
                store_get.unwrap(),
                id.clone(),
                params.max_connections,
                params.max_connections_per_layer,
                params.ef_construction,
                params.m_l,
                params.extend_candidates,
                params.keep_pruned_connections,
            )?;
        }
        Ok(())
    }

    fn search(
        &self,
        store_get: &dyn VestoStoreTrait,
        query: &Vector,
        top_k: usize,
        params: Option<&serde_json::Value>,
    ) -> Result<Vec<(Score, EntityId)>, VestoError> {
        let params = self.extract_from_json_params(params);
        let results = self.data.knn_search(
            store_get,
            &query,
            top_k,
            params.ef_construction,
            params.extend_candidates,
            params.keep_pruned_connections,
        )?;
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

        let mut hnsw = VestoHNSWIndex::new("h", crate::metrics::MetricsName::L2, None);
        hnsw.insert(ids.clone(), Some(&store), None).unwrap();

        let query = array![0.95, 0.08]; // clearly closest to id 1 [0.9, 0.1]
        let results = hnsw.search(&store, &query, 2, None).unwrap();
        assert_eq!(results[0].1, ids[1]);
    }

    #[test]
    fn heuristic_favors_diversity_over_closest_pair() {
        // A1 and A2 sit right next to each other in the same direction from
        // the query; B is a bit farther from the query but points a
        // different way. Simple top-M selection should pick the closest
        // pair (A1, A2), even though A2 is redundant with A1. The heuristic
        // should prune the redundant A2 in favor of the more diverse B.
        let mut store = VestoStore::new(2);
        let ids = store
            .insert(vec![
                array![1.0, 0.0],  // id 0 = A1, dist to query = 1.0
                array![1.0, 0.2],  // id 1 = A2, dist to query ~= 1.02, clustered next to A1
                array![0.0, 1.05], // id 2 = B,  dist to query = 1.05, different direction
            ])
            .unwrap();
        let query = array![0.0, 0.0];

        let mut graph = HNSWGraph {
            layers: Vec::new(),
            metric: Metric::new(MetricsName::L2),
            entry_point: None,
            use_heuristic_selection: false,
            heuristic_extend_candidates: false,
            heuristic_keep_pruned_connections: false,
        };

        let simple: HashSet<_> = graph
            .select_neighbors(&store, &query, &ids, 2, None, None)
            .unwrap()
            .into_iter()
            .map(|(_, id)| id)
            .collect();
        assert_eq!(simple, HashSet::from([ids[0], ids[1]]));

        graph.use_heuristic_selection = true;
        let heuristic: HashSet<_> = graph
            .select_neighbors(&store, &query, &ids, 2, None, None)
            .unwrap()
            .into_iter()
            .map(|(_, id)| id)
            .collect();
        assert_eq!(heuristic, HashSet::from([ids[0], ids[2]]));
    }

    #[test]
    fn hnsw_recall_vs_bruteforce() {
        let dim = 10;
        let n = 1000;
        let mut store = VestoStore::new(dim);

        // random vectors
        let vecs: Vec<_> = (0..n)
            .map(|_| ndarray::Array1::from_shape_fn(dim, |_| rand::random::<f32>()))
            .collect();
        let ids = store.insert(vecs).unwrap();

        // build both indexes
        let mut flat = VestoFlatIndex::new("flat", MetricsName::L2);
        flat.insert(ids.clone(), Some(&store), None).unwrap();
        let mut hnsw = VestoHNSWIndex::new("hnsw", MetricsName::L2, None);
        let t = std::time::Instant::now();
        hnsw.insert(ids.clone(), Some(&store), None).unwrap();
        println!("hnsw build: {:?}", t.elapsed());

        // query with several stored vectors, compare top-10
        let k = 10;
        let mut hits = 0;
        let mut total = 0;
        let mut avg_flat_search_time = 0.0;
        let mut avg_hnsw_search_time = 0.0;
        for &qid in ids.iter().take(30) {
            let q = store.get(&qid).unwrap();
            let t = std::time::Instant::now();
            let truth: HashSet<_> = flat
                .search(&store, &q, k, None)
                .unwrap()
                .into_iter()
                .map(|(_, id)| id)
                .collect();
            avg_flat_search_time += t.elapsed().as_millis() as f32;
            let t = std::time::Instant::now();
            let got: HashSet<_> = hnsw
                .search(&store, &q, k, None)
                .unwrap()
                .into_iter()
                .map(|(_, id)| id)
                .collect();
            avg_hnsw_search_time += t.elapsed().as_millis() as f32;
            hits += truth.intersection(&got).count();
            total += truth.len();
        }
        let recall = hits as f32 / total as f32;
        avg_flat_search_time /= 30.0;
        avg_hnsw_search_time /= 30.0;
        println!("avg flat search time: {avg_flat_search_time:.3} ms");
        println!("avg hnsw search time: {avg_hnsw_search_time:.3} ms");
        println!(
            "speedup: {:.2}x",
            avg_flat_search_time / avg_hnsw_search_time
        );
        println!("recall@{k} = {recall:.3}");
        assert!(recall > 0.8, "recall too low: {recall}");
    }
}
