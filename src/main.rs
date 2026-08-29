use std::fmt::Display;

use ndarray::{ArrayD, linalg::Dot};

pub fn cosine_similarity(vec1: &ArrayD<f32>, vec2: &ArrayD<f32>) -> f32 {
    let norm1 = vec1.mapv(|x| x.powi(2)).sum().sqrt();
    let norm2 = vec2.mapv(|x| x.powi(2)).sum().sqrt();

    let dot_product: f32 = vec1.dot(vec2).sum();

    dot_product / (norm1 * norm2)
}

struct VecStore {
    vecs: Vec<ArrayD<f32>>,
}
impl VecStore {
    fn new() -> Self {
        VecStore { vecs: Vec::new() }
    }

    fn add_vector(&mut self, vec: ArrayD<f32>) {
        self.vecs.push(vec);
    }

    fn get_vector(&self, index: usize) -> Option<&ArrayD<f32>> {
        self.vecs.get(index)
    }

    fn query(&self, query_vec: &ArrayD<f32>, top_k: usize) -> Vec<(usize, f32)> {
        let mut results: Vec<(usize, f32)> = self
            .vecs
            .iter()
            .enumerate()
            .map(|(i, vec)| (i, cosine_similarity(&vec, &query_vec)))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.into_iter().take(top_k).collect()
    }
}
impl Display for VecStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VecStore with {} vectors", self.vecs.len())
    }
}
fn main() {
    let mut store = VecStore::new();
    store.add_vector(ArrayD::from_elem(ndarray::IxDyn(&[3]), 1.0));
    println!("Store: {}", store);
    println!("Vector 0: {}", store.get_vector(0).unwrap());
    println!("Query results: {:?}", store.query(&ArrayD::from_elem(ndarray::IxDyn(&[3]), 1.0), 1));
}
