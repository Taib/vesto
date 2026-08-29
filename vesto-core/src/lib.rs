use ndarray::{Array1, linalg::Dot};

pub struct Vesto {
    store: Vec<Array1<f32>>,
}

impl Vesto {
    pub fn new() -> Self {
        Self { store: Vec::new() }
    }

    pub fn insert(&mut self, data: Vec<Array1<f32>>) -> Result<(), String> {
        self.store.extend_from_slice(&data);
        Ok(())
    }
    pub fn search(&self, query: &Array1<f32>, top_k: usize) -> Result<Vec<(f32, usize)>, String> {
        let mut scores = self
            .store
            .iter()
            .enumerate()
            .map(|(index, vector)| cosine_similarity(vector, query).map(|score| (score, index)))
            .collect::<Result<Vec<_>, _>>()?;
        scores.sort_by(
            |a, b| b.0.total_cmp(&a.0), // descending, safely handles NaN);
        );
        scores.truncate(top_k.min(scores.len()));
        Ok(scores)
    }
    pub fn get(&self, index: usize) -> Result<Array1<f32>, String> {
        if index >= self.store.len() {
            return Err(String::from("Index out of bounds"));
        }
        Ok(self.store[index].clone())
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }
}
#[derive(Debug)]
pub enum VestoError {
    DimensionMismatch { expected: usize, received: usize },
    ZeroVector,
}
impl std::fmt::Display for VestoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, received } => write!(
                f,
                "Dimension mismatch: expected {}, got {}.",
                expected, received
            ),
            Self::ZeroVector => write!(f, "Cannot compute with zero vector."),
        }
    }
}

pub fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> Result<f32, String> {
    if a.dim() != b.dim() {
        return Err(String::from("Arrays must be of same dim. Got {} and {}."));
    }
    let a_norm = a.dot(a).sqrt();
    let b_norm = b.dot(b).sqrt();

    if a_norm == 0.0 || b_norm == 0.0 {
        return Err("Cannot compute cosine similarity with a zero vector".into());
    }
    let a_dot_b = a.dot(b);
    Ok(a_dot_b / (a_norm * b_norm))
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::array;

    #[test]
    fn cosine_similarity_works() {
        let a = array![1.0, 0.0];
        let b = array![1.0, 0.0];

        let score = cosine_similarity(&a, &b).unwrap();

        assert!((score - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn search_returns_best_match() {
        println!("Vector Store: Vesto");
        let mut store = Vesto::new();
        let data = vec![
            array![1.0, 0.0, 0.0],
            array![0.9, 0.1, 0.0],
            array![0.0, 1.0, 0.0],
            array![0.0, 0.0, 1.0],
        ];
        store.insert(data).unwrap();

        let query = array![1.0, 0.1, 0.0];
        let results = store.search(&query, 2).unwrap();

        assert_eq!(results[0].1, 1);
    }
}
