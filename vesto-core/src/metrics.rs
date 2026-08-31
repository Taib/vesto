use ndarray::{ArrayBase, Data, Ix1};

use crate::{error::VestoError, types::Score};

#[derive(Debug, Clone, Copy)]
pub enum MetricsName {
    L2,
    Cosine,
}
pub fn parse_metric_from_str(name: &str) -> MetricsName {
    match name.to_lowercase().as_str() {
        "l2" => MetricsName::L2,
        _ => MetricsName::Cosine,
    }
}
pub struct Metric {
    name: MetricsName,
}
impl Metric {
    pub fn new(name: MetricsName) -> Self {
        Self { name }
    }
    pub fn distance<S1, S2>(
        &self,
        a: &ArrayBase<S1, Ix1>,
        b: &ArrayBase<S2, Ix1>,
    ) -> Result<Score, VestoError>
    where
        S1: Data<Elem = f32>,
        S2: Data<Elem = f32>,
    {
        match self.name {
            MetricsName::Cosine => cosine_similarity(a, b),
            _ => Ok(a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>()
                .sqrt()),
        }
    }
}

// Returns 1.0 - cosine_similarity(a, b):
// To match the behavior of other metrics, where lower is better.
pub fn cosine_similarity<S1, S2>(
    a: &ArrayBase<S1, Ix1>,
    b: &ArrayBase<S2, Ix1>,
) -> Result<Score, VestoError>
where
    S1: Data<Elem = f32>,
    S2: Data<Elem = f32>,
{
    if a.dim() != b.dim() {
        return Err(VestoError::DimensionMismatch {
            expected: a.len(),
            received: b.len(),
        });
    }
    let a_norm = a.dot(a).sqrt();
    let b_norm = b.dot(b).sqrt();

    if a_norm == 0.0 || b_norm == 0.0 {
        return Err(VestoError::ZeroVector);
    }
    let a_dot_b = a.dot(b);
    Ok(1.0 - a_dot_b / (a_norm * b_norm))
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

        assert!((score - 0.0).abs() < 1e-6);
    }
}
