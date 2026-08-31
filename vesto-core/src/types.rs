use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};

pub type Vector = Array1<f32>;
pub type VectorView<'a> = ArrayView1<'a, f32>;
pub type Score = f32;

#[derive(Serialize, Deserialize, Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct EntityId(pub u64);
