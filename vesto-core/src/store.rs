use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::{collections::HashMap, path::Path};

use ndarray::{Array2, Axis};

use crate::types::VectorView;
use crate::{
    error::VestoError,
    types::{EntityId, Vector},
};
pub trait VestoStoreTrait {
    fn insert(&mut self, vectors: Vec<Vector>) -> Result<Vec<EntityId>, VestoError>;
    fn get(&self, id: &EntityId) -> Option<Vector>;
    fn get_view(&self, id: &EntityId) -> Option<VectorView<'_>>;
    // fn delete(&mut self, id: EntityId) -> Result<(), VestoError>;
}

const MAGIC: &[u8; 4] = b"VSTO";
const VERSION: u32 = 1;

pub struct VestoStore {
    ids: Vec<EntityId>,                  // position -> id
    vectors: Array2<f32>,                // position -> row, contiguous
    id_to_pos: HashMap<EntityId, usize>, // id -> position (boundary only)
    next_id: u64,
    //deleted: Vec<bool>,
}
// Summary of a VestoStore
#[derive(Serialize, Deserialize)]
struct StoreMeta {
    ids: Vec<EntityId>,
    next_id: u64,
    // deleted: Vec<bool>,
}

impl VestoStoreTrait for VestoStore {
    fn insert(&mut self, vectors: Vec<Vector>) -> Result<Vec<EntityId>, VestoError> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        let dim = self.vectors.ncols();
        let n = vectors.len();

        // Making on contiguious array to be appended once later
        let mut flat = Vec::with_capacity(n * dim);
        for v in &vectors {
            if v.len() != dim {
                return Err(VestoError::DimensionMismatch {
                    expected: dim,
                    received: v.len(),
                });
            }
            flat.extend(v.iter().copied());
        }
        let block = Array2::from_shape_vec((n, dim), flat).map_err(|_| VestoError::ShapeError)?;

        // Appending to the vectors 2-d array
        let start_pos = self.vectors.nrows();
        self.vectors
            .append(Axis(0), block.view())
            .map_err(|_| VestoError::ShapeError)?;

        // Saving ids and increment next_id
        let mut ids = Vec::with_capacity(n);
        for i in 0..n {
            let pos = start_pos + i;
            let id = EntityId(self.next_id);
            self.next_id += 1;
            self.ids.push(id);
            self.id_to_pos.insert(id, pos);
            ids.push(id);
        }
        Ok(ids)
    }

    fn get(&self, id: &EntityId) -> Option<Vector> {
        let &pos = self.id_to_pos.get(id)?;
        Some(self.vectors.row(pos).to_owned())
    }

    fn get_view(&self, id: &EntityId) -> Option<VectorView<'_>> {
        let pos = self.id_to_pos.get(id)?;
        Some(self.vectors.row(*pos))
    }
}

impl VestoStore {
    pub fn new(dim: usize) -> Self {
        Self {
            ids: Vec::new(),
            vectors: Array2::default((0, dim)),
            id_to_pos: HashMap::new(),
            next_id: 0,
        }
    }
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), VestoError> {
        let nrows = self.vectors.nrows();
        let dim = self.vectors.ncols();

        // metadata -> postcard
        let meta = StoreMeta {
            ids: self.ids.clone(),
            next_id: self.next_id,
        };
        let meta_bytes = postcard::to_allocvec(&meta).map_err(|_| VestoError::SerializeError)?;

        // matrix -> raw f32 bytes. Ensure row-major contiguous first.
        let standard = self.vectors.as_standard_layout(); // Cow: copies only if needed
        let floats = standard.as_slice().ok_or(VestoError::SerializeError)?;
        let matrix_bytes: &[u8] = bytemuck::cast_slice(floats); // &[f32] -> &[u8]: always sound

        let mut w = BufWriter::new(File::create(path).map_err(|_| VestoError::IoError)?);
        w.write_all(MAGIC).map_err(|_| VestoError::IoError)?;
        w.write_all(&VERSION.to_le_bytes())
            .map_err(|_| VestoError::IoError)?;
        w.write_all(&(dim as u64).to_le_bytes())
            .map_err(|_| VestoError::IoError)?;
        w.write_all(&(nrows as u64).to_le_bytes())
            .map_err(|_| VestoError::IoError)?;
        w.write_all(&(meta_bytes.len() as u64).to_le_bytes())
            .map_err(|_| VestoError::IoError)?;
        w.write_all(&meta_bytes).map_err(|_| VestoError::IoError)?;
        w.write_all(matrix_bytes).map_err(|_| VestoError::IoError)?;
        w.flush().map_err(|_| VestoError::IoError)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, VestoError> {
        let mut buf = Vec::new();
        BufReader::new(File::open(path).map_err(|_| VestoError::IoError)?)
            .read_to_end(&mut buf)
            .map_err(|_| VestoError::IoError)?;

        let mut c = 0usize;
        let take = |c: &mut usize, n: usize| {
            let s = &buf[*c..*c + n];
            *c += n;
            s
        };

        if take(&mut c, 4) != MAGIC {
            return Err(VestoError::BadHeader);
        }
        let version = u32::from_le_bytes(take(&mut c, 4).try_into().unwrap());
        if version != VERSION {
            return Err(VestoError::BadVersion);
        }
        let dim = u64::from_le_bytes(take(&mut c, 8).try_into().unwrap()) as usize;
        let nrows = u64::from_le_bytes(take(&mut c, 8).try_into().unwrap()) as usize;
        let mlen = u64::from_le_bytes(take(&mut c, 8).try_into().unwrap()) as usize;

        let meta: StoreMeta =
            postcard::from_bytes(take(&mut c, mlen)).map_err(|_| VestoError::DeserializeError)?;

        // matrix: DON'T cast_slice here — the offset isn't guaranteed f32-aligned.
        let expected = nrows * dim * std::mem::size_of::<f32>();
        let floats: Vec<f32> = take(&mut c, expected)
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let vectors =
            Array2::from_shape_vec((nrows, dim), floats).map_err(|_| VestoError::ShapeError)?;

        // id_to_pos is derivable from ids — rebuild rather than store it.
        let mut id_to_pos = HashMap::with_capacity(meta.ids.len());
        for (pos, &id) in meta.ids.iter().enumerate() {
            id_to_pos.insert(id, pos);
        }

        Ok(Self {
            ids: meta.ids,
            vectors,
            id_to_pos,
            next_id: meta.next_id,
        })
    }
}
