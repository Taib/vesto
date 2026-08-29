#[derive(Debug)]
pub enum VestoError {
    DimensionMismatch { expected: usize, received: usize },
    ZeroVector,
    ShapeError,
    SerializeError,
    IoError,
    BadHeader,
    BadVersion,
    DeserializeError,
    DuplicateIndex,
    KeyNotFound,
    DuplicateCollection,
    UnknownIndexType,
    RequiredParameterMissing { param: String },
    EmptyIndex
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
            Self::ShapeError => write!(f, "Shape error."),
            Self::SerializeError => write!(f, "Serialization error"),
            Self::DeserializeError => write!(f, "Deserialization error"),
            Self::IoError => write!(f, "IO error."),
            Self::BadHeader => write!(f, "Bad Header."),
            Self::BadVersion => write!(f, "Bad Version."),
            Self::DuplicateIndex => write!(f, "Duplicated Index"),
            Self::KeyNotFound => write!(f, "Key Not Found"),
            Self::DuplicateCollection => write!(f, "Duplicate Collection"),
            Self::UnknownIndexType => write!(f, "Unknown Index Type"),
            Self::RequiredParameterMissing { param } => {
                write!(f, "Required parameter {param:?} is missing.")
            },
            Self::EmptyIndex => write!(f, "Empty Index"),
        }
    }
}

impl std::error::Error for VestoError {}
