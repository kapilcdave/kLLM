//! Minimal, allocation-bounded GGUF metadata and tensor inspector.

use std::{
    fmt,
    fs::File,
    io::{self, BufReader, Read, Seek, SeekFrom},
    path::Path,
};

const MAGIC: &[u8; 4] = b"GGUF";
const MAX_STRING_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ITEMS: u64 = 1_000_000;
const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufInfo {
    pub version: u32,
    pub metadata_count: u64,
    pub tensor_count: u64,
    pub alignment: u64,
    pub data_offset: u64,
    pub architecture: Option<String>,
    pub tensors: Vec<TensorInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub relative_offset: u64,
}

#[derive(Debug)]
pub enum GgufError {
    Io(io::Error),
    Invalid(&'static str),
    UnsupportedVersion(u32),
    UnsupportedValueType(u32),
    LimitExceeded(&'static str),
    ArithmeticOverflow,
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Invalid(message) => write!(f, "invalid GGUF: {message}"),
            Self::UnsupportedVersion(version) => write!(f, "unsupported GGUF version {version}"),
            Self::UnsupportedValueType(kind) => write!(f, "unsupported GGUF value type {kind}"),
            Self::LimitExceeded(what) => write!(f, "GGUF {what} exceeds inspector limit"),
            Self::ArithmeticOverflow => write!(f, "GGUF offset arithmetic overflow"),
        }
    }
}

impl std::error::Error for GgufError {}

impl From<io::Error> for GgufError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

pub fn inspect_file(path: impl AsRef<Path>) -> Result<GgufInfo, GgufError> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    inspect(BufReader::new(file), file_len)
}

pub fn inspect<R: Read + Seek>(mut reader: R, file_len: u64) -> Result<GgufInfo, GgufError> {
    let mut magic = [0; 4];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(GgufError::Invalid("bad magic"));
    }
    let version = read_u32(&mut reader)?;
    if !(2..=3).contains(&version) {
        return Err(GgufError::UnsupportedVersion(version));
    }
    let tensor_count = read_u64(&mut reader)?;
    let metadata_count = read_u64(&mut reader)?;
    check_count(tensor_count, "tensor count")?;
    check_count(metadata_count, "metadata count")?;

    let mut architecture = None;
    let mut alignment = DEFAULT_ALIGNMENT;
    for _ in 0..metadata_count {
        let key = read_string(&mut reader)?;
        let value_type = read_u32(&mut reader)?;
        if key == "general.architecture" && value_type == 8 {
            architecture = Some(read_string(&mut reader)?);
        } else if key == "general.alignment" && value_type == 4 {
            alignment = u64::from(read_u32(&mut reader)?);
        } else {
            skip_value(&mut reader, value_type, 0)?;
        }
    }
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(GgufError::Invalid(
            "general.alignment must be a nonzero power of two",
        ));
    }

    let tensor_capacity =
        usize::try_from(tensor_count).map_err(|_| GgufError::LimitExceeded("tensor count"))?;
    let mut tensors = Vec::with_capacity(tensor_capacity);
    for _ in 0..tensor_count {
        let name = read_string(&mut reader)?;
        let dimension_count = read_u32(&mut reader)?;
        if dimension_count > 4 {
            return Err(GgufError::Invalid("tensor has more than four dimensions"));
        }
        let mut dimensions = Vec::with_capacity(dimension_count as usize);
        for _ in 0..dimension_count {
            dimensions.push(read_u64(&mut reader)?);
        }
        let ggml_type = read_u32(&mut reader)?;
        let relative_offset = read_u64(&mut reader)?;
        tensors.push(TensorInfo {
            name,
            dimensions,
            ggml_type,
            relative_offset,
        });
    }

    let descriptors_end = reader.stream_position()?;
    let data_offset = align_up(descriptors_end, alignment)?;
    if data_offset > file_len {
        return Err(GgufError::Invalid("tensor data begins beyond end of file"));
    }
    for tensor in &tensors {
        let absolute = data_offset
            .checked_add(tensor.relative_offset)
            .ok_or(GgufError::ArithmeticOverflow)?;
        if absolute > file_len {
            return Err(GgufError::Invalid("tensor offset is beyond end of file"));
        }
    }

    Ok(GgufInfo {
        version,
        metadata_count,
        tensor_count,
        alignment,
        data_offset,
        architecture,
        tensors,
    })
}

fn skip_value<R: Read + Seek>(reader: &mut R, kind: u32, depth: u8) -> Result<(), GgufError> {
    if depth > 8 {
        return Err(GgufError::LimitExceeded("array nesting"));
    }
    let byte_count = match kind {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4 | 5 | 6 => 4,
        10 | 11 | 12 => 8,
        8 => {
            let length = read_u64(reader)?;
            check_string_length(length)?;
            seek_forward(reader, length)?;
            return Ok(());
        }
        9 => {
            let element_type = read_u32(reader)?;
            let count = read_u64(reader)?;
            check_count(count, "array length")?;
            for _ in 0..count {
                skip_value(reader, element_type, depth + 1)?;
            }
            return Ok(());
        }
        _ => return Err(GgufError::UnsupportedValueType(kind)),
    };
    seek_forward(reader, byte_count)
}

fn read_string<R: Read>(reader: &mut R) -> Result<String, GgufError> {
    let length = read_u64(reader)?;
    check_string_length(length)?;
    let length = usize::try_from(length).map_err(|_| GgufError::LimitExceeded("string"))?;
    let mut bytes = vec![0; length];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|_| GgufError::Invalid("string is not UTF-8"))
}

fn check_string_length(length: u64) -> Result<(), GgufError> {
    if length > MAX_STRING_BYTES {
        Err(GgufError::LimitExceeded("string"))
    } else {
        Ok(())
    }
}

fn check_count(count: u64, what: &'static str) -> Result<(), GgufError> {
    if count > MAX_ITEMS {
        Err(GgufError::LimitExceeded(what))
    } else {
        Ok(())
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64, GgufError> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .ok_or(GgufError::ArithmeticOverflow)
}

fn seek_forward<R: Seek>(reader: &mut R, amount: u64) -> Result<(), GgufError> {
    let amount = i64::try_from(amount).map_err(|_| GgufError::ArithmeticOverflow)?;
    reader.seek(SeekFrom::Current(amount))?;
    Ok(())
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32, GgufError> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64, GgufError> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn push_string(output: &mut Vec<u8>, value: &str) {
        output.extend_from_slice(&(value.len() as u64).to_le_bytes());
        output.extend_from_slice(value.as_bytes());
    }

    fn fixture() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        push_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        push_string(&mut bytes, "llama");
        push_string(&mut bytes, "token_embd.weight");
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&16_u64.to_le_bytes());
        bytes.extend_from_slice(&32_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        while bytes.len() % DEFAULT_ALIGNMENT as usize != 0 {
            bytes.push(0);
        }
        bytes
    }

    #[test]
    fn inspects_header_metadata_and_tensor() {
        let bytes = fixture();
        let info = inspect(Cursor::new(&bytes), bytes.len() as u64).unwrap();
        assert_eq!(info.version, 3);
        assert_eq!(info.architecture.as_deref(), Some("llama"));
        assert_eq!(info.tensors[0].dimensions, vec![16, 32]);
        assert_eq!(info.data_offset % DEFAULT_ALIGNMENT, 0);
    }

    #[test]
    fn rejects_bad_magic() {
        let bytes = vec![0; 24];
        assert!(matches!(
            inspect(Cursor::new(&bytes), bytes.len() as u64),
            Err(GgufError::Invalid("bad magic"))
        ));
    }
}
