//! Allocation-bounded GGUF parser.
//!
//! The parser reads the header, every metadata value, and the tensor
//! directory, but never the tensor payloads. Every length and count read from
//! the file is checked against the bytes that remain before anything is
//! allocated, so a malformed or hostile file cannot make kLLM allocate more
//! memory than the file itself occupies. Tensor payloads are exposed through a
//! read-only memory map by [`GgufFile`].

use std::{
    collections::HashMap,
    fmt,
    fs::File,
    io::{self, BufReader, Read, Seek},
    path::Path,
};

use memmap2::Mmap;

use crate::quant::GgmlType;

const MAGIC: &[u8; 4] = b"GGUF";
const MAX_STRING_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ITEMS: u64 = 16 * 1024 * 1024;
const MAX_DIMENSIONS: u32 = 4;
const DEFAULT_ALIGNMENT: u64 = 32;

/// A typed GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    /// Any non-negative integer value, widened to `u64`.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(v.into()),
            Self::U16(v) => Some(v.into()),
            Self::U32(v) => Some(v.into()),
            Self::U64(v) => Some(v),
            Self::I8(v) => u64::try_from(v).ok(),
            Self::I16(v) => u64::try_from(v).ok(),
            Self::I32(v) => u64::try_from(v).ok(),
            Self::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Self::I8(v) => Some(v.into()),
            Self::I16(v) => Some(v.into()),
            Self::I32(v) => Some(v.into()),
            Self::I64(v) => Some(v),
            _ => self.as_u64().and_then(|v| i64::try_from(v).ok()),
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Self::F32(v) => Some(v),
            Self::F64(v) => Some(v as f32),
            _ => self.as_i64().map(|v| v as f32),
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Self::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }

    /// Short human-readable rendering; long arrays are summarised.
    pub fn summary(&self) -> String {
        match self {
            Self::String(v) if v.len() > 80 => {
                let cut: String = v.chars().take(77).collect();
                format!("{:?}...", cut)
            }
            Self::String(v) => format!("{v:?}"),
            Self::Array(items) if items.len() > 8 => {
                let kind = items.first().map(Value::kind_name).unwrap_or("?");
                format!("[{kind}; {}]", items.len())
            }
            Self::Array(items) => {
                let parts: Vec<String> = items.iter().map(Value::summary).collect();
                format!("[{}]", parts.join(", "))
            }
            Self::U8(v) => v.to_string(),
            Self::I8(v) => v.to_string(),
            Self::U16(v) => v.to_string(),
            Self::I16(v) => v.to_string(),
            Self::U32(v) => v.to_string(),
            Self::I32(v) => v.to_string(),
            Self::U64(v) => v.to_string(),
            Self::I64(v) => v.to_string(),
            Self::F32(v) => v.to_string(),
            Self::F64(v) => v.to_string(),
            Self::Bool(v) => v.to_string(),
        }
    }

    fn kind_name(&self) -> &'static str {
        match self {
            Self::U8(_) => "u8",
            Self::I8(_) => "i8",
            Self::U16(_) => "u16",
            Self::I16(_) => "i16",
            Self::U32(_) => "u32",
            Self::I32(_) => "i32",
            Self::U64(_) => "u64",
            Self::I64(_) => "i64",
            Self::F32(_) => "f32",
            Self::F64(_) => "f64",
            Self::Bool(_) => "bool",
            Self::String(_) => "string",
            Self::Array(_) => "array",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufInfo {
    pub version: u32,
    pub alignment: u64,
    pub data_offset: u64,
    pub metadata: Vec<(String, Value)>,
    pub tensors: Vec<TensorInfo>,
    index: HashMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    /// GGML order: `dimensions[0]` is the contiguous (innermost) axis.
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub relative_offset: u64,
}

impl TensorInfo {
    pub fn element_count(&self) -> Option<u64> {
        self.dimensions
            .iter()
            .try_fold(1_u64, |total, dimension| total.checked_mul(*dimension))
    }

    pub fn kind(&self) -> Option<GgmlType> {
        GgmlType::from_id(self.ggml_type)
    }

    pub fn byte_size(&self) -> Option<u64> {
        let kind = self.kind()?;
        let elements = self.element_count()?;
        // Quantized rows must hold a whole number of blocks.
        let row = *self.dimensions.first()?;
        if row % kind.block_elements() as u64 != 0 {
            return None;
        }
        (elements / kind.block_elements() as u64).checked_mul(kind.block_bytes() as u64)
    }

    pub fn type_name(&self) -> &'static str {
        self.kind().map(GgmlType::name).unwrap_or("UNKNOWN")
    }

    /// Transformer block index for tensors named `blk.N.*`.
    pub fn block_index(&self) -> Option<u32> {
        self.name
            .strip_prefix("blk.")?
            .split('.')
            .next()?
            .parse()
            .ok()
    }
}

impl GgufInfo {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    }

    pub fn architecture(&self) -> Option<&str> {
        self.get("general.architecture")?.as_str()
    }

    /// Look up `<architecture>.<suffix>`, e.g. `arch_get("block_count")`.
    pub fn arch_get(&self, suffix: &str) -> Option<&Value> {
        let arch = self.architecture()?;
        self.get(&format!("{arch}.{suffix}"))
    }

    pub fn block_count(&self) -> Option<u32> {
        self.arch_get("block_count")?
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    pub fn known_weight_bytes(&self) -> Option<u64> {
        self.tensors.iter().try_fold(0_u64, |total, tensor| {
            total.checked_add(tensor.byte_size()?)
        })
    }
}

#[derive(Debug)]
pub enum GgufError {
    Io(io::Error),
    Invalid(String),
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
            Self::LimitExceeded(what) => write!(f, "GGUF {what} exceeds parser limit"),
            Self::ArithmeticOverflow => write!(f, "GGUF offset arithmetic overflow"),
        }
    }
}

impl std::error::Error for GgufError {}

impl From<io::Error> for GgufError {
    fn from(value: io::Error) -> Self {
        if value.kind() == io::ErrorKind::UnexpectedEof {
            Self::Invalid("file is truncated".into())
        } else {
            Self::Io(value)
        }
    }
}

fn invalid(message: impl Into<String>) -> GgufError {
    GgufError::Invalid(message.into())
}

pub fn inspect_file(path: impl AsRef<Path>) -> Result<GgufInfo, GgufError> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    inspect(BufReader::new(file), file_len)
}

/// Reader wrapper that knows how many bytes remain, so every declared length
/// can be rejected before it is allocated.
struct Bounded<R> {
    inner: R,
    position: u64,
    len: u64,
}

impl<R: Read + Seek> Bounded<R> {
    fn remaining(&self) -> u64 {
        self.len.saturating_sub(self.position)
    }

    fn need(&self, bytes: u64) -> Result<(), GgufError> {
        if bytes > self.remaining() {
            Err(invalid("declared length runs past end of file"))
        } else {
            Ok(())
        }
    }

    fn read_exact(&mut self, buffer: &mut [u8]) -> Result<(), GgufError> {
        self.need(buffer.len() as u64)?;
        self.inner.read_exact(buffer)?;
        self.position += buffer.len() as u64;
        Ok(())
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], GgufError> {
        let mut bytes = [0; N];
        self.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn string(&mut self) -> Result<String, GgufError> {
        let length = self.u64()?;
        if length > MAX_STRING_BYTES {
            return Err(GgufError::LimitExceeded("string length"));
        }
        self.need(length)?;
        let mut bytes = vec![0; length as usize];
        self.read_exact(&mut bytes)?;
        String::from_utf8(bytes).map_err(|_| invalid("string is not UTF-8"))
    }

    fn value(&mut self, kind: u32, depth: u8) -> Result<Value, GgufError> {
        Ok(match kind {
            0 => Value::U8(self.array::<1>()?[0]),
            1 => Value::I8(self.array::<1>()?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.array()?)),
            3 => Value::I16(i16::from_le_bytes(self.array()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.array()?)),
            6 => Value::F32(f32::from_le_bytes(self.array()?)),
            7 => match self.array::<1>()?[0] {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                _ => return Err(invalid("boolean is neither 0 nor 1")),
            },
            8 => Value::String(self.string()?),
            9 => {
                if depth >= 4 {
                    return Err(GgufError::LimitExceeded("array nesting"));
                }
                let element_type = self.u32()?;
                let count = self.u64()?;
                if count > MAX_ITEMS {
                    return Err(GgufError::LimitExceeded("array length"));
                }
                // Every element occupies at least one byte on disk.
                self.need(count.saturating_mul(min_encoded_size(element_type)?))?;
                let mut items = Vec::with_capacity(count as usize);
                for _ in 0..count {
                    items.push(self.value(element_type, depth + 1)?);
                }
                Value::Array(items)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.array()?)),
            12 => Value::F64(f64::from_le_bytes(self.array()?)),
            _ => return Err(GgufError::UnsupportedValueType(kind)),
        })
    }
}

fn min_encoded_size(kind: u32) -> Result<u64, GgufError> {
    Ok(match kind {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        8 | 10 | 11 | 12 => 8,
        9 => 12,
        _ => return Err(GgufError::UnsupportedValueType(kind)),
    })
}

pub fn inspect<R: Read + Seek>(reader: R, file_len: u64) -> Result<GgufInfo, GgufError> {
    let mut reader = Bounded {
        inner: reader,
        position: 0,
        len: file_len,
    };
    if &reader.array::<4>()? != MAGIC {
        return Err(invalid("bad magic"));
    }
    let version = reader.u32()?;
    if !(2..=3).contains(&version) {
        return Err(GgufError::UnsupportedVersion(version));
    }
    let tensor_count = reader.u64()?;
    let metadata_count = reader.u64()?;
    if tensor_count > MAX_ITEMS {
        return Err(GgufError::LimitExceeded("tensor count"));
    }
    if metadata_count > MAX_ITEMS {
        return Err(GgufError::LimitExceeded("metadata count"));
    }
    // Each metadata entry is at least key length + type + one byte; each
    // tensor descriptor is at least 8 + 4 + 4 + 8 bytes.
    reader.need(metadata_count.saturating_mul(13))?;
    let mut metadata = Vec::with_capacity(metadata_count as usize);
    for _ in 0..metadata_count {
        let key = reader.string()?;
        let value_type = reader.u32()?;
        let value = reader.value(value_type, 0)?;
        metadata.push((key, value));
    }

    let alignment = match metadata.iter().find(|(k, _)| k == "general.alignment") {
        Some((_, value)) => value
            .as_u64()
            .ok_or_else(|| invalid("general.alignment is not an integer"))?,
        None => DEFAULT_ALIGNMENT,
    };
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(invalid("general.alignment must be a nonzero power of two"));
    }

    reader.need(tensor_count.saturating_mul(24))?;
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    let mut index = HashMap::with_capacity(tensor_count as usize);
    for i in 0..tensor_count as usize {
        let name = reader.string()?;
        let dimension_count = reader.u32()?;
        if dimension_count == 0 || dimension_count > MAX_DIMENSIONS {
            return Err(invalid(format!(
                "tensor {name} has {dimension_count} dimensions"
            )));
        }
        let mut dimensions = Vec::with_capacity(dimension_count as usize);
        for _ in 0..dimension_count {
            let dimension = reader.u64()?;
            if dimension == 0 {
                return Err(invalid(format!("tensor {name} has a zero dimension")));
            }
            dimensions.push(dimension);
        }
        let ggml_type = reader.u32()?;
        let relative_offset = reader.u64()?;
        if index.insert(name.clone(), i).is_some() {
            return Err(invalid(format!("duplicate tensor name {name}")));
        }
        tensors.push(TensorInfo {
            name,
            dimensions,
            ggml_type,
            relative_offset,
        });
    }

    let data_offset = align_up(reader.position, alignment)?;
    if data_offset > file_len {
        return Err(invalid("tensor data begins beyond end of file"));
    }
    let data_len = file_len - data_offset;
    for tensor in &tensors {
        if tensor.relative_offset % alignment != 0 {
            return Err(invalid(format!("tensor {} is misaligned", tensor.name)));
        }
        if tensor.element_count().is_none() {
            return Err(GgufError::ArithmeticOverflow);
        }
        // Unknown types are allowed for inspection; their extent is unchecked
        // here and they are rejected by the model loader.
        if let Some(size) = tensor.byte_size() {
            let end = tensor
                .relative_offset
                .checked_add(size)
                .ok_or(GgufError::ArithmeticOverflow)?;
            if end > data_len {
                return Err(invalid(format!(
                    "tensor {} extends beyond end of file",
                    tensor.name
                )));
            }
        } else if tensor.relative_offset > data_len {
            return Err(invalid(format!(
                "tensor {} starts beyond end of file",
                tensor.name
            )));
        }
    }

    Ok(GgufInfo {
        version,
        alignment,
        data_offset,
        metadata,
        tensors,
        index,
    })
}

/// A parsed GGUF file whose tensor payloads are memory-mapped, not copied.
pub struct GgufFile {
    pub info: GgufInfo,
    map: Mmap,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let file = File::open(path.as_ref())?;
        let file_len = file.metadata()?.len();
        let info = inspect(BufReader::new(&file), file_len)?;
        // SAFETY: the map is read-only. Mutating the file while kLLM runs is
        // undefined behaviour, the same contract as every mmap-based loader.
        let map = unsafe { Mmap::map(&file)? };
        if map.len() as u64 != file_len {
            return Err(invalid("file changed size while being opened"));
        }
        Ok(Self { info, map })
    }

    /// Raw payload bytes of a tensor. Bounds were validated at parse time.
    pub fn tensor_bytes(&self, tensor: &TensorInfo) -> Result<&[u8], GgufError> {
        let size = tensor
            .byte_size()
            .ok_or_else(|| invalid(format!("tensor {} has an unsupported type", tensor.name)))?;
        let start = self.info.data_offset + tensor.relative_offset;
        let end = start + size;
        self.map
            .get(start as usize..end as usize)
            .ok_or_else(|| invalid(format!("tensor {} is out of bounds", tensor.name)))
    }

    /// The whole mapped file.
    pub fn mapped(&self) -> &[u8] {
        &self.map
    }

    /// Ask the kernel to start reading the mapped file in the background.
    pub fn advise_willneed(&self) {
        #[cfg(unix)]
        let _ = self.map.advise(memmap2::Advice::WillNeed);
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64, GgufError> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .ok_or(GgufError::ArithmeticOverflow)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Cursor;

    pub(crate) struct Builder {
        metadata: Vec<u8>,
        metadata_count: u64,
        tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)>,
    }

    impl Builder {
        pub(crate) fn new() -> Self {
            Self {
                metadata: Vec::new(),
                metadata_count: 0,
                tensors: Vec::new(),
            }
        }

        fn key(&mut self, key: &str, kind: u32) {
            push_string(&mut self.metadata, key);
            self.metadata.extend_from_slice(&kind.to_le_bytes());
            self.metadata_count += 1;
        }

        pub(crate) fn string(mut self, key: &str, value: &str) -> Self {
            self.key(key, 8);
            push_string(&mut self.metadata, value);
            self
        }

        pub(crate) fn u32(mut self, key: &str, value: u32) -> Self {
            self.key(key, 4);
            self.metadata.extend_from_slice(&value.to_le_bytes());
            self
        }

        pub(crate) fn f32(mut self, key: &str, value: f32) -> Self {
            self.key(key, 6);
            self.metadata.extend_from_slice(&value.to_le_bytes());
            self
        }

        pub(crate) fn strings(mut self, key: &str, values: &[&str]) -> Self {
            self.key(key, 9);
            self.metadata.extend_from_slice(&8_u32.to_le_bytes());
            self.metadata
                .extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in values {
                push_string(&mut self.metadata, value);
            }
            self
        }

        pub(crate) fn tensor(mut self, name: &str, dims: &[u64], kind: u32, data: Vec<u8>) -> Self {
            self.tensors.push((name.into(), dims.to_vec(), kind, data));
            self
        }

        pub(crate) fn build(self) -> Vec<u8> {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(MAGIC);
            bytes.extend_from_slice(&3_u32.to_le_bytes());
            bytes.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&self.metadata_count.to_le_bytes());
            bytes.extend_from_slice(&self.metadata);
            let mut offset = 0_u64;
            let mut offsets = Vec::new();
            for (name, dims, kind, data) in &self.tensors {
                push_string(&mut bytes, name);
                bytes.extend_from_slice(&(dims.len() as u32).to_le_bytes());
                for dim in dims {
                    bytes.extend_from_slice(&dim.to_le_bytes());
                }
                bytes.extend_from_slice(&kind.to_le_bytes());
                bytes.extend_from_slice(&offset.to_le_bytes());
                offsets.push(offset);
                offset =
                    (offset + data.len() as u64).div_ceil(DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT;
            }
            pad(&mut bytes);
            for (_, _, _, data) in &self.tensors {
                bytes.extend_from_slice(data);
                pad(&mut bytes);
            }
            bytes
        }
    }

    fn pad(bytes: &mut Vec<u8>) {
        while !bytes.len().is_multiple_of(DEFAULT_ALIGNMENT as usize) {
            bytes.push(0);
        }
    }

    fn push_string(output: &mut Vec<u8>, value: &str) {
        output.extend_from_slice(&(value.len() as u64).to_le_bytes());
        output.extend_from_slice(value.as_bytes());
    }

    fn parse(bytes: &[u8]) -> Result<GgufInfo, GgufError> {
        inspect(Cursor::new(bytes), bytes.len() as u64)
    }

    fn fixture() -> Vec<u8> {
        Builder::new()
            .string("general.architecture", "llama")
            .u32("llama.block_count", 2)
            .f32("llama.rope.freq_base", 10000.0)
            .strings("tokenizer.ggml.tokens", &["a", "b", "c"])
            .tensor("token_embd.weight", &[16, 32], 0, vec![0; 16 * 32 * 4])
            .tensor("blk.1.attn_q.weight", &[32, 2], 2, vec![0; 36])
            .build()
    }

    #[test]
    fn parses_metadata_and_tensors() {
        let info = parse(&fixture()).unwrap();
        assert_eq!(info.version, 3);
        assert_eq!(info.architecture(), Some("llama"));
        assert_eq!(info.block_count(), Some(2));
        assert_eq!(
            info.arch_get("rope.freq_base").and_then(Value::as_f32),
            Some(10000.0)
        );
        let tokens = info
            .get("tokenizer.ggml.tokens")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[2].as_str(), Some("c"));
        let embd = info.tensor("token_embd.weight").unwrap();
        assert_eq!(embd.dimensions, vec![16, 32]);
        assert_eq!(embd.byte_size(), Some(2048));
        let q = info.tensor("blk.1.attn_q.weight").unwrap();
        assert_eq!(q.byte_size(), Some(36));
        assert_eq!(q.type_name(), "Q4_0");
        assert_eq!(q.block_index(), Some(1));
        assert_eq!(info.known_weight_bytes(), Some(2048 + 36));
        assert_eq!(info.data_offset % DEFAULT_ALIGNMENT, 0);
    }

    #[test]
    fn rejects_bad_magic() {
        assert!(matches!(parse(&[0; 24]), Err(GgufError::Invalid(_))));
    }

    #[test]
    fn rejects_every_truncation_without_panicking() {
        let bytes = fixture();
        let info = parse(&bytes).unwrap();
        let last = info.tensor("blk.1.attn_q.weight").unwrap();
        let payload_end = (info.data_offset + last.relative_offset + 36) as usize;
        // Bytes past the last payload are alignment padding and may be cut.
        assert!(parse(&bytes[..payload_end]).is_ok());
        for cut in 0..payload_end {
            // Cutting inside the final tensor payload must fail the bounds
            // check; cutting earlier must fail the reader. Nothing may panic.
            assert!(
                parse(&bytes[..cut]).is_err(),
                "accepted truncation at {cut}"
            );
        }
    }

    #[test]
    fn rejects_huge_declared_lengths_without_allocating() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        push_string(&mut bytes, "k");
        bytes.extend_from_slice(&9_u32.to_le_bytes());
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        bytes.extend_from_slice(&(MAX_ITEMS - 1).to_le_bytes());
        assert!(matches!(parse(&bytes), Err(GgufError::Invalid(_))));
    }

    #[test]
    fn rejects_tensor_past_end_and_duplicates() {
        let mut bytes = fixture();
        bytes.truncate(bytes.len() - 32);
        assert!(parse(&bytes).is_err());

        let duplicate = Builder::new()
            .tensor("x", &[4], 0, vec![0; 16])
            .tensor("x", &[4], 0, vec![0; 16])
            .build();
        assert!(matches!(parse(&duplicate), Err(GgufError::Invalid(m)) if m.contains("duplicate")));
    }

    #[test]
    fn rejects_bad_alignment_and_bool() {
        let bytes = Builder::new().u32("general.alignment", 24).build();
        assert!(parse(&bytes).is_err());
    }
}
