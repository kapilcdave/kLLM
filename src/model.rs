//! Decoder-only transformer (Llama family) over memory-mapped GGUF weights.
//!
//! Supported architectures: `llama` (also Mistral, SmolLM, TinyLlama and
//! Llama 3 conversions), `qwen2` (q/k/v biases), and `qwen3` (per-head q/k
//! RMSNorm). All three share RMSNorm, rotary position embedding, grouped-
//! query attention and a SwiGLU feed-forward block.

use std::{
    fmt,
    path::Path,
    time::{Duration, Instant},
};

use crate::backend::{Backend, BackendError, Job, Matrix};
use crate::cpu::{CpuBackend, rms_norm_in_place};
use crate::gguf::{GgufError, GgufFile, GgufInfo, TensorInfo, Value};
use crate::kernels;
use crate::quant::{GgmlType, dequantize_row};

#[derive(Debug)]
pub enum ModelError {
    Gguf(GgufError),
    Backend(BackendError),
    Unsupported(String),
    Missing(String),
    Shape(String),
    ContextFull { context: usize },
    Gpu(String),
    BadToken(u32),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gguf(e) => write!(f, "{e}"),
            Self::Backend(e) => write!(f, "{e}"),
            Self::Unsupported(m) => write!(f, "unsupported model: {m}"),
            Self::Missing(m) => write!(f, "model is missing {m}"),
            Self::Shape(m) => write!(f, "bad tensor shape: {m}"),
            Self::ContextFull { context } => {
                write!(f, "context window of {context} tokens is full")
            }
            Self::Gpu(m) => write!(f, "GPU: {m}"),
            Self::BadToken(t) => write!(f, "token id {t} is outside the vocabulary"),
        }
    }
}

impl std::error::Error for ModelError {}

impl From<GgufError> for ModelError {
    fn from(e: GgufError) -> Self {
        Self::Gguf(e)
    }
}

impl From<BackendError> for ModelError {
    fn from(e: BackendError) -> Self {
        Self::Backend(e)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    Llama,
    Qwen2,
    Qwen3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeStyle {
    /// Rotate adjacent pairs (x[2i], x[2i+1]).
    Norm,
    /// Rotate halves (x[i], x[i + d/2]).
    Neox,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub arch: Arch,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    pub rms_eps: f32,
    pub rope_base: f32,
    pub rope_dim: usize,
    pub rope_scale: f32,
    pub rope_style: RopeStyle,
    pub train_context: usize,
}

impl Config {
    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }

    /// KV-cache bytes per token of context with f16 storage.
    pub fn kv_bytes_per_token(&self) -> u64 {
        2 * self.n_layer as u64 * self.kv_dim() as u64 * 2
    }

    pub fn from_gguf(info: &GgufInfo) -> Result<Self, ModelError> {
        let arch_name = info
            .architecture()
            .ok_or_else(|| ModelError::Missing("general.architecture".into()))?;
        let arch = match arch_name {
            "llama" | "mistral" => Arch::Llama,
            "qwen2" => Arch::Qwen2,
            "qwen3" => Arch::Qwen3,
            other => {
                return Err(ModelError::Unsupported(format!(
                    "architecture {other:?} (supported: llama, qwen2, qwen3)"
                )));
            }
        };
        let int = |key: &str| -> Result<usize, ModelError> {
            info.arch_get(key)
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| ModelError::Missing(format!("{arch_name}.{key}")))
        };
        let opt_int = |key: &str| {
            info.arch_get(key)
                .and_then(Value::as_u64)
                .map(|v| v as usize)
        };
        let float = |key: &str| info.arch_get(key).and_then(Value::as_f32);

        let n_embd = int("embedding_length")?;
        let n_head = int("attention.head_count")?;
        let n_head_kv = opt_int("attention.head_count_kv").unwrap_or(n_head);
        if n_head == 0 || n_head_kv == 0 || n_head % n_head_kv != 0 {
            return Err(ModelError::Shape(format!(
                "{n_head} query heads cannot share {n_head_kv} KV heads"
            )));
        }
        let head_dim = opt_int("attention.key_length").unwrap_or(n_embd / n_head);
        if opt_int("attention.value_length").is_some_and(|v| v != head_dim) {
            return Err(ModelError::Unsupported(
                "key and value head sizes differ".into(),
            ));
        }
        let rope_dim = opt_int("rope.dimension_count").unwrap_or(head_dim);
        if rope_dim > head_dim || rope_dim % 2 != 0 {
            return Err(ModelError::Shape(format!("rope dimension {rope_dim}")));
        }
        let rope_scale = match info.arch_get("rope.scaling.type").and_then(Value::as_str) {
            None | Some("none") => 1.0,
            Some("linear") => 1.0 / float("rope.scaling.factor").unwrap_or(1.0),
            Some(other) => {
                return Err(ModelError::Unsupported(format!("rope scaling {other:?}")));
            }
        };
        let n_vocab = match opt_int("vocab_size") {
            Some(v) => v,
            None => info
                .tensor("token_embd.weight")
                .and_then(|t| t.dimensions.get(1).copied())
                .ok_or_else(|| ModelError::Missing("token_embd.weight".into()))?
                as usize,
        };
        let config = Self {
            arch,
            n_embd,
            n_layer: int("block_count")?,
            n_head,
            n_head_kv,
            head_dim,
            n_ff: int("feed_forward_length")?,
            n_vocab,
            rms_eps: float("attention.layer_norm_rms_epsilon").unwrap_or(1e-5),
            rope_base: float("rope.freq_base").unwrap_or(10_000.0),
            rope_dim,
            rope_scale,
            rope_style: match arch {
                Arch::Llama => RopeStyle::Norm,
                Arch::Qwen2 | Arch::Qwen3 => RopeStyle::Neox,
            },
            train_context: opt_int("context_length").unwrap_or(4096),
        };
        if config.n_layer == 0 || config.n_embd == 0 || config.n_ff == 0 || config.head_dim == 0 {
            return Err(ModelError::Shape("zero-sized hyperparameter".into()));
        }
        Ok(config)
    }
}

/// A weight matrix located in the mapped file.
#[derive(Debug, Clone, Copy)]
struct Weight {
    kind: GgmlType,
    rows: usize,
    cols: usize,
    offset: usize,
    len: usize,
    /// Index into `Model::packed` of an x8-repacked copy, if one was made.
    packed: Option<usize>,
}

struct Layer {
    attn_norm: Vec<f32>,
    wq: Weight,
    wk: Weight,
    wv: Weight,
    wo: Weight,
    bq: Option<Vec<f32>>,
    bk: Option<Vec<f32>>,
    bv: Option<Vec<f32>>,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    ffn_norm: Vec<f32>,
    w_gate: Weight,
    w_up: Weight,
    w_down: Weight,
}

#[cfg(feature = "cuda")]
mod gpu;
#[cfg(feature = "cuda")]
pub use gpu::Offload;

pub struct Model {
    file: GgufFile,
    pub config: Config,
    token_embd: Weight,
    output_norm: Vec<f32>,
    output: Weight,
    layers: Vec<Layer>,
    rope_freq_factors: Option<Vec<f32>>,
    /// x8-repacked weight copies (see `kernels::repack_x8`).
    packed: Vec<Vec<u8>>,
    backend: CpuBackend,
    /// Blocks (and head) offloaded to a GPU, if any.
    #[cfg(feature = "cuda")]
    gpu: Option<gpu::GpuStack>,
}

impl Model {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ModelError> {
        let file = GgufFile::open(path)?;
        let config = Config::from_gguf(&file.info)?;
        let c = &config;
        let loader = Loader {
            file: &file,
            packed: std::cell::RefCell::new(Vec::new()),
            repack: std::env::var_os("KLLM_NO_REPACK").is_none(),
        };
        let token_embd = loader.matrix("token_embd.weight", c.n_vocab, c.n_embd)?;
        let output = match file.info.tensor("output.weight") {
            Some(_) => loader.matrix("output.weight", c.n_vocab, c.n_embd)?,
            None => token_embd, // tied embeddings
        };
        let output_norm = loader.vector("output_norm.weight", c.n_embd)?;
        let rope_freq_factors = loader.optional_vector("rope_freqs.weight", c.rope_dim / 2)?;
        let mut layers = Vec::with_capacity(c.n_layer);
        for i in 0..c.n_layer {
            let name = |suffix: &str| format!("blk.{i}.{suffix}");
            let qk_norm = c.arch == Arch::Qwen3;
            layers.push(Layer {
                attn_norm: loader.vector(&name("attn_norm.weight"), c.n_embd)?,
                wq: loader.matrix(&name("attn_q.weight"), c.q_dim(), c.n_embd)?,
                wk: loader.matrix(&name("attn_k.weight"), c.kv_dim(), c.n_embd)?,
                wv: loader.matrix(&name("attn_v.weight"), c.kv_dim(), c.n_embd)?,
                wo: loader.matrix(&name("attn_output.weight"), c.n_embd, c.q_dim())?,
                bq: loader.optional_vector(&name("attn_q.bias"), c.q_dim())?,
                bk: loader.optional_vector(&name("attn_k.bias"), c.kv_dim())?,
                bv: loader.optional_vector(&name("attn_v.bias"), c.kv_dim())?,
                q_norm: if qk_norm {
                    Some(loader.vector(&name("attn_q_norm.weight"), c.head_dim)?)
                } else {
                    None
                },
                k_norm: if qk_norm {
                    Some(loader.vector(&name("attn_k_norm.weight"), c.head_dim)?)
                } else {
                    None
                },
                ffn_norm: loader.vector(&name("ffn_norm.weight"), c.n_embd)?,
                w_gate: loader.matrix(&name("ffn_gate.weight"), c.n_ff, c.n_embd)?,
                w_up: loader.matrix(&name("ffn_up.weight"), c.n_ff, c.n_embd)?,
                w_down: loader.matrix(&name("ffn_down.weight"), c.n_embd, c.n_ff)?,
            });
        }
        file.advise_willneed();
        let packed = loader.packed.into_inner();
        Ok(Self {
            file,
            config,
            token_embd,
            output_norm,
            output,
            layers,
            rope_freq_factors,
            packed,
            backend: CpuBackend,
            #[cfg(feature = "cuda")]
            gpu: None,
        })
    }

    pub fn info(&self) -> &GgufInfo {
        &self.file.info
    }

    fn matrix(&self, w: Weight) -> Matrix<'_> {
        Matrix {
            kind: w.kind,
            rows: w.rows,
            cols: w.cols,
            data: &self.file.mapped()[w.offset..w.offset + w.len],
            packed: w.packed.map(|i| self.packed[i].as_slice()),
        }
    }

    /// Bytes of mapped weights the forward pass reads per token.
    pub fn weight_bytes(&self) -> u64 {
        let mut total = self.output.len as u64;
        for l in &self.layers {
            for w in [l.wq, l.wk, l.wv, l.wo, l.w_gate, l.w_up, l.w_down] {
                total += w.len as u64;
            }
        }
        total
    }

    /// Weight bytes of each transformer block.
    pub fn layer_bytes(&self) -> Vec<u64> {
        self.layers
            .iter()
            .map(|l| {
                [l.wq, l.wk, l.wv, l.wo, l.w_gate, l.w_up, l.w_down]
                    .iter()
                    .map(|w| w.len as u64)
                    .sum::<u64>()
                    + 4 * (l.attn_norm.len() + l.ffn_norm.len()) as u64
            })
            .collect()
    }

    /// Bytes of heap memory held by x8-repacked weight copies.
    pub fn packed_bytes(&self) -> u64 {
        self.packed.iter().map(|p| p.len() as u64).sum()
    }

    /// Distinct weight types used by the matrices, for reporting.
    pub fn weight_types(&self) -> Vec<&'static str> {
        let mut kinds: Vec<&'static str> = std::iter::once(self.output)
            .chain(std::iter::once(self.token_embd))
            .chain(
                self.layers
                    .iter()
                    .flat_map(|l| [l.wq, l.wk, l.wv, l.wo, l.w_gate, l.w_up, l.w_down]),
            )
            .map(|w| w.kind.name())
            .collect();
        kinds.sort_unstable();
        kinds.dedup();
        kinds
    }

    pub fn new_session(&self, context: usize, max_batch: usize) -> Result<Session<'_>, ModelError> {
        Session::new(self, context, max_batch)
    }

    /// Blocks that run on the CPU (all of them unless some were offloaded).
    pub fn cpu_layers(&self) -> usize {
        #[cfg(feature = "cuda")]
        if let Some(stack) = &self.gpu {
            return stack.first;
        }
        self.config.n_layer
    }
}

struct Loader<'f> {
    file: &'f GgufFile,
    packed: std::cell::RefCell<Vec<Vec<u8>>>,
    repack: bool,
}

impl Loader<'_> {
    fn tensor(&self, name: &str) -> Result<&TensorInfo, ModelError> {
        self.file
            .info
            .tensor(name)
            .ok_or_else(|| ModelError::Missing(format!("tensor {name}")))
    }

    fn matrix(&self, name: &str, rows: usize, cols: usize) -> Result<Weight, ModelError> {
        let t = self.tensor(name)?;
        let kind = t
            .kind()
            .filter(|k| k.is_supported())
            .ok_or_else(|| ModelError::Unsupported(format!("{name} has type {}", t.type_name())))?;
        if t.dimensions != [cols as u64, rows as u64] {
            return Err(ModelError::Shape(format!(
                "{name} is {:?}, expected [{cols}, {rows}]",
                t.dimensions
            )));
        }
        let bytes = self.file.tensor_bytes(t)?;
        let offset = bytes.as_ptr() as usize - self.file.mapped().as_ptr() as usize;
        let packed = if self.repack {
            kernels::repack_x8(kind, bytes, rows, cols).map(|copy| {
                let mut packed = self.packed.borrow_mut();
                packed.push(copy);
                packed.len() - 1
            })
        } else {
            None
        };
        Ok(Weight {
            kind,
            rows,
            cols,
            offset,
            len: bytes.len(),
            packed,
        })
    }

    fn vector(&self, name: &str, len: usize) -> Result<Vec<f32>, ModelError> {
        let t = self.tensor(name)?;
        let kind = t
            .kind()
            .filter(|k| k.is_supported())
            .ok_or_else(|| ModelError::Unsupported(format!("{name} has type {}", t.type_name())))?;
        if t.element_count() != Some(len as u64) {
            return Err(ModelError::Shape(format!(
                "{name} is {:?}, expected {len} elements",
                t.dimensions
            )));
        }
        let mut out = vec![0.0; len];
        dequantize_row(kind, self.file.tensor_bytes(t)?, &mut out);
        Ok(out)
    }

    fn optional_vector(&self, name: &str, len: usize) -> Result<Option<Vec<f32>>, ModelError> {
        match self.file.info.tensor(name) {
            Some(_) => self.vector(name, len).map(Some),
            None => Ok(None),
        }
    }
}

/// Phases timed by the optional profiler.
pub const PHASES: [&str; 8] = [
    "norms+misc",
    "qkv proj",
    "rope+kv store",
    "attention",
    "out proj",
    "ffn gate/up",
    "ffn down",
    "logits",
];

/// Wall-clock time per forward-pass phase (enable with
/// [`Session::set_profiling`]). Timing adds a few clock reads per layer.
#[derive(Debug, Clone)]
pub struct Profiler {
    enabled: bool,
    last: Instant,
    pub totals: [Duration; 8],
}

impl Profiler {
    #[inline]
    fn start(&mut self) {
        if self.enabled {
            self.last = Instant::now();
        }
    }

    #[inline]
    fn lap(&mut self, phase: usize) {
        if self.enabled {
            let now = Instant::now();
            self.totals[phase] += now - self.last;
            self.last = now;
        }
    }
}

/// Which logits a forward pass should produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Logits {
    /// Only for the final token (generation).
    Last,
    /// For every token in the batch (perplexity).
    All,
}

/// Per-conversation state: KV cache and preallocated scratch buffers.
pub struct Session<'m> {
    model: &'m Model,
    context: usize,
    max_batch: usize,
    pos: usize,
    /// f16 bits, laid out [layer][position][kv_dim].
    k_cache: Vec<u16>,
    v_cache: Vec<u16>,
    x: Vec<f32>,
    xn: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    proj: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    scores: Vec<f32>,
    rope: Vec<(f32, f32)>,
    logits: Vec<f32>,
    all_logits: Vec<f32>,
    pub profiler: Profiler,
    #[cfg(feature = "cuda")]
    gpu: Option<gpu::GpuState>,
}

impl<'m> Session<'m> {
    fn new(model: &'m Model, context: usize, max_batch: usize) -> Result<Self, ModelError> {
        let c = &model.config;
        let context = context.max(1);
        let b = max_batch.clamp(1, context);
        // Offloaded blocks keep their KV cache in VRAM.
        let cache = model.cpu_layers() * context * c.kv_dim();
        #[cfg(feature = "cuda")]
        let gpu = model
            .gpu
            .as_ref()
            .map(|stack| gpu::GpuState::new(stack, c, context, b))
            .transpose()?;
        Ok(Self {
            model,
            context,
            max_batch: b,
            pos: 0,
            k_cache: vec![0; cache],
            v_cache: vec![0; cache],
            x: vec![0.0; b * c.n_embd],
            xn: vec![0.0; b * c.n_embd],
            q: vec![0.0; b * c.q_dim()],
            k: vec![0.0; b * c.kv_dim()],
            v: vec![0.0; b * c.kv_dim()],
            attn: vec![0.0; b * c.q_dim()],
            proj: vec![0.0; b * c.n_embd],
            gate: vec![0.0; b * c.n_ff],
            up: vec![0.0; b * c.n_ff],
            scores: vec![0.0; b * c.n_head * context],
            rope: vec![(0.0, 0.0); b * c.rope_dim / 2],
            logits: vec![0.0; b * c.n_vocab],
            all_logits: Vec::new(),
            profiler: Profiler {
                enabled: false,
                last: Instant::now(),
                totals: [Duration::ZERO; 8],
            },
            #[cfg(feature = "cuda")]
            gpu,
        })
    }

    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiler.enabled = enabled;
        self.profiler.totals = [Duration::ZERO; 8];
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn context(&self) -> usize {
        self.context
    }

    /// Bytes held by the KV cache and scratch buffers.
    pub fn allocated_bytes(&self) -> u64 {
        let f32s = self.x.len()
            + self.xn.len()
            + self.q.len()
            + self.k.len()
            + self.v.len()
            + self.attn.len()
            + self.proj.len()
            + self.gate.len()
            + self.up.len()
            + self.scores.len()
            + self.logits.len()
            + self.rope.len() * 2;
        (self.k_cache.len() + self.v_cache.len()) as u64 * 2 + f32s as u64 * 4
    }

    /// What [`Model::new_session`] will allocate, computed without
    /// allocating: `(kv_cache_bytes, scratch_bytes)`.
    pub fn estimate(config: &Config, context: usize, max_batch: usize) -> (u64, u64) {
        let c = config;
        let context = context.max(1) as u64;
        let b = max_batch.clamp(1, context as usize) as u64;
        let kv = 2 * 2 * c.n_layer as u64 * context * c.kv_dim() as u64;
        let f32s = b
            * (3 * c.n_embd as u64
                + 2 * c.q_dim() as u64
                + 2 * c.kv_dim() as u64
                + 2 * c.n_ff as u64
                + c.n_head as u64 * context
                + c.rope_dim as u64
                + c.n_vocab as u64);
        (kv, f32s * 4)
    }

    /// Discard positions `>= pos` so generation can resume from `pos`.
    pub fn truncate(&mut self, pos: usize) {
        self.pos = self.pos.min(pos);
    }

    /// Feed `tokens`, returning logits for the last one (or for every token
    /// with [`Logits::All`], as `tokens.len() * n_vocab` values).
    pub fn forward(&mut self, tokens: &[u32], want: Logits) -> Result<&[f32], ModelError> {
        let n_vocab = self.model.config.n_vocab;
        if tokens.is_empty() {
            return Err(ModelError::Shape("empty token batch".into()));
        }
        if let Some(&bad) = tokens.iter().find(|&&t| t as usize >= n_vocab) {
            return Err(ModelError::BadToken(bad));
        }
        if self.pos + tokens.len() > self.context {
            return Err(ModelError::ContextFull {
                context: self.context,
            });
        }
        let chunk_count = tokens.len().div_ceil(self.max_batch);
        if want == Logits::All {
            self.all_logits.clear();
            self.all_logits.reserve(tokens.len() * n_vocab);
        }
        for (i, chunk) in tokens.chunks(self.max_batch).enumerate() {
            let rows = match want {
                Logits::All => Rows::All,
                Logits::Last if i + 1 == chunk_count => Rows::Last,
                Logits::Last => Rows::None,
            };
            self.forward_batch(chunk, rows)?;
            if want == Logits::All {
                self.all_logits
                    .extend_from_slice(&self.logits[..chunk.len() * n_vocab]);
            }
        }
        Ok(match want {
            Logits::All => &self.all_logits,
            Logits::Last => &self.logits[..n_vocab],
        })
    }

    fn forward_batch(&mut self, tokens: &[u32], rows: Rows) -> Result<(), ModelError> {
        let model = self.model;
        let c = &model.config;
        let be = &model.backend;
        let n = tokens.len();
        let (d, qd, kvd, hd) = (c.n_embd, c.q_dim(), c.kv_dim(), c.head_dim);
        let start = self.pos;
        self.profiler.start();

        // Embedding lookup.
        let embd = model.matrix(model.token_embd);
        for (i, &t) in tokens.iter().enumerate() {
            dequantize_row(
                embd.kind,
                embd.row(t as usize),
                &mut self.x[i * d..(i + 1) * d],
            );
        }

        // Rotary tables for each position in the batch (ggml's recurrence).
        let half_rope = c.rope_dim / 2;
        let theta_scale = c.rope_base.powf(-2.0 / c.rope_dim as f32);
        for i in 0..n {
            let mut theta = (start + i) as f32 * c.rope_scale;
            for j in 0..half_rope {
                let factor = model.rope_freq_factors.as_ref().map_or(1.0, |f| f[j]);
                let t = theta / factor;
                self.rope[i * half_rope + j] = (t.cos(), t.sin());
                theta *= theta_scale;
            }
        }

        let scale = 1.0 / (hd as f32).sqrt();
        let group = c.n_head / c.n_head_kv;
        let layer_stride = self.context * kvd;
        let query_block = query_block(c, n, crate::pool::global().threads());

        for (li, layer) in model.layers.iter().enumerate().take(model.cpu_layers()) {
            // --- attention ---
            for i in 0..n {
                let (x, xn) = (
                    &self.x[i * d..(i + 1) * d],
                    &mut self.xn[i * d..(i + 1) * d],
                );
                xn.copy_from_slice(x);
                rms_norm_in_place(xn, &layer.attn_norm, c.rms_eps);
            }
            self.profiler.lap(0);
            be.matmul_many(
                &mut [
                    Job {
                        matrix: model.matrix(layer.wq),
                        output: &mut self.q[..n * qd],
                    },
                    Job {
                        matrix: model.matrix(layer.wk),
                        output: &mut self.k[..n * kvd],
                    },
                    Job {
                        matrix: model.matrix(layer.wv),
                        output: &mut self.v[..n * kvd],
                    },
                ],
                &self.xn[..n * d],
                n,
            )?;
            self.profiler.lap(1);
            for i in 0..n {
                let q = &mut self.q[i * qd..(i + 1) * qd];
                let k = &mut self.k[i * kvd..(i + 1) * kvd];
                let v = &mut self.v[i * kvd..(i + 1) * kvd];
                add_bias(q, layer.bq.as_deref());
                add_bias(k, layer.bk.as_deref());
                add_bias(v, layer.bv.as_deref());
                if let Some(w) = &layer.q_norm {
                    q.chunks_exact_mut(hd)
                        .for_each(|h| rms_norm_in_place(h, w, c.rms_eps));
                }
                if let Some(w) = &layer.k_norm {
                    k.chunks_exact_mut(hd)
                        .for_each(|h| rms_norm_in_place(h, w, c.rms_eps));
                }
                let table = &self.rope[i * half_rope..(i + 1) * half_rope];
                q.chunks_exact_mut(hd)
                    .for_each(|h| rope(h, table, c.rope_style));
                k.chunks_exact_mut(hd)
                    .for_each(|h| rope(h, table, c.rope_style));
                let at = li * layer_stride + (start + i) * kvd;
                kernels::store_f16(k, &mut self.k_cache[at..at + kvd]);
                kernels::store_f16(v, &mut self.v_cache[at..at + kvd]);
            }

            self.profiler.lap(2);
            // Causal attention: token i sees positions 0..=start+i.
            //
            // Blocked over query tokens: one task owns a block of query
            // tokens and one KV head, so each cache row is read once and
            // reused by every query in the block and every query head in the
            // group. A row-per-(token, head) loop instead re-reads the whole
            // cache n_head times per token, which is the dominant cost of
            // prefill attention. The arithmetic per (token, head) is
            // unchanged, so results are bit-identical either way.
            let k_layer = &self.k_cache[li * layer_stride..(li + 1) * layer_stride];
            let v_layer = &self.v_cache[li * layer_stride..(li + 1) * layer_stride];
            let q_all = &self.q[..n * qd];
            let context = self.context;
            let attn = OutPtr(self.attn.as_mut_ptr());
            let scores_base = OutPtr(self.scores.as_mut_ptr());
            let blocks = n.div_ceil(query_block);
            crate::pool::global().for_each(blocks * c.n_head_kv, |task| {
                let (block, kv_head) = (task / c.n_head_kv, task % c.n_head_kv);
                let i0 = block * query_block;
                let i1 = (i0 + query_block).min(n);
                let kv_off = kv_head * hd;
                let heads = kv_head * group..(kv_head + 1) * group;
                // SAFETY: this task owns attn[(i * n_head + h) * hd ..] and
                // scores[(i * n_head + h) * context ..] for its own query
                // tokens i and query heads h, which no other task touches.
                let out = |i: usize, h: usize| unsafe {
                    std::slice::from_raw_parts_mut(attn.at(i * qd + h * hd), hd)
                };
                let row = |i: usize, h: usize| unsafe {
                    std::slice::from_raw_parts_mut(
                        scores_base.at((i * c.n_head + h) * context),
                        context,
                    )
                };
                // Scores: read each key once and dot it against the queries
                // that attend to it, a tile at a time so one widened key
                // serves several queries and each has its own accumulators.
                const Q_TILE: usize = 4;
                for t in 0..start + i1 {
                    let key = &k_layer[t * kvd + kv_off..t * kvd + kv_off + hd];
                    // Query token i attends to position t only once
                    // t <= start + i, so the earliest query that sees t is
                    // t - start.
                    let first = i0.max(t.saturating_sub(start));
                    let pairs = (i1 - first) * group;
                    // Pair p is query token first + p / group, query head
                    // kv_head * group + p % group.
                    let pair = |p: usize| (first + p / group, kv_head * group + p % group);
                    let query = |p: usize| {
                        let (i, h) = pair(p);
                        &q_all[i * qd + h * hd..i * qd + (h + 1) * hd]
                    };
                    let mut p = 0;
                    while p + Q_TILE <= pairs {
                        let got = kernels::dot_f16_f32_many(
                            key,
                            std::array::from_fn::<_, Q_TILE, _>(|o| query(p + o)),
                        );
                        for (o, value) in got.iter().enumerate() {
                            let (i, h) = pair(p + o);
                            row(i, h)[t] = value * scale;
                        }
                        p += Q_TILE;
                    }
                    while p < pairs {
                        let (i, h) = pair(p);
                        row(i, h)[t] = kernels::dot_f16_f32_many(key, [query(p)])[0] * scale;
                        p += 1;
                    }
                }
                // Softmax each (token, head) row over the positions it sees.
                for i in i0..i1 {
                    for h in heads.clone() {
                        let scores = &mut row(i, h)[..start + i + 1];
                        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let inv = 1.0 / kernels::exp_shifted_sum(scores, max);
                        for s in scores.iter_mut() {
                            *s *= inv;
                        }
                        out(i, h).fill(0.0);
                    }
                }
                // Values: read each value row once and accumulate it into
                // every output that weights it.
                for t in 0..start + i1 {
                    let value = &v_layer[t * kvd + kv_off..t * kvd + kv_off + hd];
                    for i in i0.max(t.saturating_sub(start))..i1 {
                        for h in heads.clone() {
                            kernels::axpy_f16(out(i, h), row(i, h)[t], value);
                        }
                    }
                }
            });
            self.profiler.lap(3);
            be.matmul(
                model.matrix(layer.wo),
                &self.attn[..n * qd],
                &mut self.proj[..n * d],
                n,
            )?;
            self.profiler.lap(4);
            add_assign(&mut self.x[..n * d], &self.proj[..n * d]);

            // --- feed-forward ---
            for i in 0..n {
                let (x, xn) = (
                    &self.x[i * d..(i + 1) * d],
                    &mut self.xn[i * d..(i + 1) * d],
                );
                xn.copy_from_slice(x);
                rms_norm_in_place(xn, &layer.ffn_norm, c.rms_eps);
            }
            let ff = c.n_ff;
            self.profiler.lap(0);
            be.matmul_many(
                &mut [
                    Job {
                        matrix: model.matrix(layer.w_gate),
                        output: &mut self.gate[..n * ff],
                    },
                    Job {
                        matrix: model.matrix(layer.w_up),
                        output: &mut self.up[..n * ff],
                    },
                ],
                &self.xn[..n * d],
                n,
            )?;
            self.profiler.lap(5);
            kernels::swiglu_in_place(&mut self.gate[..n * ff], &self.up[..n * ff]);
            self.profiler.lap(0);
            be.matmul(
                model.matrix(layer.w_down),
                &self.gate[..n * ff],
                &mut self.proj[..n * d],
                n,
            )?;
            self.profiler.lap(6);
            add_assign(&mut self.x[..n * d], &self.proj[..n * d]);
        }
        #[cfg(feature = "cuda")]
        let logits_done = self.forward_gpu(n, start, rows)?;
        #[cfg(not(feature = "cuda"))]
        let logits_done = false;
        self.pos += n;
        if logits_done {
            self.profiler.lap(7);
            return Ok(());
        }

        let first = match rows {
            Rows::None => return Ok(()),
            Rows::Last => n - 1,
            Rows::All => 0,
        };
        for i in first..n {
            rms_norm_in_place(
                &mut self.x[i * d..(i + 1) * d],
                &model.output_norm,
                c.rms_eps,
            );
        }
        let count = n - first;
        be.matmul(
            model.matrix(model.output),
            &self.x[first * d..n * d],
            &mut self.logits[..count * c.n_vocab],
            count,
        )?;
        self.profiler.lap(7);
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Rows {
    None,
    Last,
    All,
}

/// Raw pointer to a scratch buffer whose disjoint parts parallel tasks fill.
#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
// SAFETY: each task writes only its own sub-slice (see the attention loop).
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

impl OutPtr {
    /// A method (not field access) so closures capture the `Sync` wrapper.
    fn at(self, offset: usize) -> *mut f32 {
        // SAFETY: callers stay within the buffer the pointer came from.
        unsafe { self.0.add(offset) }
    }
}

/// Query tokens per attention task.
///
/// A block reads each KV-cache row once and shares it across every query in
/// the block and every query head in the group, so a larger block means less
/// cache traffic. But the block's queries are re-read for every position it
/// attends to, so once they stop fitting in L1 the saved key traffic comes
/// straight back as query traffic -- that is a real regression, not a
/// diminishing return (it cost 7% on a 128-wide head before this bound).
/// So: the largest block whose queries and output accumulators fit in half of
/// a 32 KiB L1, and never more than leaves about two tasks per thread.
fn query_block(config: &Config, tokens: usize, threads: usize) -> usize {
    if let Some(forced) = std::env::var("KLLM_QUERY_BLOCK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
    {
        return forced.min(tokens.max(1));
    }
    let group = config.n_head / config.n_head_kv;
    // Each query in the block needs one query vector and one output
    // accumulator per query head in the group, both f32.
    let per_query = 2 * group * config.head_dim * 4;
    let by_cache = (16 * 1024 / per_query.max(1)).max(1);
    let by_threads = (tokens * config.n_head_kv / (2 * threads.max(1))).max(1);
    by_cache.min(by_threads).min(tokens.max(1))
}

fn add_bias(values: &mut [f32], bias: Option<&[f32]>) {
    if let Some(bias) = bias {
        add_assign(values, bias);
    }
}

fn add_assign(target: &mut [f32], source: &[f32]) {
    for (t, s) in target.iter_mut().zip(source) {
        *t += s;
    }
}

/// Rotate the first `2 * table.len()` dimensions of one head in place.
fn rope(head: &mut [f32], table: &[(f32, f32)], style: RopeStyle) {
    let n = table.len();
    match style {
        RopeStyle::Norm => {
            for (j, &(cos, sin)) in table.iter().enumerate() {
                let (a, b) = (head[2 * j], head[2 * j + 1]);
                head[2 * j] = a * cos - b * sin;
                head[2 * j + 1] = a * sin + b * cos;
            }
        }
        RopeStyle::Neox => {
            for (j, &(cos, sin)) in table.iter().enumerate() {
                let (a, b) = (head[j], head[j + n]);
                head[j] = a * cos - b * sin;
                head[j + n] = a * sin + b * cos;
            }
        }
    }
}
