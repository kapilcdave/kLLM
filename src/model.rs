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
    Qwen3Moe,
    /// Qwen3.5: three linear-attention blocks to each attention block.
    Qwen35,
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
    /// Present when the feed-forward block is a mixture of experts.
    pub moe: Option<Moe>,
    /// Present when some layers mix positions with a linear-attention
    /// recurrence instead of softmax attention.
    pub ssm: Option<SsmConfig>,
    /// Whether each q head also emits a sigmoid output gate, doubling the
    /// projection width with interleaved query/gate heads (Qwen3.5).
    pub attn_gated: bool,
}

/// Linear-attention parameters (Qwen3.5's gated DeltaNet).
#[derive(Debug, Clone, Copy)]
pub struct SsmConfig {
    /// Taps of the causal depthwise convolution.
    pub conv_kernel: usize,
    /// Key and value head dimension.
    pub state_size: usize,
    pub group_count: usize,
    pub time_step_rank: usize,
    /// Combined value-head width (the Q/K channels are additional).
    pub inner_size: usize,
    /// Every `interval`-th block uses softmax attention instead.
    pub interval: usize,
}

impl SsmConfig {
    fn conv_width(self) -> usize {
        2 * self.group_count * self.state_size + self.inner_size
    }
    fn state_floats(self) -> usize {
        self.time_step_rank * self.state_size * self.state_size
            + self.conv_width() * self.conv_kernel
    }
}

/// How the router turns hidden states into expert weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gating {
    Softmax,
    Sigmoid,
}

/// Mixture-of-experts feed-forward parameters.
#[derive(Debug, Clone, Copy)]
pub struct Moe {
    /// Experts per layer.
    pub n_expert: usize,
    /// Experts each token is routed to.
    pub n_expert_used: usize,
    /// One expert's feed-forward width (not `Config::n_ff`).
    pub n_ff_exp: usize,
    /// Width of the always-on shared expert; zero when there is none.
    pub n_ff_shared: usize,
    /// Renormalize the chosen experts' weights to sum to one.
    pub weights_norm: bool,
    pub weights_scale: f32,
    pub gating: Gating,
}

impl Moe {
    /// Weight bytes of one layer's experts that a single token reads, given
    /// the bytes of the whole expert block. Routing is what makes an MoE
    /// cheap per token and expensive to hold, so the planner needs both.
    pub fn active_fraction(&self) -> f64 {
        self.n_expert_used as f64 / self.n_expert.max(1) as f64
    }
}

impl Config {
    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }

    /// Widest feed-forward intermediate the forward pass will hold: the
    /// dense width, or an expert's (a routed model's `n_ff` is unused and
    /// some files leave it zero).
    pub fn ffn_width(&self) -> usize {
        let moe = self.moe.map_or(0, |m| m.n_ff_exp.max(m.n_ff_shared));
        self.n_ff.max(moe)
    }

    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }

    /// KV-cache bytes per token of context with f16 storage.
    pub fn kv_bytes_per_token(&self) -> u64 {
        2 * self.kv_layers() as u64 * self.kv_dim() as u64 * 2
    }

    /// Layers that keep a KV cache.
    ///
    /// Every layer, unless the architecture is a hybrid: Qwen3.5 mixes one
    /// attention block into every `full_attention_interval`, and the linear
    /// attention blocks hold a fixed-size recurrent state instead of a cache
    /// that grows with the context. That is the whole reason a long context
    /// is affordable there -- 20 KiB a token against 96 for a dense Qwen3 of
    /// the same depth.
    pub fn kv_layers(&self) -> usize {
        match self.ssm {
            Some(ssm) if ssm.interval > 0 && ssm.interval <= self.n_layer => {
                self.n_layer / ssm.interval
            }
            _ => self.n_layer,
        }
    }

    /// Which KV slot layer `index` uses, if it keeps a cache at all. The
    /// attention block is the last of each interval.
    pub fn kv_slot(&self, index: usize) -> Option<usize> {
        match self.ssm {
            Some(ssm) if ssm.interval > 0 && ssm.interval <= self.n_layer => (index + 1)
                .is_multiple_of(ssm.interval)
                .then(|| index / ssm.interval),
            _ => Some(index),
        }
    }

    pub fn from_gguf(info: &GgufInfo) -> Result<Self, ModelError> {
        let arch_name = info
            .architecture()
            .ok_or_else(|| ModelError::Missing("general.architecture".into()))?;
        let arch = match arch_name {
            "llama" | "mistral" => Arch::Llama,
            "qwen2" => Arch::Qwen2,
            "qwen3" => Arch::Qwen3,
            "qwen3moe" => Arch::Qwen3Moe,
            "qwen35" | "qwen35moe" => Arch::Qwen35,
            other => {
                return Err(ModelError::Unsupported(format!(
                    "architecture {other:?} (supported: llama, qwen2, qwen3, qwen3moe, qwen35)"
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
        // Mixture-of-experts parameters. The architecture decides how the
        // router's weights are post-processed, because llama.cpp hardcodes
        // that per architecture rather than writing it into the file; any
        // key that *is* present overrides the default.
        let moe = match opt_int("expert_count").filter(|n| *n > 0) {
            None => None,
            Some(n_expert) => {
                let n_expert_used = opt_int("expert_used_count")
                    .filter(|k| *k > 0 && *k <= n_expert)
                    .ok_or_else(|| {
                        ModelError::Shape(format!(
                            "{arch_name}.expert_used_count must be in 1..={n_expert}"
                        ))
                    })?;
                let (norm, scale, gating) = match arch {
                    Arch::Qwen3Moe | Arch::Qwen35 => (true, 1.0, Gating::Softmax),
                    _ => (false, 1.0, Gating::Softmax),
                };
                Some(Moe {
                    n_expert,
                    n_expert_used,
                    n_ff_exp: opt_int("expert_feed_forward_length").ok_or_else(|| {
                        ModelError::Missing(format!("{arch_name}.expert_feed_forward_length"))
                    })?,
                    n_ff_shared: opt_int("expert_shared_feed_forward_length").unwrap_or(0),
                    weights_norm: info
                        .arch_get("expert_weights_norm")
                        .and_then(Value::as_bool)
                        .unwrap_or(norm),
                    weights_scale: float("expert_weights_scale").unwrap_or(scale),
                    gating: match info.arch_get("expert_gating_func").and_then(Value::as_u64) {
                        // llama.cpp: 1 = softmax, 2 = sigmoid.
                        Some(2) => Gating::Sigmoid,
                        _ => gating,
                    },
                })
            }
        };
        let config = Self {
            arch,
            n_embd,
            n_layer: int("block_count")?,
            n_head,
            n_head_kv,
            head_dim,
            n_ff: opt_int("feed_forward_length").unwrap_or(0),
            n_vocab,
            rms_eps: float("attention.layer_norm_rms_epsilon").unwrap_or(1e-5),
            rope_base: float("rope.freq_base").unwrap_or(10_000.0),
            rope_dim,
            rope_scale,
            rope_style: match arch {
                Arch::Llama => RopeStyle::Norm,
                Arch::Qwen2 | Arch::Qwen3 | Arch::Qwen3Moe | Arch::Qwen35 => RopeStyle::Neox,
            },
            train_context: opt_int("context_length").unwrap_or(4096),
            moe,
            // By shape, not by architecture: each Q head has a companion
            // output gate, doubling the Q projection width.
            attn_gated: (0..int("block_count")?).any(|i| {
                info.tensor(&format!("blk.{i}.attn_q.weight"))
                    .and_then(|t| t.dimensions.get(1).copied())
                    .is_some_and(|rows| rows == 2 * (n_head * head_dim) as u64)
            }),
            ssm: opt_int("ssm.state_size").map(|state_size| SsmConfig {
                conv_kernel: opt_int("ssm.conv_kernel").unwrap_or(4),
                state_size,
                group_count: opt_int("ssm.group_count").unwrap_or(1),
                time_step_rank: opt_int("ssm.time_step_rank").unwrap_or(0),
                inner_size: opt_int("ssm.inner_size").unwrap_or(0),
                interval: opt_int("full_attention_interval").unwrap_or(usize::MAX),
            }),
        };
        // A routed model's dense `feed_forward_length` is unused, and some
        // files set it to zero, so only require it when there are no experts.
        let n_ff_required = config.moe.map_or(config.n_ff, |m| m.n_ff_exp);
        if config.n_layer == 0 || config.n_embd == 0 || n_ff_required == 0 || config.head_dim == 0 {
            return Err(ModelError::Shape("zero-sized hyperparameter".into()));
        }
        if let Some(sc) = config.ssm
            && (sc.conv_kernel == 0
                || sc.state_size == 0
                || sc.group_count == 0
                || sc.time_step_rank == 0
                || sc.time_step_rank % sc.group_count != 0
                || sc.inner_size != sc.time_step_rank * sc.state_size
                || sc.interval == 0)
        {
            return Err(ModelError::Shape(
                "invalid gated DeltaNet hyperparameters".into(),
            ));
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
    mixer: Mixer,
    ffn_norm: Vec<f32>,
    ffn: Ffn,
    /// Read from disk every token instead of held in RAM (see `LoadOptions`).
    streamed: bool,
}

/// How a layer mixes across positions.
///
/// Softmax attention for every layer of the architectures kLLM started with;
/// Qwen3.5 instead alternates, three linear-attention blocks to one
/// attention block (`full_attention_interval`).
#[allow(clippy::large_enum_variant)]
enum Mixer {
    Attention(Attention),
    /// A gated DeltaNet: see `docs/QWEN35.md`.
    Ssm(Ssm),
}

struct Attention {
    wq: Weight,
    wk: Weight,
    wv: Weight,
    wo: Weight,
    bq: Option<Vec<f32>>,
    bk: Option<Vec<f32>>,
    bv: Option<Vec<f32>>,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
}

/// Gated DeltaNet weights. GGUF stores `a` as -exp(A_log).
struct Ssm {
    qkv: Weight,
    gate: Weight,
    out: Weight,
    conv1d: Vec<f32>,
    a: Vec<f32>,
    dt_bias: Vec<f32>,
    alpha: Weight,
    beta: Weight,
    norm: Vec<f32>,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

struct SsmState {
    conv: Vec<f32>,
    matrix: Vec<f32>,
    mixed: Vec<f32>,
}

impl SsmState {
    fn new(c: SsmConfig) -> Self {
        let width = 2 * c.group_count * c.state_size + c.inner_size;
        Self {
            conv: vec![0.0; width * (c.conv_kernel - 1)],
            matrix: vec![0.0; c.time_step_rank * c.state_size * c.state_size],
            mixed: vec![0.0; width],
        }
    }
    fn reset(&mut self) {
        self.conv.fill(0.0);
        self.matrix.fill(0.0);
    }

    // Reference: llama.cpp src/models/qwen35.cpp and gated_delta_net.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        c: SsmConfig,
        w: &Ssm,
        eps: f32,
        x: &[f32],
        z: &[f32],
        alpha: &[f32],
        beta: &[f32],
        out: &mut [f32],
    ) {
        let taps = c.conv_kernel;
        let hd = c.state_size;
        let kd = c.group_count * hd;
        for (ch, &current) in x.iter().enumerate() {
            let history = &mut self.conv[ch * (taps - 1)..(ch + 1) * (taps - 1)];
            let kernel = &w.conv1d[ch * taps..(ch + 1) * taps];
            let mut value = current * kernel[taps - 1];
            for j in 0..taps - 1 {
                value += history[j] * kernel[j];
            }
            if taps > 1 {
                history.rotate_left(1);
                history[taps - 2] = current;
            }
            self.mixed[ch] = value * sigmoid(value);
        }
        for head in self.mixed[..2 * kd].chunks_exact_mut(hd) {
            let inv = 1.0 / (head.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
            for v in head {
                *v *= inv;
            }
        }
        assert_eq!(self.matrix.len(), c.time_step_rank * hd * hd);
        assert_eq!(out.len(), c.time_step_rank * hd);
        let matrix = OutPtr(self.matrix.as_mut_ptr());
        let output = OutPtr(out.as_mut_ptr());
        let mixed = &self.mixed;
        crate::pool::global().for_each(c.time_step_rank, |h| {
            // GGUF reorders V heads for tiled Q/K broadcasting.
            let kh = h % c.group_count;
            let q = &mixed[kh * hd..(kh + 1) * hd];
            let k = &mixed[kd + kh * hd..kd + (kh + 1) * hd];
            let v = &mixed[2 * kd + h * hd..2 * kd + (h + 1) * hd];
            let a = alpha[h] + w.dt_bias[h];
            let softplus = a.max(0.0) + (-a.abs()).exp().ln_1p();
            let decay = (w.a[h] * softplus).exp();
            let b = sigmoid(beta[h]);
            // SAFETY: task h exclusively owns matrix head h and output
            // head h; the pool joins all tasks before these buffers are reused.
            let state = unsafe { std::slice::from_raw_parts_mut(matrix.at(h * hd * hd), hd * hd) };
            let o = unsafe { std::slice::from_raw_parts_mut(output.at(h * hd), hd) };
            for j in 0..hd {
                let row = &mut state[j * hd..(j + 1) * hd];
                for s in row.iter_mut() {
                    *s *= decay;
                }
                let delta = b * (v[j] - row.iter().zip(k).map(|(s, k)| s * k).sum::<f32>());
                let mut value = 0.0;
                for t in 0..hd {
                    row[t] += k[t] * delta;
                    value += row[t] * q[t];
                }
                o[j] = value / (hd as f32).sqrt();
            }
            rms_norm_in_place(o, &w.norm, eps);
            for j in 0..hd {
                let gate = z[h * hd + j];
                o[j] *= gate * sigmoid(gate);
            }
        });
    }
}

/// One layer's feed-forward block.
// The routed variant is the larger one, which costs `n_layer` times the
// difference -- a few kilobytes against a model measured in gigabytes.
#[allow(clippy::large_enum_variant)]
enum Ffn {
    Dense {
        gate: Weight,
        up: Weight,
        down: Weight,
    },
    /// A mixture of experts: a router picks `n_expert_used` of `n_expert`
    /// slices of each expert tensor, optionally alongside a shared expert
    /// that every token uses.
    Routed {
        router: Weight,
        gate: Experts,
        up: Experts,
        down: Experts,
        shared: Option<[Weight; 3]>,
        shared_gate: Option<Vec<f32>>,
    },
}

/// A 3-D expert tensor: `count` matrices of `rows` x `cols` laid out back to
/// back, so expert `e` starts at `offset + e * stride`.
#[derive(Debug, Clone, Copy)]
struct Experts {
    kind: GgmlType,
    rows: usize,
    cols: usize,
    offset: usize,
    stride: usize,
    count: usize,
}

impl Layer {
    /// Every 2-D weight in the layer. Expert tensors are not here: they are
    /// 3-D and only a routed fraction of them is read per token, so callers
    /// that care about bytes ask `expert_bytes` separately.
    fn matrices(&self) -> Vec<Weight> {
        let mut all = match &self.mixer {
            Mixer::Attention(a) => vec![a.wq, a.wk, a.wv, a.wo],
            Mixer::Ssm(m) => vec![m.qkv, m.gate, m.out, m.alpha, m.beta],
        };
        match &self.ffn {
            Ffn::Dense { gate, up, down } => all.extend([*gate, *up, *down]),
            Ffn::Routed { router, shared, .. } => {
                // The shared expert's gate is a dequantized vector, not a
                // mapped matrix, so it holds no bytes here.
                all.push(*router);
                all.extend(shared.iter().flatten().copied());
            }
        }
        all
    }

    /// Bytes held by this layer's 3-D expert tensors, and the bytes of them
    /// one token reads.
    fn expert_bytes(&self, moe: Option<Moe>) -> (u64, u64) {
        let Ffn::Routed { gate, up, down, .. } = &self.ffn else {
            return (0, 0);
        };
        let total: u64 = [gate, up, down]
            .iter()
            .map(|e| (e.stride * e.count) as u64)
            .sum();
        let used = moe.map_or(0, |m| m.n_expert_used as u64);
        let active: u64 = [gate, up, down]
            .iter()
            .map(|e| e.stride as u64 * used)
            .sum();
        (total, active)
    }
}

/// How to load a model.
#[derive(Debug, Clone, Copy, Default)]
pub struct LoadOptions {
    /// Bytes of RAM the weights may occupy. A model larger than this keeps
    /// the output head and a prefix of blocks resident and streams the rest
    /// from disk every token, with read-ahead of the next blocks and
    /// immediate release of each streamed block, so the stream never evicts
    /// the resident set. `None` keeps everything resident.
    pub ram_budget: Option<u64>,
}

/// Which weights stay in RAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Residency {
    /// Blocks `0..first_streamed` are resident; the rest stream from disk.
    pub first_streamed: usize,
    pub head_streamed: bool,
    /// Weight bytes read from disk per token.
    pub streamed_bytes: u64,
    pub resident_bytes: u64,
    /// Bytes the plan does not manage at all: a routed model's expert
    /// tensors, which are demand-paged because only a routed fraction of
    /// them is touched per token. Planning them as resident would reserve
    /// budget for pages that are never prefetched, which is what made an
    /// MoE stream blocks it had no need to.
    pub demand_bytes: u64,
    /// True when those expert pages cannot all be held: every byte of RAM
    /// is then worth more as page cache for them than as anything else.
    pub demand_paged: bool,
}

impl Residency {
    /// Plan residency from the tensor directory alone, before loading.
    pub fn plan(info: &GgufInfo, n_layer: usize, budget: Option<u64>) -> Self {
        let mut layer = vec![0u64; n_layer];
        let (mut head, mut other, mut demand) = (0u64, 0u64, 0u64);
        let tied = info.tensor("output.weight").is_none();
        for t in &info.tensors {
            let size = t.byte_size().unwrap_or(0);
            // Expert tensors are demand-paged, not planned: see `demand_bytes`
            // and `Model::stream_ahead`.
            if t.name.ends_with("_exps.weight") {
                demand += size;
                continue;
            }
            match t.block_index() {
                Some(i) if (i as usize) < n_layer => layer[i as usize] += size,
                _ if t.name == "output.weight" || (tied && t.name == "token_embd.weight") => {
                    head += size
                }
                _ => other += size,
            }
        }
        let total: u64 = layer.iter().sum::<u64>() + head + other;
        let contended = budget.is_some_and(|b| total + demand > b);
        let all = Self {
            first_streamed: n_layer,
            head_streamed: false,
            streamed_bytes: 0,
            resident_bytes: total,
            demand_bytes: demand,
            demand_paged: demand > 0 && contended,
        };
        let Some(budget) = budget else { return all };
        if total <= budget {
            return all;
        }
        // The head is read in full by every token, like a block, but it sits
        // at the end of the pass: keep it resident first, then a prefix.
        let mut remaining = budget.saturating_sub(other);
        let head_streamed = head > remaining;
        if !head_streamed {
            remaining -= head;
        }
        let mut first_streamed = 0;
        while first_streamed < n_layer && layer[first_streamed] <= remaining {
            remaining -= layer[first_streamed];
            first_streamed += 1;
        }
        let streamed_bytes =
            layer[first_streamed..].iter().sum::<u64>() + if head_streamed { head } else { 0 };
        Self {
            first_streamed,
            head_streamed,
            streamed_bytes,
            resident_bytes: total - streamed_bytes,
            demand_bytes: demand,
            demand_paged: demand > 0,
        }
    }
}

/// Blocks to read ahead of the one being computed.
const STREAM_LOOKAHEAD: usize = 2;

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
    pub residency: Residency,
    /// Blocks (and head) offloaded to a GPU, if any.
    #[cfg(feature = "cuda")]
    gpu: Option<gpu::GpuStack>,
}

impl Model {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ModelError> {
        Self::load_with(path, LoadOptions::default())
    }

    pub fn load_with(path: impl AsRef<Path>, options: LoadOptions) -> Result<Self, ModelError> {
        let file = GgufFile::open(path)?;
        let config = Config::from_gguf(&file.info)?;
        let c = &config;
        let residency = Residency::plan(&file.info, c.n_layer, options.ram_budget);
        let loader = Loader {
            file: &file,
            packed: std::cell::RefCell::new(Vec::new()),
            // An x8-repacked copy is a second copy, on the heap. When the
            // experts have to be demand-paged, that gigabyte is worth more
            // as page cache for them: measured on Qwen3-30B-A3B, repacking
            // costs 19% of decode (mean 8.9 against 10.6 tok/s over four
            // interleaved rounds) and prefill too.
            repack: std::env::var_os("KLLM_NO_REPACK").is_none() && !residency.demand_paged,
            streaming: std::cell::Cell::new(false),
            released: std::cell::RefCell::new(Vec::new()),
        };
        loader.streaming.set(residency.head_streamed);
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
            let streamed = i >= residency.first_streamed;
            loader.streaming.set(streamed);
            layers.push(Layer {
                streamed,
                attn_norm: loader.vector(&name("attn_norm.weight"), c.n_embd)?,
                // Attention or linear attention, by which tensors the block
                // has rather than by arithmetic on `full_attention_interval`.
                mixer: match c
                    .ssm
                    .filter(|_| loader.file.info.tensor(&name("ssm_a")).is_some())
                {
                    None => Mixer::Attention(Attention {
                        wq: loader.matrix(
                            &name("attn_q.weight"),
                            if c.attn_gated {
                                2 * c.q_dim()
                            } else {
                                c.q_dim()
                            },
                            c.n_embd,
                        )?,
                        wk: loader.matrix(&name("attn_k.weight"), c.kv_dim(), c.n_embd)?,
                        wv: loader.matrix(&name("attn_v.weight"), c.kv_dim(), c.n_embd)?,
                        wo: loader.matrix(&name("attn_output.weight"), c.n_embd, c.q_dim())?,
                        bq: loader.optional_vector(&name("attn_q.bias"), c.q_dim())?,
                        bk: loader.optional_vector(&name("attn_k.bias"), c.kv_dim())?,
                        bv: loader.optional_vector(&name("attn_v.bias"), c.kv_dim())?,
                        // Presence, not architecture: Qwen3-MoE carries
                        // these just as Qwen3 does, and deciding by
                        // architecture silently dropped them.
                        q_norm: loader.optional_vector(&name("attn_q_norm.weight"), c.head_dim)?,
                        k_norm: loader.optional_vector(&name("attn_k_norm.weight"), c.head_dim)?,
                    }),
                    Some(ssm) => Mixer::Ssm(Ssm {
                        qkv: loader.matrix(
                            &name("attn_qkv.weight"),
                            2 * ssm.group_count * ssm.state_size + ssm.inner_size,
                            c.n_embd,
                        )?,
                        gate: loader.matrix(&name("attn_gate.weight"), ssm.inner_size, c.n_embd)?,
                        out: loader.matrix(&name("ssm_out.weight"), c.n_embd, ssm.inner_size)?,
                        conv1d: loader.vector(
                            &name("ssm_conv1d.weight"),
                            ssm.conv_kernel
                                * (2 * ssm.group_count * ssm.state_size + ssm.inner_size),
                        )?,
                        a: loader.vector(&name("ssm_a"), ssm.time_step_rank)?,
                        dt_bias: loader.vector(&name("ssm_dt.bias"), ssm.time_step_rank)?,
                        alpha: loader.matrix(
                            &name("ssm_alpha.weight"),
                            ssm.time_step_rank,
                            c.n_embd,
                        )?,
                        beta: loader.matrix(
                            &name("ssm_beta.weight"),
                            ssm.time_step_rank,
                            c.n_embd,
                        )?,
                        norm: loader.vector(&name("ssm_norm.weight"), ssm.state_size)?,
                    }),
                },
                // The norm before the feed-forward block. Qwen3.5 has no
                // `ffn_norm` and calls it `post_attention_norm` instead --
                // which is the same position in the block, after the mixer
                // and before the feed-forward, and there is no other
                // candidate in the file.
                ffn_norm: match loader.optional_vector(&name("ffn_norm.weight"), c.n_embd)? {
                    Some(v) => v,
                    None => loader.vector(&name("post_attention_norm.weight"), c.n_embd)?,
                },
                // Routed or dense per layer, by which tensors are present:
                // several MoE architectures keep the first blocks dense, and
                // a whole-model switch would fail to load them.
                ffn: match c.moe.filter(|_| {
                    loader
                        .file
                        .info
                        .tensor(&name("ffn_gate_exps.weight"))
                        .is_some()
                }) {
                    None => Ffn::Dense {
                        gate: loader.matrix(&name("ffn_gate.weight"), c.n_ff, c.n_embd)?,
                        up: loader.matrix(&name("ffn_up.weight"), c.n_ff, c.n_embd)?,
                        down: loader.matrix(&name("ffn_down.weight"), c.n_embd, c.n_ff)?,
                    },
                    Some(moe) => {
                        let (n, ff) = (moe.n_expert, moe.n_ff_exp);
                        let shared = if moe.n_ff_shared > 0 {
                            let sh = moe.n_ff_shared;
                            Some([
                                loader.matrix(&name("ffn_gate_shexp.weight"), sh, c.n_embd)?,
                                loader.matrix(&name("ffn_up_shexp.weight"), sh, c.n_embd)?,
                                loader.matrix(&name("ffn_down_shexp.weight"), c.n_embd, sh)?,
                            ])
                        } else {
                            None
                        };
                        Ffn::Routed {
                            router: loader.matrix(&name("ffn_gate_inp.weight"), n, c.n_embd)?,
                            gate: loader.experts(&name("ffn_gate_exps.weight"), ff, c.n_embd, n)?,
                            up: loader.experts(&name("ffn_up_exps.weight"), ff, c.n_embd, n)?,
                            down: loader.experts(&name("ffn_down_exps.weight"), c.n_embd, ff, n)?,
                            shared,
                            // Stored 1-D as [n_embd], so it is a vector and
                            // not a one-row matrix.
                            shared_gate: loader
                                .optional_vector(&name("ffn_gate_inp_shexp.weight"), c.n_embd)?,
                        }
                    }
                },
            });
        }
        // Read ahead only what stays resident; streamed blocks are read
        // block by block during the forward pass.
        if residency.first_streamed == c.n_layer
            && !residency.head_streamed
            && residency.demand_bytes == 0
        {
            file.advise_willneed();
        }
        // Repacked weights never read their row-major pages again (except the
        // embedding table, which is looked up by row): give that memory back.
        for w in loader.released.borrow().iter() {
            file.release(w.offset, w.len);
        }
        let packed = loader.packed.into_inner();
        let model = Self {
            file,
            config,
            token_embd,
            output_norm,
            output,
            layers,
            rope_freq_factors,
            packed,
            backend: CpuBackend,
            residency,
            #[cfg(feature = "cuda")]
            gpu: None,
        };
        model.pin_managed_weights();
        Ok(model)
    }

    /// Read ahead everything a routed model touches every token, which is
    /// everything except the experts.
    ///
    /// Without this the attention and head weights are demand-paged like the
    /// experts, and the experts evict them: there are sixteen gigabytes of
    /// expert pages competing for the cache against one gigabyte that every
    /// single token needs. It showed up as `qkv proj` costing 10% of a token
    /// to move six megabytes of weights -- 0.8 G MAC/s where the experts
    /// managed 20.
    fn pin_managed_weights(&self) {
        if self.residency.demand_bytes == 0 {
            return; // a dense model is already handled by `advise_willneed`.
        }
        for layer in &self.layers {
            if layer.streamed {
                continue; // deliberately read block by block instead.
            }
            for w in layer.matrices() {
                self.file.prefetch(w.offset, w.len);
            }
        }
        if !self.residency.head_streamed {
            self.file.prefetch(self.output.offset, self.output.len);
        }
        self.file
            .prefetch(self.token_embd.offset, self.token_embd.len);
    }

    pub fn info(&self) -> &GgufInfo {
        &self.file.info
    }

    /// Expert `index` of a 3-D expert tensor.
    ///
    /// Deliberately unpacked: the x8 layouts are a second copy of the
    /// weights, and the expert tensors are nearly all of an MoE's bytes, so
    /// repacking them would double the resident set of the one thing that
    /// already does not fit.
    fn expert(&self, e: Experts, index: usize) -> Matrix<'_> {
        debug_assert!(index < e.count);
        let at = e.offset + index * e.stride;
        Matrix {
            kind: e.kind,
            rows: e.rows,
            cols: e.cols,
            data: &self.file.mapped()[at..at + e.stride],
            packed: None,
        }
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
            total += l.expert_bytes(self.config.moe).1;
            for w in l.matrices() {
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
                l.matrices().iter().map(|w| w.len as u64).sum::<u64>()
                    + l.expert_bytes(self.config.moe).0
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
            .chain(self.layers.iter().flat_map(|l| l.matrices()))
            .map(|w| w.kind.name())
            .collect();
        kinds.sort_unstable();
        kinds.dedup();
        kinds
    }

    pub fn new_session(&self, context: usize, max_batch: usize) -> Result<Session<'_>, ModelError> {
        Session::new(self, context, max_batch)
    }

    /// Read ahead the streamed blocks that follow block `layer`.
    ///
    /// `matrices()` deliberately excludes the 3-D expert tensors, so experts
    /// are never read ahead and never released. Both halves matter for an
    /// MoE. Reading a block ahead would fetch all `n_expert` experts when a
    /// token wants `n_expert_used` of them -- for Qwen3-30B-A3B that is 128
    /// instead of 8, turning ~1.9 GiB of real traffic per token into ~18
    /// GiB. And not releasing them leaves their pages to the kernel's own
    /// LRU, which is a usable expert cache for free; whether its hit rate is
    /// good enough is the open question in docs/MOE.md, and it should be
    /// measured before anything cleverer is built.
    fn stream_ahead(&self, layer: usize) {
        let n = self.layers.len();
        for next in layer + 1..(layer + 1 + STREAM_LOOKAHEAD).min(n) {
            if self.layers[next].streamed {
                for w in self.layers[next].matrices() {
                    self.file.prefetch(w.offset, w.len);
                }
            }
        }
        if self.residency.head_streamed && layer + STREAM_LOOKAHEAD >= n {
            self.file.prefetch(self.output.offset, self.output.len);
        }
    }

    /// Return a streamed block's pages once it has been computed.
    fn stream_done(&self, layer: usize) {
        if self.layers[layer].streamed {
            for w in self.layers[layer].matrices() {
                self.file.release(w.offset, w.len);
            }
        }
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
    /// Whether the matrices being loaded now are streamed from disk.
    streaming: std::cell::Cell<bool>,
    /// Row-major originals made redundant by a repacked copy.
    released: std::cell::RefCell<Vec<Weight>>,
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

    /// A 3-D expert tensor of `count` matrices, each `rows` x `cols`.
    fn experts(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
        count: usize,
    ) -> Result<Experts, ModelError> {
        let t = self.tensor(name)?;
        let kind = t
            .kind()
            .filter(|k| k.is_supported())
            .ok_or_else(|| ModelError::Unsupported(format!("{name} has type {}", t.type_name())))?;
        if t.dimensions != [cols as u64, rows as u64, count as u64] {
            return Err(ModelError::Shape(format!(
                "{name} is {:?}, expected [{cols}, {rows}, {count}]",
                t.dimensions
            )));
        }
        let bytes = self.file.tensor_bytes(t)?;
        let offset = bytes.as_ptr() as usize - self.file.mapped().as_ptr() as usize;
        let stride = bytes.len() / count;
        let row_bytes = kind.row_bytes(cols).ok_or(ModelError::Shape(format!(
            "{name}: {cols} columns are not a whole number of {} blocks",
            kind.name()
        )))?;
        if stride != row_bytes * rows {
            return Err(ModelError::Shape(format!(
                "{name}: {stride} bytes per expert, expected {}",
                row_bytes * rows
            )));
        }
        Ok(Experts {
            kind,
            rows,
            cols,
            offset,
            stride,
            count,
        })
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
        // Streamed weights stay row-major in the mapping: a repacked copy
        // would be heap memory, which is exactly what they must not take.
        let packed = if self.repack && !self.streaming.get() {
            kernels::repack_x8(kind, bytes, rows, cols).map(|copy| {
                let mut packed = self.packed.borrow_mut();
                packed.push(copy);
                packed.len() - 1
            })
        } else {
            None
        };
        let weight = Weight {
            kind,
            rows,
            cols,
            offset,
            len: bytes.len(),
            packed,
        };
        if weight.packed.is_some() && name != "token_embd.weight" {
            self.released.borrow_mut().push(weight);
        }
        Ok(weight)
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
    /// Router logits, `n_expert` per token (empty for dense models).
    router: Vec<f32>,
    /// One expert's down-projection output, accumulated into `proj`.
    moe_proj: Vec<f32>,
    /// Activations gathered for one expert's token group, and that group's
    /// output, when a batch is sorted by expert.
    moe_x: Vec<f32>,
    moe_y: Vec<f32>,
    scores: Vec<f32>,
    rope: Vec<(f32, f32)>,
    logits: Vec<f32>,
    all_logits: Vec<f32>,
    ssm_states: Vec<Option<SsmState>>,
    history: Vec<u32>,
    replay: bool,
    q_full: Vec<f32>,
    attn_gate: Vec<f32>,
    ssm_qkv: Vec<f32>,
    ssm_z: Vec<f32>,
    ssm_alpha: Vec<f32>,
    ssm_beta: Vec<f32>,
    ssm_out: Vec<f32>,
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
        // Only the layers that keep a cache get a slot (see
        // `Config::kv_layers`), which for a hybrid is one in four.
        let cached_layers = (0..model.cpu_layers())
            .filter(|i| c.kv_slot(*i).is_some())
            .count();
        let cache = cached_layers * context * c.kv_dim();
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
            gate: vec![0.0; b * c.ffn_width()],
            up: vec![0.0; b * c.ffn_width()],
            router: vec![0.0; b * c.moe.map_or(0, |m| m.n_expert)],
            moe_proj: vec![0.0; c.n_embd],
            moe_x: vec![0.0; if c.moe.is_some() { b * c.n_embd } else { 0 }],
            moe_y: vec![0.0; if c.moe.is_some() { b * c.n_embd } else { 0 }],
            scores: vec![0.0; crate::pool::global().threads() * score_rows_per_worker(c) * context],
            rope: vec![(0.0, 0.0); b * c.rope_dim / 2],
            logits: vec![0.0; b * c.n_vocab],
            all_logits: Vec::new(),
            ssm_states: model
                .layers
                .iter()
                .map(|l| match l.mixer {
                    Mixer::Ssm(_) => Some(SsmState::new(c.ssm.unwrap())),
                    _ => None,
                })
                .collect(),
            history: Vec::new(),
            replay: false,
            q_full: vec![0.0; if c.attn_gated { b * 2 * c.q_dim() } else { 0 }],
            attn_gate: vec![0.0; if c.attn_gated { b * c.q_dim() } else { 0 }],
            ssm_qkv: vec![0.0; b * c.ssm.map_or(0, |sc| sc.conv_width())],
            ssm_z: vec![0.0; b * c.ssm.map_or(0, |sc| sc.inner_size)],
            ssm_alpha: vec![0.0; b * c.ssm.map_or(0, |sc| sc.time_step_rank)],
            ssm_beta: vec![0.0; b * c.ssm.map_or(0, |sc| sc.time_step_rank)],
            ssm_out: vec![0.0; b * c.ssm.map_or(0, |sc| sc.inner_size)],
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
            + self.rope.len() * 2
            + self.q_full.len()
            + self.attn_gate.len()
            + self.ssm_qkv.len()
            + self.ssm_z.len()
            + self.ssm_alpha.len()
            + self.ssm_beta.len()
            + self.ssm_out.len()
            + self
                .ssm_states
                .iter()
                .flatten()
                .map(|s| s.conv.len() + s.matrix.len() + s.mixed.len())
                .sum::<usize>();
        (self.k_cache.len() + self.v_cache.len()) as u64 * 2 + f32s as u64 * 4
    }

    /// What [`Model::new_session`] will allocate, computed without
    /// allocating: `(kv_cache_bytes, scratch_bytes)`.
    pub fn estimate(config: &Config, context: usize, max_batch: usize) -> (u64, u64) {
        let c = config;
        let context = context.max(1) as u64;
        let b = max_batch.clamp(1, context as usize) as u64;
        // Through `kv_bytes_per_token`, so this cannot drift from what
        // `Session::new` actually allocates again: it counted every layer
        // while the allocation gives a slot only to the layers that keep a
        // cache, and on a 40-block hybrid with one attention block in four
        // that is a 4x over-estimate -- enough for the planner to refuse a
        // 128k context that fits in 2.5 GiB.
        let kv = c.kv_bytes_per_token() * context;
        let f32s = b
            * (3 * c.n_embd as u64
                + 2 * c.q_dim() as u64
                + 2 * c.kv_dim() as u64
                + 2 * c.ffn_width() as u64
                + c.rope_dim as u64
                + c.n_vocab as u64);
        let gated = if c.attn_gated {
            b * 3 * c.q_dim() as u64
        } else {
            0
        };
        // Attention's score rows are sized per worker, not per token.
        let scores =
            crate::pool::global().threads() as u64 * score_rows_per_worker(c) as u64 * context;
        let ssm = c.ssm.map_or(0, |sc| {
            let layers = c.n_layer - c.n_layer / sc.interval;
            layers as u64 * sc.state_floats() as u64
                + b * (sc.conv_width() + 2 * sc.inner_size + 2 * sc.time_step_rank) as u64
        });
        (kv, (f32s + scores + gated + ssm) * 4)
    }

    /// Discard positions `>= pos` so generation can resume from `pos`.
    /// Recurrent models rebuild a retained prefix on the next forward pass;
    /// resetting to zero clears the convolution and DeltaNet state immediately.
    pub fn truncate(&mut self, pos: usize) {
        let next = self.pos.min(pos);
        if next < self.pos && self.model.config.ssm.is_some() {
            self.history.truncate(next);
            for state in self.ssm_states.iter_mut().flatten() {
                state.reset();
            }
            self.replay = next != 0;
        }
        self.pos = next;
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
        if self.replay {
            let prefix = self.history.clone();
            self.pos = 0;
            self.history.clear();
            self.replay = false;
            self.forward(&prefix, Logits::Last)?;
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
            if self.model.config.ssm.is_some() {
                self.history.extend_from_slice(chunk);
            }
            if want == Logits::All {
                self.all_logits
                    .extend_from_slice(&self.logits[..chunk.len() * n_vocab]);
            }
        }
        // Flush per forward pass so a trace survives an interrupted run.
        if router_trace::enabled() {
            router_trace::flush();
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
        // Reused by every routed layer; `route` clears it.
        let mut chosen: Vec<(usize, f32)> =
            Vec::with_capacity(c.moe.map_or(0, |m| m.n_expert_used));
        // Tokens grouped by the expert they chose, for batches larger than
        // one. Allocated once here rather than per layer.
        let mut assign: Vec<Vec<(usize, f32)>> = vec![
            Vec::new();
            if n > 1 {
                c.moe.map_or(0, |m| m.n_expert)
            } else {
                0
            }
        ];

        let streaming = model.residency.first_streamed < c.n_layer || model.residency.head_streamed;
        if streaming {
            // The first streamed block is read while the resident ones run.
            if let Some(first) = model.layers.iter().position(|l| l.streamed) {
                model.stream_ahead(first.saturating_sub(1));
            }
        }
        for (li, layer) in model.layers.iter().enumerate().take(model.cpu_layers()) {
            if streaming {
                model.stream_ahead(li);
            }
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
            if let Mixer::Ssm(weights) = &layer.mixer {
                let sc = c.ssm.unwrap();
                let width = weights.qkv.rows;
                let qkv = &mut self.ssm_qkv[..n * width];
                let z = &mut self.ssm_z[..n * sc.inner_size];
                let alpha = &mut self.ssm_alpha[..n * sc.time_step_rank];
                let beta = &mut self.ssm_beta[..n * sc.time_step_rank];
                be.matmul_many(
                    &mut [
                        Job {
                            matrix: model.matrix(weights.qkv),
                            output: qkv,
                        },
                        Job {
                            matrix: model.matrix(weights.gate),
                            output: z,
                        },
                        Job {
                            matrix: model.matrix(weights.alpha),
                            output: alpha,
                        },
                        Job {
                            matrix: model.matrix(weights.beta),
                            output: beta,
                        },
                    ],
                    &self.xn[..n * d],
                    n,
                )?;
                self.profiler.lap(1);
                let out = &mut self.ssm_out[..n * sc.inner_size];
                let state = self.ssm_states[li].as_mut().unwrap();
                for i in 0..n {
                    state.step(
                        sc,
                        weights,
                        c.rms_eps,
                        &qkv[i * width..(i + 1) * width],
                        &z[i * sc.inner_size..(i + 1) * sc.inner_size],
                        &alpha[i * sc.time_step_rank..(i + 1) * sc.time_step_rank],
                        &beta[i * sc.time_step_rank..(i + 1) * sc.time_step_rank],
                        &mut out[i * sc.inner_size..(i + 1) * sc.inner_size],
                    );
                }
                self.profiler.lap(3);
                be.matmul(model.matrix(weights.out), out, &mut self.proj[..n * d], n)?;
            } else {
                let Mixer::Attention(weights) = &layer.mixer else {
                    unreachable!()
                };
                // A hybrid gives a cache slot only to its attention blocks,
                // so the slot is not the layer index.
                let kv_slot = c.kv_slot(li).expect("an attention block has a KV slot");
                be.matmul_many(
                    &mut [
                        Job {
                            matrix: model.matrix(weights.wq),
                            output: if c.attn_gated {
                                &mut self.q_full[..n * 2 * qd]
                            } else {
                                &mut self.q[..n * qd]
                            },
                        },
                        Job {
                            matrix: model.matrix(weights.wk),
                            output: &mut self.k[..n * kvd],
                        },
                        Job {
                            matrix: model.matrix(weights.wv),
                            output: &mut self.v[..n * kvd],
                        },
                    ],
                    &self.xn[..n * d],
                    n,
                )?;
                if c.attn_gated {
                    for i in 0..n {
                        for h in 0..c.n_head {
                            let src = i * 2 * qd + h * 2 * hd;
                            let dst = i * qd + h * hd;
                            self.q[dst..dst + hd].copy_from_slice(&self.q_full[src..src + hd]);
                            self.attn_gate[dst..dst + hd]
                                .copy_from_slice(&self.q_full[src + hd..src + 2 * hd]);
                        }
                    }
                }
                self.profiler.lap(1);
                for i in 0..n {
                    let q = &mut self.q[i * qd..(i + 1) * qd];
                    let k = &mut self.k[i * kvd..(i + 1) * kvd];
                    let v = &mut self.v[i * kvd..(i + 1) * kvd];
                    add_bias(q, weights.bq.as_deref());
                    add_bias(k, weights.bk.as_deref());
                    add_bias(v, weights.bv.as_deref());
                    if let Some(w) = &weights.q_norm {
                        q.chunks_exact_mut(hd)
                            .for_each(|h| rms_norm_in_place(h, w, c.rms_eps));
                    }
                    if let Some(w) = &weights.k_norm {
                        k.chunks_exact_mut(hd)
                            .for_each(|h| rms_norm_in_place(h, w, c.rms_eps));
                    }
                    let table = &self.rope[i * half_rope..(i + 1) * half_rope];
                    q.chunks_exact_mut(hd)
                        .for_each(|h| rope(h, table, c.rope_style));
                    k.chunks_exact_mut(hd)
                        .for_each(|h| rope(h, table, c.rope_style));
                    let at = kv_slot * layer_stride + (start + i) * kvd;
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
                let k_layer = &self.k_cache[kv_slot * layer_stride..(kv_slot + 1) * layer_stride];
                let v_layer = &self.v_cache[kv_slot * layer_stride..(kv_slot + 1) * layer_stride];
                let q_all = &self.q[..n * qd];
                let context = self.context;
                let attn = OutPtr(self.attn.as_mut_ptr());
                let scores_base = OutPtr(self.scores.as_mut_ptr());
                let blocks = n.div_ceil(query_block);
                let rows_per_worker = score_rows_per_worker(c);
                crate::pool::global().for_each_on(blocks * c.n_head_kv, |worker, task| {
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
                    // Scores live in this worker's slot, not at an index
                    // derived from the token: only `threads` tasks run at
                    // once, so the buffer is sized for that rather than for
                    // the batch (see `score_rows_per_worker`).
                    let row = |i: usize, h: usize| unsafe {
                        let local = (i - i0) * group + (h - kv_head * group);
                        debug_assert!(local < rows_per_worker);
                        std::slice::from_raw_parts_mut(
                            scores_base.at((worker * rows_per_worker + local) * context),
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
                if c.attn_gated {
                    for (a, g) in self.attn[..n * qd].iter_mut().zip(&self.attn_gate) {
                        *a *= sigmoid(*g);
                    }
                }
                self.profiler.lap(3);
                be.matmul(
                    model.matrix(weights.wo),
                    &self.attn[..n * qd],
                    &mut self.proj[..n * d],
                    n,
                )?;
            }
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
            self.profiler.lap(0);
            match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    let ff = c.n_ff;
                    be.matmul_many(
                        &mut [
                            Job {
                                matrix: model.matrix(*gate),
                                output: &mut self.gate[..n * ff],
                            },
                            Job {
                                matrix: model.matrix(*up),
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
                        model.matrix(*down),
                        &self.gate[..n * ff],
                        &mut self.proj[..n * d],
                        n,
                    )?;
                    self.profiler.lap(6);
                }
                Ffn::Routed {
                    router,
                    gate,
                    up,
                    down,
                    shared,
                    shared_gate,
                } => {
                    let moe = c.moe.expect("a routed layer needs expert parameters");
                    let ne = moe.n_expert;
                    let ff = moe.n_ff_exp;
                    // One router matvec covers the whole batch.
                    be.matmul(
                        model.matrix(*router),
                        &self.xn[..n * d],
                        &mut self.router[..n * ne],
                        n,
                    )?;
                    self.profiler.lap(5);
                    self.proj[..n * d].fill(0.0);

                    // More than one token: group the tokens by the expert
                    // they chose, so each expert's weights are read once for
                    // the batch instead of once per token. Prefill is
                    // bandwidth-bound like decode, and this is the whole
                    // difference: at batch 64 with 8 of 128 experts, a layer
                    // reads 64 x 8 experts naively against 128 sorted, so
                    // about four times less.
                    if n > 1 {
                        for list in assign.iter_mut() {
                            list.clear();
                        }
                        for i in 0..n {
                            route(&self.router[i * ne..(i + 1) * ne], &moe, &mut chosen);
                            if router_trace::enabled() {
                                router_trace::record(start + i, li, &chosen);
                            }
                            for &(expert, weight) in chosen.iter() {
                                assign[expert].push((i, weight));
                            }
                        }
                        for (expert, group) in assign.iter().enumerate() {
                            let rows = group.len();
                            if rows == 0 {
                                continue;
                            }
                            for (j, &(i, _)) in group.iter().enumerate() {
                                self.moe_x[j * d..(j + 1) * d]
                                    .copy_from_slice(&self.xn[i * d..(i + 1) * d]);
                            }
                            // Not repacked. An expert is used once per
                            // sorted batch, so repacking its weights eight
                            // rows at a time into a scratch buffer looks
                            // free -- but it is a read and a write over the
                            // slice (about 8 MiB of traffic against 2.65 for
                            // reducing it in place), and a group of ~16
                            // columns cannot amortize that. Measured, it
                            // cost prefill a third: pp256 6.7 and 8.3 tok/s
                            // against 10.7 and 10.3. It would only pay if
                            // the repacked form were cached across batches,
                            // which is a second copy of the model again.
                            be.matmul_many(
                                &mut [
                                    Job {
                                        matrix: model.expert(*gate, expert),
                                        output: &mut self.gate[..rows * ff],
                                    },
                                    Job {
                                        matrix: model.expert(*up, expert),
                                        output: &mut self.up[..rows * ff],
                                    },
                                ],
                                &self.moe_x[..rows * d],
                                rows,
                            )?;
                            kernels::swiglu_in_place(
                                &mut self.gate[..rows * ff],
                                &self.up[..rows * ff],
                            );
                            be.matmul(
                                model.expert(*down, expert),
                                &self.gate[..rows * ff],
                                &mut self.moe_y[..rows * d],
                                rows,
                            )?;
                            for (j, &(i, weight)) in group.iter().enumerate() {
                                for (p, v) in self.proj[i * d..(i + 1) * d]
                                    .iter_mut()
                                    .zip(&self.moe_y[j * d..(j + 1) * d])
                                {
                                    *p += weight * v;
                                }
                            }
                        }
                        // The shared expert runs for every token at once.
                        if let Some([sg, su, sd]) = shared {
                            let sh = moe.n_ff_shared;
                            be.matmul_many(
                                &mut [
                                    Job {
                                        matrix: model.matrix(*sg),
                                        output: &mut self.gate[..n * sh],
                                    },
                                    Job {
                                        matrix: model.matrix(*su),
                                        output: &mut self.up[..n * sh],
                                    },
                                ],
                                &self.xn[..n * d],
                                n,
                            )?;
                            kernels::swiglu_in_place(&mut self.gate[..n * sh], &self.up[..n * sh]);
                            be.matmul(
                                model.matrix(*sd),
                                &self.gate[..n * sh],
                                &mut self.moe_y[..n * d],
                                n,
                            )?;
                            for i in 0..n {
                                let scale = shared_gate.as_ref().map_or(1.0, |g| {
                                    let logit: f32 = g
                                        .iter()
                                        .zip(&self.xn[i * d..(i + 1) * d])
                                        .map(|(w, v)| w * v)
                                        .sum();
                                    1.0 / (1.0 + (-logit).exp())
                                });
                                for (p, v) in self.proj[i * d..(i + 1) * d]
                                    .iter_mut()
                                    .zip(&self.moe_y[i * d..(i + 1) * d])
                                {
                                    *p += scale * v;
                                }
                            }
                        }
                        self.profiler.lap(6);
                        add_assign(&mut self.x[..n * d], &self.proj[..n * d]);
                        continue;
                    }

                    for i in 0..n {
                        route(&self.router[i * ne..(i + 1) * ne], &moe, &mut chosen);
                        if router_trace::enabled() {
                            router_trace::record(start + i, li, &chosen);
                        }
                        for &(expert, weight) in chosen.iter() {
                            be.matmul_many(
                                &mut [
                                    Job {
                                        matrix: model.expert(*gate, expert),
                                        output: &mut self.gate[..ff],
                                    },
                                    Job {
                                        matrix: model.expert(*up, expert),
                                        output: &mut self.up[..ff],
                                    },
                                ],
                                &self.xn[i * d..(i + 1) * d],
                                1,
                            )?;
                            kernels::swiglu_in_place(&mut self.gate[..ff], &self.up[..ff]);
                            be.matmul(
                                model.expert(*down, expert),
                                &self.gate[..ff],
                                &mut self.moe_proj[..d],
                                1,
                            )?;
                            for (p, v) in self.proj[i * d..(i + 1) * d]
                                .iter_mut()
                                .zip(&self.moe_proj[..d])
                            {
                                *p += weight * v;
                            }
                        }
                        // The shared expert runs for every token. Its own
                        // gate, when the model has one, is a sigmoid.
                        if let Some([sg, su, sd]) = shared {
                            let sh = moe.n_ff_shared;
                            be.matmul_many(
                                &mut [
                                    Job {
                                        matrix: model.matrix(*sg),
                                        output: &mut self.gate[..sh],
                                    },
                                    Job {
                                        matrix: model.matrix(*su),
                                        output: &mut self.up[..sh],
                                    },
                                ],
                                &self.xn[i * d..(i + 1) * d],
                                1,
                            )?;
                            kernels::swiglu_in_place(&mut self.gate[..sh], &self.up[..sh]);
                            be.matmul(
                                model.matrix(*sd),
                                &self.gate[..sh],
                                &mut self.moe_proj[..d],
                                1,
                            )?;
                            let mut scale = 1.0;
                            if let Some(g) = shared_gate {
                                let logit: f32 = g
                                    .iter()
                                    .zip(&self.xn[i * d..(i + 1) * d])
                                    .map(|(w, v)| w * v)
                                    .sum();
                                scale = 1.0 / (1.0 + (-logit).exp());
                            }
                            for (p, v) in self.proj[i * d..(i + 1) * d]
                                .iter_mut()
                                .zip(&self.moe_proj[..d])
                            {
                                *p += scale * v;
                            }
                        }
                    }
                    self.profiler.lap(6);
                }
            }
            add_assign(&mut self.x[..n * d], &self.proj[..n * d]);
            if streaming {
                model.stream_done(li);
            }
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
        if model.residency.head_streamed && model.output.packed.is_none() {
            model.file.release(model.output.offset, model.output.len);
        }
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

/// Score rows one worker can have in flight: the widest query block times
/// the query heads sharing a KV head.
///
/// Attention keeps a row of scores per (query token, head) while it
/// softmaxes, and a row is as long as the context. Sizing that by the batch
/// is what made a long context impossible -- 256 tokens by 32 heads by 128k
/// positions is 4.3 GiB of scratch before a weight is read. Only `threads`
/// tasks run at once, so a per-worker slot is enough.
fn score_rows_per_worker(config: &Config) -> usize {
    let group = config.n_head / config.n_head_kv.max(1);
    widest_query_block(config) * group
}

/// The largest block `query_block` can ever return for this model, whatever
/// the batch. `score_rows_per_worker` sizes attention's scratch from this,
/// so the two must not disagree -- including under the override, where a
/// larger block would write past the end of a worker's slot.
fn widest_query_block(config: &Config) -> usize {
    if let Some(forced) = std::env::var("KLLM_QUERY_BLOCK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
    {
        return forced;
    }
    let group = config.n_head / config.n_head_kv.max(1);
    let per_query = 2 * group * config.head_dim * 4;
    (16 * 1024 / per_query.max(1)).max(1)
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
    let by_threads = (tokens * config.n_head_kv / (2 * threads.max(1))).max(1);
    widest_query_block(config)
        .min(by_threads)
        .min(tokens.max(1))
}

/// Optional log of which experts each token routed to.
///
/// This exists to answer the question in docs/MOE.md: how concentrated is
/// expert selection across a single decode stream? That decides whether a
/// 30B MoE can run from disk on a small machine, and it is a property of the
/// data, not of the code -- so it has to be measured rather than assumed.
/// Set `KLLM_ROUTER_TRACE` to a path; each line is
/// `position layer expert,expert,...`.
mod router_trace {
    use std::io::Write;
    use std::sync::{Mutex, OnceLock};

    static SINK: OnceLock<Option<Mutex<std::io::BufWriter<std::fs::File>>>> = OnceLock::new();

    fn sink() -> Option<&'static Mutex<std::io::BufWriter<std::fs::File>>> {
        SINK.get_or_init(|| {
            let path = std::env::var_os("KLLM_ROUTER_TRACE")?;
            match std::fs::File::create(&path) {
                Ok(file) => Some(Mutex::new(std::io::BufWriter::new(file))),
                Err(e) => {
                    eprintln!("kllm: cannot write router trace: {e}");
                    None
                }
            }
        })
        .as_ref()
    }

    pub fn enabled() -> bool {
        sink().is_some()
    }

    pub fn record(position: usize, layer: usize, chosen: &[(usize, f32)]) {
        let Some(sink) = sink() else { return };
        let mut out = sink.lock().unwrap_or_else(|e| e.into_inner());
        let _ = write!(out, "{position} {layer}");
        for (i, (expert, _)) in chosen.iter().enumerate() {
            let _ = write!(out, "{}{expert}", if i == 0 { " " } else { "," });
        }
        let _ = writeln!(out);
    }

    /// Flush on the way out, so a trace is usable even if the run is killed.
    pub fn flush() {
        if let Some(sink) = sink() {
            let _ = sink.lock().unwrap_or_else(|e| e.into_inner()).flush();
        }
    }
}

/// Expert weights for one token: gate the router's logits, take the
/// `n_expert_used` largest, then renormalize and scale as the architecture
/// asks.
///
/// This follows llama.cpp's order exactly, because the order is observable:
/// gating over *all* experts and then selecting is not the same as selecting
/// and then gating unless the weights are renormalized afterwards. Ties go
/// to the lower expert index.
fn route(logits: &[f32], moe: &Moe, out: &mut Vec<(usize, f32)>) {
    out.clear();
    // Gate first, over every expert.
    let probability = |v: f32, max: f32| match moe.gating {
        Gating::Softmax => (v - max).exp(),
        Gating::Sigmoid => 1.0 / (1.0 + (-v).exp()),
    };
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total: f32 = match moe.gating {
        Gating::Softmax => logits.iter().map(|&v| (v - max).exp()).sum(),
        Gating::Sigmoid => 1.0,
    };
    // Then the k largest. Selection is O(k * n_expert) with no allocation,
    // and k is a handful.
    let mut taken = 0_u64;
    let wide = logits.len() > 64;
    for _ in 0..moe.n_expert_used.min(logits.len()) {
        let mut best = usize::MAX;
        let mut best_value = f32::NEG_INFINITY;
        for (e, &v) in logits.iter().enumerate() {
            let already = if wide {
                out.iter().any(|(j, _)| *j == e)
            } else {
                taken & (1 << e) != 0
            };
            if already || v <= best_value {
                continue;
            }
            best_value = v;
            best = e;
        }
        if best == usize::MAX {
            break;
        }
        if !wide {
            taken |= 1 << best;
        }
        out.push((best, probability(best_value, max) / total));
    }
    if moe.weights_norm {
        let sum: f32 = out.iter().map(|(_, w)| *w).sum();
        if sum != 0.0 {
            for (_, w) in out.iter_mut() {
                *w /= sum;
            }
        }
    }
    if moe.weights_scale != 1.0 {
        for (_, w) in out.iter_mut() {
            *w *= moe.weights_scale;
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::excessive_precision)]
    fn deltanet_two_steps_match_matrix_reference() {
        // Independent float64 matrix calculation of decay, delta update,
        // q/sqrt(d) readout and gated RMSNorm. Two K heads and four V heads
        // distinguish GGUF's tiled head layout from repeat_interleave.
        let c = SsmConfig {
            conv_kernel: 2,
            state_size: 2,
            group_count: 2,
            time_step_rank: 4,
            inner_size: 8,
            interval: 4,
        };
        let unused = Weight {
            kind: GgmlType::F32,
            rows: 0,
            cols: 0,
            offset: 0,
            len: 0,
            packed: None,
        };
        let w = Ssm {
            qkv: unused,
            gate: unused,
            out: unused,
            alpha: unused,
            beta: unused,
            conv1d: [0.25, 0.75].repeat(16),
            a: vec![-0.2, -0.4, -0.6, -0.8],
            dt_bias: vec![0.1, -0.1, 0.2, -0.2],
            norm: vec![0.7, 1.3],
        };
        let inputs: [Vec<f32>; 2] = [
            (0..16).map(|i| 0.1 * (i as f32 - 5.0)).collect(),
            (0..16).map(|i| 0.07 * (8.0 - i as f32)).collect(),
        ];
        let expected = [
            [
                -0.13402966,
                0.35352146,
                -0.34338953,
                0.18236619,
                0.07055382,
                0.54209078,
                0.10481338,
                -0.88364250,
            ],
            [
                0.058649996,
                -0.39217433,
                0.37416972,
                -0.16981628,
                -0.06435756,
                -0.56974379,
                -0.10716131,
                0.86273280,
            ],
        ];
        let mut state = SsmState::new(c);
        for (input, expected) in inputs.iter().zip(expected) {
            let mut out = [0.0; 8];
            state.step(
                c,
                &w,
                1e-5,
                input,
                &[-0.7, 0.4, 0.8, -0.3, 0.2, 0.6, -0.4, 0.9],
                &[0.3, -0.5, 0.7, -0.2],
                &[-0.8, 0.2, 0.6, -0.4],
                &mut out,
            );
            for (got, expected) in out.iter().zip(expected) {
                assert!((got - expected).abs() < 2e-6, "{got} vs {expected}");
            }
        }
    }

    fn moe(n_expert: usize, n_expert_used: usize) -> Moe {
        Moe {
            n_expert,
            n_expert_used,
            n_ff_exp: 8,
            n_ff_shared: 0,
            weights_norm: true,
            weights_scale: 1.0,
            gating: Gating::Softmax,
        }
    }

    #[test]
    fn router_picks_the_largest_logits() {
        let mut out = Vec::new();
        route(&[0.1, 3.0, -1.0, 2.0, 0.5], &moe(5, 2), &mut out);
        assert_eq!(out.iter().map(|(e, _)| *e).collect::<Vec<_>>(), [1, 3]);
        // Normalized, so the two weights sum to one.
        let sum: f32 = out.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 1e-6, "weights sum to {sum}");
        // And they are softmax over just those two, because renormalizing
        // cancels the full normalizer: exp(3)/(exp(3)+exp(2)).
        let expected = 1.0 / (1.0 + (2.0_f32 - 3.0).exp());
        assert!(
            (out[0].1 - expected).abs() < 1e-6,
            "{} vs {expected}",
            out[0].1
        );
    }

    /// Selection must be stable where logits tie, or two engines disagree on
    /// which expert ran.
    #[test]
    fn router_breaks_ties_toward_the_lower_index() {
        let mut out = Vec::new();
        route(&[1.0, 5.0, 5.0, 5.0], &moe(4, 2), &mut out);
        assert_eq!(out.iter().map(|(e, _)| *e).collect::<Vec<_>>(), [1, 2]);
    }

    #[test]
    fn router_handles_more_than_sixty_four_experts() {
        // Past 64 the "already taken" test cannot be a u64 mask.
        let mut logits = vec![0.0_f32; 128];
        logits[100] = 9.0;
        logits[7] = 8.0;
        let mut out = Vec::new();
        route(&logits, &moe(128, 3), &mut out);
        let picked: Vec<usize> = out.iter().map(|(e, _)| *e).collect();
        assert_eq!(picked[..2], [100, 7]);
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|(_, w)| w.is_finite() && *w >= 0.0));
    }

    #[test]
    fn router_applies_sigmoid_gating_and_scale() {
        let mut config = moe(4, 2);
        config.gating = Gating::Sigmoid;
        config.weights_norm = false;
        config.weights_scale = 2.0;
        let mut out = Vec::new();
        route(&[0.0, 1.0, -1.0, 2.0], &config, &mut out);
        assert_eq!(out.iter().map(|(e, _)| *e).collect::<Vec<_>>(), [3, 1]);
        let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
        assert!((out[0].1 - 2.0 * sigmoid(2.0)).abs() < 1e-6);
        assert!((out[1].1 - 2.0 * sigmoid(1.0)).abs() < 1e-6);
    }

    /// Asking for every expert must still produce a valid distribution.
    #[test]
    fn router_can_select_all_experts() {
        let mut out = Vec::new();
        route(&[1.0, 2.0, 3.0], &moe(3, 3), &mut out);
        assert_eq!(out.len(), 3);
        let sum: f32 = out.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
