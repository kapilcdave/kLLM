//! GPU-resident layers (milestone M6).
//!
//! [`Model::offload`] uploads the last `n` transformer blocks, and optionally
//! the output head, to the GPU. A forward pass then runs the CPU blocks,
//! copies the hidden state to the device once, runs the GPU blocks there
//! (their KV cache lives in VRAM), and copies back either the logits (head on
//! the GPU) or the hidden state. Offloading a *suffix* keeps the GPU blocks
//! adjacent to the head, so a token crosses the bus once each way.

use std::sync::Arc;

use super::{Config, Layer, Model, ModelError, Rows, Session};
use crate::cuda::{CudaError, DeviceBuffer, Gpu, GpuMatrix};

impl From<CudaError> for ModelError {
    fn from(e: CudaError) -> Self {
        ModelError::Gpu(e.to_string())
    }
}

struct GpuLayer {
    attn_norm: DeviceBuffer,
    mixer: GpuMixer,
    ffn_norm: DeviceBuffer,
    w_gate: GpuMatrix,
    w_up: GpuMatrix,
    w_down: GpuMatrix,
}

enum GpuMixer {
    Attention {
        wq: GpuMatrix,
        wk: GpuMatrix,
        wv: GpuMatrix,
        wo: GpuMatrix,
        bq: Option<DeviceBuffer>,
        bk: Option<DeviceBuffer>,
        bv: Option<DeviceBuffer>,
        q_norm: Option<DeviceBuffer>,
        k_norm: Option<DeviceBuffer>,
        /// Compact slot among the offloaded attention blocks.
        kv_slot: usize,
    },
    Ssm {
        qkv: GpuMatrix,
        gate: GpuMatrix,
        out: GpuMatrix,
        conv1d: DeviceBuffer,
        a: DeviceBuffer,
        dt_bias: DeviceBuffer,
        alpha: GpuMatrix,
        beta: GpuMatrix,
        norm: DeviceBuffer,
        /// Compact slot among the offloaded recurrent blocks.
        state_slot: usize,
    },
}

pub(super) struct GpuStack {
    pub(super) gpu: Arc<Gpu>,
    /// Index of the first GPU-resident block; blocks before it run on the CPU.
    pub(super) first: usize,
    layers: Vec<GpuLayer>,
    kv_layers: usize,
    ssm_layers: usize,
    head: Option<(DeviceBuffer, GpuMatrix)>,
    /// Device bytes held by weights.
    pub(super) bytes: u64,
}

/// Per-session device state: the GPU blocks' KV cache and scratch buffers.
pub(super) struct GpuState {
    k_cache: DeviceBuffer,
    v_cache: DeviceBuffer,
    x: DeviceBuffer,
    xn: DeviceBuffer,
    q: DeviceBuffer,
    q_full: DeviceBuffer,
    attn_gate: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    attn: DeviceBuffer,
    proj: DeviceBuffer,
    gate: DeviceBuffer,
    up: DeviceBuffer,
    scores: DeviceBuffer,
    rope: DeviceBuffer,
    ssm_qkv: DeviceBuffer,
    ssm_z: DeviceBuffer,
    ssm_alpha: DeviceBuffer,
    ssm_beta: DeviceBuffer,
    ssm_out: DeviceBuffer,
    ssm_conv: DeviceBuffer,
    ssm_matrix: DeviceBuffer,
    reset_ssm: bool,
    logits: Option<DeviceBuffer>,
    rope_host: Vec<f32>,
}

/// What an offload placed where.
#[derive(Debug, Clone, Copy)]
pub struct Offload {
    pub gpu_layers: usize,
    pub head: bool,
    pub weight_bytes: u64,
}

impl Model {
    /// Move the last `layers` blocks (and the output head, if `head`) to
    /// `gpu`. Replaces any earlier offload. `layers == 0 && !head` returns the
    /// model to pure CPU execution.
    pub fn offload(
        &mut self,
        gpu: Arc<Gpu>,
        layers: usize,
        head: bool,
    ) -> Result<Offload, ModelError> {
        self.gpu = None; // free the previous upload first
        let c = &self.config;
        let layers = layers.min(c.n_layer);
        if layers == 0 && !head {
            return Ok(Offload {
                gpu_layers: 0,
                head: false,
                weight_bytes: 0,
            });
        }
        if c.head_dim > 256 {
            return Err(ModelError::Unsupported(format!(
                "GPU attention needs head_dim <= 256, model has {}",
                c.head_dim
            )));
        }
        let first = c.n_layer - layers;
        let mut bytes = 0u64;
        let mut matrix = |w: super::Weight| -> Result<GpuMatrix, ModelError> {
            if !gpu.supports(w.kind) {
                return Err(ModelError::Unsupported(format!(
                    "no GPU kernel for {}",
                    w.kind.name()
                )));
            }
            let m = self.matrix(w);
            let up = gpu.upload_matrix(m.kind, m.data, m.rows, m.cols)?;
            bytes += up.buffer.bytes as u64;
            Ok(up)
        };
        let mut uploaded = Vec::with_capacity(layers);
        let (mut kv_layers, mut ssm_layers) = (0, 0);
        for layer in &self.layers[first..] {
            let kv_slot = kv_layers;
            let state_slot = ssm_layers;
            uploaded.push(upload_layer(&gpu, layer, &mut matrix, kv_slot, state_slot)?);
            match layer.mixer {
                super::Mixer::Attention(_) => kv_layers += 1,
                super::Mixer::Ssm(_) => ssm_layers += 1,
            }
        }
        let head = if head {
            Some((gpu.upload(&self.output_norm)?, matrix(self.output)?))
        } else {
            None
        };
        let has_head = head.is_some();
        self.gpu = Some(GpuStack {
            gpu,
            first,
            layers: uploaded,
            kv_layers,
            ssm_layers,
            head,
            bytes,
        });
        Ok(Offload {
            gpu_layers: layers,
            head: has_head,
            weight_bytes: bytes,
        })
    }

    /// Blocks and head currently on the GPU.
    pub fn offloaded(&self) -> Offload {
        match &self.gpu {
            Some(s) => Offload {
                gpu_layers: s.layers.len(),
                head: s.head.is_some(),
                weight_bytes: s.bytes,
            },
            None => Offload {
                gpu_layers: 0,
                head: false,
                weight_bytes: 0,
            },
        }
    }
}

fn upload_layer(
    gpu: &Arc<Gpu>,
    layer: &Layer,
    matrix: &mut impl FnMut(super::Weight) -> Result<GpuMatrix, ModelError>,
    kv_slot: usize,
    state_slot: usize,
) -> Result<GpuLayer, ModelError> {
    let vector = |v: &[f32]| gpu.upload(v);
    let optional = |v: &Option<Vec<f32>>| v.as_deref().map(|v| gpu.upload(v)).transpose();
    let mixer = match &layer.mixer {
        super::Mixer::Attention(attn) => GpuMixer::Attention {
            wq: matrix(attn.wq)?,
            wk: matrix(attn.wk)?,
            wv: matrix(attn.wv)?,
            wo: matrix(attn.wo)?,
            bq: optional(&attn.bq)?,
            bk: optional(&attn.bk)?,
            bv: optional(&attn.bv)?,
            q_norm: optional(&attn.q_norm)?,
            k_norm: optional(&attn.k_norm)?,
            kv_slot,
        },
        super::Mixer::Ssm(ssm) => GpuMixer::Ssm {
            qkv: matrix(ssm.qkv)?,
            gate: matrix(ssm.gate)?,
            out: matrix(ssm.out)?,
            conv1d: vector(&ssm.conv1d)?,
            a: vector(&ssm.a)?,
            dt_bias: vector(&ssm.dt_bias)?,
            alpha: matrix(ssm.alpha)?,
            beta: matrix(ssm.beta)?,
            norm: vector(&ssm.norm)?,
            state_slot,
        },
    };
    Ok(GpuLayer {
        attn_norm: vector(&layer.attn_norm)?,
        mixer,
        ffn_norm: vector(&layer.ffn_norm)?,
        // The GPU stack is dense-only for now: a routed layer would need
        // the router, per-expert slicing and a gather on the device.
        w_gate: match &layer.ffn {
            super::Ffn::Dense { gate, .. } => matrix(*gate)?,
            super::Ffn::Routed { .. } => {
                return Err(ModelError::Unsupported(
                    "mixture-of-experts layers on the GPU".into(),
                ));
            }
        },
        w_up: match &layer.ffn {
            super::Ffn::Dense { up, .. } => matrix(*up)?,
            super::Ffn::Routed { .. } => unreachable!("rejected above"),
        },
        w_down: match &layer.ffn {
            super::Ffn::Dense { down, .. } => matrix(*down)?,
            super::Ffn::Routed { .. } => unreachable!("rejected above"),
        },
    })
}

impl GpuState {
    pub(super) fn new(
        stack: &GpuStack,
        c: &Config,
        context: usize,
        batch: usize,
    ) -> Result<Self, ModelError> {
        let g = &stack.gpu;
        let f = |n: usize| g.alloc(n * 4);
        let cache = stack.kv_layers * context * c.kv_dim() * 2;
        let sc = c.ssm;
        let ssm_state = sc.map_or(0, |s| s.time_step_rank * s.state_size * s.state_size);
        let ssm_conv = sc.map_or(0, |s| s.conv_width() * (s.conv_kernel - 1));
        let state = Self {
            k_cache: g.alloc(cache)?,
            v_cache: g.alloc(cache)?,
            x: f(batch * c.n_embd)?,
            xn: f(batch * c.n_embd)?,
            q: f(batch * c.q_dim())?,
            q_full: f(if c.attn_gated {
                batch * 2 * c.q_dim()
            } else {
                0
            })?,
            attn_gate: f(if c.attn_gated { batch * c.q_dim() } else { 0 })?,
            k: f(batch * c.kv_dim())?,
            v: f(batch * c.kv_dim())?,
            attn: f(batch * c.q_dim())?,
            proj: f(batch * c.n_embd)?,
            gate: f(batch * c.n_ff)?,
            up: f(batch * c.n_ff)?,
            scores: f(batch * c.n_head * context)?,
            rope: f(batch * c.rope_dim)?,
            ssm_qkv: f(batch * sc.map_or(0, |s| s.conv_width()))?,
            ssm_z: f(batch * sc.map_or(0, |s| s.inner_size))?,
            ssm_alpha: f(batch * sc.map_or(0, |s| s.time_step_rank))?,
            ssm_beta: f(batch * sc.map_or(0, |s| s.time_step_rank))?,
            ssm_out: f(batch * sc.map_or(0, |s| s.inner_size))?,
            ssm_conv: f(stack.ssm_layers * ssm_conv)?,
            ssm_matrix: f(stack.ssm_layers * ssm_state)?,
            reset_ssm: false,
            logits: if stack.head.is_some() {
                Some(f(batch * c.n_vocab)?)
            } else {
                None
            },
            rope_host: Vec::with_capacity(batch * c.rope_dim),
        };
        state.ssm_conv.zero()?;
        state.ssm_matrix.zero()?;
        Ok(state)
    }

    pub(super) fn has_ssm(&self) -> bool {
        self.ssm_matrix.bytes != 0
    }

    pub(super) fn reset_ssm_on_next_forward(&mut self) {
        self.reset_ssm = true;
    }
}

impl Session<'_> {
    /// Run the GPU blocks (and head) for a batch whose CPU blocks are done:
    /// `self.x` holds their output. Returns true if the logits are already
    /// in `self.logits` (head on the GPU).
    pub(super) fn forward_gpu(
        &mut self,
        n: usize,
        start: usize,
        rows: Rows,
    ) -> Result<bool, ModelError> {
        let model = self.model;
        let (Some(stack), Some(st)) = (&model.gpu, &mut self.gpu) else {
            return Ok(false);
        };
        let g = &stack.gpu;
        let c = &model.config;
        let (d, qd, kvd, hd, ff) = (c.n_embd, c.q_dim(), c.kv_dim(), c.head_dim, c.n_ff);
        let neox = c.rope_style == super::RopeStyle::Neox;
        if st.reset_ssm {
            st.ssm_conv.zero()?;
            st.ssm_matrix.zero()?;
            st.reset_ssm = false;
        }
        st.x.write(&self.x[..n * d])?;
        st.rope_host.clear();
        for &(cos, sin) in &self.rope[..n * c.rope_dim / 2] {
            st.rope_host.extend([cos, sin]);
        }
        st.rope.write(&st.rope_host)?;
        let scale = 1.0 / (hd as f32).sqrt();
        let layer_cache = self.context * kvd * 2;
        for l in &stack.layers {
            g.rms_norm(st.x.ptr, l.attn_norm.ptr, st.xn.ptr, n, d, c.rms_eps)?;
            match &l.mixer {
                GpuMixer::Attention {
                    wq,
                    wk,
                    wv,
                    wo,
                    bq,
                    bk,
                    bv,
                    q_norm,
                    k_norm,
                    kv_slot,
                } => {
                    if c.attn_gated {
                        g.matmul_matrix(wq, st.xn.ptr, st.q_full.ptr, n)?;
                        g.split_q_gate(st.q_full.ptr, st.q.ptr, st.attn_gate.ptr, hd, n * qd)?;
                    } else {
                        g.matmul_matrix(wq, st.xn.ptr, st.q.ptr, n)?;
                    }
                    g.matmul_matrix(wk, st.xn.ptr, st.k.ptr, n)?;
                    g.matmul_matrix(wv, st.xn.ptr, st.v.ptr, n)?;
                    for (bias, buf, width) in [(bq, &st.q, qd), (bk, &st.k, kvd), (bv, &st.v, kvd)]
                    {
                        if let Some(b) = bias {
                            g.add_bias(buf.ptr, b.ptr, width, n)?;
                        }
                    }
                    if let Some(w) = q_norm {
                        g.rms_norm(st.q.ptr, w.ptr, st.q.ptr, n * c.n_head, hd, c.rms_eps)?;
                    }
                    if let Some(w) = k_norm {
                        g.rms_norm(st.k.ptr, w.ptr, st.k.ptr, n * c.n_head_kv, hd, c.rms_eps)?;
                    }
                    g.rope(st.q.ptr, st.rope.ptr, n, c.n_head, hd, c.rope_dim, neox)?;
                    g.rope(st.k.ptr, st.rope.ptr, n, c.n_head_kv, hd, c.rope_dim, neox)?;
                    let (kc, vc) = (
                        st.k_cache.at(kv_slot * layer_cache),
                        st.v_cache.at(kv_slot * layer_cache),
                    );
                    g.kv_store(st.k.ptr, st.v.ptr, kc, vc, kvd, start, n)?;
                    g.attention(
                        st.q.ptr,
                        kc,
                        vc,
                        st.attn.ptr,
                        st.scores.ptr,
                        n,
                        c.n_head,
                        c.n_head_kv,
                        hd,
                        start,
                        self.context,
                        scale,
                    )?;
                    if c.attn_gated {
                        g.sigmoid_mul(st.attn.ptr, st.attn_gate.ptr, n * qd)?;
                    }
                    g.matmul_matrix(wo, st.attn.ptr, st.proj.ptr, n)?;
                }
                GpuMixer::Ssm {
                    qkv,
                    gate,
                    out,
                    conv1d,
                    a,
                    dt_bias,
                    alpha,
                    beta,
                    norm,
                    state_slot,
                } => {
                    let sc = c.ssm.expect("an SSM GPU layer needs its config");
                    let width = sc.conv_width();
                    let shd = sc.state_size;
                    g.matmul_matrix(qkv, st.xn.ptr, st.ssm_qkv.ptr, n)?;
                    g.matmul_matrix(gate, st.xn.ptr, st.ssm_z.ptr, n)?;
                    g.matmul_matrix(alpha, st.xn.ptr, st.ssm_alpha.ptr, n)?;
                    g.matmul_matrix(beta, st.xn.ptr, st.ssm_beta.ptr, n)?;
                    let conv_stride = width * (sc.conv_kernel - 1) * 4;
                    let matrix_stride = sc.time_step_rank * shd * shd * 4;
                    let conv = st.ssm_conv.at(state_slot * conv_stride);
                    let matrix = st.ssm_matrix.at(state_slot * matrix_stride);
                    for i in 0..n {
                        let mixed = st.ssm_qkv.at(i * width * 4);
                        let z = st.ssm_z.at(i * sc.inner_size * 4);
                        let alpha_row = st.ssm_alpha.at(i * sc.time_step_rank * 4);
                        let beta_row = st.ssm_beta.at(i * sc.time_step_rank * 4);
                        let out_row = st.ssm_out.at(i * sc.inner_size * 4);
                        g.conv_silu(mixed, conv, conv1d.ptr, width, sc.conv_kernel)?;
                        g.l2_norm_heads(mixed, 2 * sc.group_count, shd, c.rms_eps)?;
                        g.deltanet_step(
                            mixed,
                            z,
                            alpha_row,
                            beta_row,
                            a.ptr,
                            dt_bias.ptr,
                            norm.ptr,
                            matrix,
                            out_row,
                            sc.group_count,
                            sc.time_step_rank,
                            shd,
                            c.rms_eps,
                        )?;
                    }
                    g.matmul_matrix(out, st.ssm_out.ptr, st.proj.ptr, n)?;
                }
            }
            g.add_inplace(st.x.ptr, st.proj.ptr, n * d)?;
            g.rms_norm(st.x.ptr, l.ffn_norm.ptr, st.xn.ptr, n, d, c.rms_eps)?;
            g.matmul_matrix(&l.w_gate, st.xn.ptr, st.gate.ptr, n)?;
            g.matmul_matrix(&l.w_up, st.xn.ptr, st.up.ptr, n)?;
            g.swiglu(st.gate.ptr, st.up.ptr, n * ff)?;
            g.matmul_matrix(&l.w_down, st.gate.ptr, st.proj.ptr, n)?;
            g.add_inplace(st.x.ptr, st.proj.ptr, n * d)?;
        }
        let first = match rows {
            Rows::None => return Ok(true),
            Rows::Last => n - 1,
            Rows::All => 0,
        };
        let count = n - first;
        match (&stack.head, &st.logits) {
            (Some((norm, head)), Some(logits)) => {
                let x = st.x.at(first * d * 4);
                g.rms_norm(x, norm.ptr, st.xn.ptr, count, d, c.rms_eps)?;
                g.matmul_matrix(head, st.xn.ptr, logits.ptr, count)?;
                logits.read(&mut self.logits[..count * c.n_vocab])?;
                Ok(true)
            }
            _ => {
                st.x.read(&mut self.x[..n * d])?;
                Ok(false)
            }
        }
    }
}
