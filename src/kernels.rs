//! Dot-product kernels for batch-one decoding.
//!
//! Like llama.cpp, quantized weights are not dequantized to f32 for the dot
//! product. Instead the activation vector is quantized once per matvec to an
//! 8-bit companion format (`Q8_0` for the 32-element block types, `Q8_K` for
//! the K-quants) and each weight row is reduced with integer multiply-adds.
//!
//! Every SIMD kernel has a portable scalar twin with identical integer
//! arithmetic; the tests check that the two agree and that both agree with
//! the f32 reference in [`crate::quant`].

use crate::quant::{GgmlType, QK_K, bf16_to_f32, dot_row_reference, f16_to_f32, scale_min_k4};

#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockQ8_0 {
    pub d: f32,
    /// `d * sum(qs)`, needed by the types with a minimum (Q4_1, Q5_1); this
    /// is ggml's `block_q8_1.s`.
    pub s: f32,
    pub qs: [i8; 32],
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; QK_K],
    pub bsums: [i16; 16],
}

/// Which activation format a weight type is reduced against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DotInput {
    F32,
    Q8_0,
    Q8K,
}

pub fn dot_input(kind: GgmlType) -> DotInput {
    match kind {
        GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q8_0 => {
            DotInput::Q8_0
        }
        GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K => DotInput::Q8K,
        _ => DotInput::F32,
    }
}

/// An activation vector prepared for one weight type.
pub enum Prepared<'a> {
    F32(&'a [f32]),
    Q8_0(Vec<BlockQ8_0>),
    Q8K(Vec<BlockQ8K>),
}

pub fn prepare<'a>(kind: GgmlType, x: &'a [f32]) -> Prepared<'a> {
    match dot_input(kind) {
        DotInput::F32 => Prepared::F32(x),
        DotInput::Q8_0 => Prepared::Q8_0(quantize_q8_0(x)),
        DotInput::Q8K => Prepared::Q8K(quantize_q8k(x)),
    }
}

#[inline]
fn half(bytes: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
}

pub fn quantize_q8_0(x: &[f32]) -> Vec<BlockQ8_0> {
    debug_assert_eq!(x.len() % 32, 0);
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        let mut out = Vec::with_capacity(x.len() / 32);
        // SAFETY: features detected; `x` is a whole number of 32-float
        // blocks and `out` has room for every block.
        unsafe {
            avx2::quantize_q8_0_into(x, out.spare_capacity_mut());
            out.set_len(x.len() / 32);
        }
        return out;
    }
    quantize_q8_0_scalar(x)
}

/// The portable reference the SIMD path must match value for value.
pub fn quantize_q8_0_scalar(x: &[f32]) -> Vec<BlockQ8_0> {
    debug_assert_eq!(x.len() % 32, 0);
    x.chunks_exact(32)
        .map(|chunk| {
            let amax = chunk.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
            let mut qs = [0_i8; 32];
            for (q, v) in qs.iter_mut().zip(chunk) {
                *q = (v * id).round_ties_even() as i8;
            }
            // ggml stores this scale as f16; round it the same way so both
            // engines see the same effective activations.
            let d = f16_to_f32(crate::quant::f32_to_f16(d));
            let sum: i32 = qs.iter().map(|&q| q as i32).sum();
            BlockQ8_0 {
                d,
                s: d * sum as f32,
                qs,
            }
        })
        .collect()
}

pub fn quantize_q8k(x: &[f32]) -> Vec<BlockQ8K> {
    debug_assert_eq!(x.len() % QK_K, 0);
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        let mut out = Vec::with_capacity(x.len() / QK_K);
        // SAFETY: features detected; `x` is a whole number of QK_K blocks
        // and `out` has room for every block.
        unsafe {
            avx2::quantize_q8k_into(x, out.spare_capacity_mut());
            out.set_len(x.len() / QK_K);
        }
        return out;
    }
    quantize_q8k_scalar(x)
}

/// The portable reference the SIMD path must match value for value.
pub fn quantize_q8k_scalar(x: &[f32]) -> Vec<BlockQ8K> {
    debug_assert_eq!(x.len() % QK_K, 0);
    x.chunks_exact(QK_K)
        .map(|chunk| {
            let mut max = 0.0_f32;
            let mut amax = 0.0_f32;
            for &v in chunk {
                if v.abs() > amax {
                    amax = v.abs();
                    max = v;
                }
            }
            let mut block = BlockQ8K {
                d: 0.0,
                qs: [0; QK_K],
                bsums: [0; 16],
            };
            if amax == 0.0 {
                return block;
            }
            let iscale = -127.0 / max;
            for (q, v) in block.qs.iter_mut().zip(chunk) {
                *q = ((iscale * v).round() as i32).min(127) as i8;
            }
            for (sum, group) in block.bsums.iter_mut().zip(block.qs.chunks_exact(16)) {
                *sum = group.iter().map(|&q| q as i16).sum();
            }
            block.d = 1.0 / iscale;
            block
        })
        .collect()
}

#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    use std::sync::OnceLock;
    static AVX2: OnceLock<bool> = OnceLock::new();
    *AVX2.get_or_init(|| {
        std::env::var_os("KLLM_NO_SIMD").is_none()
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
    })
}

/// Name of the kernel set in use, for logs and benchmarks.
pub fn kernel_name() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        return "avx2+fma+f16c";
    }
    "scalar"
}

/// Dot product of one stored row with a prepared activation vector.
pub fn dot(kind: GgmlType, row: &[u8], x: &Prepared<'_>) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: the required CPU features were detected at runtime.
        unsafe {
            match (kind, x) {
                (GgmlType::F32, Prepared::F32(x)) => return avx2::dot_f32(row, x),
                (GgmlType::F16, Prepared::F32(x)) => return avx2::dot_f16(row, x),
                (GgmlType::BF16, Prepared::F32(x)) => return avx2::dot_bf16(row, x),
                (GgmlType::Q4_0, Prepared::Q8_0(x)) => return avx2::q4_0_n(row, [x])[0],
                (GgmlType::Q8_0, Prepared::Q8_0(x)) => return avx2::q8_0_n(row, [x])[0],
                (GgmlType::Q4_1, Prepared::Q8_0(x)) => return avx2::q4_1_n(row, [x])[0],
                (GgmlType::Q5_0, Prepared::Q8_0(x)) => return avx2::q5_0_n(row, [x])[0],
                (GgmlType::Q5_1, Prepared::Q8_0(x)) => return avx2::q5_1_n(row, [x])[0],
                (GgmlType::Q4K, Prepared::Q8K(x)) => return avx2::q4k_n(row, [x])[0],
                (GgmlType::Q5K, Prepared::Q8K(x)) => return avx2::dot_q5k(row, x),
                (GgmlType::Q6K, Prepared::Q8K(x)) => return avx2::q6k_n(row, [x])[0],
                _ => {}
            }
        }
    }
    dot_scalar(kind, row, x)
}

/// Dot one stored row with several prepared columns: `out[j] = row · xs[j]`.
/// Each weight block is unpacked once per group of up to four columns.
/// `scratch` holds a dequantized row for the float types.
pub fn dot_multi(
    kind: GgmlType,
    row: &[u8],
    xs: &[Prepared<'_>],
    out: &mut [f32],
    scratch: &mut Vec<f32>,
) {
    debug_assert_eq!(xs.len(), out.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: features detected; every prepared column matches `kind`
        // (checked by the pattern match) and has one block per weight block.
        unsafe {
            if dot_input(kind) == DotInput::F32 && xs.len() == 1 {
                out[0] = dot(kind, row, &xs[0]);
                return;
            }
            if dot_input(kind) == DotInput::F32 {
                let cols = xs.first().map_or(0, |x| match x {
                    Prepared::F32(v) => v.len(),
                    _ => 0,
                });
                scratch.resize(cols, 0.0);
                match kind {
                    GgmlType::F16 => load_f16(row, scratch),
                    GgmlType::BF16 => load_bf16(row, scratch),
                    _ => crate::quant::dequantize_row(kind, row, scratch),
                }
                for (o, x) in out.iter_mut().zip(xs) {
                    let Prepared::F32(x) = x else { unreachable!() };
                    *o = avx2::dot_f32_f32(scratch, x);
                }
                return;
            }
            macro_rules! groups {
                ($variant:ident, $kernel:ident) => {{
                    // No allocation here: this runs once per weight row.
                    let col = |j: usize| match &xs[j] {
                        Prepared::$variant(v) => v.as_slice(),
                        _ => unreachable!("activation prepared for the wrong weight type"),
                    };
                    let mut j = 0;
                    while j + 8 <= xs.len() {
                        let r = avx2::$kernel(
                            row,
                            [
                                col(j),
                                col(j + 1),
                                col(j + 2),
                                col(j + 3),
                                col(j + 4),
                                col(j + 5),
                                col(j + 6),
                                col(j + 7),
                            ],
                        );
                        out[j..j + 8].copy_from_slice(&r);
                        j += 8;
                    }
                    while j + 4 <= xs.len() {
                        let r = avx2::$kernel(row, [col(j), col(j + 1), col(j + 2), col(j + 3)]);
                        out[j..j + 4].copy_from_slice(&r);
                        j += 4;
                    }
                    while j < xs.len() {
                        out[j] = avx2::$kernel(row, [col(j)])[0];
                        j += 1;
                    }
                    return;
                }};
            }
            match kind {
                GgmlType::Q4_0 => groups!(Q8_0, q4_0_n),
                GgmlType::Q8_0 => groups!(Q8_0, q8_0_n),
                GgmlType::Q4_1 => groups!(Q8_0, q4_1_n),
                GgmlType::Q5_0 => groups!(Q8_0, q5_0_n),
                GgmlType::Q5_1 => groups!(Q8_0, q5_1_n),
                GgmlType::Q4K => groups!(Q8K, q4k_n),
                GgmlType::Q6K => groups!(Q8K, q6k_n),
                _ => {}
            }
        }
    }
    for (o, x) in out.iter_mut().zip(xs) {
        *o = dot(kind, row, x);
    }
}

/// Portable kernels with the same integer arithmetic as the SIMD ones.
pub fn dot_scalar(kind: GgmlType, row: &[u8], x: &Prepared<'_>) -> f32 {
    match (kind, x) {
        (GgmlType::F32, Prepared::F32(x)) => row
            .chunks_exact(4)
            .zip(*x)
            .map(|(b, v)| f32::from_le_bytes(b.try_into().unwrap()) * v)
            .sum(),
        (GgmlType::F16, Prepared::F32(x)) => row
            .chunks_exact(2)
            .zip(*x)
            .map(|(b, v)| f16_to_f32(u16::from_le_bytes([b[0], b[1]])) * v)
            .sum(),
        (GgmlType::BF16, Prepared::F32(x)) => row
            .chunks_exact(2)
            .zip(*x)
            .map(|(b, v)| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])) * v)
            .sum(),
        (GgmlType::Q4_0, Prepared::Q8_0(x)) => row
            .chunks_exact(18)
            .zip(x)
            .map(|(w, a)| {
                let mut sum = 0_i32;
                for j in 0..16 {
                    sum += ((w[2 + j] & 0x0f) as i32 - 8) * a.qs[j] as i32;
                    sum += ((w[2 + j] >> 4) as i32 - 8) * a.qs[j + 16] as i32;
                }
                sum as f32 * half(w, 0) * a.d
            })
            .sum(),
        (GgmlType::Q4_1, Prepared::Q8_0(x)) => row
            .chunks_exact(20)
            .zip(x)
            .map(|(w, a)| {
                let mut sum = 0_i32;
                for j in 0..16 {
                    sum += (w[4 + j] & 0x0f) as i32 * a.qs[j] as i32;
                    sum += (w[4 + j] >> 4) as i32 * a.qs[j + 16] as i32;
                }
                half(w, 0) * a.d * sum as f32 + half(w, 2) * a.s
            })
            .sum(),
        (GgmlType::Q5_0, Prepared::Q8_0(x)) => row
            .chunks_exact(22)
            .zip(x)
            .map(|(w, a)| {
                let qh = u32::from_le_bytes(w[2..6].try_into().unwrap());
                let mut sum = 0_i32;
                for j in 0..16 {
                    let lo = ((w[6 + j] & 0x0f) as u32 | (((qh >> j) & 1) << 4)) as i32 - 16;
                    let hi = ((w[6 + j] >> 4) as u32 | (((qh >> (j + 16)) & 1) << 4)) as i32 - 16;
                    sum += lo * a.qs[j] as i32 + hi * a.qs[j + 16] as i32;
                }
                half(w, 0) * a.d * sum as f32
            })
            .sum(),
        (GgmlType::Q5_1, Prepared::Q8_0(x)) => row
            .chunks_exact(24)
            .zip(x)
            .map(|(w, a)| {
                let qh = u32::from_le_bytes(w[4..8].try_into().unwrap());
                let mut sum = 0_i32;
                for j in 0..16 {
                    let lo = ((w[8 + j] & 0x0f) as u32 | (((qh >> j) & 1) << 4)) as i32;
                    let hi = ((w[8 + j] >> 4) as u32 | (((qh >> (j + 16)) & 1) << 4)) as i32;
                    sum += lo * a.qs[j] as i32 + hi * a.qs[j + 16] as i32;
                }
                half(w, 0) * a.d * sum as f32 + half(w, 2) * a.s
            })
            .sum(),
        (GgmlType::Q8_0, Prepared::Q8_0(x)) => row
            .chunks_exact(34)
            .zip(x)
            .map(|(w, a)| {
                let sum: i32 = w[2..34]
                    .iter()
                    .zip(&a.qs)
                    .map(|(&q, &y)| (q as i8) as i32 * y as i32)
                    .sum();
                sum as f32 * half(w, 0) * a.d
            })
            .sum(),
        (GgmlType::Q4K, Prepared::Q8K(x)) => row
            .chunks_exact(144)
            .zip(x)
            .map(|(w, a)| {
                let scales = &w[4..16];
                let qs = &w[16..144];
                let mut sumi = 0_i32;
                let mut summ = 0_i32;
                for j in 0..8 {
                    let (sc, m) = scale_min_k4(j, scales);
                    let q = &qs[(j / 2) * 32..(j / 2) * 32 + 32];
                    let y = &a.qs[j * 32..j * 32 + 32];
                    let mut s = 0_i32;
                    for l in 0..32 {
                        let v = if j % 2 == 0 { q[l] & 0x0f } else { q[l] >> 4 };
                        s += v as i32 * y[l] as i32;
                    }
                    sumi += s * sc as i32;
                    summ += m as i32 * (a.bsums[2 * j] as i32 + a.bsums[2 * j + 1] as i32);
                }
                a.d * (half(w, 0) * sumi as f32 - half(w, 2) * summ as f32)
            })
            .sum(),
        (GgmlType::Q5K, Prepared::Q8K(x)) => row
            .chunks_exact(176)
            .zip(x)
            .map(|(w, a)| {
                let scales = &w[4..16];
                let qh = &w[16..48];
                let qs = &w[48..176];
                let mut sumi = 0_i32;
                let mut summ = 0_i32;
                for j in 0..8 {
                    let (sc, m) = scale_min_k4(j, scales);
                    let q = &qs[(j / 2) * 32..(j / 2) * 32 + 32];
                    let y = &a.qs[j * 32..j * 32 + 32];
                    let mut s = 0_i32;
                    for l in 0..32 {
                        let low = if j % 2 == 0 { q[l] & 0x0f } else { q[l] >> 4 };
                        let high = (qh[l] >> j) & 1;
                        s += (low | (high << 4)) as i32 * y[l] as i32;
                    }
                    sumi += s * sc as i32;
                    summ += m as i32 * (a.bsums[2 * j] as i32 + a.bsums[2 * j + 1] as i32);
                }
                a.d * (half(w, 0) * sumi as f32 - half(w, 2) * summ as f32)
            })
            .sum(),
        (GgmlType::Q6K, Prepared::Q8K(x)) => row
            .chunks_exact(210)
            .zip(x)
            .map(|(w, a)| {
                let mut sumi = 0_i32;
                for n in 0..2 {
                    let ql = &w[n * 64..n * 64 + 64];
                    let qh = &w[128 + n * 32..128 + n * 32 + 32];
                    let sc = &w[192 + n * 8..192 + n * 8 + 8];
                    let y = &a.qs[n * 128..n * 128 + 128];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0x0f) | ((qh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        sumi += (sc[is] as i8) as i32 * q1 * y[l] as i32;
                        sumi += (sc[is + 2] as i8) as i32 * q2 * y[l + 32] as i32;
                        sumi += (sc[is + 4] as i8) as i32 * q3 * y[l + 64] as i32;
                        sumi += (sc[is + 6] as i8) as i32 * q4 * y[l + 96] as i32;
                    }
                }
                a.d * half(w, 208) * sumi as f32
            })
            .sum(),
        (kind, Prepared::F32(x)) => dot_row_reference(kind, row, x),
        _ => unreachable!("activation prepared for the wrong weight type"),
    }
}

/// `sum(k[i] * q[i])` where `k` holds f16 bits (the KV cache format).
pub fn dot_f16_f32(k: &[u16], q: &[f32]) -> f32 {
    debug_assert_eq!(k.len(), q.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && k.len().is_multiple_of(8) {
        // SAFETY: features detected; length is a multiple of 8.
        return unsafe { avx2::dot_f16_bits(k, q) };
    }
    k.iter().zip(q).map(|(&a, b)| f16_to_f32(a) * b).sum()
}

/// `out[j] = sum(k[i] * qs[j][i])` for several query vectors against one
/// f16 row (a key from the KV cache).
///
/// Two things make this much faster than calling [`dot_f16_f32`] per query.
/// Each query gets two accumulators, so a 64-wide head is two chains of four
/// FMAs rather than one chain of eight -- the single-accumulator form is
/// latency-bound, not throughput-bound. And the key is widened from f16 once
/// and reused by every query, which is the whole point of reading it once.
///
/// Each query's accumulation order is fixed, so a result does not depend on
/// how many queries happened to share the call.
pub fn dot_f16_f32_many<const N: usize>(k: &[u16], qs: [&[f32]; N]) -> [f32; N] {
    debug_assert!(qs.iter().all(|q| q.len() == k.len()));
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: features detected; every query is as long as `k`.
        return unsafe { avx2::dot_f16_bits_many(k, qs) };
    }
    std::array::from_fn(|j| {
        let (mut a, mut b) = (0.0_f32, 0.0_f32);
        for (c, chunk) in k.chunks(8).enumerate() {
            let acc = if c % 2 == 0 { &mut a } else { &mut b };
            for (i, &bits) in chunk.iter().enumerate() {
                *acc += f16_to_f32(bits) * qs[j][c * 8 + i];
            }
        }
        a + b
    })
}

/// `out[i] += a * v[i]` where `v` holds f16 bits.
pub fn axpy_f16(out: &mut [f32], a: f32, v: &[u16]) {
    debug_assert_eq!(out.len(), v.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && v.len().is_multiple_of(8) {
        // SAFETY: features detected; length is a multiple of 8.
        return unsafe { avx2::axpy_f16_bits(out, a, v) };
    }
    for (o, &b) in out.iter_mut().zip(v) {
        *o += a * f16_to_f32(b);
    }
}

// ---------------------------------------------------------------------------
// x8 packing: eight weight rows interleaved so that each 32-bit SIMD lane
// holds one row. A single vector op then applies all eight row scales and the
// eight results store directly, with no horizontal sums. This is the layout
// trick behind llama.cpp's repacked CPU kernels. Packed blocks, per group of
// eight rows and per block index (all multi-byte fields little-endian):
//
// Q4_0x8 (144 B = 8 x 18): f16 d[8]; then for m in 0..4 the bytes
//   qs[4m..4m+4] of rows 0..7. A Q4_0 byte qs[j] holds elements j (low
//   nibble) and j + 16 (high), so lane r of chunk m carries elements
//   4m..4m+4 and 16+4m..16+4m+4 of row r.
// Q8_0x8 (272 B = 8 x 34): f16 d[8]; then for m in 0..8, qs[4m..4m+4] of
//   rows 0..7.
// Q4_Kx8 (1248 B, 8 x 144 + 96): f16 d[8], f16 dmin[8]; u8 scale[8 sub][8
//   rows]; i16 (min[2p], min[2p+1]) pairs [4 p][8 rows]; then for k in 0..4,
//   m in 0..8, qs[32k+4m..+4] of rows 0..7. The 6-bit scales and mins are
//   unpacked once here instead of on every block.
// Q6_Kx8 (1680 B = 8 x 210): f16 d[8]; i8 scale[16 groups][8 rows]; then
//   for c in 0..32, ql[4c..4c+4] of rows 0..7; then for c in 0..16,
//   qh[4c..4c+4] of rows 0..7.

/// Packed bytes per (group of eight rows, block index).
fn packed_block_bytes(kind: GgmlType) -> Option<usize> {
    match kind {
        GgmlType::Q4_0 => Some(144),
        GgmlType::Q8_0 => Some(272),
        GgmlType::Q4K => Some(1248),
        GgmlType::Q6K => Some(1680),
        _ => None,
    }
}

/// Whether `kind` has an x8 layout and `rows` allows it. Packed kernels exist
/// only for AVX2; without it the row-major scalar kernels are used.
pub fn packable(kind: GgmlType, rows: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        has_avx2() && packed_block_bytes(kind).is_some() && rows.is_multiple_of(8)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (kind, rows);
        false
    }
}

/// Packed size for `len` bytes of row-major `kind` weights.
pub fn packed_len(kind: GgmlType, len: usize) -> usize {
    match packed_block_bytes(kind) {
        Some(packed) => len / (8 * kind.block_bytes()) * packed,
        None => 0,
    }
}

/// Repack a row-major matrix eight rows at a time, or `None` if the type or
/// shape has no x8 layout.
pub fn repack_x8(kind: GgmlType, data: &[u8], rows: usize, cols: usize) -> Option<Vec<u8>> {
    if !packable(kind, rows) {
        return None;
    }
    let row_bytes = kind.row_bytes(cols)?;
    let bb = kind.block_bytes();
    let pb = packed_block_bytes(kind)?;
    let blocks = row_bytes / bb;
    let mut out = vec![0_u8; packed_len(kind, data.len())];
    for group in 0..rows / 8 {
        for b in 0..blocks {
            let dst = &mut out[(group * blocks + b) * pb..][..pb];
            let src = |r: usize| &data[(group * 8 + r) * row_bytes + b * bb..][..bb];
            // Interleave `chunks` 4-byte chunks starting at `from` in each
            // source block into `dst[at..]`.
            let interleave = |dst: &mut [u8], at: usize, from: usize, chunks: usize| {
                for c in 0..chunks {
                    for r in 0..8 {
                        dst[at + 32 * c + 4 * r..][..4]
                            .copy_from_slice(&src(r)[from + 4 * c..][..4]);
                    }
                }
            };
            match kind {
                GgmlType::Q4_0 | GgmlType::Q8_0 => {
                    for r in 0..8 {
                        dst[2 * r..2 * r + 2].copy_from_slice(&src(r)[0..2]);
                    }
                    interleave(dst, 16, 2, (bb - 2) / 4);
                }
                GgmlType::Q4K => {
                    for r in 0..8 {
                        let s = src(r);
                        dst[2 * r..2 * r + 2].copy_from_slice(&s[0..2]);
                        dst[16 + 2 * r..16 + 2 * r + 2].copy_from_slice(&s[2..4]);
                        for j in 0..8 {
                            let (sc, mn) = scale_min_k4(j, &s[4..16]);
                            dst[32 + 8 * j + r] = sc;
                            if j % 2 == 0 {
                                let at = 96 + 32 * (j / 2) + 4 * r;
                                dst[at..at + 2].copy_from_slice(&(mn as i16).to_le_bytes());
                            } else {
                                let at = 96 + 32 * (j / 2) + 4 * r + 2;
                                dst[at..at + 2].copy_from_slice(&(mn as i16).to_le_bytes());
                            }
                        }
                    }
                    interleave(dst, 224, 16, 32);
                }
                GgmlType::Q6K => {
                    for r in 0..8 {
                        let s = src(r);
                        dst[2 * r..2 * r + 2].copy_from_slice(&s[208..210]);
                        for g in 0..16 {
                            dst[16 + 8 * g + r] = s[192 + g];
                        }
                    }
                    interleave(dst, 144, 0, 32);
                    interleave(dst, 1168, 128, 16);
                }
                _ => unreachable!(),
            }
        }
    }
    Some(out)
}

/// Eight rows (one packed group) against `xs.len()` columns:
/// `out[c * 8 + r] = row_r · xs[c]`. Only valid where [`packable`] holds.
pub fn dot_x8(kind: GgmlType, group: &[u8], xs: &[Prepared<'_>], out: &mut [f32]) {
    debug_assert_eq!(out.len(), xs.len() * 8);
    #[cfg(target_arch = "x86_64")]
    {
        assert!(has_avx2(), "x8-packed weights need AVX2");
        macro_rules! run {
            ($variant:ident, $kernel:ident, $tile:literal) => {{
                let col = |j: usize| match &xs[j] {
                    Prepared::$variant(v) => v.as_slice(),
                    _ => unreachable!("activation prepared for the wrong weight type"),
                };
                // SAFETY: AVX2 checked above; every column has one block per
                // weight block (shapes validated by the backend).
                unsafe {
                    let mut j = 0;
                    while j + $tile <= xs.len() {
                        let cols: [&[_]; $tile] = std::array::from_fn(|c| col(j + c));
                        let r = avx2::$kernel::<$tile>(group, cols);
                        for (c, v) in r.iter().enumerate() {
                            out[(j + c) * 8..(j + c + 1) * 8].copy_from_slice(v);
                        }
                        j += $tile;
                    }
                    while j < xs.len() {
                        let r = avx2::$kernel::<1>(group, [col(j)]);
                        out[j * 8..(j + 1) * 8].copy_from_slice(&r[0]);
                        j += 1;
                    }
                }
            }};
        }
        match kind {
            GgmlType::Q4_0 => run!(Q8_0, q4_0x8, 4),
            GgmlType::Q8_0 => run!(Q8_0, q8_0x8, 4),
            GgmlType::Q4K => run!(Q8K, q4kx8, 2),
            GgmlType::Q6K => run!(Q8K, q6kx8, 2),
            other => unreachable!("{} has no x8 layout", other.name()),
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (kind, group, xs, out);
        unreachable!("x8 packing is AVX2-only");
    }
}

/// `gate[i] = silu(gate[i]) * up[i]` (the SwiGLU activation), vectorized.
pub fn swiglu_in_place(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: features detected; only whole 8-lane chunks are touched.
        done = unsafe { avx2::swiglu(gate, up) };
    }
    for (g, u) in gate[done..].iter_mut().zip(&up[done..]) {
        *g = *g / (1.0 + (-*g).exp()) * u;
    }
}

/// `values[i] = exp(values[i] - max)`, returning the sum (softmax numerator).
pub fn exp_shifted_sum(values: &mut [f32], max: f32) -> f32 {
    let mut sum = 0.0;
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: features detected; only whole 8-lane chunks are touched.
        let (n, s) = unsafe { avx2::exp_shifted_sum(values, max) };
        done = n;
        sum = s;
    }
    for v in &mut values[done..] {
        *v = (*v - max).exp();
        sum += *v;
    }
    sum
}

/// Widen f16 bits to f32 in bulk.
pub fn load_f16(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len() * 2);
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: features detected; only whole 8-lane chunks are touched.
        done = unsafe { avx2::load_f16(src, dst) };
    }
    for (d, b) in dst[done..].iter_mut().zip(src[done * 2..].chunks_exact(2)) {
        *d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
    }
}

/// Widen bf16 bits to f32 in bulk.
pub fn load_bf16(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len() * 2);
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: features detected; only whole 8-lane chunks are touched.
        done = unsafe { avx2::load_bf16(src, dst) };
    }
    for (d, b) in dst[done..].iter_mut().zip(src[done * 2..].chunks_exact(2)) {
        *d = bf16_to_f32(u16::from_le_bytes([b[0], b[1]]));
    }
}

/// Convert f32 values to f16 bits (round to nearest even).
pub fn store_f16(src: &[f32], dst: &mut [u16]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && src.len().is_multiple_of(8) {
        // SAFETY: features detected; length is a multiple of 8.
        return unsafe { avx2::store_f16_bits(src, dst) };
    }
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = crate::quant::f32_to_f16(s);
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)] // every item in this module is an intrinsic wrapper
mod avx2 {
    use super::{BlockQ8_0, BlockQ8K};
    use crate::quant::scale_min_k4;
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn hsum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn half8(p: *const u8) -> __m256 {
        _mm256_cvtph_ps(_mm_set1_epi16(std::ptr::read_unaligned(p as *const i16)))
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn half(p: *const u8) -> f32 {
        // `_mm_loadu_si16` zero-extends into a fresh register. Building the
        // vector from a scalar let LLVM emit `vpinsrw` into a live register
        // (the accumulator), chaining every block on the previous FMA and
        // costing ~20 cycles per block.
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_loadu_si16(p)))
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_f32(row: &[u8], x: &[f32]) -> f32 {
        let n = x.len();
        let w = row.as_ptr() as *const f32;
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i + 32 <= n {
            for (k, a) in acc.iter_mut().enumerate() {
                let wv = _mm256_loadu_ps(w.add(i + 8 * k));
                *a = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x.as_ptr().add(i + 8 * k)), *a);
            }
            i += 32;
        }
        let mut sum = hsum(_mm256_add_ps(
            _mm256_add_ps(acc[0], acc[1]),
            _mm256_add_ps(acc[2], acc[3]),
        ));
        while i < n {
            sum += std::ptr::read_unaligned(w.add(i)) * x[i];
            i += 1;
        }
        sum
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_f16(row: &[u8], x: &[f32]) -> f32 {
        let n = x.len();
        let w = row.as_ptr();
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i + 32 <= n {
            for (k, a) in acc.iter_mut().enumerate() {
                let h = _mm_loadu_si128(w.add(2 * (i + 8 * k)) as *const __m128i);
                *a = _mm256_fmadd_ps(
                    _mm256_cvtph_ps(h),
                    _mm256_loadu_ps(x.as_ptr().add(i + 8 * k)),
                    *a,
                );
            }
            i += 32;
        }
        let mut sum = hsum(_mm256_add_ps(
            _mm256_add_ps(acc[0], acc[1]),
            _mm256_add_ps(acc[2], acc[3]),
        ));
        while i < n {
            sum += half(w.add(2 * i)) * x[i];
            i += 1;
        }
        sum
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_f16_bits(k: &[u16], q: &[f32]) -> f32 {
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i < k.len() {
            let h = _mm_loadu_si128(k.as_ptr().add(i) as *const __m128i);
            acc = _mm256_fmadd_ps(_mm256_cvtph_ps(h), _mm256_loadu_ps(q.as_ptr().add(i)), acc);
            i += 8;
        }
        hsum(acc)
    }

    /// `N` query vectors against one f16 key row. Two accumulators per query
    /// break the FMA dependency chain; the widened key is shared.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_f16_bits_many<const N: usize>(k: &[u16], qs: [&[f32]; N]) -> [f32; N] {
        let n = k.len();
        let mut acc = [[_mm256_setzero_ps(); 2]; N];
        let kp = k.as_ptr();
        let mut i = 0;
        while i + 16 <= n {
            let k0 = _mm256_cvtph_ps(_mm_loadu_si128(kp.add(i) as *const __m128i));
            let k1 = _mm256_cvtph_ps(_mm_loadu_si128(kp.add(i + 8) as *const __m128i));
            for j in 0..N {
                let q = qs[j].as_ptr().add(i);
                acc[j][0] = _mm256_fmadd_ps(k0, _mm256_loadu_ps(q), acc[j][0]);
                acc[j][1] = _mm256_fmadd_ps(k1, _mm256_loadu_ps(q.add(8)), acc[j][1]);
            }
            i += 16;
        }
        if i + 8 <= n {
            let k0 = _mm256_cvtph_ps(_mm_loadu_si128(kp.add(i) as *const __m128i));
            for j in 0..N {
                acc[j][0] = _mm256_fmadd_ps(k0, _mm256_loadu_ps(qs[j].as_ptr().add(i)), acc[j][0]);
            }
            i += 8;
        }
        let mut out = [0.0_f32; N];
        for (j, pair) in acc.iter().enumerate() {
            let mut sum = hsum(_mm256_add_ps(pair[0], pair[1]));
            for (bits, value) in k[i..n].iter().zip(&qs[j][i..n]) {
                sum += crate::quant::f16_to_f32(*bits) * value;
            }
            out[j] = sum;
        }
        out
    }

    /// Quantize whole 32-float blocks into `out`, matching
    /// [`super::quantize_q8_0_scalar`] exactly for finite inputs.
    ///
    /// `vcvtps2dq` rounds with MXCSR, which is round-to-nearest-even by
    /// default and so is already the scalar's `round_ties_even`; no explicit
    /// rounding step is needed. The block sum is taken from the i32 vectors
    /// before they are packed, where it cannot overflow (32 values of at
    /// most 127).
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn quantize_q8_0_into(
        x: &[f32],
        out: &mut [std::mem::MaybeUninit<super::BlockQ8_0>],
    ) {
        let blocks = x.len() / 32;
        debug_assert!(out.len() >= blocks);
        let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
        // Restores lane order after the two packing steps.
        let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        for b in 0..blocks {
            let p = x.as_ptr().add(b * 32);
            let v: [__m256; 4] = std::array::from_fn(|i| _mm256_loadu_ps(p.add(8 * i)));
            // amax over the block. The scalar fold starts at 0.0 and uses
            // f32::max, which agrees with vmaxps on finite inputs.
            let mut m = _mm256_and_ps(v[0], sign_mask);
            for part in &v[1..] {
                m = _mm256_max_ps(m, _mm256_and_ps(*part, sign_mask));
            }
            let half = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
            let half = _mm_max_ps(half, _mm_movehl_ps(half, half));
            let amax = _mm_cvtss_f32(_mm_max_ss(half, _mm_shuffle_ps(half, half, 1)));

            let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
            let scale = _mm256_set1_ps(id);
            let i32s: [__m256i; 4] =
                std::array::from_fn(|i| _mm256_cvtps_epi32(_mm256_mul_ps(v[i], scale)));

            let mut total = _mm256_add_epi32(
                _mm256_add_epi32(i32s[0], i32s[1]),
                _mm256_add_epi32(i32s[2], i32s[3]),
            );
            total = _mm256_hadd_epi32(total, total);
            total = _mm256_hadd_epi32(total, total);
            let sum = _mm256_extract_epi32(total, 0) + _mm256_extract_epi32(total, 4);

            // i32 -> i16 -> i8, then undo the lane interleave.
            let lo = _mm256_packs_epi32(i32s[0], i32s[1]);
            let hi = _mm256_packs_epi32(i32s[2], i32s[3]);
            let packed = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(lo, hi), order);

            let d = super::f16_to_f32(crate::quant::f32_to_f16(amax / 127.0));
            let slot = out.get_unchecked_mut(b).as_mut_ptr();
            (*slot).d = d;
            (*slot).s = d * sum as f32;
            _mm256_storeu_si256((*slot).qs.as_mut_ptr() as *mut __m256i, packed);
        }
    }

    /// Quantize whole 256-float blocks into `out`, matching
    /// [`super::quantize_q8k_scalar`] exactly for finite inputs.
    ///
    /// Two details decide correctness. The scale's sign comes from the
    /// *first* value attaining the largest magnitude, so after finding the
    /// magnitude the blocks are rescanned in order for it. And the scalar
    /// rounds with `f32::round`, half away from zero, which `vcvtps2dq`
    /// does not do; `trunc(t + copysign(0.5, t))` does, exactly, for the
    /// |t| <= 127 this produces.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn quantize_q8k_into(x: &[f32], out: &mut [std::mem::MaybeUninit<super::BlockQ8K>]) {
        const K: usize = crate::quant::QK_K;
        let blocks = x.len() / K;
        debug_assert!(out.len() >= blocks);
        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
        let sign_bit = _mm256_castsi256_ps(_mm256_set1_epi32(-0x8000_0000));
        let half = _mm256_set1_ps(0.5);
        let cap = _mm256_set1_epi32(127);
        let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        for b in 0..blocks {
            let p = x.as_ptr().add(b * K);
            let slot = out.get_unchecked_mut(b).as_mut_ptr();

            let mut m = _mm256_setzero_ps();
            for i in 0..K / 8 {
                m = _mm256_max_ps(m, _mm256_and_ps(_mm256_loadu_ps(p.add(8 * i)), abs_mask));
            }
            let lo = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
            let lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
            let amax = _mm_cvtss_f32(_mm_max_ss(lo, _mm_shuffle_ps(lo, lo, 1)));

            if amax == 0.0 {
                (*slot).d = 0.0;
                (*slot).qs = [0; K];
                (*slot).bsums = [0; 16];
                continue;
            }
            // The first value whose magnitude equals amax fixes the sign.
            let target = _mm256_set1_ps(amax);
            let mut max = 0.0_f32;
            for i in 0..K / 8 {
                let v = _mm256_loadu_ps(p.add(8 * i));
                let hit = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_EQ_OQ>(
                    _mm256_and_ps(v, abs_mask),
                    target,
                ));
                if hit != 0 {
                    max = *p.add(8 * i + hit.trailing_zeros() as usize);
                    break;
                }
            }
            let iscale = -127.0 / max;
            let scale = _mm256_set1_ps(iscale);

            let mut packed = [_mm256_setzero_si256(); K / 32];
            let mut group_sums = [0_i32; 16];
            for q in 0..K / 32 {
                // 32 values -> four i32 vectors -> one packed i8 vector.
                let mut parts = [_mm256_setzero_si256(); 4];
                for (j, part) in parts.iter_mut().enumerate() {
                    let t = _mm256_mul_ps(_mm256_loadu_ps(p.add(32 * q + 8 * j)), scale);
                    let bump = _mm256_or_ps(_mm256_and_ps(t, sign_bit), half);
                    let r = _mm256_cvttps_epi32(_mm256_add_ps(t, bump));
                    *part = _mm256_min_epi32(r, cap);
                }
                // bsums covers groups of 16 values, two per 32-value chunk.
                for g in 0..2 {
                    let pair = _mm256_add_epi32(parts[2 * g], parts[2 * g + 1]);
                    let folded = _mm256_hadd_epi32(pair, pair);
                    let folded = _mm256_hadd_epi32(folded, folded);
                    group_sums[2 * q + g] =
                        _mm256_extract_epi32(folded, 0) + _mm256_extract_epi32(folded, 4);
                }
                let a = _mm256_packs_epi32(parts[0], parts[1]);
                let c = _mm256_packs_epi32(parts[2], parts[3]);
                packed[q] = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(a, c), order);
            }
            for (q, value) in packed.iter().enumerate() {
                _mm256_storeu_si256((*slot).qs.as_mut_ptr().add(32 * q) as *mut __m256i, *value);
            }
            for (sum, value) in (*slot).bsums.iter_mut().zip(&group_sums) {
                *sum = *value as i16;
            }
            (*slot).d = 1.0 / iscale;
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn axpy_f16_bits(out: &mut [f32], a: f32, v: &[u16]) {
        let av = _mm256_set1_ps(a);
        let mut i = 0;
        while i < v.len() {
            let h = _mm_loadu_si128(v.as_ptr().add(i) as *const __m128i);
            let o = out.as_mut_ptr().add(i);
            _mm256_storeu_ps(
                o,
                _mm256_fmadd_ps(av, _mm256_cvtph_ps(h), _mm256_loadu_ps(o)),
            );
            i += 8;
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn store_f16_bits(src: &[f32], dst: &mut [u16]) {
        let mut i = 0;
        while i < src.len() {
            let h =
                _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm256_loadu_ps(src.as_ptr().add(i)));
            _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, h);
            i += 8;
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn k_mins(mins: &[u8; 8], bsums: &[i16; 16]) -> i32 {
        let mut s = 0_i32;
        for j in 0..8 {
            s += mins[j] as i32 * (bsums[2 * j] as i32 + bsums[2 * j + 1] as i32);
        }
        s
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q5k(row: &[u8], x: &[BlockQ8K]) -> f32 {
        let mask = _mm256_set1_epi8(0x0f);
        let one = _mm256_set1_epi8(1);
        let mut acc = _mm256_setzero_ps();
        let mut acc_m = _mm256_setzero_ps();
        let w = row.as_ptr();
        for (b, a) in x.iter().enumerate() {
            let block = std::slice::from_raw_parts(w.add(b * 176), 176);
            let mut sc = [0_u8; 8];
            let mut mn = [0_u8; 8];
            for j in 0..8 {
                (sc[j], mn[j]) = scale_min_k4(j, &block[4..16]);
            }
            let qh = _mm256_loadu_si256(block.as_ptr().add(16) as *const __m256i);
            let qs = block.as_ptr().add(48);
            let mut sumi = _mm256_setzero_si256();
            for k in 0..4 {
                let q = _mm256_loadu_si256(qs.add(32 * k) as *const __m256i);
                let sh_lo = _mm_cvtsi32_si128((2 * k) as i32);
                let sh_hi = _mm_cvtsi32_si128((2 * k + 1) as i32);
                let h_lo = _mm256_slli_epi16(_mm256_and_si256(_mm256_srl_epi16(qh, sh_lo), one), 4);
                let h_hi = _mm256_slli_epi16(_mm256_and_si256(_mm256_srl_epi16(qh, sh_hi), one), 4);
                let lo = _mm256_or_si256(_mm256_and_si256(q, mask), h_lo);
                let hi = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q, 4), mask), h_hi);
                let ya = _mm256_loadu_si256(a.qs.as_ptr().add(64 * k) as *const __m256i);
                let yb = _mm256_loadu_si256(a.qs.as_ptr().add(64 * k + 32) as *const __m256i);
                let pa = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(lo, ya),
                    _mm256_set1_epi16(sc[2 * k] as i16),
                );
                let pb = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(hi, yb),
                    _mm256_set1_epi16(sc[2 * k + 1] as i16),
                );
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(pa, pb));
            }
            let ad = _mm256_set1_ps(a.d);
            let d = _mm256_mul_ps(ad, half8(block.as_ptr()));
            let dmin = _mm256_mul_ps(ad, half8(block.as_ptr().add(2)));
            acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi), acc);
            let m = _mm256_set1_ps(k_mins(&mn, &a.bsums) as f32);
            acc_m = _mm256_fmadd_ps(dmin, m, acc_m);
        }
        hsum(acc) - _mm256_cvtss_f32(acc_m)
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q4_0_n<const N: usize>(row: &[u8], xs: [&[BlockQ8_0]; N]) -> [f32; N] {
        let mask = _mm256_set1_epi8(0x0f);
        let eight = _mm256_set1_epi8(8);
        let mut acc = [_mm256_setzero_ps(); N];
        let w = row.as_ptr();
        for b in 0..row.len() / 18 {
            let block = w.add(b * 18);
            let dw = half8(block);
            let qs = _mm_loadu_si128(block.add(2) as *const __m128i);
            let q = _mm256_sub_epi8(
                _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(qs, 4), qs), mask),
                eight,
            );
            let aq = _mm256_sign_epi8(q, q);
            for j in 0..N {
                let a = xs[j].get_unchecked(b);
                let y = _mm256_loadu_si256(a.qs.as_ptr() as *const __m256i);
                let p = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(aq, _mm256_sign_epi8(y, q)),
                    _mm256_set1_epi16(1),
                );
                acc[j] = _mm256_fmadd_ps(
                    _mm256_mul_ps(dw, _mm256_set1_ps(a.d)),
                    _mm256_cvtepi32_ps(p),
                    acc[j],
                );
            }
        }
        acc.map(|a| hsum(a))
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q8_0_n<const N: usize>(row: &[u8], xs: [&[BlockQ8_0]; N]) -> [f32; N] {
        let mut acc = [_mm256_setzero_ps(); N];
        let w = row.as_ptr();
        for b in 0..row.len() / 34 {
            let block = w.add(b * 34);
            let dw = half8(block);
            let q = _mm256_loadu_si256(block.add(2) as *const __m256i);
            let aq = _mm256_sign_epi8(q, q);
            for j in 0..N {
                let a = xs[j].get_unchecked(b);
                let y = _mm256_loadu_si256(a.qs.as_ptr() as *const __m256i);
                let p = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(aq, _mm256_sign_epi8(y, q)),
                    _mm256_set1_epi16(1),
                );
                acc[j] = _mm256_fmadd_ps(
                    _mm256_mul_ps(dw, _mm256_set1_ps(a.d)),
                    _mm256_cvtepi32_ps(p),
                    acc[j],
                );
            }
        }
        acc.map(|a| hsum(a))
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q4k_n<const N: usize>(row: &[u8], xs: [&[BlockQ8K]; N]) -> [f32; N] {
        let mask = _mm256_set1_epi8(0x0f);
        let mut acc = [_mm256_setzero_ps(); N];
        let mut acc_m = [_mm256_setzero_ps(); N];
        let w = row.as_ptr();
        for b in 0..row.len() / 144 {
            let block = std::slice::from_raw_parts(w.add(b * 144), 144);
            let mut sc = [0_u8; 8];
            let mut mn = [0_u8; 8];
            for j in 0..8 {
                (sc[j], mn[j]) = scale_min_k4(j, &block[4..16]);
            }
            let qs = block.as_ptr().add(16);
            let mut sumi = [_mm256_setzero_si256(); N];
            for k in 0..4 {
                let q4 = _mm256_loadu_si256(qs.add(32 * k) as *const __m256i);
                let lo = _mm256_and_si256(q4, mask);
                let hi = _mm256_and_si256(_mm256_srli_epi16(q4, 4), mask);
                let s_lo = _mm256_set1_epi16(sc[2 * k] as i16);
                let s_hi = _mm256_set1_epi16(sc[2 * k + 1] as i16);
                for j in 0..N {
                    let a = xs[j].get_unchecked(b);
                    let ya = _mm256_loadu_si256(a.qs.as_ptr().add(64 * k) as *const __m256i);
                    let yb = _mm256_loadu_si256(a.qs.as_ptr().add(64 * k + 32) as *const __m256i);
                    let pa = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, ya), s_lo);
                    let pb = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, yb), s_hi);
                    sumi[j] = _mm256_add_epi32(sumi[j], _mm256_add_epi32(pa, pb));
                }
            }
            let d = half8(block.as_ptr());
            let dmin = half8(block.as_ptr().add(2));
            for j in 0..N {
                let a = xs[j].get_unchecked(b);
                let ad = _mm256_set1_ps(a.d);
                acc[j] = _mm256_fmadd_ps(_mm256_mul_ps(ad, d), _mm256_cvtepi32_ps(sumi[j]), acc[j]);
                let m = _mm256_set1_ps(k_mins(&mn, &a.bsums) as f32);
                acc_m[j] = _mm256_fmadd_ps(_mm256_mul_ps(ad, dmin), m, acc_m[j]);
            }
        }
        let mut out = [0.0_f32; N];
        for j in 0..N {
            // acc_m holds the min correction in every lane; take one lane.
            out[j] = hsum(acc[j]) - _mm256_cvtss_f32(acc_m[j]);
        }
        out
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q6k_n<const N: usize>(row: &[u8], xs: [&[BlockQ8K]; N]) -> [f32; N] {
        let m4 = _mm256_set1_epi8(0x0f);
        let m3 = _mm256_set1_epi8(3);
        let mut acc = [_mm256_setzero_ps(); N];
        let w = row.as_ptr();
        for b in 0..row.len() / 210 {
            let block = w.add(b * 210);
            let scales = std::slice::from_raw_parts(block.add(192), 16);
            let mut sumi = [_mm256_setzero_si256(); N];
            for n in 0..2 {
                let ql0 = _mm256_loadu_si256(block.add(64 * n) as *const __m256i);
                let ql1 = _mm256_loadu_si256(block.add(64 * n + 32) as *const __m256i);
                let qh = _mm256_loadu_si256(block.add(128 + 32 * n) as *const __m256i);
                let high = |shift: i32| -> __m256i {
                    let s = _mm_cvtsi32_si128(shift);
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srl_epi16(qh, s), m3), 4)
                };
                let q = [
                    _mm256_or_si256(_mm256_and_si256(ql0, m4), high(0)),
                    _mm256_or_si256(_mm256_and_si256(ql1, m4), high(2)),
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), m4), high(4)),
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), m4), high(6)),
                ];
                for (i, qi) in q.iter().enumerate() {
                    let s0 = scales[8 * n + 2 * i] as i8 as i16;
                    let s1 = scales[8 * n + 2 * i + 1] as i8 as i16;
                    let sv = _mm256_set_m128i(_mm_set1_epi16(s1), _mm_set1_epi16(s0));
                    for j in 0..N {
                        let a = xs[j].get_unchecked(b);
                        let y = _mm256_loadu_si256(
                            a.qs.as_ptr().add(128 * n + 32 * i) as *const __m256i
                        );
                        sumi[j] = _mm256_add_epi32(
                            sumi[j],
                            _mm256_madd_epi16(_mm256_maddubs_epi16(*qi, y), sv),
                        );
                    }
                }
            }
            let d = half8(block.add(208));
            for j in 0..N {
                let a = xs[j].get_unchecked(b);
                let mut corr = 0_i32;
                for g in 0..16 {
                    corr += scales[g] as i8 as i32 * a.bsums[g] as i32;
                }
                // Subtract the 32 offset in integer space, spread over lanes.
                let total = _mm256_sub_epi32(sumi[j], _mm256_set1_epi32(4 * corr));
                acc[j] = _mm256_fmadd_ps(
                    _mm256_mul_ps(_mm256_set1_ps(a.d), d),
                    _mm256_cvtepi32_ps(total),
                    acc[j],
                );
            }
        }
        acc.map(|a| hsum(a))
    }

    /// 32 bits -> 32 bytes, 0xFF where the bit is set (ggml's
    /// `bytes_from_bits_32`).
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn bytes_from_bits(p: *const u8) -> __m256i {
        let bits = std::ptr::read_unaligned(p as *const u32);
        let shuffle = _mm256_set_epi64x(
            0x0303_0303_0303_0303,
            0x0202_0202_0202_0202,
            0x0101_0101_0101_0101,
            0,
        );
        let bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(bits as i32), shuffle);
        let bytes = _mm256_or_si256(bytes, _mm256_set1_epi64x(0x7fbf_dfef_f7fb_fdfe));
        _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1))
    }

    /// The 16 packed nibbles at `p` as 32 bytes (low nibbles first).
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn nibbles(p: *const u8) -> __m256i {
        let qs = _mm_loadu_si128(p as *const __m128i);
        _mm256_and_si256(
            _mm256_set_m128i(_mm_srli_epi16(qs, 4), qs),
            _mm256_set1_epi8(0x0f),
        )
    }

    /// Signed weights: dot via the sign trick.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn dot_signed(q: __m256i, y: __m256i) -> __m256 {
        let p = _mm256_maddubs_epi16(_mm256_sign_epi8(q, q), _mm256_sign_epi8(y, q));
        _mm256_cvtepi32_ps(_mm256_madd_epi16(p, _mm256_set1_epi16(1)))
    }

    /// Unsigned weights (0..=31): `maddubs` takes them directly.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn dot_unsigned(q: __m256i, y: __m256i) -> __m256 {
        let p = _mm256_maddubs_epi16(q, y);
        _mm256_cvtepi32_ps(_mm256_madd_epi16(p, _mm256_set1_epi16(1)))
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q4_1_n<const N: usize>(row: &[u8], xs: [&[BlockQ8_0]; N]) -> [f32; N] {
        let mut acc = [_mm256_setzero_ps(); N];
        let mut acc_m = [_mm256_setzero_ps(); N];
        let w = row.as_ptr();
        for b in 0..row.len() / 20 {
            let block = w.add(b * 20);
            let (d, m) = (half8(block), half8(block.add(2)));
            let q = nibbles(block.add(4));
            for j in 0..N {
                let a = xs[j].get_unchecked(b);
                let y = _mm256_loadu_si256(a.qs.as_ptr() as *const __m256i);
                acc[j] = _mm256_fmadd_ps(
                    _mm256_mul_ps(d, _mm256_set1_ps(a.d)),
                    dot_unsigned(q, y),
                    acc[j],
                );
                acc_m[j] = _mm256_fmadd_ps(m, _mm256_set1_ps(a.s), acc_m[j]);
            }
        }
        let mut out = [0.0; N];
        for j in 0..N {
            out[j] = hsum(acc[j]) + _mm256_cvtss_f32(acc_m[j]);
        }
        out
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q5_0_n<const N: usize>(row: &[u8], xs: [&[BlockQ8_0]; N]) -> [f32; N] {
        let mut acc = [_mm256_setzero_ps(); N];
        let w = row.as_ptr();
        let high = _mm256_set1_epi8(0xf0_u8 as i8);
        for b in 0..row.len() / 22 {
            let block = w.add(b * 22);
            let d = half8(block);
            // (nibble | 0xF0 where the fifth bit is clear) == value - 16.
            let hi = _mm256_andnot_si256(bytes_from_bits(block.add(2)), high);
            let q = _mm256_or_si256(nibbles(block.add(6)), hi);
            for j in 0..N {
                let a = xs[j].get_unchecked(b);
                let y = _mm256_loadu_si256(a.qs.as_ptr() as *const __m256i);
                acc[j] = _mm256_fmadd_ps(
                    _mm256_mul_ps(d, _mm256_set1_ps(a.d)),
                    dot_signed(q, y),
                    acc[j],
                );
            }
        }
        acc.map(|a| hsum(a))
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q5_1_n<const N: usize>(row: &[u8], xs: [&[BlockQ8_0]; N]) -> [f32; N] {
        let mut acc = [_mm256_setzero_ps(); N];
        let mut acc_m = [_mm256_setzero_ps(); N];
        let w = row.as_ptr();
        let fifth = _mm256_set1_epi8(0x10);
        for b in 0..row.len() / 24 {
            let block = w.add(b * 24);
            let (d, m) = (half8(block), half8(block.add(2)));
            let hi = _mm256_and_si256(bytes_from_bits(block.add(4)), fifth);
            let q = _mm256_or_si256(nibbles(block.add(8)), hi);
            for j in 0..N {
                let a = xs[j].get_unchecked(b);
                let y = _mm256_loadu_si256(a.qs.as_ptr() as *const __m256i);
                acc[j] = _mm256_fmadd_ps(
                    _mm256_mul_ps(d, _mm256_set1_ps(a.d)),
                    dot_unsigned(q, y),
                    acc[j],
                );
                acc_m[j] = _mm256_fmadd_ps(m, _mm256_set1_ps(a.s), acc_m[j]);
            }
        }
        let mut out = [0.0; N];
        for j in 0..N {
            out[j] = hsum(acc[j]) + _mm256_cvtss_f32(acc_m[j]);
        }
        out
    }

    /// Q4_0x8 against C columns; result `[c][r]`.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q4_0x8<const C: usize>(group: &[u8], xs: [&[BlockQ8_0]; C]) -> [[f32; 8]; C] {
        let mask = _mm256_set1_epi8(0x0f);
        let ones = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_ps(); C];
        let g = group.as_ptr();
        for b in 0..group.len() / 144 {
            let block = g.add(b * 144);
            let d = _mm256_cvtph_ps(_mm_loadu_si128(block as *const __m128i));
            // Unsigned nibbles 0..15 feed maddubs directly; the -8 offset is
            // applied once per block below via the activation sum. Eight
            // maddubs results stay within i16: 8 * 2 * 15 * 127 = 30480.
            let mut sum16 = [_mm256_setzero_si256(); C];
            for m in 0..4 {
                let w = _mm256_loadu_si256(block.add(16 + 32 * m) as *const __m256i);
                let lo = _mm256_and_si256(w, mask);
                let hi = _mm256_and_si256(_mm256_srli_epi16(w, 4), mask);
                for c in 0..C {
                    let a = xs[c].get_unchecked(b);
                    let ylo = _mm256_set1_epi32(std::ptr::read_unaligned(
                        a.qs.as_ptr().add(4 * m) as *const i32
                    ));
                    let yhi = _mm256_set1_epi32(std::ptr::read_unaligned(
                        a.qs.as_ptr().add(16 + 4 * m) as *const i32,
                    ));
                    sum16[c] = _mm256_add_epi16(sum16[c], _mm256_maddubs_epi16(lo, ylo));
                    sum16[c] = _mm256_add_epi16(sum16[c], _mm256_maddubs_epi16(hi, yhi));
                }
            }
            for c in 0..C {
                let a = xs[c].get_unchecked(b);
                let p = _mm256_cvtepi32_ps(_mm256_madd_epi16(sum16[c], ones));
                acc[c] = _mm256_fmadd_ps(_mm256_mul_ps(d, _mm256_set1_ps(a.d)), p, acc[c]);
                // sum((q - 8) * y) = sum(q * y) - 8 * sum(y); a.s = a.d * sum(y).
                acc[c] = _mm256_fmadd_ps(d, _mm256_set1_ps(-8.0 * a.s), acc[c]);
            }
        }
        let mut out = [[0.0_f32; 8]; C];
        for c in 0..C {
            _mm256_storeu_ps(out[c].as_mut_ptr(), acc[c]);
        }
        out
    }

    /// `(sc, sc)` i16 pairs per lane from eight u8 or i8 scales.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn pair_u8(p: *const u8) -> __m256i {
        let x = _mm256_cvtepu8_epi32(_mm_loadl_epi64(p as *const __m128i));
        _mm256_or_si256(x, _mm256_slli_epi32(x, 16))
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn pair_i8(p: *const u8) -> __m256i {
        let x = _mm256_cvtepi8_epi32(_mm_loadl_epi64(p as *const __m128i));
        _mm256_or_si256(
            _mm256_and_si256(x, _mm256_set1_epi32(0xffff)),
            _mm256_slli_epi32(x, 16),
        )
    }

    /// Broadcast the i16 pair (lo, hi) to every 32-bit lane.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn pair16(lo: i32, hi: i32) -> __m256i {
        _mm256_set1_epi32((hi << 16) | (lo & 0xffff))
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn bcast4(p: *const i8) -> __m256i {
        _mm256_set1_epi32(std::ptr::read_unaligned(p as *const i32))
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q8_0x8<const C: usize>(group: &[u8], xs: [&[BlockQ8_0]; C]) -> [[f32; 8]; C] {
        let ones = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_ps(); C];
        let g = group.as_ptr();
        for b in 0..group.len() / 272 {
            let block = g.add(b * 272);
            let d = _mm256_cvtph_ps(_mm_loadu_si128(block as *const __m128i));
            let mut sum = [_mm256_setzero_si256(); C];
            for m in 0..8 {
                let w = _mm256_loadu_si256(block.add(16 + 32 * m) as *const __m256i);
                let aw = _mm256_sign_epi8(w, w);
                for c in 0..C {
                    let y = bcast4(xs[c].get_unchecked(b).qs.as_ptr().add(4 * m));
                    let p = _mm256_maddubs_epi16(aw, _mm256_sign_epi8(y, w));
                    sum[c] = _mm256_add_epi32(sum[c], _mm256_madd_epi16(p, ones));
                }
            }
            for c in 0..C {
                let a = xs[c].get_unchecked(b);
                acc[c] = _mm256_fmadd_ps(
                    _mm256_mul_ps(d, _mm256_set1_ps(a.d)),
                    _mm256_cvtepi32_ps(sum[c]),
                    acc[c],
                );
            }
        }
        let mut out = [[0.0_f32; 8]; C];
        for c in 0..C {
            _mm256_storeu_ps(out[c].as_mut_ptr(), acc[c]);
        }
        out
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q4kx8<const C: usize>(group: &[u8], xs: [&[BlockQ8K]; C]) -> [[f32; 8]; C] {
        let mask = _mm256_set1_epi8(0x0f);
        let mut acc = [_mm256_setzero_ps(); C];
        let g = group.as_ptr();
        for b in 0..group.len() / 1248 {
            let block = g.add(b * 1248);
            let d = _mm256_cvtph_ps(_mm_loadu_si128(block as *const __m128i));
            let dmin = _mm256_cvtph_ps(_mm_loadu_si128(block.add(16) as *const __m128i));
            let qs = block.add(224);
            let mut sumi = [_mm256_setzero_si256(); C];
            for k in 0..4 {
                // Eight maddubs of 4-bit x 8-bit stay within i16: 30480.
                let mut slo = [_mm256_setzero_si256(); C];
                let mut shi = [_mm256_setzero_si256(); C];
                for m in 0..8 {
                    let w = _mm256_loadu_si256(qs.add(256 * k + 32 * m) as *const __m256i);
                    let lo = _mm256_and_si256(w, mask);
                    let hi = _mm256_and_si256(_mm256_srli_epi16(w, 4), mask);
                    for c in 0..C {
                        let y = xs[c].get_unchecked(b).qs.as_ptr();
                        slo[c] = _mm256_add_epi16(
                            slo[c],
                            _mm256_maddubs_epi16(lo, bcast4(y.add(64 * k + 4 * m))),
                        );
                        shi[c] = _mm256_add_epi16(
                            shi[c],
                            _mm256_maddubs_epi16(hi, bcast4(y.add(64 * k + 32 + 4 * m))),
                        );
                    }
                }
                let sc_lo = pair_u8(block.add(32 + 8 * (2 * k)));
                let sc_hi = pair_u8(block.add(32 + 8 * (2 * k + 1)));
                for c in 0..C {
                    let p = _mm256_add_epi32(
                        _mm256_madd_epi16(slo[c], sc_lo),
                        _mm256_madd_epi16(shi[c], sc_hi),
                    );
                    sumi[c] = _mm256_add_epi32(sumi[c], p);
                }
            }
            for c in 0..C {
                let a = xs[c].get_unchecked(b);
                let bsum = |j: usize| a.bsums[2 * j] as i32 + a.bsums[2 * j + 1] as i32;
                let mut mins = _mm256_setzero_si256();
                for p in 0..4 {
                    let pairs = _mm256_loadu_si256(block.add(96 + 32 * p) as *const __m256i);
                    mins = _mm256_add_epi32(
                        mins,
                        _mm256_madd_epi16(pairs, pair16(bsum(2 * p), bsum(2 * p + 1))),
                    );
                }
                let ad = _mm256_set1_ps(a.d);
                acc[c] = _mm256_fmadd_ps(_mm256_mul_ps(d, ad), _mm256_cvtepi32_ps(sumi[c]), acc[c]);
                acc[c] =
                    _mm256_fnmadd_ps(_mm256_mul_ps(dmin, ad), _mm256_cvtepi32_ps(mins), acc[c]);
            }
        }
        let mut out = [[0.0_f32; 8]; C];
        for c in 0..C {
            _mm256_storeu_ps(out[c].as_mut_ptr(), acc[c]);
        }
        out
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q6kx8<const C: usize>(group: &[u8], xs: [&[BlockQ8K]; C]) -> [[f32; 8]; C] {
        let m4 = _mm256_set1_epi8(0x0f);
        let m3 = _mm256_set1_epi8(3);
        let mut acc = [_mm256_setzero_ps(); C];
        let g = group.as_ptr();
        for b in 0..group.len() / 1680 {
            let block = g.add(b * 1680);
            let d = _mm256_cvtph_ps(_mm_loadu_si128(block as *const __m128i));
            let sc = block.add(16);
            let ql = block.add(144);
            let qh = block.add(1168);
            let mut sumi = [_mm256_setzero_si256(); C];
            for n in 0..2 {
                // Element type 0..4 of this half: (ql chunk base, low/high
                // nibble, qh shift, element offset).
                for ty in 0..4 {
                    let ql_base = 16 * n + if ty % 2 == 1 { 8 } else { 0 };
                    let qh_shift = _mm_cvtsi32_si128(2 * ty as i32);
                    let high_nibble = ty >= 2;
                    let e_base = 128 * n + 32 * ty;
                    for half in 0..2 {
                        let scv = pair_i8(sc.add(8 * (8 * n + 2 * ty + half)));
                        // Two maddubs of 6-bit x 8-bit stay within i16: 32004.
                        for pair in 0..2 {
                            let mut s16 = [_mm256_setzero_si256(); C];
                            for t in 4 * half + 2 * pair..4 * half + 2 * pair + 2 {
                                let l = _mm256_loadu_si256(
                                    ql.add(32 * (ql_base + t)) as *const __m256i
                                );
                                let l = if high_nibble {
                                    _mm256_srli_epi16(l, 4)
                                } else {
                                    l
                                };
                                let h =
                                    _mm256_loadu_si256(qh.add(32 * (8 * n + t)) as *const __m256i);
                                let h = _mm256_and_si256(_mm256_srl_epi16(h, qh_shift), m3);
                                let q = _mm256_or_si256(
                                    _mm256_and_si256(l, m4),
                                    _mm256_slli_epi16(h, 4),
                                );
                                for c in 0..C {
                                    let y = bcast4(
                                        xs[c].get_unchecked(b).qs.as_ptr().add(e_base + 4 * t),
                                    );
                                    s16[c] = _mm256_add_epi16(s16[c], _mm256_maddubs_epi16(q, y));
                                }
                            }
                            for c in 0..C {
                                sumi[c] = _mm256_add_epi32(sumi[c], _mm256_madd_epi16(s16[c], scv));
                            }
                        }
                    }
                }
            }
            for c in 0..C {
                let a = xs[c].get_unchecked(b);
                // q was taken as unsigned 0..63: subtract 32 * sum(sc * y).
                let mut corr = _mm256_setzero_si256();
                for p in 0..8 {
                    let lo = _mm_loadl_epi64(sc.add(8 * (2 * p)) as *const __m128i);
                    let hi = _mm_loadl_epi64(sc.add(8 * (2 * p + 1)) as *const __m128i);
                    let pairs = _mm256_cvtepi8_epi16(_mm_unpacklo_epi8(lo, hi));
                    let bs = pair16(a.bsums[2 * p] as i32, a.bsums[2 * p + 1] as i32);
                    corr = _mm256_add_epi32(corr, _mm256_madd_epi16(pairs, bs));
                }
                let total = _mm256_sub_epi32(sumi[c], _mm256_slli_epi32(corr, 5));
                acc[c] = _mm256_fmadd_ps(
                    _mm256_mul_ps(d, _mm256_set1_ps(a.d)),
                    _mm256_cvtepi32_ps(total),
                    acc[c],
                );
            }
        }
        let mut out = [[0.0_f32; 8]; C];
        for c in 0..C {
            _mm256_storeu_ps(out[c].as_mut_ptr(), acc[c]);
        }
        out
    }

    /// exp(x) for 8 lanes: Cephes-style range reduction and degree-5
    /// polynomial, max relative error ~2 ulp on [-87, 88]; inputs below the
    /// range flush to 0, above saturate to f32::MAX.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn exp8(x: __m256) -> __m256 {
        let x = _mm256_min_ps(
            _mm256_max_ps(x, _mm256_set1_ps(-87.336_54)),
            _mm256_set1_ps(88.376_26),
        );
        // x = n * ln2 + r, |r| <= ln2 / 2
        let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            _mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E)),
        );
        let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693_359_4), x);
        let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.121_944_4e-4), r);
        let mut p = _mm256_set1_ps(1.987_569_1e-4);
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.398_199_9e-3));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.333_452e-3));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.166_579_6e-2));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.666_666_5e-1));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5));
        let r2 = _mm256_mul_ps(r, r);
        let y = _mm256_add_ps(_mm256_fmadd_ps(p, r2, r), _mm256_set1_ps(1.0));
        // Scale by 2^n via the exponent field.
        let pow2 = _mm256_slli_epi32::<23>(_mm256_add_epi32(
            _mm256_cvtps_epi32(n),
            _mm256_set1_epi32(127),
        ));
        _mm256_mul_ps(y, _mm256_castsi256_ps(pow2))
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn swiglu(gate: &mut [f32], up: &[f32]) -> usize {
        let n = gate.len() / 8 * 8;
        let one = _mm256_set1_ps(1.0);
        let zero = _mm256_setzero_ps();
        for i in (0..n).step_by(8) {
            let g = _mm256_loadu_ps(gate.as_ptr().add(i));
            let u = _mm256_loadu_ps(up.as_ptr().add(i));
            let e = exp8(_mm256_sub_ps(zero, g));
            let silu = _mm256_div_ps(g, _mm256_add_ps(one, e));
            _mm256_storeu_ps(gate.as_mut_ptr().add(i), _mm256_mul_ps(silu, u));
        }
        n
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn exp_shifted_sum(values: &mut [f32], max: f32) -> (usize, f32) {
        let n = values.len() / 8 * 8;
        let m = _mm256_set1_ps(max);
        let mut sum = _mm256_setzero_ps();
        for i in (0..n).step_by(8) {
            let e = exp8(_mm256_sub_ps(_mm256_loadu_ps(values.as_ptr().add(i)), m));
            _mm256_storeu_ps(values.as_mut_ptr().add(i), e);
            sum = _mm256_add_ps(sum, e);
        }
        (n, hsum(sum))
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn load_f16(src: &[u8], dst: &mut [f32]) -> usize {
        let n = dst.len() / 8 * 8;
        for i in (0..n).step_by(8) {
            let h = _mm_loadu_si128(src.as_ptr().add(2 * i) as *const __m128i);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_cvtph_ps(h));
        }
        n
    }

    /// bf16 is the top half of an f32, so widening is a shift, not a
    /// conversion -- no f16c needed and exact by construction.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn bf16x8(p: *const u8) -> __m256 {
        let bits = _mm256_cvtepu16_epi32(_mm_loadu_si128(p as *const __m128i));
        _mm256_castsi256_ps(_mm256_slli_epi32(bits, 16))
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn load_bf16(src: &[u8], dst: &mut [f32]) -> usize {
        let n = dst.len() / 8 * 8;
        for i in (0..n).step_by(8) {
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), bf16x8(src.as_ptr().add(2 * i)));
        }
        n
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_bf16(row: &[u8], x: &[f32]) -> f32 {
        let n = x.len();
        let w = row.as_ptr();
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i + 32 <= n {
            for (k, a) in acc.iter_mut().enumerate() {
                *a = _mm256_fmadd_ps(
                    bf16x8(w.add(2 * (i + 8 * k))),
                    _mm256_loadu_ps(x.as_ptr().add(i + 8 * k)),
                    *a,
                );
            }
            i += 32;
        }
        let mut sum = hsum(_mm256_add_ps(
            _mm256_add_ps(acc[0], acc[1]),
            _mm256_add_ps(acc[2], acc[3]),
        ));
        while i < n {
            sum += crate::quant::bf16_to_f32(std::ptr::read_unaligned(w.add(2 * i) as *const u16))
                * x[i];
            i += 1;
        }
        sum
    }

    /// f32 · f32 (used after dequantizing a row once for several columns).
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_f32_f32(a: &[f32], x: &[f32]) -> f32 {
        let n = a.len();
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i + 32 <= n {
            for (k, s) in acc.iter_mut().enumerate() {
                *s = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a.as_ptr().add(i + 8 * k)),
                    _mm256_loadu_ps(x.as_ptr().add(i + 8 * k)),
                    *s,
                );
            }
            i += 32;
        }
        let mut sum = hsum(_mm256_add_ps(
            _mm256_add_ps(acc[0], acc[1]),
            _mm256_add_ps(acc[2], acc[3]),
        ));
        while i < n {
            sum += a[i] * x[i];
            i += 1;
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{
        dequantize_row, encode_f16, encode_f32, quantize_q4_0, quantize_q8_0 as q8w,
    };

    /// Deterministic pseudo-random bytes and values.
    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    fn random_block(kind: GgmlType, seed: &mut u64) -> Vec<u8> {
        let mut bytes: Vec<u8> = (0..kind.block_bytes()).map(|_| lcg(seed) as u8).collect();
        // Keep f16 scales finite and modest.
        let scale_offsets: &[usize] = match kind {
            GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q4_1 | GgmlType::Q5_1 => &[0, 2],
            GgmlType::Q5_0 => &[0],
            GgmlType::Q6K => &[208],
            _ => &[],
        };
        for &at in scale_offsets {
            let value = (lcg(seed) % 1000) as f32 / 20000.0 + 0.001;
            bytes[at..at + 2].copy_from_slice(&crate::quant::f32_to_f16(value).to_le_bytes());
        }
        bytes
    }

    /// Types whose blocks can be filled with random bytes (finite scales).
    const RANDOM_TYPES: [GgmlType; 6] = [
        GgmlType::Q4K,
        GgmlType::Q5K,
        GgmlType::Q6K,
        GgmlType::Q4_1,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
    ];

    /// A random 1024-element row.
    fn random_row(kind: GgmlType, seed: &mut u64) -> Vec<u8> {
        (0..1024 / kind.block_elements())
            .flat_map(|_| random_block(kind, seed))
            .collect()
    }

    fn input(n: usize, seed: &mut u64) -> Vec<f32> {
        (0..n)
            .map(|_| (lcg(seed) % 2001) as f32 / 1000.0 - 1.0)
            .collect()
    }

    fn check(kind: GgmlType, row: &[u8], x: &[f32], tolerance: f32) {
        let mut deq = vec![0.0; x.len()];
        dequantize_row(kind, row, &mut deq);
        let exact: f64 = deq.iter().zip(x).map(|(a, b)| *a as f64 * *b as f64).sum();
        let scale: f64 = deq
            .iter()
            .zip(x)
            .map(|(a, b)| (*a as f64 * *b as f64).abs())
            .sum::<f64>()
            .max(1e-6);
        let prepared = prepare(kind, x);
        let scalar = dot_scalar(kind, row, &prepared);
        let fast = dot(kind, row, &prepared);
        // SIMD and scalar use identical integer arithmetic.
        assert!(
            ((scalar - fast) as f64).abs() <= 1e-4 * scale,
            "{}: scalar {scalar} vs simd {fast}",
            kind.name()
        );
        // Activation quantization error relative to the magnitude of the terms.
        assert!(
            ((scalar as f64 - exact) / scale).abs() <= tolerance as f64,
            "{}: scalar {scalar} vs exact {exact} (scale {scale})",
            kind.name()
        );
    }

    #[test]
    fn kernels_agree_with_reference() {
        let mut seed = 7;
        for &kind in &RANDOM_TYPES {
            for _ in 0..20 {
                let row = random_row(kind, &mut seed);
                let x = input(1024, &mut seed);
                check(kind, &row, &x, 0.02);
            }
        }
        for _ in 0..20 {
            let values = input(1024, &mut seed);
            let x = input(1024, &mut seed);
            check(GgmlType::Q4_0, &quantize_q4_0(&values), &x, 0.02);
            check(GgmlType::Q8_0, &q8w(&values), &x, 0.02);
            check(GgmlType::F16, &encode_f16(&values), &x, 1e-5);
            check(GgmlType::F32, &encode_f32(&values), &x, 1e-5);
        }
    }

    #[test]
    fn multi_column_kernels_match_single_column() {
        let mut seed = 11;
        let mut cases: Vec<(GgmlType, Vec<u8>)> = RANDOM_TYPES
            .iter()
            .map(|&kind| (kind, random_row(kind, &mut seed)))
            .collect();
        cases.push((GgmlType::Q4_0, quantize_q4_0(&input(1024, &mut seed))));
        cases.push((GgmlType::Q8_0, q8w(&input(1024, &mut seed))));
        cases.push((GgmlType::F16, encode_f16(&input(1024, &mut seed))));
        for (kind, row) in cases {
            for columns in [1, 3, 4, 7] {
                let inputs: Vec<Vec<f32>> = (0..columns).map(|_| input(1024, &mut seed)).collect();
                let prepared: Vec<Prepared> = inputs.iter().map(|x| prepare(kind, x)).collect();
                let mut out = vec![0.0; columns];
                dot_multi(kind, &row, &prepared, &mut out, &mut Vec::new());
                for (o, p) in out.iter().zip(&prepared) {
                    let single = dot(kind, &row, p);
                    assert!(
                        (o - single).abs() <= 1e-4 * single.abs().max(1.0),
                        "{}: {o} vs {single}",
                        kind.name()
                    );
                }
            }
        }
    }

    #[test]
    fn vector_exp_and_swiglu_are_accurate() {
        // The kernel saturates outside [-87.3, 88.3]; test inside it.
        let xs: Vec<f32> = (0..4000).map(|i| -87.0 + i as f32 * 0.0437).collect();
        let mut shifted = xs.clone();
        let sum = exp_shifted_sum(&mut shifted, 0.0);
        let mut expected_sum = 0.0_f64;
        for (x, e) in xs.iter().zip(&shifted) {
            let exact = (*x as f64).exp();
            expected_sum += exact;
            if exact > 1e-30 {
                assert!(
                    ((*e as f64 - exact) / exact).abs() < 4e-7,
                    "exp({x}) = {e}, want {exact}"
                );
            }
        }
        let _ = (sum, expected_sum); // that sum overflows f32; softmax never sees one
        // Softmax subtracts the max first, so every term is <= 1.
        let mut scores: Vec<f32> = (0..1000).map(|i| -(i as f32) * 0.03).collect();
        let exact: f64 = scores.iter().map(|&x| (x as f64).exp()).sum();
        let got = exp_shifted_sum(&mut scores, 0.0);
        assert!(
            ((got as f64 - exact) / exact).abs() < 1e-6,
            "{got} vs {exact}"
        );
        let mut gate: Vec<f32> = (0..1001).map(|i| -20.0 + i as f32 * 0.04).collect();
        let up: Vec<f32> = (0..1001).map(|i| (i as f32 * 0.1).cos()).collect();
        let expected: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        swiglu_in_place(&mut gate, &up);
        for (g, e) in gate.iter().zip(&expected) {
            assert!((g - e).abs() <= 1e-6 * e.abs().max(1.0), "{g} vs {e}");
        }
    }

    #[test]
    fn packed_x8_matches_row_major() {
        if !packable(GgmlType::Q4_0, 8) {
            return; // no AVX2: packing is disabled
        }
        let mut seed = 5;
        let (rows, cols) = (24, 1024);
        let cases: Vec<(GgmlType, Vec<u8>)> = vec![
            (
                GgmlType::Q4_0,
                quantize_q4_0(&input(rows * cols, &mut seed)),
            ),
            (GgmlType::Q8_0, q8w(&input(rows * cols, &mut seed))),
            (
                GgmlType::Q4K,
                (0..rows)
                    .flat_map(|_| random_row(GgmlType::Q4K, &mut seed))
                    .collect(),
            ),
            (
                GgmlType::Q6K,
                (0..rows)
                    .flat_map(|_| random_row(GgmlType::Q6K, &mut seed))
                    .collect(),
            ),
        ];
        for (kind, data) in cases {
            let packed = repack_x8(kind, &data, rows, cols).unwrap();
            assert_eq!(packed.len(), packed_len(kind, data.len()));
            let row_bytes = data.len() / rows;
            let group_bytes = packed.len() / (rows / 8);
            for columns in [1, 2, 3, 4, 5] {
                let inputs: Vec<Vec<f32>> = (0..columns).map(|_| input(cols, &mut seed)).collect();
                let prepared: Vec<Prepared> = inputs.iter().map(|x| prepare(kind, x)).collect();
                for group in 0..rows / 8 {
                    let g = &packed[group * group_bytes..(group + 1) * group_bytes];
                    let mut out = vec![0.0; columns * 8];
                    dot_x8(kind, g, &prepared, &mut out);
                    for (c, p) in prepared.iter().enumerate() {
                        for r in 0..8 {
                            let row = &data[(group * 8 + r) * row_bytes..][..row_bytes];
                            let reference = dot_scalar(kind, row, p);
                            let got = out[c * 8 + r];
                            assert!(
                                (got - reference).abs() <= 1e-4 * reference.abs().max(1.0),
                                "{} row {r} col {c}: packed {got} vs {reference}",
                                kind.name()
                            );
                        }
                    }
                }
            }
        }
    }

    /// The SIMD activation quantizer must agree with the scalar reference
    /// bit for bit: these values feed every dot product, so a one-step
    /// difference here would show up as a perplexity difference that looks
    /// like a kernel bug.
    #[test]
    fn simd_q8_0_quantization_matches_scalar() {
        let mut seed = 12345_u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let mut x: Vec<f32> = Vec::new();
        // A spread of magnitudes, exact ties (0.5 steps, where
        // round-half-to-even differs from round-half-away), zeros, and
        // negatives.
        for _ in 0..64 {
            x.push((next() % 2001) as f32 / 1000.0 - 1.0);
        }
        for i in 0..32 {
            x.push((i as f32 - 16.0) * 0.5);
        }
        x.extend([0.0; 32]);
        for i in 0..32 {
            x.push(if i % 2 == 0 { 1e-30 } else { -1e-30 });
        }
        for i in 0..32 {
            x.push(if i == 7 { -65504.0 } else { 1.0 });
        }
        let fast = quantize_q8_0(&x);
        let reference = quantize_q8_0_scalar(&x);
        assert_eq!(fast.len(), reference.len());
        for (b, (a, e)) in fast.iter().zip(&reference).enumerate() {
            assert_eq!(a.d.to_bits(), e.d.to_bits(), "block {b} scale");
            assert_eq!(a.s.to_bits(), e.s.to_bits(), "block {b} sum");
            assert_eq!(a.qs, e.qs, "block {b} quants");
        }
    }

    /// Same contract for the K-quant activation format, including the cases
    /// its two awkward details turn on: ties under round-half-away-from-zero,
    /// and which value attains the maximum magnitude first (it sets the
    /// scale's sign).
    #[test]
    fn simd_q8k_quantization_matches_scalar() {
        let mut seed = 777_u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let mut x: Vec<f32> = Vec::new();
        for _ in 0..crate::quant::QK_K {
            x.push((next() % 4001) as f32 / 1000.0 - 2.0);
        }
        // Positive maximum first, then an equal negative one.
        let mut block = vec![0.5_f32; crate::quant::QK_K];
        block[10] = 3.0;
        block[200] = -3.0;
        x.extend(&block);
        // Negative maximum first: the scale's sign flips.
        block[10] = -3.0;
        block[200] = 3.0;
        x.extend(&block);
        // All zeros, which takes the early-out path.
        x.extend(vec![0.0_f32; crate::quant::QK_K]);
        // Values that land exactly on .5 after scaling.
        let mut ties = vec![0.0_f32; crate::quant::QK_K];
        for (i, v) in ties.iter_mut().enumerate() {
            *v = (i as f32 - 128.0) / 254.0;
        }
        x.extend(&ties);
        let fast = quantize_q8k(&x);
        let reference = quantize_q8k_scalar(&x);
        assert_eq!(fast.len(), reference.len());
        for (b, (a, e)) in fast.iter().zip(&reference).enumerate() {
            assert_eq!(a.d.to_bits(), e.d.to_bits(), "block {b} scale");
            assert_eq!(a.qs, e.qs, "block {b} quants");
            assert_eq!(a.bsums, e.bsums, "block {b} group sums");
        }
    }

    #[test]
    fn q8k_extremes_do_not_overflow() {
        let mut x = vec![1.0_f32; 256];
        x[0] = -1.0; // max-magnitude value negative -> iscale positive
        let blocks = quantize_q8k(&x);
        assert_eq!(blocks[0].qs[0], -127);
        assert_eq!(blocks[0].qs[1], 127);
    }

    #[test]
    fn f16_kv_helpers() {
        let mut seed = 3;
        let v = input(64, &mut seed);
        let q = input(64, &mut seed);
        let mut bits = vec![0_u16; 64];
        store_f16(&v, &mut bits);
        for (b, x) in bits.iter().zip(&v) {
            assert_eq!(*b, crate::quant::f32_to_f16(*x));
        }
        let expected: f32 = bits.iter().zip(&q).map(|(b, x)| f16_to_f32(*b) * x).sum();
        assert!((dot_f16_f32(&bits, &q) - expected).abs() < 1e-4);
        let mut out = vec![1.0_f32; 64];
        axpy_f16(&mut out, 0.5, &bits);
        for ((o, b), _) in out.iter().zip(&bits).zip(0..) {
            assert!((o - (1.0 + 0.5 * f16_to_f32(*b))).abs() < 1e-6);
        }
    }
}
