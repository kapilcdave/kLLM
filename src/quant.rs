//! GGML tensor types: block layouts, f16/bf16 conversion, and reference
//! (scalar) dequantization.
//!
//! The scalar code here is the correctness reference. Faster kernels in
//! [`crate::kernels`] are tested against it.

/// GGML tensor element types kLLM knows the layout of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

pub const QK_K: usize = 256;

impl GgmlType {
    pub fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            30 => Self::BF16,
            _ => return None,
        })
    }

    pub fn id(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            Self::BF16 => 30,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
        }
    }

    /// Elements per block.
    pub fn block_elements(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => QK_K,
        }
    }

    /// Bytes per block.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
        }
    }

    /// Whether kLLM can compute with this type (not merely size it).
    pub fn is_supported(self) -> bool {
        !matches!(self, Self::Q8_1 | Self::Q3K | Self::Q8K)
    }

    /// Bytes needed for one row of `columns` elements.
    pub fn row_bytes(self, columns: usize) -> Option<usize> {
        if !columns.is_multiple_of(self.block_elements()) {
            return None;
        }
        (columns / self.block_elements()).checked_mul(self.block_bytes())
    }
}

/// IEEE 754 binary16 to binary32, exact for every input including
/// subnormals, infinities, and NaN payloads.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x03ff);
    let out = match (exponent, mantissa) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: value = mantissa * 2^-24.
            let value = mantissa as f32 * f32::from_bits(0x3380_0000);
            return if sign != 0 { -value } else { value };
        }
        (0x1f, _) => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(out)
}

/// binary32 to binary16 with round-to-nearest-even.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0xff {
        let nan = if mantissa != 0 { 0x0200 } else { 0 };
        return sign | 0x7c00 | nan | (mantissa >> 13) as u16;
    }
    let unbiased = exponent - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased >= -14 {
        let half_exponent = (unbiased + 15) as u32;
        let mut half = (half_exponent << 10) | (mantissa >> 13);
        let rest = mantissa & 0x1fff;
        if rest > 0x1000 || (rest == 0x1000 && (half & 1) == 1) {
            half += 1; // May carry into the exponent, which is correct.
        }
        return sign | half as u16;
    }
    if unbiased < -25 {
        return sign;
    }
    // Subnormal half.
    let full = mantissa | 0x0080_0000;
    let shift = (-unbiased - 14 + 13) as u32;
    let mut half = full >> shift;
    let rest = full & ((1 << shift) - 1);
    let halfway = 1 << (shift - 1);
    if rest > halfway || (rest == halfway && (half & 1) == 1) {
        half += 1;
    }
    sign | half as u16
}

#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[inline]
fn half(bytes: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
}

/// Dequantize one block of `kind` from `block` into `out`
/// (`out.len() == kind.block_elements()`).
pub fn dequantize_block(kind: GgmlType, block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), kind.block_bytes());
    debug_assert_eq!(out.len(), kind.block_elements());
    match kind {
        GgmlType::F32 => out[0] = f32::from_le_bytes(block[..4].try_into().unwrap()),
        GgmlType::F16 => out[0] = half(block, 0),
        GgmlType::BF16 => out[0] = bf16_to_f32(u16::from_le_bytes([block[0], block[1]])),
        GgmlType::Q4_0 => {
            let d = half(block, 0);
            let qs = &block[2..18];
            for j in 0..16 {
                out[j] = ((qs[j] & 0x0f) as i32 - 8) as f32 * d;
                out[j + 16] = ((qs[j] >> 4) as i32 - 8) as f32 * d;
            }
        }
        GgmlType::Q4_1 => {
            let d = half(block, 0);
            let m = half(block, 2);
            let qs = &block[4..20];
            for j in 0..16 {
                out[j] = (qs[j] & 0x0f) as f32 * d + m;
                out[j + 16] = (qs[j] >> 4) as f32 * d + m;
            }
        }
        GgmlType::Q5_0 => {
            let d = half(block, 0);
            let qh = u32::from_le_bytes(block[2..6].try_into().unwrap());
            let qs = &block[6..22];
            for j in 0..16 {
                let h0 = ((qh >> j) & 1) << 4;
                let h1 = ((qh >> (j + 16)) & 1) << 4;
                out[j] = (((qs[j] & 0x0f) as u32 | h0) as i32 - 16) as f32 * d;
                out[j + 16] = (((qs[j] >> 4) as u32 | h1) as i32 - 16) as f32 * d;
            }
        }
        GgmlType::Q5_1 => {
            let d = half(block, 0);
            let m = half(block, 2);
            let qh = u32::from_le_bytes(block[4..8].try_into().unwrap());
            let qs = &block[8..24];
            for j in 0..16 {
                let h0 = ((qh >> j) & 1) << 4;
                let h1 = ((qh >> (j + 16)) & 1) << 4;
                out[j] = ((qs[j] & 0x0f) as u32 | h0) as f32 * d + m;
                out[j + 16] = ((qs[j] >> 4) as u32 | h1) as f32 * d + m;
            }
        }
        GgmlType::Q8_0 => {
            let d = half(block, 0);
            for (o, &q) in out.iter_mut().zip(&block[2..34]) {
                *o = (q as i8) as f32 * d;
            }
        }
        GgmlType::Q2K => dequantize_q2k(block, out),
        GgmlType::Q4K => dequantize_q4k(block, out),
        GgmlType::Q5K => dequantize_q5k(block, out),
        GgmlType::Q6K => dequantize_q6k(block, out),
        GgmlType::Q8_1 | GgmlType::Q3K | GgmlType::Q8K => {
            unreachable!("{} is not a supported compute type", kind.name())
        }
    }
}

/// Unpack the 6-bit (scale, min) pair `j` of a Q4_K/Q5_K block.
#[inline]
pub fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn dequantize_q2k(block: &[u8], out: &mut [f32]) {
    // Layout: scales[16], qs[64], d, dmin.
    let scales = &block[0..16];
    let qs = &block[16..80];
    let d = half(block, 80);
    let dmin = half(block, 82);
    let mut is = 0;
    let mut y = 0;
    for n in 0..2 {
        let q = &qs[n * 32..n * 32 + 32];
        for shift in [0, 2, 4, 6] {
            for half_index in 0..2 {
                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0x0f) as f32;
                let ml = dmin * (sc >> 4) as f32;
                for l in 0..16 {
                    let value = (q[half_index * 16 + l] >> shift) & 3;
                    out[y] = dl * value as f32 - ml;
                    y += 1;
                }
            }
        }
    }
}

fn dequantize_q4k(block: &[u8], out: &mut [f32]) {
    // Layout: d, dmin, scales[12], qs[128].
    let d = half(block, 0);
    let dmin = half(block, 2);
    let scales = &block[4..16];
    let qs = &block[16..144];
    for j in 0..4 {
        let (s1, m1) = scale_min_k4(2 * j, scales);
        let (s2, m2) = scale_min_k4(2 * j + 1, scales);
        let (d1, m1) = (d * s1 as f32, dmin * m1 as f32);
        let (d2, m2) = (d * s2 as f32, dmin * m2 as f32);
        let q = &qs[j * 32..j * 32 + 32];
        let y = &mut out[j * 64..j * 64 + 64];
        for l in 0..32 {
            y[l] = d1 * (q[l] & 0x0f) as f32 - m1;
            y[l + 32] = d2 * (q[l] >> 4) as f32 - m2;
        }
    }
}

fn dequantize_q5k(block: &[u8], out: &mut [f32]) {
    // Layout: d, dmin, scales[12], qh[32], qs[128].
    let d = half(block, 0);
    let dmin = half(block, 2);
    let scales = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];
    for j in 0..4 {
        let (s1, m1) = scale_min_k4(2 * j, scales);
        let (s2, m2) = scale_min_k4(2 * j + 1, scales);
        let (d1, m1) = (d * s1 as f32, dmin * m1 as f32);
        let (d2, m2) = (d * s2 as f32, dmin * m2 as f32);
        let u1 = 1_u8 << (2 * j);
        let u2 = 2_u8 << (2 * j);
        let q = &qs[j * 32..j * 32 + 32];
        let y = &mut out[j * 64..j * 64 + 64];
        for l in 0..32 {
            let hi1 = if qh[l] & u1 != 0 { 16 } else { 0 };
            let hi2 = if qh[l] & u2 != 0 { 16 } else { 0 };
            y[l] = d1 * ((q[l] & 0x0f) + hi1) as f32 - m1;
            y[l + 32] = d2 * ((q[l] >> 4) + hi2) as f32 - m2;
        }
    }
}

fn dequantize_q6k(block: &[u8], out: &mut [f32]) {
    // Layout: ql[128], qh[64], scales[16] (i8), d.
    let d = half(block, 208);
    for n in 0..2 {
        let ql = &block[n * 64..n * 64 + 64];
        let qh = &block[128 + n * 32..128 + n * 32 + 32];
        let sc = &block[192 + n * 8..192 + n * 8 + 8];
        let y = &mut out[n * 128..n * 128 + 128];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0x0f) | ((qh[l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
            y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
            y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
            y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
            y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
        }
    }
}

/// Dequantize a whole row. `row.len()` must equal `kind.row_bytes(out.len())`.
pub fn dequantize_row(kind: GgmlType, row: &[u8], out: &mut [f32]) {
    let be = kind.block_elements();
    let bb = kind.block_bytes();
    debug_assert_eq!(Some(row.len()), kind.row_bytes(out.len()));
    match kind {
        GgmlType::F32 => {
            for (o, b) in out.iter_mut().zip(row.chunks_exact(4)) {
                *o = f32::from_le_bytes(b.try_into().unwrap());
            }
        }
        GgmlType::F16 => {
            for (o, b) in out.iter_mut().zip(row.chunks_exact(2)) {
                *o = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            }
        }
        _ => {
            for (o, b) in out.chunks_exact_mut(be).zip(row.chunks_exact(bb)) {
                dequantize_block(kind, b, o);
            }
        }
    }
}

/// Reference dot product of one stored row with an f32 vector.
pub fn dot_row_reference(kind: GgmlType, row: &[u8], x: &[f32]) -> f32 {
    let be = kind.block_elements();
    let bb = kind.block_bytes();
    let mut scratch = [0.0_f32; QK_K];
    let mut sum = 0.0_f32;
    if be == 1 {
        let mut tmp = [0.0_f32; 64];
        for (rows, xs) in row.chunks(64 * bb).zip(x.chunks(64)) {
            let n = xs.len();
            dequantize_row(kind, rows, &mut tmp[..n]);
            sum += tmp[..n].iter().zip(xs).map(|(a, b)| a * b).sum::<f32>();
        }
        return sum;
    }
    for (block, xs) in row.chunks_exact(bb).zip(x.chunks_exact(be)) {
        let out = &mut scratch[..be];
        dequantize_block(kind, block, out);
        sum += out.iter().zip(xs).map(|(a, b)| a * b).sum::<f32>();
    }
    sum
}

// ---------------------------------------------------------------------------
// Reference quantizers. These exist so tests can build quantized tensors from
// known f32 data; they follow the ggml reference rounding rules.

/// Quantize to Q8_0 (ggml `quantize_row_q8_0_ref`).
pub fn quantize_q8_0(values: &[f32]) -> Vec<u8> {
    assert_eq!(values.len() % 32, 0);
    let mut out = Vec::with_capacity(values.len() / 32 * 34);
    for block in values.chunks_exact(32) {
        let amax = block.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        for v in block {
            out.push((v * id).round() as i8 as u8);
        }
    }
    out
}

/// Quantize to Q4_0 (ggml `quantize_row_q4_0_ref`).
pub fn quantize_q4_0(values: &[f32]) -> Vec<u8> {
    assert_eq!(values.len() % 32, 0);
    let mut out = Vec::with_capacity(values.len() / 32 * 18);
    for block in values.chunks_exact(32) {
        let mut amax = 0.0_f32;
        let mut max = 0.0_f32;
        for &v in block {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        for j in 0..16 {
            let q0 = ((block[j] * id + 8.5) as i32).clamp(0, 15) as u8;
            let q1 = ((block[j + 16] * id + 8.5) as i32).clamp(0, 15) as u8;
            out.push(q0 | (q1 << 4));
        }
    }
    out
}

pub fn encode_f16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| f32_to_f16(*v).to_le_bytes())
        .collect()
}

pub fn encode_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trips_every_finite_value() {
        for bits in 0..=u16::MAX {
            let value = f16_to_f32(bits);
            if value.is_nan() {
                assert!(f16_to_f32(f32_to_f16(value)).is_nan());
            } else {
                assert_eq!(f32_to_f16(value), bits, "bits {bits:#06x} value {value}");
            }
        }
    }

    #[test]
    fn f16_known_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert_eq!(f32_to_f16(65520.0), 0x7c00);
        assert_eq!(f32_to_f16(1.0 + 1.0 / 2048.0), 0x3c00); // tie to even
    }

    fn ramp(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 13.0)
            .collect()
    }

    #[test]
    fn q8_0_round_trip_error_is_bounded() {
        let values = ramp(128);
        let bytes = quantize_q8_0(&values);
        let mut out = vec![0.0; 128];
        dequantize_row(GgmlType::Q8_0, &bytes, &mut out);
        for (a, b) in values.iter().zip(&out) {
            assert!((a - b).abs() < 0.02, "{a} vs {b}");
        }
    }

    #[test]
    fn q4_0_round_trip_error_is_bounded() {
        let values = ramp(64);
        let bytes = quantize_q4_0(&values);
        let mut out = vec![0.0; 64];
        dequantize_row(GgmlType::Q4_0, &bytes, &mut out);
        // Q4_0 spans [-8, 7] * d with d = -max / 8, so the far end of the
        // range is clipped by up to amax / 8; everything else is within d / 2.
        for (block, deq) in values.chunks(32).zip(out.chunks(32)) {
            let amax = block.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
            for (a, b) in block.iter().zip(deq) {
                assert!((a - b).abs() <= amax / 8.0 + 1e-3, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn q4_k_block_layout() {
        // d = 1, dmin = 0.5, scale/min pair j = (j + 1, j).
        let mut block = vec![0_u8; 144];
        block[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        block[2..4].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
        let scales: Vec<(u8, u8)> = (0..8).map(|j| (j as u8 + 1, j as u8)).collect();
        let packed = &mut block[4..16];
        for j in 0..4 {
            packed[j] = scales[j].0 | ((scales[j + 4].0 >> 4) << 6);
            packed[j + 4] = scales[j].1 | ((scales[j + 4].1 >> 4) << 6);
            packed[j + 8] = (scales[j + 4].0 & 0x0f) | ((scales[j + 4].1 & 0x0f) << 4);
        }
        for (j, expected) in scales.iter().enumerate() {
            assert_eq!(scale_min_k4(j, &block[4..16]), *expected);
        }
        for (i, b) in block[16..144].iter_mut().enumerate() {
            *b = (i % 16) as u8 | (((i + 3) % 16) as u8) << 4;
        }
        let mut out = vec![0.0; 256];
        dequantize_row(GgmlType::Q4K, &block, &mut out);
        // Element 0: low nibble 0, scale 1, min 0 -> 0.
        assert_eq!(out[0], 0.0);
        // Element 32+5 (high nibbles of chunk 0): q = (5+3)%16 = 8, pair 1 = (2, 1)
        assert_eq!(out[37], 2.0 * 8.0 - 0.5);
        // Element 64+1 (chunk 1 low): byte 33 low = 1, pair 2 = (3, 2)
        assert_eq!(out[65], 3.0 * 1.0 - 1.0);
    }

    #[test]
    fn q6_k_block_layout() {
        let mut block = vec![0_u8; 210];
        block[208..210].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
        for (i, s) in block[192..208].iter_mut().enumerate() {
            *s = (i as i8 - 4) as u8;
        }
        // Element 0: ql[0] low | qh[0] bits 0..1. Set q = 33 -> value 1.
        block[0] = 33 & 0x0f;
        block[128] = (33 >> 4) & 3;
        let mut out = vec![0.0; 256];
        dequantize_row(GgmlType::Q6K, &block, &mut out);
        assert_eq!(out[0], 0.5 * -4.0 * 1.0);
        // Element 1 is q = 0 -> -32 with scale index 0.
        assert_eq!(out[1], 0.5 * -4.0 * -32.0);
    }

    #[test]
    fn reference_dot_matches_dequantized_dot() {
        let values = ramp(256);
        let x: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
        for (kind, bytes) in [
            (GgmlType::F32, encode_f32(&values)),
            (GgmlType::F16, encode_f16(&values)),
            (GgmlType::Q8_0, quantize_q8_0(&values)),
            (GgmlType::Q4_0, quantize_q4_0(&values)),
        ] {
            let mut deq = vec![0.0; 256];
            dequantize_row(kind, &bytes, &mut deq);
            let expected: f32 = deq.iter().zip(&x).map(|(a, b)| a * b).sum();
            let got = dot_row_reference(kind, &bytes, &x);
            assert!(
                (expected - got).abs() < 1e-3,
                "{}: {expected} vs {got}",
                kind.name()
            );
        }
    }
}
