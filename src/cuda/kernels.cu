// kLLM CUDA kernels for legacy NVIDIA GPUs (compute 5.0 and newer).
//
// Compiled to PTX by scripts/build_ptx.py; no CUDA headers are used, so the
// file builds with bare NVRTC. Everything here is plain FP32 arithmetic on
// CUDA cores: Maxwell has neither tensor cores nor dp4a.
//
// Layouts match the CPU backend: matrices are row-major in their GGML block
// format; activations are [batch][n] f32; the KV cache is [pos][kv_dim] f16.

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef signed char i8;

#define WARP 32
#define ROWS_PER_BLOCK 4 // one warp per output row

__device__ __forceinline__ float h2f(u16 h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

__device__ __forceinline__ u16 f2h(float f) {
    u16 h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}

// Unaligned-safe little-endian loads (GGML blocks are packed, e.g. 18 bytes).
__device__ __forceinline__ u16 ld16(const u8* p) { return (u16)p[0] | ((u16)p[1] << 8); }
__device__ __forceinline__ u32 ld32(const u8* p) {
    return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = WARP / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Block-wide sum for blockDim.x <= 1024; `scratch` needs 32 floats.
__device__ float block_sum(float v, float* scratch) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    int warps = (blockDim.x + 31) / 32;
    v = (threadIdx.x < warps) ? scratch[threadIdx.x] : 0.0f;
    if (warp == 0) v = warp_sum(v);
    if (threadIdx.x == 0) scratch[0] = v;
    __syncthreads();
    return scratch[0];
}

__device__ float block_max(float v, float* scratch) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    int warps = (blockDim.x + 31) / 32;
    v = (threadIdx.x < warps) ? scratch[threadIdx.x] : -3.402823466e38f;
    if (warp == 0) v = warp_max(v);
    if (threadIdx.x == 0) scratch[0] = v;
    __syncthreads();
    return scratch[0];
}

// ---------------------------------------------------------------------------
// Block dot products: one quantized block against the matching activations.

__device__ __forceinline__ float dot_q4_0(const u8* b, const float* x) {
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; j++) {
        u8 q = b[2 + j];
        s += (float)((int)(q & 15) - 8) * x[j] + (float)((int)(q >> 4) - 8) * x[j + 16];
    }
    return s * h2f(ld16(b));
}

__device__ __forceinline__ float dot_q4_1(const u8* b, const float* x) {
    float s = 0.0f, xs = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; j++) {
        u8 q = b[4 + j];
        s += (float)(q & 15) * x[j] + (float)(q >> 4) * x[j + 16];
        xs += x[j] + x[j + 16];
    }
    return s * h2f(ld16(b)) + xs * h2f(ld16(b + 2));
}

__device__ __forceinline__ float dot_q5_0(const u8* b, const float* x) {
    u32 qh = ld32(b + 2);
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; j++) {
        u8 q = b[6 + j];
        int lo = (int)((q & 15) | (((qh >> j) & 1) << 4)) - 16;
        int hi = (int)((q >> 4) | (((qh >> (j + 16)) & 1) << 4)) - 16;
        s += (float)lo * x[j] + (float)hi * x[j + 16];
    }
    return s * h2f(ld16(b));
}

__device__ __forceinline__ float dot_q5_1(const u8* b, const float* x) {
    u32 qh = ld32(b + 4);
    float s = 0.0f, xs = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; j++) {
        u8 q = b[8 + j];
        int lo = (int)((q & 15) | (((qh >> j) & 1) << 4));
        int hi = (int)((q >> 4) | (((qh >> (j + 16)) & 1) << 4));
        s += (float)lo * x[j] + (float)hi * x[j + 16];
        xs += x[j] + x[j + 16];
    }
    return s * h2f(ld16(b)) + xs * h2f(ld16(b + 2));
}

__device__ __forceinline__ float dot_q8_0(const u8* b, const float* x) {
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; j++) s += (float)(i8)b[2 + j] * x[j];
    return s * h2f(ld16(b));
}

__device__ __forceinline__ void scale_min_k4(int j, const u8* q, int* sc, int* m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

__device__ __forceinline__ float dot_q4_k(const u8* b, const float* x) {
    float d = h2f(ld16(b)), dmin = h2f(ld16(b + 2));
    const u8* scales = b + 4;
    const u8* qs = b + 16;
    float s = 0.0f;
    for (int k = 0; k < 4; k++) {
        int sc1, m1, sc2, m2;
        scale_min_k4(2 * k, scales, &sc1, &m1);
        scale_min_k4(2 * k + 1, scales, &sc2, &m2);
        const u8* q = qs + 32 * k;
        const float* xa = x + 64 * k;
        float a = 0.0f, bsum = 0.0f, xa_sum = 0.0f, xb_sum = 0.0f;
        for (int l = 0; l < 32; l++) {
            a += (float)(q[l] & 15) * xa[l];
            bsum += (float)(q[l] >> 4) * xa[l + 32];
            xa_sum += xa[l];
            xb_sum += xa[l + 32];
        }
        s += d * (sc1 * a + sc2 * bsum) - dmin * (m1 * xa_sum + m2 * xb_sum);
    }
    return s;
}

__device__ __forceinline__ float dot_q5_k(const u8* b, const float* x) {
    float d = h2f(ld16(b)), dmin = h2f(ld16(b + 2));
    const u8* scales = b + 4;
    const u8* qh = b + 16;
    const u8* qs = b + 48;
    float s = 0.0f;
    for (int k = 0; k < 4; k++) {
        int sc1, m1, sc2, m2;
        scale_min_k4(2 * k, scales, &sc1, &m1);
        scale_min_k4(2 * k + 1, scales, &sc2, &m2);
        const u8* q = qs + 32 * k;
        const float* xa = x + 64 * k;
        float a = 0.0f, bsum = 0.0f, xa_sum = 0.0f, xb_sum = 0.0f;
        for (int l = 0; l < 32; l++) {
            int lo = (q[l] & 15) | (((qh[l] >> (2 * k)) & 1) << 4);
            int hi = (q[l] >> 4) | (((qh[l] >> (2 * k + 1)) & 1) << 4);
            a += (float)lo * xa[l];
            bsum += (float)hi * xa[l + 32];
            xa_sum += xa[l];
            xb_sum += xa[l + 32];
        }
        s += d * (sc1 * a + sc2 * bsum) - dmin * (m1 * xa_sum + m2 * xb_sum);
    }
    return s;
}

__device__ __forceinline__ float dot_q6_k(const u8* b, const float* x) {
    float d = h2f(ld16(b + 208));
    float s = 0.0f;
    for (int n = 0; n < 2; n++) {
        const u8* ql = b + 64 * n;
        const u8* qh = b + 128 + 32 * n;
        const i8* sc = (const i8*)(b + 192 + 8 * n);
        const float* y = x + 128 * n;
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int q1 = (int)((ql[l] & 15) | ((qh[l] & 3) << 4)) - 32;
            int q2 = (int)((ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            s += sc[is] * q1 * y[l] + sc[is + 2] * q2 * y[l + 32] + sc[is + 4] * q3 * y[l + 64] +
                 sc[is + 6] * q4 * y[l + 96];
        }
    }
    return d * s;
}

__device__ __forceinline__ float dot_q2_k(const u8* b, const float* x) {
    const u8* scales = b;
    const u8* qs = b + 16;
    float d = h2f(ld16(b + 80)), dmin = h2f(ld16(b + 82));
    float s = 0.0f;
    int is = 0, y = 0;
    for (int n = 0; n < 2; n++) {
        const u8* q = qs + 32 * n;
        for (int shift = 0; shift < 8; shift += 2) {
            for (int h = 0; h < 2; h++) {
                u8 sc = scales[is++];
                float dl = d * (sc & 15), ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; l++) {
                    s += (dl * ((q[16 * h + l] >> shift) & 3) - ml) * x[y++];
                }
            }
        }
    }
    return s;
}

// Matvec / batched matmul. One warp per output row; lanes stride over blocks
// (quantized) or elements (float types). grid = (ceil(rows / 4), batch).
// y[b * rows + r] = W[r] . x[b * cols ..]

#define BLOCK_MATVEC(NAME, BYTES, ELEMS, DOT)                                         \
    extern "C" __global__ void NAME(const u8* w, const float* x, float* y, int rows,   \
                                    int cols) {                                       \
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;                         \
        int row = blockIdx.x * ROWS_PER_BLOCK + warp;                                 \
        if (row >= rows) return;                                                      \
        int nb = cols / ELEMS;                                                        \
        const u8* r = w + (size_t)row * nb * BYTES;                                   \
        const float* xb = x + (size_t)blockIdx.y * cols;                              \
        float sum = 0.0f;                                                             \
        for (int b = lane; b < nb; b += WARP) sum += DOT(r + (size_t)b * BYTES, xb + b * ELEMS); \
        sum = warp_sum(sum);                                                          \
        if (lane == 0) y[(size_t)blockIdx.y * rows + row] = sum;                     \
    }

BLOCK_MATVEC(matvec_q4_0, 18, 32, dot_q4_0)
BLOCK_MATVEC(matvec_q4_1, 20, 32, dot_q4_1)
BLOCK_MATVEC(matvec_q5_0, 22, 32, dot_q5_0)
BLOCK_MATVEC(matvec_q5_1, 24, 32, dot_q5_1)
BLOCK_MATVEC(matvec_q8_0, 34, 32, dot_q8_0)
BLOCK_MATVEC(matvec_q2_k, 84, 256, dot_q2_k)
BLOCK_MATVEC(matvec_q4_k, 144, 256, dot_q4_k)
BLOCK_MATVEC(matvec_q5_k, 176, 256, dot_q5_k)
BLOCK_MATVEC(matvec_q6_k, 210, 256, dot_q6_k)

extern "C" __global__ void matvec_f16(const u8* w, const float* x, float* y, int rows, int cols) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * ROWS_PER_BLOCK + warp;
    if (row >= rows) return;
    const u16* r = (const u16*)w + (size_t)row * cols; // f16 rows are 2-byte aligned
    const float* xb = x + (size_t)blockIdx.y * cols;
    float sum = 0.0f;
    for (int i = lane; i < cols; i += WARP) sum += h2f(r[i]) * xb[i];
    sum = warp_sum(sum);
    if (lane == 0) y[(size_t)blockIdx.y * rows + row] = sum;
}

extern "C" __global__ void matvec_bf16(const u8* w, const float* x, float* y, int rows, int cols) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * ROWS_PER_BLOCK + warp;
    if (row >= rows) return;
    const u16* r = (const u16*)w + (size_t)row * cols;
    const float* xb = x + (size_t)blockIdx.y * cols;
    float sum = 0.0f;
    for (int i = lane; i < cols; i += WARP) sum += __uint_as_float((u32)r[i] << 16) * xb[i];
    sum = warp_sum(sum);
    if (lane == 0) y[(size_t)blockIdx.y * rows + row] = sum;
}

extern "C" __global__ void matvec_f32(const u8* w, const float* x, float* y, int rows, int cols) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * ROWS_PER_BLOCK + warp;
    if (row >= rows) return;
    const float* r = (const float*)w + (size_t)row * cols;
    const float* xb = x + (size_t)blockIdx.y * cols;
    float sum = 0.0f;
    for (int i = lane; i < cols; i += WARP) sum += r[i] * xb[i];
    sum = warp_sum(sum);
    if (lane == 0) y[(size_t)blockIdx.y * rows + row] = sum;
}

// ---------------------------------------------------------------------------
// Elementwise and normalisation kernels.

// RMSNorm over `rows` rows of length n: y = x / rms(x) * w. In-place safe.
// grid = rows, block = 256.
extern "C" __global__ void rms_norm(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float scratch[32];
    const float* xr = x + (size_t)blockIdx.x * n;
    float* yr = y + (size_t)blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, scratch);
    float inv = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * inv * w[i];
}

// x[b * n + i] += v[i] (bias) for every batch row. grid covers n * batch.
extern "C" __global__ void add_bias(float* x, const float* v, int n, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) x[i] += v[i % n];
}

// x[i] += y[i].
extern "C" __global__ void add_inplace(float* x, const float* y, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) x[i] += y[i];
}

// gate[i] = silu(gate[i]) * up[i].
extern "C" __global__ void swiglu(float* gate, const float* up, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        float g = gate[i];
        gate[i] = g / (1.0f + expf(-g)) * up[i];
    }
}

// Rotary embedding in place. `x` is [batch][heads][head_dim]; `table` holds
// (cos, sin) for [batch][rope_dim / 2], computed on the host exactly as the
// CPU backend does. grid = batch * heads, block = rope_dim / 2.
extern "C" __global__ void rope(float* x, const float* table, int heads, int head_dim, int rope_dim,
                                int neox) {
    int bh = blockIdx.x, j = threadIdx.x, half = rope_dim / 2;
    if (j >= half) return;
    int b = bh / heads;
    float* h = x + (size_t)bh * head_dim;
    float c = table[((size_t)b * half + j) * 2], s = table[((size_t)b * half + j) * 2 + 1];
    int i0 = neox ? j : 2 * j, i1 = neox ? j + half : 2 * j + 1;
    float a = h[i0], bb = h[i1];
    h[i0] = a * c - bb * s;
    h[i1] = a * s + bb * c;
}

// Write `batch` rows of k and v (f32, kv_dim wide) into the f16 cache at
// positions start..start+batch. grid covers batch * kv_dim.
extern "C" __global__ void kv_store(const float* k, const float* v, u16* k_cache, u16* v_cache,
                                    int kv_dim, int start, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    size_t at = (size_t)start * kv_dim + i;
    k_cache[at] = f2h(k[i]);
    v_cache[at] = f2h(v[i]);
}

// Causal attention for one (token, head). q is [batch][heads][head_dim];
// token b sees cache positions 0..start+b. `scores` is scratch of
// [batch * heads][context]. grid = (heads, batch), block = 128.
extern "C" __global__ void attention(const float* q, const u16* k_cache, const u16* v_cache,
                                     float* out, float* scores, int heads, int kv_heads,
                                     int head_dim, int start, int context, float scale) {
    __shared__ float scratch[32];
    __shared__ float qs[256];
    int h = blockIdx.x, b = blockIdx.y;
    int kv_dim = kv_heads * head_dim;
    int kv_off = (h / (heads / kv_heads)) * head_dim;
    int len = start + b + 1;
    const float* qh = q + ((size_t)b * heads + h) * head_dim;
    float* sc = scores + ((size_t)b * heads + h) * context;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) qs[i] = qh[i];
    __syncthreads();
    float mx = -3.402823466e38f;
    for (int t = threadIdx.x; t < len; t += blockDim.x) {
        const u16* kr = k_cache + (size_t)t * kv_dim + kv_off;
        float s = 0.0f;
        for (int i = 0; i < head_dim; i++) s += h2f(kr[i]) * qs[i];
        s *= scale;
        sc[t] = s;
        mx = fmaxf(mx, s);
    }
    mx = block_max(mx, scratch);
    float sum = 0.0f;
    for (int t = threadIdx.x; t < len; t += blockDim.x) {
        float e = expf(sc[t] - mx);
        sc[t] = e;
        sum += e;
    }
    sum = block_sum(sum, scratch);
    float inv = 1.0f / sum;
    __syncthreads();
    float* o = out + ((size_t)b * heads + h) * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < len; t++) acc += sc[t] * h2f(v_cache[(size_t)t * kv_dim + kv_off + i]);
        o[i] = acc * inv;
    }
}

// ---------------------------------------------------------------------------
// Coalesced matvecs over a struct-of-arrays weight layout.
//
// The row-major GGML kernels above walk a whole block per lane with byte
// loads, which neither coalesces nor uses the load width; on a clock-starved
// GPU they reach under 1 GB/s. Here the host splits each matrix into aligned
// arrays (quants apart from scales) on upload, lanes share a block and issue
// aligned 32-bit loads, integers become floats with a byte permute and one
// subtraction (Maxwell's int->float convert runs at quarter rate), and one
// pass over the weights serves N activation columns.
//
// grid = (ceil(rows / 4), ceil(batch / N)), block = 128 (one warp per row).

// Float with value 2^23 + byte i of `v`: PRMT builds the bit pattern
// 0x4B0000vv. Subtract (2^23 + offset) to get `byte - offset`.
__device__ __forceinline__ float byte_float(u32 v, int i) {
    return __int_as_float(__byte_perm(v, 0x4B000000u, 0x7540 + i));
}

#define SOA_PROLOGUE                                                          \
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;                     \
    int row = blockIdx.x * ROWS_PER_BLOCK + warp;                             \
    if (row >= rows) return;                                                  \
    int col0 = blockIdx.y * N;                                                \
    float acc[N];                                                             \
    _Pragma("unroll") for (int n = 0; n < N; n++) acc[n] = 0.0f;

#define SOA_EPILOGUE                                                          \
    _Pragma("unroll") for (int n = 0; n < N; n++) {                           \
        float s = warp_sum(acc[n]);                                           \
        if (lane == 0 && col0 + n < batch) y[(size_t)(col0 + n) * rows + row] = s; \
    }

// Q4_0: quants [rows * nb][16 B], scales [rows * nb] f16.
// Four lanes per block (4 bytes = elements 4c..4c+3 and 16+4c..16+4c+3).
template <int N>
__device__ void mv_q4_0(const u8* qs, const u16* sc, const float* x, float* y, int rows, int cols,
                        int batch) {
    SOA_PROLOGUE
    int nb = cols / 32, c = lane & 3;
    for (int base = 0; base < nb; base += 8) {
        int b = base + (lane >> 2);
        if (b >= nb) continue;
        size_t blk = (size_t)row * nb + b;
        u32 q = *(const u32*)(qs + blk * 16 + 4 * c);
        u32 lo = q & 0x0F0F0F0Fu, hi = (q >> 4) & 0x0F0F0F0Fu;
        float d = h2f(sc[blk]);
        float wl[4], wh[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            wl[i] = byte_float(lo, i) - 8388616.0f; // 2^23 + 8
            wh[i] = byte_float(hi, i) - 8388616.0f;
        }
#pragma unroll
        for (int n = 0; n < N; n++) {
            if (col0 + n >= batch) break;
            const float* xb = x + (size_t)(col0 + n) * cols + b * 32;
            float4 a = *(const float4*)(xb + 4 * c);
            float4 e = *(const float4*)(xb + 16 + 4 * c);
            float s = wl[0] * a.x + wl[1] * a.y + wl[2] * a.z + wl[3] * a.w + wh[0] * e.x +
                      wh[1] * e.y + wh[2] * e.z + wh[3] * e.w;
            acc[n] += d * s;
        }
    }
    SOA_EPILOGUE
}

// Q8_0: quants [rows * nb][32 B], scales [rows * nb] f16. Eight lanes per block.
template <int N>
__device__ void mv_q8_0(const u8* qs, const u16* sc, const float* x, float* y, int rows, int cols,
                        int batch) {
    SOA_PROLOGUE
    int nb = cols / 32, c = lane & 7;
    for (int base = 0; base < nb; base += 4) {
        int b = base + (lane >> 3);
        if (b >= nb) continue;
        size_t blk = (size_t)row * nb + b;
        u32 q = *(const u32*)(qs + blk * 32 + 4 * c) ^ 0x80808080u; // signed -> biased
        float d = h2f(sc[blk]);
        float w[4];
#pragma unroll
        for (int i = 0; i < 4; i++) w[i] = byte_float(q, i) - 8388736.0f; // 2^23 + 128
#pragma unroll
        for (int n = 0; n < N; n++) {
            if (col0 + n >= batch) break;
            float4 a = *(const float4*)(x + (size_t)(col0 + n) * cols + b * 32 + 4 * c);
            acc[n] += d * (w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w);
        }
    }
    SOA_EPILOGUE
}

// Q4_K: quants [rows * nb][128 B], meta [rows * nb][16 B] = d, dmin, scales[12].
// One warp per 256-element block: lane t holds qs[4t..4t+4], i.e. elements
// 4(t%8).. of sub-blocks 2(t/8) (low nibbles) and 2(t/8)+1 (high nibbles).
template <int N>
__device__ void mv_q4_k(const u8* qs, const u8* meta, const float* x, float* y, int rows, int cols,
                        int batch) {
    SOA_PROLOGUE
    int nb = cols / 256, k = lane >> 3, l = 4 * (lane & 7);
    for (int b = 0; b < nb; b++) {
        size_t blk = (size_t)row * nb + b;
        const u8* m = meta + blk * 16;
        float d = h2f(ld16(m)), dmin = h2f(ld16(m + 2));
        int sc1, m1, sc2, m2;
        scale_min_k4(2 * k, m + 4, &sc1, &m1);
        scale_min_k4(2 * k + 1, m + 4, &sc2, &m2);
        u32 q = *(const u32*)(qs + blk * 128 + 4 * lane);
        u32 lo = q & 0x0F0F0F0Fu, hi = (q >> 4) & 0x0F0F0F0Fu;
        float wl[4], wh[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            wl[i] = byte_float(lo, i) - 8388608.0f;
            wh[i] = byte_float(hi, i) - 8388608.0f;
        }
        float dl = d * sc1, ml = dmin * m1, dh = d * sc2, mh = dmin * m2;
#pragma unroll
        for (int n = 0; n < N; n++) {
            if (col0 + n >= batch) break;
            const float* xb = x + (size_t)(col0 + n) * cols + b * 256 + 64 * k + l;
            float4 a = *(const float4*)xb;
            float4 e = *(const float4*)(xb + 32);
            float sl = wl[0] * a.x + wl[1] * a.y + wl[2] * a.z + wl[3] * a.w;
            float sh = wh[0] * e.x + wh[1] * e.y + wh[2] * e.z + wh[3] * e.w;
            float xl = a.x + a.y + a.z + a.w, xh = e.x + e.y + e.z + e.w;
            acc[n] += dl * sl - ml * xl + dh * sh - mh * xh;
        }
    }
    SOA_EPILOGUE
}

// Q6_K: ql [rows * nb][128 B], qh [rows * nb][64 B], meta [rows * nb][32 B] =
// i8 scales[16], f16 d. Lane t takes half n = t / 16 and positions l, l + 1
// with l = 2 (t % 16); each position yields four elements (l, l+32, l+64,
// l+96 within the half).
template <int N>
__device__ void mv_q6_k(const u8* ql, const u8* qh, const u8* meta, const float* x, float* y,
                        int rows, int cols, int batch) {
    SOA_PROLOGUE
    int nb = cols / 256, half = lane >> 4, l = 2 * (lane & 15);
    for (int b = 0; b < nb; b++) {
        size_t blk = (size_t)row * nb + b;
        const u8* m = meta + blk * 32;
        const i8* sc = (const i8*)m + 8 * half + l / 16;
        float d = h2f(ld16(m + 16));
        u32 a0 = *(const u16*)(ql + blk * 128 + 64 * half + l);      // positions l, l+1
        u32 a1 = *(const u16*)(ql + blk * 128 + 64 * half + 32 + l); // l+32, l+33
        u32 h = *(const u16*)(qh + blk * 64 + 32 * half + l);
        // Assemble the 6-bit values of the four element groups, two per byte pair.
        u32 g0 = (a0 & 0x0F0Fu) | ((h & 0x0303u) << 4);
        u32 g1 = (a1 & 0x0F0Fu) | (((h >> 2) & 0x0303u) << 4);
        u32 g2 = ((a0 >> 4) & 0x0F0Fu) | (((h >> 4) & 0x0303u) << 4);
        u32 g3 = ((a1 >> 4) & 0x0F0Fu) | (((h >> 6) & 0x0303u) << 4);
        float w[4][2];
#pragma unroll
        for (int i = 0; i < 2; i++) {
            w[0][i] = byte_float(g0, i) - 8388640.0f; // 2^23 + 32
            w[1][i] = byte_float(g1, i) - 8388640.0f;
            w[2][i] = byte_float(g2, i) - 8388640.0f;
            w[3][i] = byte_float(g3, i) - 8388640.0f;
        }
        float s0 = d * sc[0], s1 = d * sc[2], s2 = d * sc[4], s3 = d * sc[6];
#pragma unroll
        for (int n = 0; n < N; n++) {
            if (col0 + n >= batch) break;
            const float* xb = x + (size_t)(col0 + n) * cols + b * 256 + 128 * half + l;
            float2 e0 = *(const float2*)xb, e1 = *(const float2*)(xb + 32);
            float2 e2 = *(const float2*)(xb + 64), e3 = *(const float2*)(xb + 96);
            acc[n] += s0 * (w[0][0] * e0.x + w[0][1] * e0.y) + s1 * (w[1][0] * e1.x + w[1][1] * e1.y) +
                      s2 * (w[2][0] * e2.x + w[2][1] * e2.y) + s3 * (w[3][0] * e3.x + w[3][1] * e3.y);
        }
    }
    SOA_EPILOGUE
}

// Entry points: (a, b, c) are the layout regions, unused ones ignored.
extern "C" __global__ void soa_q4_0_1(const u8* a, const u8* b, const u8* c, const float* x, float* y, int rows, int cols, int batch) { mv_q4_0<1>(a, (const u16*)b, x, y, rows, cols, batch); }
extern "C" __global__ void soa_q4_0_4(const u8* a, const u8* b, const u8* c, const float* x, float* y, int rows, int cols, int batch) { mv_q4_0<4>(a, (const u16*)b, x, y, rows, cols, batch); }
extern "C" __global__ void soa_q8_0_1(const u8* a, const u8* b, const u8* c, const float* x, float* y, int rows, int cols, int batch) { mv_q8_0<1>(a, (const u16*)b, x, y, rows, cols, batch); }
extern "C" __global__ void soa_q8_0_4(const u8* a, const u8* b, const u8* c, const float* x, float* y, int rows, int cols, int batch) { mv_q8_0<4>(a, (const u16*)b, x, y, rows, cols, batch); }
extern "C" __global__ void soa_q4_k_1(const u8* a, const u8* b, const u8* c, const float* x, float* y, int rows, int cols, int batch) { mv_q4_k<1>(a, b, x, y, rows, cols, batch); }
extern "C" __global__ void soa_q4_k_4(const u8* a, const u8* b, const u8* c, const float* x, float* y, int rows, int cols, int batch) { mv_q4_k<4>(a, b, x, y, rows, cols, batch); }
extern "C" __global__ void soa_q6_k_1(const u8* a, const u8* b, const u8* c, const float* x, float* y, int rows, int cols, int batch) { mv_q6_k<1>(a, b, c, x, y, rows, cols, batch); }
extern "C" __global__ void soa_q6_k_4(const u8* a, const u8* b, const u8* c, const float* x, float* y, int rows, int cols, int batch) { mv_q6_k<4>(a, b, c, x, y, rows, cols, batch); }
