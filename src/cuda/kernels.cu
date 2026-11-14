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
