//! CUDA backend for legacy NVIDIA GPUs.
//!
//! kLLM talks to the CUDA *driver* API directly, loading `libcuda.so` at run
//! time. The kernels in `kernels.cu` ship pre-compiled to PTX for compute 5.0
//! (Maxwell); the driver JIT-compiles them for the installed GPU. So neither
//! building nor running kLLM needs a CUDA toolkit, and a machine without an
//! NVIDIA driver simply reports that no GPU is available.
//!
//! All work is issued on the legacy default stream, which serialises kernels
//! and copies; correctness first, overlap later.

use std::{
    ffi::{CStr, c_char, c_int, c_uint, c_void},
    fmt,
    sync::{Arc, OnceLock},
};

use libloading::Library;

use crate::backend::{Backend, BackendError, Matrix, require_len};
use crate::quant::GgmlType;

const PTX: &str = include_str!("kernels.ptx");

type CuResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
pub type DevicePtr = u64;

const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
const CU_DEVICE_ATTRIBUTE_CLOCK_RATE: c_int = 13;
const CU_JIT_ERROR_LOG_BUFFER: c_int = 5;
const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_int = 6;

#[derive(Debug, Clone)]
pub struct CudaError {
    pub what: String,
    pub code: i32,
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.code == 0 {
            write!(f, "CUDA: {}", self.what)
        } else {
            write!(f, "CUDA: {} failed with error {}", self.what, self.code)
        }
    }
}

impl std::error::Error for CudaError {}

impl From<CudaError> for BackendError {
    fn from(_: CudaError) -> Self {
        BackendError::InvalidParameter("CUDA operation failed")
    }
}

#[allow(non_snake_case)]
struct Api {
    cuInit: unsafe extern "C" fn(c_uint) -> CuResult,
    cuDeviceGetCount: unsafe extern "C" fn(*mut c_int) -> CuResult,
    cuDeviceGet: unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult,
    cuDeviceGetName: unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult,
    cuDeviceGetAttribute: unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> CuResult,
    cuDeviceTotalMem: unsafe extern "C" fn(*mut usize, CuDevice) -> CuResult,
    cuDevicePrimaryCtxRetain: unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult,
    cuDevicePrimaryCtxRelease: unsafe extern "C" fn(CuDevice) -> CuResult,
    cuCtxSetCurrent: unsafe extern "C" fn(CuContext) -> CuResult,
    cuCtxSynchronize: unsafe extern "C" fn() -> CuResult,
    cuModuleLoadDataEx: unsafe extern "C" fn(
        *mut CuModule,
        *const c_void,
        c_uint,
        *mut c_int,
        *mut *mut c_void,
    ) -> CuResult,
    cuModuleUnload: unsafe extern "C" fn(CuModule) -> CuResult,
    cuModuleGetFunction: unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult,
    cuMemAlloc: unsafe extern "C" fn(*mut DevicePtr, usize) -> CuResult,
    cuMemFree: unsafe extern "C" fn(DevicePtr) -> CuResult,
    cuMemcpyHtoD: unsafe extern "C" fn(DevicePtr, *const c_void, usize) -> CuResult,
    cuMemcpyDtoH: unsafe extern "C" fn(*mut c_void, DevicePtr, usize) -> CuResult,
    cuMemcpyDtoD: unsafe extern "C" fn(DevicePtr, DevicePtr, usize) -> CuResult,
    cuMemsetD8: unsafe extern "C" fn(DevicePtr, u8, usize) -> CuResult,
    cuMemGetInfo: unsafe extern "C" fn(*mut usize, *mut usize) -> CuResult,
    cuLaunchKernel: unsafe extern "C" fn(
        CuFunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult,
    _lib: Library,
}

impl Api {
    fn load() -> Result<Self, CudaError> {
        let fail = |what: String| CudaError { what, code: 0 };
        // SAFETY: loading the system's NVIDIA driver library.
        let lib = unsafe { Library::new("libcuda.so.1").or_else(|_| Library::new("libcuda.so")) }
            .map_err(|e| {
            fail(format!(
                "cannot load libcuda.so ({e}); is the NVIDIA driver installed?"
            ))
        })?;
        macro_rules! sym {
            ($name:literal) => {{
                // SAFETY: the signature matches the CUDA driver API header.
                let s = unsafe { lib.get(concat!($name, "\0").as_bytes()) }
                    .map_err(|e| fail(format!("libcuda.so lacks {} ({e})", $name)))?;
                *s
            }};
        }
        Ok(Self {
            cuInit: sym!("cuInit"),
            cuDeviceGetCount: sym!("cuDeviceGetCount"),
            cuDeviceGet: sym!("cuDeviceGet"),
            cuDeviceGetName: sym!("cuDeviceGetName"),
            cuDeviceGetAttribute: sym!("cuDeviceGetAttribute"),
            cuDeviceTotalMem: sym!("cuDeviceTotalMem_v2"),
            cuDevicePrimaryCtxRetain: sym!("cuDevicePrimaryCtxRetain"),
            cuDevicePrimaryCtxRelease: sym!("cuDevicePrimaryCtxRelease_v2"),
            cuCtxSetCurrent: sym!("cuCtxSetCurrent"),
            cuCtxSynchronize: sym!("cuCtxSynchronize"),
            cuModuleLoadDataEx: sym!("cuModuleLoadDataEx"),
            cuModuleUnload: sym!("cuModuleUnload"),
            cuModuleGetFunction: sym!("cuModuleGetFunction"),
            cuMemAlloc: sym!("cuMemAlloc_v2"),
            cuMemFree: sym!("cuMemFree_v2"),
            cuMemcpyHtoD: sym!("cuMemcpyHtoD_v2"),
            cuMemcpyDtoH: sym!("cuMemcpyDtoH_v2"),
            cuMemcpyDtoD: sym!("cuMemcpyDtoD_v2"),
            cuMemsetD8: sym!("cuMemsetD8_v2"),
            cuMemGetInfo: sym!("cuMemGetInfo_v2"),
            cuLaunchKernel: sym!("cuLaunchKernel"),
            _lib: lib,
        })
    }
}

fn check(code: CuResult, what: &str) -> Result<(), CudaError> {
    if code == 0 {
        Ok(())
    } else {
        Err(CudaError {
            what: what.to_string(),
            code,
        })
    }
}

/// Every kernel in `kernels.cu`, resolved once at start-up.
#[derive(Clone, Copy)]
struct Kernels {
    matvec: [(GgmlType, CuFunction); 13],
    /// Coalesced struct-of-arrays matvecs: (type, 1-column, 4-column).
    soa: [(GgmlType, CuFunction, CuFunction); 4],
    rms_norm: CuFunction,
    add_bias: CuFunction,
    add_inplace: CuFunction,
    swiglu: CuFunction,
    split_q_gate: CuFunction,
    sigmoid_mul: CuFunction,
    conv_silu: CuFunction,
    l2_norm_heads: CuFunction,
    deltanet_step: CuFunction,
    rope: CuFunction,
    kv_store: CuFunction,
    attention: CuFunction,
}

/// An open GPU: driver handle, context, and the loaded kernel module.
pub struct Gpu {
    api: Api,
    device: CuDevice,
    context: CuContext,
    module: CuModule,
    kernels: Kernels,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub multiprocessors: i32,
    pub clock_mhz: i32,
    pub total_memory: u64,
}

// SAFETY: the driver API is thread-safe; every entry point first makes this
// GPU's context current on the calling thread (`bind`).
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

/// A kernel argument.
#[derive(Clone, Copy)]
pub enum Arg {
    Ptr(DevicePtr),
    I32(i32),
    F32(f32),
}

impl Gpu {
    /// Open GPU `ordinal`, load the kernels, and make its context current.
    pub fn open(ordinal: i32) -> Result<Arc<Self>, CudaError> {
        let api = Api::load()?;
        // SAFETY: plain driver API calls with valid out-pointers.
        unsafe {
            check((api.cuInit)(0), "cuInit")?;
            let mut count = 0;
            check((api.cuDeviceGetCount)(&mut count), "cuDeviceGetCount")?;
            if ordinal >= count {
                return Err(CudaError {
                    what: format!("GPU {ordinal} requested but {count} present"),
                    code: 0,
                });
            }
            let mut device = 0;
            check((api.cuDeviceGet)(&mut device, ordinal), "cuDeviceGet")?;
            let mut name = [0 as c_char; 256];
            check(
                (api.cuDeviceGetName)(name.as_mut_ptr(), 256, device),
                "cuDeviceGetName",
            )?;
            let name = CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned();
            let attr = |a| {
                let mut v = 0;
                check(
                    (api.cuDeviceGetAttribute)(&mut v, a, device),
                    "cuDeviceGetAttribute",
                )
                .map(|_| v)
            };
            let cc = (
                attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
                attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
            );
            let multiprocessors = attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
            let clock_mhz = attr(CU_DEVICE_ATTRIBUTE_CLOCK_RATE)? / 1000;
            let mut total = 0usize;
            check(
                (api.cuDeviceTotalMem)(&mut total, device),
                "cuDeviceTotalMem",
            )?;
            if cc.0 < 5 {
                return Err(CudaError {
                    what: format!(
                        "{name} is compute {}.{}; kLLM needs 5.0 or newer",
                        cc.0, cc.1
                    ),
                    code: 0,
                });
            }
            let mut context = std::ptr::null_mut();
            check(
                (api.cuDevicePrimaryCtxRetain)(&mut context, device),
                "cuDevicePrimaryCtxRetain",
            )?;
            check((api.cuCtxSetCurrent)(context), "cuCtxSetCurrent")?;

            let ptx = std::ffi::CString::new(PTX).expect("PTX contains no NUL");
            let mut log = vec![0_u8; 16 * 1024];
            let mut options = [CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES];
            let mut values = [log.as_mut_ptr() as *mut c_void, log.len() as *mut c_void];
            let mut module = std::ptr::null_mut();
            let code = (api.cuModuleLoadDataEx)(
                &mut module,
                ptx.as_ptr() as *const c_void,
                2,
                options.as_mut_ptr(),
                values.as_mut_ptr(),
            );
            if code != 0 {
                let text =
                    String::from_utf8_lossy(&log[..log.iter().position(|&b| b == 0).unwrap_or(0)])
                        .into_owned();
                return Err(CudaError {
                    what: format!("loading kernels (JIT log: {text})"),
                    code,
                });
            }
            let get = |name: &str| -> Result<CuFunction, CudaError> {
                let c = std::ffi::CString::new(name).unwrap();
                let mut f = std::ptr::null_mut();
                check((api.cuModuleGetFunction)(&mut f, module, c.as_ptr()), name)?;
                Ok(f)
            };
            let matvec = [
                (GgmlType::F32, get("matvec_f32")?),
                (GgmlType::F16, get("matvec_f16")?),
                (GgmlType::BF16, get("matvec_bf16")?),
                (GgmlType::Q4_0, get("matvec_q4_0")?),
                (GgmlType::Q4_1, get("matvec_q4_1")?),
                (GgmlType::Q5_0, get("matvec_q5_0")?),
                (GgmlType::Q5_1, get("matvec_q5_1")?),
                (GgmlType::Q8_0, get("matvec_q8_0")?),
                (GgmlType::Q2K, get("matvec_q2_k")?),
                (GgmlType::Q4K, get("matvec_q4_k")?),
                (GgmlType::Q5K, get("matvec_q5_k")?),
                (GgmlType::Q6K, get("matvec_q6_k")?),
                (GgmlType::Q8_1, std::ptr::null_mut()),
            ];
            let kernels = Kernels {
                matvec,
                soa: [
                    (GgmlType::Q4_0, get("soa_q4_0_1")?, get("soa_q4_0_4")?),
                    (GgmlType::Q8_0, get("soa_q8_0_1")?, get("soa_q8_0_4")?),
                    (GgmlType::Q4K, get("soa_q4_k_1")?, get("soa_q4_k_4")?),
                    (GgmlType::Q6K, get("soa_q6_k_1")?, get("soa_q6_k_4")?),
                ],
                rms_norm: get("rms_norm")?,
                add_bias: get("add_bias")?,
                add_inplace: get("add_inplace")?,
                swiglu: get("swiglu")?,
                split_q_gate: get("split_q_gate")?,
                sigmoid_mul: get("sigmoid_mul")?,
                conv_silu: get("conv_silu")?,
                l2_norm_heads: get("l2_norm_heads")?,
                deltanet_step: get("deltanet_step")?,
                rope: get("rope")?,
                kv_store: get("kv_store")?,
                attention: get("attention")?,
            };
            Ok(Arc::new(Self {
                api,
                device,
                context,
                module,
                kernels,
                name,
                compute_capability: cc,
                multiprocessors,
                clock_mhz,
                total_memory: total as u64,
            }))
        }
    }

    /// Make this GPU's context current on the calling thread.
    fn bind(&self) -> Result<(), CudaError> {
        // SAFETY: the context is valid for the lifetime of `self`.
        check(
            unsafe { (self.api.cuCtxSetCurrent)(self.context) },
            "cuCtxSetCurrent",
        )
    }

    /// (free, total) device memory in bytes.
    pub fn memory_info(&self) -> Result<(u64, u64), CudaError> {
        self.bind()?;
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: valid out-pointers.
        check(
            unsafe { (self.api.cuMemGetInfo)(&mut free, &mut total) },
            "cuMemGetInfo",
        )?;
        Ok((free as u64, total as u64))
    }

    pub fn supports(&self, kind: GgmlType) -> bool {
        self.kernels
            .matvec
            .iter()
            .any(|(k, f)| *k == kind && !f.is_null())
    }

    pub fn alloc(self: &Arc<Self>, bytes: usize) -> Result<DeviceBuffer, CudaError> {
        self.bind()?;
        let mut ptr = 0;
        // SAFETY: valid out-pointer; a zero-byte request is rounded up.
        check(
            unsafe { (self.api.cuMemAlloc)(&mut ptr, bytes.max(4)) },
            "cuMemAlloc",
        )
        .map_err(|e| CudaError {
            what: format!(
                "allocating {:.1} MiB ({})",
                bytes as f64 / 1048576.0,
                e.what
            ),
            code: e.code,
        })?;
        Ok(DeviceBuffer {
            gpu: Arc::clone(self),
            ptr,
            bytes,
        })
    }

    pub fn upload<T: Copy>(self: &Arc<Self>, data: &[T]) -> Result<DeviceBuffer, CudaError> {
        let buffer = self.alloc(std::mem::size_of_val(data))?;
        buffer.write(data)?;
        Ok(buffer)
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.bind()?;
        // SAFETY: no arguments.
        check(unsafe { (self.api.cuCtxSynchronize)() }, "kernel execution")
    }

    fn launch(
        &self,
        f: CuFunction,
        grid: (u32, u32),
        block: u32,
        args: &[Arg],
    ) -> Result<(), CudaError> {
        if grid.0 == 0 || grid.1 == 0 {
            return Ok(());
        }
        self.bind()?;
        let mut values: Vec<[u8; 8]> = args
            .iter()
            .map(|a| match *a {
                Arg::Ptr(p) => p.to_ne_bytes(),
                Arg::I32(v) => {
                    let mut b = [0; 8];
                    b[..4].copy_from_slice(&v.to_ne_bytes());
                    b
                }
                Arg::F32(v) => {
                    let mut b = [0; 8];
                    b[..4].copy_from_slice(&v.to_ne_bytes());
                    b
                }
            })
            .collect();
        let mut pointers: Vec<*mut c_void> = values
            .iter_mut()
            .map(|v| v.as_mut_ptr() as *mut c_void)
            .collect();
        // SAFETY: `pointers` holds one pointer per kernel parameter, each to
        // storage of the parameter's size, alive for the call.
        check(
            unsafe {
                (self.api.cuLaunchKernel)(
                    f,
                    grid.0,
                    grid.1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    pointers.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }

    fn blocks(n: usize, per: usize) -> u32 {
        n.div_ceil(per) as u32
    }

    /// `y[b] = W · x[b]` for `batch` columns; all pointers on the device.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        &self,
        kind: GgmlType,
        w: DevicePtr,
        rows: usize,
        cols: usize,
        x: DevicePtr,
        y: DevicePtr,
        batch: usize,
    ) -> Result<(), CudaError> {
        let f = self
            .kernels
            .matvec
            .iter()
            .find(|(k, f)| *k == kind && !f.is_null())
            .map(|(_, f)| *f)
            .ok_or_else(|| CudaError {
                what: format!("no GPU kernel for {}", kind.name()),
                code: 0,
            })?;
        self.launch(
            f,
            (Self::blocks(rows, 4), batch as u32),
            128,
            &[
                Arg::Ptr(w),
                Arg::Ptr(x),
                Arg::Ptr(y),
                Arg::I32(rows as i32),
                Arg::I32(cols as i32),
            ],
        )
    }

    /// Upload a weight matrix in the layout its kernel wants: struct-of-arrays
    /// for the types with coalesced kernels, row-major GGML otherwise.
    pub fn upload_matrix(
        self: &Arc<Self>,
        kind: GgmlType,
        data: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<GpuMatrix, CudaError> {
        match soa_layout(kind, data, rows, cols) {
            Some((bytes, regions)) => Ok(GpuMatrix {
                buffer: self.upload(&bytes)?,
                kind,
                rows,
                cols,
                regions: Some(regions),
            }),
            None => Ok(GpuMatrix {
                buffer: self.upload(data)?,
                kind,
                rows,
                cols,
                regions: None,
            }),
        }
    }

    /// `y[b] = W · x[b]` for `batch` columns of an uploaded matrix.
    pub fn matmul_matrix(
        &self,
        w: &GpuMatrix,
        x: DevicePtr,
        y: DevicePtr,
        batch: usize,
    ) -> Result<(), CudaError> {
        let Some(regions) = w.regions else {
            return self.matmul(w.kind, w.buffer.ptr, w.rows, w.cols, x, y, batch);
        };
        let &(_, one, four) = self
            .kernels
            .soa
            .iter()
            .find(|(k, _, _)| *k == w.kind)
            .expect("SoA layout implies a kernel");
        let (f, per) = if batch >= 4 { (four, 4) } else { (one, 1) };
        let region = |i: usize| Arg::Ptr(w.buffer.ptr + regions[i] as u64);
        self.launch(
            f,
            (Self::blocks(w.rows, 4), batch.div_ceil(per) as u32),
            128,
            &[
                region(0),
                region(1),
                region(2),
                Arg::Ptr(x),
                Arg::Ptr(y),
                Arg::I32(w.rows as i32),
                Arg::I32(w.cols as i32),
                Arg::I32(batch as i32),
            ],
        )
    }

    /// In-place-safe RMSNorm of `rows` rows of length `n`.
    pub fn rms_norm(
        &self,
        x: DevicePtr,
        w: DevicePtr,
        y: DevicePtr,
        rows: usize,
        n: usize,
        eps: f32,
    ) -> Result<(), CudaError> {
        self.launch(
            self.kernels.rms_norm,
            (rows as u32, 1),
            256,
            &[
                Arg::Ptr(x),
                Arg::Ptr(w),
                Arg::Ptr(y),
                Arg::I32(n as i32),
                Arg::F32(eps),
            ],
        )
    }

    pub fn add_bias(
        &self,
        x: DevicePtr,
        bias: DevicePtr,
        n: usize,
        batch: usize,
    ) -> Result<(), CudaError> {
        let total = n * batch;
        self.launch(
            self.kernels.add_bias,
            (Self::blocks(total, 256), 1),
            256,
            &[
                Arg::Ptr(x),
                Arg::Ptr(bias),
                Arg::I32(n as i32),
                Arg::I32(total as i32),
            ],
        )
    }

    pub fn add_inplace(&self, x: DevicePtr, y: DevicePtr, total: usize) -> Result<(), CudaError> {
        self.launch(
            self.kernels.add_inplace,
            (Self::blocks(total, 256), 1),
            256,
            &[Arg::Ptr(x), Arg::Ptr(y), Arg::I32(total as i32)],
        )
    }

    pub fn swiglu(&self, gate: DevicePtr, up: DevicePtr, total: usize) -> Result<(), CudaError> {
        self.launch(
            self.kernels.swiglu,
            (Self::blocks(total, 256), 1),
            256,
            &[Arg::Ptr(gate), Arg::Ptr(up), Arg::I32(total as i32)],
        )
    }

    pub fn split_q_gate(
        &self,
        full: DevicePtr,
        q: DevicePtr,
        gate: DevicePtr,
        head_dim: usize,
        total: usize,
    ) -> Result<(), CudaError> {
        self.launch(
            self.kernels.split_q_gate,
            (Self::blocks(total, 256), 1),
            256,
            &[
                Arg::Ptr(full),
                Arg::Ptr(q),
                Arg::Ptr(gate),
                Arg::I32(head_dim as i32),
                Arg::I32(total as i32),
            ],
        )
    }

    pub fn sigmoid_mul(
        &self,
        x: DevicePtr,
        gate: DevicePtr,
        total: usize,
    ) -> Result<(), CudaError> {
        self.launch(
            self.kernels.sigmoid_mul,
            (Self::blocks(total, 256), 1),
            256,
            &[Arg::Ptr(x), Arg::Ptr(gate), Arg::I32(total as i32)],
        )
    }

    pub fn conv_silu(
        &self,
        x: DevicePtr,
        history: DevicePtr,
        weights: DevicePtr,
        width: usize,
        taps: usize,
    ) -> Result<(), CudaError> {
        self.launch(
            self.kernels.conv_silu,
            (Self::blocks(width, 256), 1),
            256,
            &[
                Arg::Ptr(x),
                Arg::Ptr(history),
                Arg::Ptr(weights),
                Arg::I32(width as i32),
                Arg::I32(taps as i32),
            ],
        )
    }

    pub fn l2_norm_heads(
        &self,
        x: DevicePtr,
        heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<(), CudaError> {
        self.launch(
            self.kernels.l2_norm_heads,
            (heads as u32, 1),
            128,
            &[Arg::Ptr(x), Arg::I32(head_dim as i32), Arg::F32(eps)],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn deltanet_step(
        &self,
        mixed: DevicePtr,
        z: DevicePtr,
        alpha: DevicePtr,
        beta: DevicePtr,
        a: DevicePtr,
        dt_bias: DevicePtr,
        norm: DevicePtr,
        state: DevicePtr,
        out: DevicePtr,
        groups: usize,
        rank: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<(), CudaError> {
        self.launch(
            self.kernels.deltanet_step,
            (rank as u32, 1),
            256,
            &[
                Arg::Ptr(mixed),
                Arg::Ptr(z),
                Arg::Ptr(alpha),
                Arg::Ptr(beta),
                Arg::Ptr(a),
                Arg::Ptr(dt_bias),
                Arg::Ptr(norm),
                Arg::Ptr(state),
                Arg::Ptr(out),
                Arg::I32(groups as i32),
                Arg::I32(rank as i32),
                Arg::I32(head_dim as i32),
                Arg::F32(eps),
            ],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        x: DevicePtr,
        table: DevicePtr,
        batch: usize,
        heads: usize,
        head_dim: usize,
        rope_dim: usize,
        neox: bool,
    ) -> Result<(), CudaError> {
        self.launch(
            self.kernels.rope,
            ((batch * heads) as u32, 1),
            (rope_dim / 2).max(1) as u32,
            &[
                Arg::Ptr(x),
                Arg::Ptr(table),
                Arg::I32(heads as i32),
                Arg::I32(head_dim as i32),
                Arg::I32(rope_dim as i32),
                Arg::I32(neox as i32),
            ],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn kv_store(
        &self,
        k: DevicePtr,
        v: DevicePtr,
        k_cache: DevicePtr,
        v_cache: DevicePtr,
        kv_dim: usize,
        start: usize,
        batch: usize,
    ) -> Result<(), CudaError> {
        let total = kv_dim * batch;
        self.launch(
            self.kernels.kv_store,
            (Self::blocks(total, 256), 1),
            256,
            &[
                Arg::Ptr(k),
                Arg::Ptr(v),
                Arg::Ptr(k_cache),
                Arg::Ptr(v_cache),
                Arg::I32(kv_dim as i32),
                Arg::I32(start as i32),
                Arg::I32(total as i32),
            ],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        q: DevicePtr,
        k_cache: DevicePtr,
        v_cache: DevicePtr,
        out: DevicePtr,
        scores: DevicePtr,
        batch: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        start: usize,
        context: usize,
        scale: f32,
    ) -> Result<(), CudaError> {
        self.launch(
            self.kernels.attention,
            (heads as u32, batch as u32),
            128,
            &[
                Arg::Ptr(q),
                Arg::Ptr(k_cache),
                Arg::Ptr(v_cache),
                Arg::Ptr(out),
                Arg::Ptr(scores),
                Arg::I32(heads as i32),
                Arg::I32(kv_heads as i32),
                Arg::I32(head_dim as i32),
                Arg::I32(start as i32),
                Arg::I32(context as i32),
                Arg::F32(scale),
            ],
        )
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: the module and context were created by `open`.
        unsafe {
            let _ = (self.api.cuCtxSetCurrent)(self.context);
            let _ = (self.api.cuModuleUnload)(self.module);
            let _ = (self.api.cuDevicePrimaryCtxRelease)(self.device);
        }
    }
}

/// A weight matrix resident on the GPU.
pub struct GpuMatrix {
    pub buffer: DeviceBuffer,
    pub kind: GgmlType,
    pub rows: usize,
    pub cols: usize,
    /// Byte offsets of the struct-of-arrays regions, or `None` for row-major.
    regions: Option<[usize; 3]>,
}

/// Split a row-major GGML matrix into the struct-of-arrays layout the
/// coalesced kernels read (see kernels.cu). Every region starts 256-byte
/// aligned. Returns the bytes and the region offsets.
fn soa_layout(
    kind: GgmlType,
    data: &[u8],
    rows: usize,
    cols: usize,
) -> Option<(Vec<u8>, [usize; 3])> {
    // (bytes of each region per block, and how to fill them from a block).
    let sizes: [usize; 3] = match kind {
        GgmlType::Q4_0 => [16, 2, 0],
        GgmlType::Q8_0 => [32, 2, 0],
        GgmlType::Q4K => [128, 16, 0],
        GgmlType::Q6K => [128, 64, 32],
        _ => return None,
    };
    let blocks = rows * kind.row_bytes(cols)? / kind.block_bytes();
    if blocks * kind.block_bytes() != data.len() {
        return None;
    }
    let mut offsets = [0usize; 3];
    let mut total = 0;
    for (o, size) in offsets.iter_mut().zip(sizes) {
        *o = total;
        total = (total + size * blocks).next_multiple_of(256);
    }
    let mut out = vec![0_u8; total];
    for (i, block) in data.chunks_exact(kind.block_bytes()).enumerate() {
        let mut put = |region: usize, bytes: &[u8]| {
            let at = offsets[region] + i * sizes[region];
            out[at..at + bytes.len()].copy_from_slice(bytes);
        };
        match kind {
            GgmlType::Q4_0 => {
                put(0, &block[2..18]);
                put(1, &block[0..2]);
            }
            GgmlType::Q8_0 => {
                put(0, &block[2..34]);
                put(1, &block[0..2]);
            }
            GgmlType::Q4K => {
                put(0, &block[16..144]);
                put(1, &block[0..16]); // d, dmin, scales[12]
            }
            GgmlType::Q6K => {
                put(0, &block[0..128]); // ql
                put(1, &block[128..192]); // qh
                let mut meta = [0_u8; 32];
                meta[0..16].copy_from_slice(&block[192..208]); // scales
                meta[16..18].copy_from_slice(&block[208..210]); // d
                put(2, &meta);
            }
            _ => unreachable!(),
        }
    }
    Some((out, offsets))
}

/// Device memory, freed on drop.
pub struct DeviceBuffer {
    gpu: Arc<Gpu>,
    pub ptr: DevicePtr,
    pub bytes: usize,
}

impl DeviceBuffer {
    pub fn write<T: Copy>(&self, data: &[T]) -> Result<(), CudaError> {
        self.write_at(0, data)
    }

    /// Copy `data` to byte offset `offset`.
    pub fn write_at<T: Copy>(&self, offset: usize, data: &[T]) -> Result<(), CudaError> {
        let bytes = std::mem::size_of_val(data);
        assert!(offset + bytes <= self.bytes, "device write out of bounds");
        self.gpu.bind()?;
        // SAFETY: bounds checked; `data` is plain old data.
        check(
            unsafe {
                (self.gpu.api.cuMemcpyHtoD)(
                    self.ptr + offset as u64,
                    data.as_ptr() as *const c_void,
                    bytes,
                )
            },
            "cuMemcpyHtoD",
        )
    }

    pub fn read<T: Copy>(&self, out: &mut [T]) -> Result<(), CudaError> {
        self.read_at(0, out)
    }

    pub fn read_at<T: Copy>(&self, offset: usize, out: &mut [T]) -> Result<(), CudaError> {
        let bytes = std::mem::size_of_val(out);
        assert!(offset + bytes <= self.bytes, "device read out of bounds");
        self.gpu.bind()?;
        // SAFETY: bounds checked; `out` is plain old data.
        check(
            unsafe {
                (self.gpu.api.cuMemcpyDtoH)(
                    out.as_mut_ptr() as *mut c_void,
                    self.ptr + offset as u64,
                    bytes,
                )
            },
            "cuMemcpyDtoH",
        )
    }

    pub fn zero(&self) -> Result<(), CudaError> {
        self.gpu.bind()?;
        // SAFETY: the whole allocation.
        check(
            unsafe { (self.gpu.api.cuMemsetD8)(self.ptr, 0, self.bytes) },
            "cuMemsetD8",
        )
    }

    pub fn copy_from(&self, other: &DeviceBuffer, bytes: usize) -> Result<(), CudaError> {
        assert!(bytes <= self.bytes && bytes <= other.bytes);
        self.gpu.bind()?;
        // SAFETY: bounds checked.
        check(
            unsafe { (self.gpu.api.cuMemcpyDtoD)(self.ptr, other.ptr, bytes) },
            "cuMemcpyDtoD",
        )
    }

    /// Device address `offset` bytes in.
    pub fn at(&self, offset: usize) -> DevicePtr {
        debug_assert!(offset <= self.bytes);
        self.ptr + offset as u64
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        let _ = self.gpu.bind();
        // SAFETY: allocated by `Gpu::alloc`.
        unsafe {
            let _ = (self.gpu.api.cuMemFree)(self.ptr);
        }
    }
}

static DEFAULT: OnceLock<Result<Arc<Gpu>, CudaError>> = OnceLock::new();

/// The first GPU, opened once per process (`KLLM_NO_GPU=1` disables it).
pub fn default_gpu() -> Result<Arc<Gpu>, CudaError> {
    DEFAULT
        .get_or_init(|| {
            if std::env::var_os("KLLM_NO_GPU").is_some() {
                return Err(CudaError {
                    what: "disabled by KLLM_NO_GPU".into(),
                    code: 0,
                });
            }
            Gpu::open(0)
        })
        .clone()
}

/// A [`Backend`] that runs each operation on the GPU, uploading inputs and
/// downloading results every call. It exists to test the kernels against
/// [`crate::cpu::CpuBackend`]; real inference will keep whole layers on the
/// device instead (milestone M6).
pub struct CudaBackend {
    pub gpu: Arc<Gpu>,
}

impl Backend for CudaBackend {
    fn matvec(
        &self,
        matrix: Matrix<'_>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), BackendError> {
        self.matmul(matrix, input, output, 1)
    }

    fn matmul(
        &self,
        matrix: Matrix<'_>,
        inputs: &[f32],
        outputs: &mut [f32],
        batch: usize,
    ) -> Result<(), BackendError> {
        require_len("matmul inputs", matrix.cols * batch, inputs.len())?;
        require_len("matmul outputs", matrix.rows * batch, outputs.len())?;
        if !self.gpu.supports(matrix.kind) {
            return Err(BackendError::UnsupportedType(matrix.kind));
        }
        let w = self
            .gpu
            .upload_matrix(matrix.kind, matrix.data, matrix.rows, matrix.cols)?;
        let x = self.gpu.upload(inputs)?;
        let y = self.gpu.alloc(outputs.len() * 4)?;
        self.gpu.matmul_matrix(&w, x.ptr, y.ptr, batch)?;
        y.read(outputs)?;
        Ok(())
    }

    fn rms_norm(
        &self,
        input: &[f32],
        weights: &[f32],
        epsilon: f32,
        output: &mut [f32],
    ) -> Result<(), BackendError> {
        require_len("RMSNorm weights", input.len(), weights.len())?;
        require_len("RMSNorm output", input.len(), output.len())?;
        let x = self.gpu.upload(input)?;
        let w = self.gpu.upload(weights)?;
        self.gpu
            .rms_norm(x.ptr, w.ptr, x.ptr, 1, input.len(), epsilon)?;
        x.read(output)?;
        Ok(())
    }

    fn swiglu(&self, gate: &[f32], up: &[f32], output: &mut [f32]) -> Result<(), BackendError> {
        require_len("SwiGLU up", gate.len(), up.len())?;
        require_len("SwiGLU output", gate.len(), output.len())?;
        let g = self.gpu.upload(gate)?;
        let u = self.gpu.upload(up)?;
        self.gpu.swiglu(g.ptr, u.ptr, gate.len())?;
        g.read(output)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;
    use crate::quant::{dequantize_row, encode_f16, encode_f32, quantize_q4_0, quantize_q8_0};

    fn gpu() -> Option<Arc<Gpu>> {
        match default_gpu() {
            Ok(g) => Some(g),
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                None
            }
        }
    }

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    fn random_k_blocks(kind: GgmlType, blocks: usize, seed: &mut u64) -> Vec<u8> {
        let mut out = Vec::new();
        for _ in 0..blocks {
            let mut b: Vec<u8> = (0..kind.block_bytes()).map(|_| lcg(seed) as u8).collect();
            let at: &[usize] = match kind {
                GgmlType::Q6K => &[208],
                GgmlType::Q2K => &[80, 82],
                _ => &[0, 2],
            };
            for &a in at {
                let v = (lcg(seed) % 1000) as f32 / 20000.0 + 0.001;
                b[a..a + 2].copy_from_slice(&crate::quant::f32_to_f16(v).to_le_bytes());
            }
            out.extend(b);
        }
        out
    }

    /// GPU matvec against an exact f64 reference on the dequantized weights
    /// (the GPU keeps activations in f32, so this is tight).
    #[test]
    fn gpu_matmul_matches_dequantized_reference() {
        let Some(gpu) = gpu() else { return };
        let backend = CudaBackend { gpu };
        let (rows, cols) = (37, 512);
        let mut seed = 1;
        let values: Vec<f32> = (0..rows * cols)
            .map(|_| (lcg(&mut seed) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let mut cases = vec![
            (GgmlType::F32, encode_f32(&values)),
            (GgmlType::F16, encode_f16(&values)),
            (GgmlType::Q4_0, quantize_q4_0(&values)),
            (GgmlType::Q8_0, quantize_q8_0(&values)),
        ];
        for kind in [
            GgmlType::Q4_1,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
            GgmlType::Q2K,
            GgmlType::Q4K,
            GgmlType::Q5K,
            GgmlType::Q6K,
        ] {
            cases.push((
                kind,
                random_k_blocks(kind, rows * cols / kind.block_elements(), &mut seed),
            ));
        }
        for (kind, data) in cases {
            let m = Matrix::new(kind, rows, cols, &data).unwrap();
            for batch in [1, 3, 4, 9] {
                let x: Vec<f32> = (0..cols * batch).map(|i| (i as f32 * 0.37).sin()).collect();
                let mut y = vec![0.0; rows * batch];
                backend.matmul(m, &x, &mut y, batch).unwrap();
                let mut row = vec![0.0; cols];
                for b in 0..batch {
                    for r in 0..rows {
                        dequantize_row(kind, m.row(r), &mut row);
                        let xs = &x[b * cols..(b + 1) * cols];
                        let exact: f64 =
                            row.iter().zip(xs).map(|(a, b)| *a as f64 * *b as f64).sum();
                        let mag: f64 = row
                            .iter()
                            .zip(xs)
                            .map(|(a, b)| (*a as f64 * *b as f64).abs())
                            .sum();
                        let got = y[b * rows + r] as f64;
                        assert!(
                            (got - exact).abs() <= 1e-5 * mag.max(1.0),
                            "{} row {r}: gpu {got} vs {exact}",
                            kind.name()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn gpu_norm_and_swiglu_match_cpu() {
        let Some(gpu) = gpu() else { return };
        let backend = CudaBackend { gpu };
        let x: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.1).sin() * 3.0).collect();
        let w: Vec<f32> = (0..1000).map(|i| 1.0 + (i % 7) as f32 * 0.1).collect();
        let (mut a, mut b) = (vec![0.0; 1000], vec![0.0; 1000]);
        backend.rms_norm(&x, &w, 1e-6, &mut a).unwrap();
        CpuBackend.rms_norm(&x, &w, 1e-6, &mut b).unwrap();
        for (p, q) in a.iter().zip(&b) {
            assert!((p - q).abs() < 1e-5 * q.abs().max(1.0), "{p} vs {q}");
        }
        backend.swiglu(&x, &w, &mut a).unwrap();
        CpuBackend.swiglu(&x, &w, &mut b).unwrap();
        for (p, q) in a.iter().zip(&b) {
            assert!((p - q).abs() < 1e-5 * q.abs().max(1.0), "{p} vs {q}");
        }
    }
}
