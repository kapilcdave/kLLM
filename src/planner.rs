use std::fmt;

const GIB: u64 = 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanInput {
    pub model_bytes: u64,
    pub layer_count: u32,
    pub vram_bytes: u64,
    pub context_tokens: u64,
    pub kv_bytes_per_token: u64,
    pub workspace_bytes: u64,
    pub safety_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryPlan {
    pub gpu_layers: u32,
    pub cpu_layers: u32,
    pub gpu_weight_bytes: u64,
    pub cpu_weight_bytes: u64,
    pub kv_cache_bytes: u64,
    pub workspace_bytes: u64,
    pub safety_bytes: u64,
    pub unused_vram_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    NoLayers,
    ArithmeticOverflow,
    FixedAllocationsExceedVram { required: u64, available: u64 },
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoLayers => write!(f, "layer count must be greater than zero"),
            Self::ArithmeticOverflow => write!(f, "memory estimate overflowed u64"),
            Self::FixedAllocationsExceedVram {
                required,
                available,
            } => write!(
                f,
                "KV cache, workspace, and safety reserve require {:.2} GiB but only {:.2} GiB is available",
                gib(*required),
                gib(*available)
            ),
        }
    }
}

impl std::error::Error for PlanError {}

pub fn plan(input: PlanInput) -> Result<MemoryPlan, PlanError> {
    if input.layer_count == 0 {
        return Err(PlanError::NoLayers);
    }

    let kv_cache_bytes = input
        .context_tokens
        .checked_mul(input.kv_bytes_per_token)
        .ok_or(PlanError::ArithmeticOverflow)?;
    let fixed = kv_cache_bytes
        .checked_add(input.workspace_bytes)
        .and_then(|n| n.checked_add(input.safety_bytes))
        .ok_or(PlanError::ArithmeticOverflow)?;

    if fixed > input.vram_bytes {
        return Err(PlanError::FixedAllocationsExceedVram {
            required: fixed,
            available: input.vram_bytes,
        });
    }

    let layer_bytes = input.model_bytes.div_ceil(u64::from(input.layer_count));
    let weight_budget = input.vram_bytes - fixed;
    let gpu_layers = if layer_bytes == 0 {
        input.layer_count
    } else {
        (weight_budget / layer_bytes).min(u64::from(input.layer_count)) as u32
    };
    let gpu_weight_bytes = layer_bytes
        .checked_mul(u64::from(gpu_layers))
        .ok_or(PlanError::ArithmeticOverflow)?
        .min(input.model_bytes);
    let used = fixed
        .checked_add(gpu_weight_bytes)
        .ok_or(PlanError::ArithmeticOverflow)?;

    Ok(MemoryPlan {
        gpu_layers,
        cpu_layers: input.layer_count - gpu_layers,
        gpu_weight_bytes,
        cpu_weight_bytes: input.model_bytes - gpu_weight_bytes,
        kv_cache_bytes,
        workspace_bytes: input.workspace_bytes,
        safety_bytes: input.safety_bytes,
        unused_vram_bytes: input.vram_bytes - used,
    })
}

pub fn bytes_from_gib(value: f64) -> Result<u64, &'static str> {
    if !value.is_finite() || value < 0.0 || value > (u64::MAX as f64 / GIB as f64) {
        return Err("GiB value must be finite, non-negative, and representable");
    }
    Ok((value * GIB as f64).round() as u64)
}

pub fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(context_tokens: u64) -> PlanInput {
        PlanInput {
            model_bytes: bytes_from_gib(4.0).unwrap(),
            layer_count: 32,
            vram_bytes: bytes_from_gib(4.0).unwrap(),
            context_tokens,
            kv_bytes_per_token: 128 * 1024,
            workspace_bytes: bytes_from_gib(0.25).unwrap(),
            safety_bytes: bytes_from_gib(0.25).unwrap(),
        }
    }

    #[test]
    fn plan_stays_within_budget() {
        let source = input(4096);
        let result = plan(source).unwrap();
        let total = result.gpu_weight_bytes
            + result.kv_cache_bytes
            + result.workspace_bytes
            + result.safety_bytes
            + result.unused_vram_bytes;
        assert_eq!(total, source.vram_bytes);
        assert_eq!(result.gpu_layers + result.cpu_layers, source.layer_count);
    }

    #[test]
    fn more_context_never_adds_gpu_layers() {
        let short = plan(input(1024)).unwrap();
        let long = plan(input(8192)).unwrap();
        assert!(long.gpu_layers <= short.gpu_layers);
    }

    #[test]
    fn rejects_fixed_allocations_over_budget() {
        let mut source = input(32768);
        source.vram_bytes = bytes_from_gib(1.0).unwrap();
        assert!(matches!(
            plan(source),
            Err(PlanError::FixedAllocationsExceedVram { .. })
        ));
    }

    #[test]
    fn rejects_zero_layers() {
        let mut source = input(1024);
        source.layer_count = 0;
        assert_eq!(plan(source), Err(PlanError::NoLayers));
    }
}
