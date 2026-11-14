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
    let gpu_layers = weight_budget
        .checked_div(layer_bytes)
        .map_or(input.layer_count, |n| {
            n.min(u64::from(input.layer_count)) as u32
        });
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

/// Inputs for planning a specific model from its real tensor sizes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPlanInput {
    /// Weight bytes of each transformer block, in order.
    pub layer_bytes: Vec<u64>,
    /// Output projection plus final norm. Placed on the GPU only once every
    /// layer fits (llama.cpp's `-ngl n_layer + 1` convention).
    pub head_bytes: u64,
    /// Token embedding table; a row lookup, so it always stays in RAM.
    pub embedding_bytes: u64,
    pub vram_bytes: u64,
    pub context_tokens: u64,
    /// KV bytes per token for one layer.
    pub kv_bytes_per_token_per_layer: u64,
    pub workspace_bytes: u64,
    pub safety_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelPlan {
    pub gpu_layers: u32,
    pub cpu_layers: u32,
    /// GPU layers are the last `gpu_layers` blocks, adjacent to the head.
    pub first_gpu_layer: u32,
    pub head_on_gpu: bool,
    pub gpu_weight_bytes: u64,
    pub cpu_weight_bytes: u64,
    pub gpu_kv_bytes: u64,
    pub cpu_kv_bytes: u64,
    pub workspace_bytes: u64,
    pub safety_bytes: u64,
    pub unused_vram_bytes: u64,
}

pub fn plan_model(input: &ModelPlanInput) -> Result<ModelPlan, PlanError> {
    let layers =
        u32::try_from(input.layer_bytes.len()).map_err(|_| PlanError::ArithmeticOverflow)?;
    if layers == 0 {
        return Err(PlanError::NoLayers);
    }
    let kv_per_layer = input
        .context_tokens
        .checked_mul(input.kv_bytes_per_token_per_layer)
        .ok_or(PlanError::ArithmeticOverflow)?;
    let fixed = input
        .workspace_bytes
        .checked_add(input.safety_bytes)
        .ok_or(PlanError::ArithmeticOverflow)?;
    if fixed > input.vram_bytes {
        return Err(PlanError::FixedAllocationsExceedVram {
            required: fixed,
            available: input.vram_bytes,
        });
    }
    let mut budget = input.vram_bytes - fixed;
    let mut gpu_layers = 0_u32;
    let mut gpu_weights = 0_u64;
    for &bytes in input.layer_bytes.iter().rev() {
        let cost = bytes
            .checked_add(kv_per_layer)
            .ok_or(PlanError::ArithmeticOverflow)?;
        if cost > budget {
            break;
        }
        budget -= cost;
        gpu_layers += 1;
        gpu_weights += bytes;
    }
    let head_on_gpu = gpu_layers == layers && input.head_bytes <= budget;
    if head_on_gpu {
        budget -= input.head_bytes;
        gpu_weights += input.head_bytes;
    }
    let total_weights = input
        .layer_bytes
        .iter()
        .try_fold(
            input.head_bytes.checked_add(input.embedding_bytes),
            |acc, b| acc.and_then(|a| a.checked_add(*b)).map(Some),
        )
        .flatten()
        .ok_or(PlanError::ArithmeticOverflow)?;
    let gpu_kv = kv_per_layer * u64::from(gpu_layers);
    Ok(ModelPlan {
        gpu_layers,
        cpu_layers: layers - gpu_layers,
        first_gpu_layer: layers - gpu_layers,
        head_on_gpu,
        gpu_weight_bytes: gpu_weights,
        cpu_weight_bytes: total_weights - gpu_weights,
        gpu_kv_bytes: gpu_kv,
        cpu_kv_bytes: kv_per_layer * u64::from(layers - gpu_layers),
        workspace_bytes: input.workspace_bytes,
        safety_bytes: input.safety_bytes,
        unused_vram_bytes: budget,
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

    fn model_input(context_tokens: u64, vram_gib: f64) -> ModelPlanInput {
        ModelPlanInput {
            layer_bytes: (0..24).map(|i| 100_000_000 + i * 1_000_000).collect(),
            head_bytes: 300_000_000,
            embedding_bytes: 300_000_000,
            vram_bytes: bytes_from_gib(vram_gib).unwrap(),
            context_tokens,
            kv_bytes_per_token_per_layer: 4096,
            workspace_bytes: bytes_from_gib(0.25).unwrap(),
            safety_bytes: bytes_from_gib(0.25).unwrap(),
        }
    }

    #[test]
    fn model_plan_accounts_for_every_byte() {
        for vram in [0.5, 1.0, 2.0, 4.0, 8.0] {
            let input = model_input(4096, vram);
            let p = plan_model(&input).unwrap();
            let used = p.gpu_weight_bytes
                + p.gpu_kv_bytes
                + p.workspace_bytes
                + p.safety_bytes
                + p.unused_vram_bytes;
            assert_eq!(used, input.vram_bytes);
            let weights: u64 =
                input.layer_bytes.iter().sum::<u64>() + input.head_bytes + input.embedding_bytes;
            assert_eq!(p.gpu_weight_bytes + p.cpu_weight_bytes, weights);
            assert_eq!(p.gpu_layers + p.cpu_layers, 24);
            assert_eq!(p.first_gpu_layer, p.cpu_layers);
            assert!(!p.head_on_gpu || p.gpu_layers == 24);
        }
        assert!(plan_model(&model_input(4096, 8.0)).unwrap().head_on_gpu);
    }

    #[test]
    fn model_plan_is_monotone_in_context_and_vram() {
        let mut previous = u32::MAX;
        for context in [512, 2048, 8192, 32768, 131072] {
            let p = plan_model(&model_input(context, 3.0)).unwrap();
            assert!(p.gpu_layers <= previous);
            previous = p.gpu_layers;
        }
        let mut previous = 0;
        for vram in [0.6, 1.0, 1.5, 2.0, 3.0] {
            let p = plan_model(&model_input(4096, vram)).unwrap();
            assert!(p.gpu_layers >= previous);
            previous = p.gpu_layers;
        }
    }

    #[test]
    fn model_plan_uses_real_sizes() {
        // Layers grow toward the end, so a suffix fits fewer of them than a
        // uniform estimate from the mean would claim.
        let mut input = model_input(0, 1.0);
        input.workspace_bytes = 0;
        input.safety_bytes = 0;
        input.vram_bytes = 123_000_000 + 122_000_000 + 121_000_000;
        let p = plan_model(&input).unwrap();
        assert_eq!(p.gpu_layers, 3);
        assert_eq!(p.unused_vram_bytes, 0);
    }
}
