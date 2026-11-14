use std::{env, process::ExitCode};

use kllm::gguf::inspect_file;
use kllm::planner::{PlanInput, bytes_from_gib, gib, plan};

const DEFAULT_WORKSPACE_GIB: f64 = 0.25;
const DEFAULT_SAFETY_GIB: f64 = 0.25;

fn main() -> ExitCode {
    match run(env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            eprintln!("\n{}", usage());
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("plan") => run_plan(&args),
        Some("plan-model") => run_plan_model(&args),
        Some("inspect") => run_inspect(&args),
        _ => Err("expected plan, plan-model, or inspect".into()),
    }
}

fn run_plan(args: &[String]) -> Result<(), String> {
    let model_gib = flag_f64(&args, "--model-gib")?;
    let layers = flag_u64(&args, "--layers")?;
    let vram_gib = flag_f64(&args, "--vram-gib")?;
    let context = flag_u64(&args, "--context")?;
    let kv_bytes_per_token = flag_u64(&args, "--kv-bytes-per-token")?;
    let workspace_gib =
        optional_flag_f64(&args, "--workspace-gib")?.unwrap_or(DEFAULT_WORKSPACE_GIB);
    let safety_gib = optional_flag_f64(&args, "--safety-gib")?.unwrap_or(DEFAULT_SAFETY_GIB);
    let layer_count = u32::try_from(layers).map_err(|_| "--layers exceeds u32".to_string())?;

    let input = PlanInput {
        model_bytes: bytes_from_gib(model_gib).map_err(str::to_string)?,
        layer_count,
        vram_bytes: bytes_from_gib(vram_gib).map_err(str::to_string)?,
        context_tokens: context,
        kv_bytes_per_token,
        workspace_bytes: bytes_from_gib(workspace_gib).map_err(str::to_string)?,
        safety_bytes: bytes_from_gib(safety_gib).map_err(str::to_string)?,
    };
    let result = plan(input).map_err(|error| error.to_string())?;

    println!("kLLM memory plan");
    println!("  model weights:    {:>6.2} GiB", gib(input.model_bytes));
    println!("  VRAM limit:       {:>6.2} GiB", gib(input.vram_bytes));
    println!(
        "  GPU layers:       {:>6} / {}",
        result.gpu_layers, input.layer_count
    );
    println!(
        "  CPU layers:       {:>6} / {}",
        result.cpu_layers, input.layer_count
    );
    println!(
        "  GPU weights:      {:>6.2} GiB",
        gib(result.gpu_weight_bytes)
    );
    println!(
        "  CPU weights:      {:>6.2} GiB",
        gib(result.cpu_weight_bytes)
    );
    println!(
        "  KV cache:         {:>6.2} GiB",
        gib(result.kv_cache_bytes)
    );
    println!(
        "  workspace:        {:>6.2} GiB",
        gib(result.workspace_bytes)
    );
    println!("  safety reserve:   {:>6.2} GiB", gib(result.safety_bytes));
    println!(
        "  unused VRAM:      {:>6.2} GiB",
        gib(result.unused_vram_bytes)
    );
    Ok(())
}

fn run_inspect(args: &[String]) -> Result<(), String> {
    let path = args.get(1).ok_or_else(|| "missing GGUF path".to_string())?;
    let info = inspect_file(path).map_err(|error| error.to_string())?;
    println!("kLLM GGUF inspection");
    println!("  version:          {}", info.version);
    println!(
        "  architecture:     {}",
        info.architecture.as_deref().unwrap_or("unknown")
    );
    println!("  metadata entries: {}", info.metadata_count);
    println!("  tensors:          {}", info.tensor_count);
    if let Some(block_count) = info.block_count {
        println!("  transformer blocks: {block_count}");
    }
    if let Some(bytes) = info.known_weight_bytes() {
        println!("  known weights:    {:.2} GiB", gib(bytes));
    }
    println!("  alignment:        {} bytes", info.alignment);
    println!("  data offset:      {} bytes", info.data_offset);
    for tensor in &info.tensors {
        println!(
            "  - {} {:?} {} bytes={} offset={}",
            tensor.name,
            tensor.dimensions,
            tensor.type_name(),
            tensor
                .byte_size()
                .map(|size| size.to_string())
                .unwrap_or_else(|| "unknown".into()),
            tensor.relative_offset
        );
    }
    Ok(())
}

fn run_plan_model(args: &[String]) -> Result<(), String> {
    let path = args.get(1).ok_or_else(|| "missing GGUF path".to_string())?;
    let info = inspect_file(path).map_err(|error| error.to_string())?;
    let model_bytes = info.known_weight_bytes().ok_or_else(|| {
        "model contains a GGML tensor type whose size kLLM does not know".to_string()
    })?;
    let layer_count = info
        .block_count
        .ok_or_else(|| "GGUF does not contain an architecture block count".to_string())?;
    let input = PlanInput {
        model_bytes,
        layer_count,
        vram_bytes: bytes_from_gib(flag_f64(args, "--vram-gib")?).map_err(str::to_string)?,
        context_tokens: flag_u64(args, "--context")?,
        kv_bytes_per_token: flag_u64(args, "--kv-bytes-per-token")?,
        workspace_bytes: bytes_from_gib(
            optional_flag_f64(args, "--workspace-gib")?.unwrap_or(DEFAULT_WORKSPACE_GIB),
        )
        .map_err(str::to_string)?,
        safety_bytes: bytes_from_gib(
            optional_flag_f64(args, "--safety-gib")?.unwrap_or(DEFAULT_SAFETY_GIB),
        )
        .map_err(str::to_string)?,
    };
    print_plan(input)
}

fn print_plan(input: PlanInput) -> Result<(), String> {
    let result = plan(input).map_err(|error| error.to_string())?;
    println!("kLLM memory plan");
    println!("  model weights:    {:>6.2} GiB", gib(input.model_bytes));
    println!("  VRAM limit:       {:>6.2} GiB", gib(input.vram_bytes));
    println!(
        "  GPU layers:       {:>6} / {}",
        result.gpu_layers, input.layer_count
    );
    println!(
        "  CPU layers:       {:>6} / {}",
        result.cpu_layers, input.layer_count
    );
    println!(
        "  GPU weights:      {:>6.2} GiB",
        gib(result.gpu_weight_bytes)
    );
    println!(
        "  CPU weights:      {:>6.2} GiB",
        gib(result.cpu_weight_bytes)
    );
    println!(
        "  KV cache:         {:>6.2} GiB",
        gib(result.kv_cache_bytes)
    );
    println!(
        "  workspace:        {:>6.2} GiB",
        gib(result.workspace_bytes)
    );
    println!("  safety reserve:   {:>6.2} GiB", gib(result.safety_bytes));
    println!(
        "  unused VRAM:      {:>6.2} GiB",
        gib(result.unused_vram_bytes)
    );
    Ok(())
}

fn value_after<'a>(args: &'a [String], flag: &str) -> Result<&'a str, String> {
    let index = args
        .iter()
        .position(|arg| arg == flag)
        .ok_or_else(|| format!("missing {flag}"))?;
    args.get(index + 1)
        .map(String::as_str)
        .ok_or_else(|| format!("missing value after {flag}"))
}

fn flag_f64(args: &[String], flag: &str) -> Result<f64, String> {
    value_after(args, flag)?
        .parse()
        .map_err(|_| format!("invalid value for {flag}"))
}

fn optional_flag_f64(args: &[String], flag: &str) -> Result<Option<f64>, String> {
    if args.iter().any(|arg| arg == flag) {
        flag_f64(args, flag).map(Some)
    } else {
        Ok(None)
    }
}

fn flag_u64(args: &[String], flag: &str) -> Result<u64, String> {
    value_after(args, flag)?
        .parse()
        .map_err(|_| format!("invalid value for {flag}"))
}

fn usage() -> &'static str {
    "usage:\n  kllm inspect MODEL.gguf\n  kllm plan-model MODEL.gguf --vram-gib N --context N --kv-bytes-per-token N\n  kllm plan --model-gib N --layers N --vram-gib N --context N \\\n       --kv-bytes-per-token N [--workspace-gib N] [--safety-gib N]"
}
