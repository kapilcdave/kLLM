use std::{
    io::{self, Write},
    process::ExitCode,
    time::Instant,
};

use kllm::eval::perplexity;
use kllm::gguf::{GgufInfo, inspect_file};
use kllm::kernels;
use kllm::model::{Config, Logits, Model, Session};
use kllm::planner::{ModelPlanInput, PlanInput, bytes_from_gib, gib, plan, plan_model};
use kllm::sampler::{Sampler, SamplerConfig};
use kllm::tokenizer::{Tokenizer, format_chat};

const DEFAULT_WORKSPACE_GIB: f64 = 0.25;
const DEFAULT_SAFETY_GIB: f64 = 0.25;
const MIB: f64 = 1024.0 * 1024.0;

fn main() -> ExitCode {
    let args = Args(std::env::args().skip(1).collect());
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            if message.starts_with("usage") || message.starts_with("expected") {
                eprintln!("\n{USAGE}");
            }
            ExitCode::FAILURE
        }
    }
}

const USAGE: &str = "\
usage:
  kllm inspect MODEL.gguf [--tensors]
  kllm plan-model MODEL.gguf --vram-gib N [--context N] [--workspace-gib N] [--safety-gib N]
  kllm plan --model-gib N --layers N --vram-gib N --context N --kv-bytes-per-token N
  kllm run MODEL.gguf --prompt TEXT [-n N] [--chat] [--system TEXT] [--context N]
           [--temp T] [--top-k K] [--top-p P] [--min-p P] [--repeat-penalty R]
           [--seed S] [--threads N] [--batch N] [--ignore-eos] [--gpu-layers N|auto]
           [--ram-gib N|auto|all]
  kllm perplexity MODEL.gguf --file TEXT [--context 512] [--chunks N] [--threads N] [--gpu-layers N|auto]
  kllm compare REFERENCE.gguf TEST.gguf --file TEXT [--context 512] [--chunks N] [--threads N]
  kllm devices
  kllm bench MODEL.gguf [-p 128] [-n 64] [--threads N] [--repeat 3] [--profile] [--gpu-layers N|auto]";

struct Args(Vec<String>);

impl Args {
    fn command(&self) -> Option<&str> {
        self.0.first().map(String::as_str)
    }

    fn positional(&self, index: usize, what: &str) -> Result<&str, String> {
        self.0
            .get(index)
            .filter(|a| !a.starts_with('-'))
            .map(String::as_str)
            .ok_or_else(|| format!("usage: missing {what}"))
    }

    fn has(&self, flag: &str) -> bool {
        self.0.iter().any(|a| a == flag)
    }

    fn value(&self, flags: &[&str]) -> Result<Option<&str>, String> {
        for (i, arg) in self.0.iter().enumerate() {
            if flags.contains(&arg.as_str()) {
                return self
                    .0
                    .get(i + 1)
                    .map(|v| Some(v.as_str()))
                    .ok_or_else(|| format!("missing value after {arg}"));
            }
        }
        Ok(None)
    }

    fn parse<T: std::str::FromStr>(&self, flags: &[&str]) -> Result<Option<T>, String> {
        match self.value(flags)? {
            Some(v) => v
                .parse()
                .map(Some)
                .map_err(|_| format!("invalid value {v:?} for {}", flags[0])),
            None => Ok(None),
        }
    }

    fn or<T: std::str::FromStr>(&self, flags: &[&str], default: T) -> Result<T, String> {
        Ok(self.parse(flags)?.unwrap_or(default))
    }

    fn required<T: std::str::FromStr>(&self, flag: &str) -> Result<T, String> {
        self.parse(&[flag])?
            .ok_or_else(|| format!("usage: missing {flag}"))
    }
}

fn run(args: &Args) -> Result<(), String> {
    match args.command() {
        Some("inspect") => cmd_inspect(args),
        Some("plan") => cmd_plan(args),
        Some("plan-model") => cmd_plan_model(args),
        Some("run") => cmd_run(args),
        Some("perplexity") => cmd_perplexity(args),
        Some("compare") => cmd_compare(args),
        Some("bench") => cmd_bench(args),
        Some("devices") => cmd_devices(),
        Some("help" | "--help" | "-h") => {
            println!("{USAGE}");
            Ok(())
        }
        _ => Err("expected a command".into()),
    }
}

fn init_threads(args: &Args) -> Result<usize, String> {
    // Hyper-threads share the memory bus; physical cores are the useful count.
    let threads = args.or(&["--threads", "-t"], kllm::pool::default_threads())?;
    if threads == 0 {
        return Err("--threads must be at least 1".into());
    }
    Ok(kllm::pool::init(threads))
}

/// Write to stdout, treating a closed pipe (`kllm ... | head`) as success.
macro_rules! out {
    ($($arg:tt)*) => {
        if writeln!(io::stdout(), $($arg)*).is_err() {
            return Ok(());
        }
    };
}

fn cmd_inspect(args: &Args) -> Result<(), String> {
    let path = args.positional(1, "GGUF path")?;
    let info = inspect_file(path).map_err(|e| e.to_string())?;
    out!("kLLM GGUF inspection");
    out!("  version:            {}", info.version);
    out!(
        "  architecture:       {}",
        info.architecture().unwrap_or("unknown")
    );
    if let Some(blocks) = info.block_count() {
        out!("  transformer blocks: {blocks}");
    }
    out!("  tensors:            {}", info.tensors.len());
    if let Some(bytes) = info.known_weight_bytes() {
        out!("  weights:            {:.2} GiB", gib(bytes));
    }
    out!("  alignment:          {} bytes", info.alignment);
    out!("  data offset:        {} bytes", info.data_offset);
    let mut types: Vec<(&str, usize, u64)> = Vec::new();
    for t in &info.tensors {
        let name = t.type_name();
        let size = t.byte_size().unwrap_or(0);
        match types.iter_mut().find(|(n, _, _)| *n == name) {
            Some(entry) => {
                entry.1 += 1;
                entry.2 += size;
            }
            None => types.push((name, 1, size)),
        }
    }
    out!("  tensor types:");
    for (name, count, bytes) in types {
        out!(
            "    {name:<6} {count:>5} tensors {:>9.1} MiB",
            bytes as f64 / MIB
        );
    }
    match Config::from_gguf(&info) {
        Ok(c) => {
            out!("  runnable:           yes");
            out!(
                "  shape:              d={} ff={} heads={}/{} head_dim={} vocab={} rope={:?} base={}",
                c.n_embd,
                c.n_ff,
                c.n_head,
                c.n_head_kv,
                c.head_dim,
                c.n_vocab,
                c.rope_style,
                c.rope_base
            );
            if let Some(moe) = c.moe {
                out!(
                    "  experts:            {} of {} per token, ff={}{}, {:?} gating{}{}",
                    moe.n_expert_used,
                    moe.n_expert,
                    moe.n_ff_exp,
                    if moe.n_ff_shared > 0 {
                        format!(" (+shared ff={})", moe.n_ff_shared)
                    } else {
                        String::new()
                    },
                    moe.gating,
                    if moe.weights_norm { ", normalized" } else { "" },
                    if moe.weights_scale != 1.0 {
                        format!(", scale={}", moe.weights_scale)
                    } else {
                        String::new()
                    },
                );
                out!(
                    "  active fraction:    {:.1}% of expert weights per token",
                    moe.active_fraction() * 100.0
                );
            }
            out!(
                "  KV cache:           {} bytes/token (f16)",
                c.kv_bytes_per_token()
            );
        }
        Err(e) => out!("  runnable:           no ({e})"),
    }
    out!("  metadata ({}):", info.metadata.len());
    for (key, value) in &info.metadata {
        out!("    {key} = {}", value.summary());
    }
    if args.has("--tensors") {
        out!("  tensors:");
        for t in &info.tensors {
            out!(
                "    {} {:?} {} bytes={} offset={}",
                t.name,
                t.dimensions,
                t.type_name(),
                t.byte_size()
                    .map_or_else(|| "unknown".into(), |s| s.to_string()),
                t.relative_offset
            );
        }
    }
    Ok(())
}

fn cmd_plan(args: &Args) -> Result<(), String> {
    let layers: u64 = args.required("--layers")?;
    let input = PlanInput {
        model_bytes: bytes_from_gib(args.required("--model-gib")?)?,
        layer_count: u32::try_from(layers).map_err(|_| "--layers exceeds u32")?,
        vram_bytes: bytes_from_gib(args.required("--vram-gib")?)?,
        context_tokens: args.required("--context")?,
        kv_bytes_per_token: args.required("--kv-bytes-per-token")?,
        workspace_bytes: bytes_from_gib(args.or(&["--workspace-gib"], DEFAULT_WORKSPACE_GIB)?)?,
        safety_bytes: bytes_from_gib(args.or(&["--safety-gib"], DEFAULT_SAFETY_GIB)?)?,
    };
    let result = plan(input).map_err(|e| e.to_string())?;
    out!("kLLM memory plan");
    out!("  model weights:    {:>6.2} GiB", gib(input.model_bytes));
    out!("  VRAM limit:       {:>6.2} GiB", gib(input.vram_bytes));
    out!(
        "  GPU layers:       {:>6} / {}",
        result.gpu_layers,
        input.layer_count
    );
    out!(
        "  CPU layers:       {:>6} / {}",
        result.cpu_layers,
        input.layer_count
    );
    out!(
        "  GPU weights:      {:>6.2} GiB",
        gib(result.gpu_weight_bytes)
    );
    out!(
        "  CPU weights:      {:>6.2} GiB",
        gib(result.cpu_weight_bytes)
    );
    out!(
        "  KV cache:         {:>6.2} GiB",
        gib(result.kv_cache_bytes)
    );
    out!(
        "  workspace:        {:>6.2} GiB",
        gib(result.workspace_bytes)
    );
    out!("  safety reserve:   {:>6.2} GiB", gib(result.safety_bytes));
    out!(
        "  unused VRAM:      {:>6.2} GiB",
        gib(result.unused_vram_bytes)
    );
    Ok(())
}

/// Planning shape for any GGUF with standard attention metadata, runnable
/// or not: `(layers, kv_bytes_per_token_per_layer, default_context)`.
fn plan_shape(info: &GgufInfo) -> Result<(usize, u64, u64), String> {
    if let Ok(c) = Config::from_gguf(info) {
        return Ok((
            c.n_layer,
            c.kv_bytes_per_token() / c.n_layer as u64,
            c.train_context as u64,
        ));
    }
    let int = |key: &str| info.arch_get(key).and_then(|v| v.as_u64());
    let layers = info.block_count().ok_or("GGUF has no block_count")? as usize;
    let heads = int("attention.head_count").ok_or("GGUF has no attention.head_count")?;
    let kv_heads = int("attention.head_count_kv").unwrap_or(heads);
    let embd = int("embedding_length").ok_or("GGUF has no embedding_length")?;
    let key = int("attention.key_length").unwrap_or(embd / heads.max(1));
    let value = int("attention.value_length").unwrap_or(key);
    // Conservative for hybrid models, where only some layers keep a KV cache.
    Ok((
        layers,
        kv_heads * (key + value) * 2,
        int("context_length").unwrap_or(4096),
    ))
}

fn model_plan_input(
    info: &GgufInfo,
    args: &Args,
    context: u64,
    vram_bytes: u64,
) -> Result<ModelPlanInput, String> {
    let (n_layer, kv_per_layer, _) = plan_shape(info)?;
    let mut layer_bytes = vec![0_u64; n_layer];
    let (mut head, mut embedding) = (0_u64, 0_u64);
    for t in &info.tensors {
        let size = t
            .byte_size()
            .ok_or_else(|| format!("tensor {} has a type kLLM cannot size", t.name))?;
        match t.block_index() {
            Some(i) if (i as usize) < layer_bytes.len() => layer_bytes[i as usize] += size,
            _ if t.name == "token_embd.weight" => embedding += size,
            _ => head += size, // output.weight, output_norm.weight, rope_freqs
        }
    }
    if info.tensor("output.weight").is_none() {
        // Tied embeddings: the head *is* the embedding table. Charge it to
        // the head; a GPU head is a device copy, RAM keeps the mapping.
        head += embedding;
        embedding = 0;
    }
    let kv_override: Option<u64> = args.parse(&["--kv-bytes-per-token"])?;
    Ok(ModelPlanInput {
        layer_bytes,
        head_bytes: head,
        embedding_bytes: embedding,
        vram_bytes,
        context_tokens: context,
        kv_bytes_per_token_per_layer: kv_override.map_or(kv_per_layer, |k| k / n_layer as u64),
        workspace_bytes: bytes_from_gib(args.or(&["--workspace-gib"], DEFAULT_WORKSPACE_GIB)?)?,
        safety_bytes: bytes_from_gib(args.or(&["--safety-gib"], DEFAULT_SAFETY_GIB)?)?,
    })
}

fn cmd_plan_model(args: &Args) -> Result<(), String> {
    let path = args.positional(1, "GGUF path")?;
    let info = inspect_file(path).map_err(|e| e.to_string())?;
    let (_, _, train_context) = plan_shape(&info)?;
    let context: u64 = args.or(&["--context", "-c"], train_context.min(4096))?;
    let input = model_plan_input(
        &info,
        args,
        context,
        bytes_from_gib(args.required("--vram-gib")?)?,
    )?;
    let p = plan_model(&input).map_err(|e| e.to_string())?;
    let layers = input.layer_bytes.len();
    out!(
        "kLLM memory plan for {}",
        info.get("general.name")
            .and_then(|v| v.as_str())
            .unwrap_or(path)
    );
    out!(
        "  architecture:     {} ({layers} layers)",
        info.architecture().unwrap_or("?")
    );
    out!(
        "  context:          {context} tokens, KV {} bytes/token",
        input.kv_bytes_per_token_per_layer * layers as u64
    );
    out!("  VRAM limit:       {:>8.2} GiB", gib(input.vram_bytes));
    if p.gpu_layers == 0 {
        out!("  GPU layers:       {:>8}", "none");
    } else {
        out!(
            "  GPU layers:       {:>8} (blocks {}..{}{})",
            p.gpu_layers,
            p.first_gpu_layer,
            layers - 1,
            if p.head_on_gpu { " + output head" } else { "" }
        );
    }
    out!("  CPU layers:       {:>8}", p.cpu_layers);
    out!("  GPU weights:      {:>8.2} GiB", gib(p.gpu_weight_bytes));
    out!("  GPU KV cache:     {:>8.2} GiB", gib(p.gpu_kv_bytes));
    out!("  workspace:        {:>8.2} GiB", gib(p.workspace_bytes));
    out!("  safety reserve:   {:>8.2} GiB", gib(p.safety_bytes));
    out!("  unused VRAM:      {:>8.2} GiB", gib(p.unused_vram_bytes));
    out!(
        "  CPU weights:      {:>8.2} GiB (memory-mapped)",
        gib(p.cpu_weight_bytes)
    );
    out!("  CPU KV cache:     {:>8.2} GiB", gib(p.cpu_kv_bytes));
    if Config::from_gguf(&info).is_err() {
        out!(
            "  note: kLLM cannot run this architecture yet; KV estimate assumes every layer has attention"
        );
    }
    if !p.head_on_gpu && p.gpu_layers as usize == layers {
        out!(
            "  note: all layers fit; the {:.0} MiB output head does not",
            input.head_bytes as f64 / MIB
        );
    }
    Ok(())
}

/// Memory this process can still use: the system's MemAvailable, capped by
/// the headroom of every enclosing cgroup-v2 limit (`memory.max` minus
/// `memory.current`). MemAvailable alone ignores a container's or systemd
/// scope's limit, and a model sized to it gets OOM-killed there.
fn mem_available() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    let mut available = kib * 1024;
    if let Some(headroom) = cgroup_headroom() {
        available = available.min(headroom);
    }
    Some(available)
}

fn cgroup_headroom() -> Option<u64> {
    let own = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = own.lines().find_map(|l| l.strip_prefix("0::"))?.trim();
    let mut dir = std::path::PathBuf::from("/sys/fs/cgroup").join(path.trim_start_matches('/'));
    let mut best: Option<u64> = None;
    loop {
        let read = |f: &str| std::fs::read_to_string(dir.join(f)).ok();
        if let (Some(max), Some(current)) = (read("memory.max"), read("memory.current"))
            && let (Ok(max), Ok(current)) =
                (max.trim().parse::<u64>(), current.trim().parse::<u64>())
        {
            let headroom = max.saturating_sub(current);
            best = Some(best.map_or(headroom, |b| b.min(headroom)));
        }
        if dir == std::path::Path::new("/sys/fs/cgroup") || !dir.pop() {
            break;
        }
    }
    best
}

fn cpu_name() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|t| {
            t.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "unknown CPU".into())
}

/// Load a model after printing (and checking) its memory plan.
/// `--gpu-layers N|auto` (alias `-ngl`): how many transformer blocks to run
/// on the GPU. N >= the block count also offloads the output head, as with
/// llama.cpp's `-ngl`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuRequest {
    Off,
    Layers(usize),
    Auto,
}

fn gpu_request(args: &Args) -> Result<GpuRequest, String> {
    Ok(match args.value(&["--gpu-layers", "-ngl"])? {
        None => GpuRequest::Off,
        Some("auto") => GpuRequest::Auto,
        Some(v) => match v.parse::<usize>() {
            Ok(0) => GpuRequest::Off,
            Ok(n) => GpuRequest::Layers(n),
            Err(_) => {
                return Err(format!(
                    "invalid value {v:?} for --gpu-layers (a count or \"auto\")"
                ));
            }
        },
    })
}

fn load_with_plan(
    args: &Args,
    gpu: GpuRequest,
    path: &str,
    context: usize,
    batch: usize,
    threads: usize,
) -> Result<Model, String> {
    let info = inspect_file(path).map_err(|e| e.to_string())?;
    let config = Config::from_gguf(&info).map_err(|e| e.to_string())?;
    let (kv, scratch) = Session::estimate(&config, context, batch);
    let weights = info.known_weight_bytes().unwrap_or(0);
    eprintln!(
        "kllm: {} | {} layers | ctx {context} | {threads} threads | kernels {}",
        info.get("general.name")
            .and_then(|v| v.as_str())
            .unwrap_or(path),
        config.n_layer,
        kernels::kernel_name()
    );
    eprintln!(
        "kllm: plan: weights {:.0} MiB mapped, KV cache {:.0} MiB, scratch {:.0} MiB, execution CPU",
        weights as f64 / MIB,
        kv as f64 / MIB,
        scratch as f64 / MIB
    );
    if let Some(available) = mem_available()
        && kv + scratch > available
    {
        return Err(format!(
            "KV cache and scratch need {:.0} MiB but only {:.0} MiB of RAM is available; \
                 lower --context (each token costs {} bytes)",
            (kv + scratch) as f64 / MIB,
            available as f64 / MIB,
            config.kv_bytes_per_token()
        ));
    }
    let budget = ram_budget(args, kv + scratch)?;
    let residency = kllm::model::Residency::plan(&info, config.n_layer, budget);
    if let Some(moe) = config.moe {
        eprintln!(
            "kllm: {} of {} experts per token ({:.0}% of {:.0} MiB of expert weights), \
             demand-paged",
            moe.n_expert_used,
            moe.n_expert,
            moe.active_fraction() * 100.0,
            residency.demand_bytes as f64 / MIB,
        );
    }
    if residency.first_streamed < config.n_layer || residency.head_streamed {
        let streamed = config.n_layer - residency.first_streamed;
        eprintln!(
            "kllm: model exceeds the RAM budget ({:.0} MiB): {} of {} blocks{} resident ({:.0} MiB), \
             {streamed} blocks{} streamed from disk ({:.0} MiB per token)",
            budget.unwrap_or(0) as f64 / MIB,
            residency.first_streamed,
            config.n_layer,
            if residency.head_streamed {
                ""
            } else {
                " + head"
            },
            residency.resident_bytes as f64 / MIB,
            if residency.head_streamed {
                " + head"
            } else {
                ""
            },
            // A routed model reads only its selected experts, so the
            // streamed figure has to be the active bytes or it overstates
            // the traffic by 1/active_fraction -- 16x for Qwen3-30B-A3B.
            config.moe.map_or(residency.streamed_bytes as f64, |m| {
                residency.streamed_bytes as f64 * m.active_fraction()
            }) / MIB
        );
    }
    let start = Instant::now();
    #[allow(unused_mut)]
    let mut model = Model::load_with(path, kllm::model::LoadOptions { ram_budget: budget })
        .map_err(|e| e.to_string())?;
    if gpu != GpuRequest::Off {
        offload(args, gpu, &info, &mut model, context, batch)?;
    }
    eprintln!("kllm: loaded in {:.2} s", start.elapsed().as_secs_f64());
    Ok(model)
}

#[cfg(not(feature = "cuda"))]
fn offload(
    _: &Args,
    _: GpuRequest,
    _: &GgufInfo,
    _: &mut Model,
    _: usize,
    _: usize,
) -> Result<(), String> {
    Err("this kLLM was built without GPU support (enable the `cuda` feature)".into())
}

#[cfg(feature = "cuda")]
fn offload(
    args: &Args,
    gpu: GpuRequest,
    info: &GgufInfo,
    model: &mut Model,
    context: usize,
    batch: usize,
) -> Result<(), String> {
    let device = kllm::cuda::default_gpu().map_err(|e| e.to_string())?;
    let (free, _) = device.memory_info().map_err(|e| e.to_string())?;
    let c = model.config.clone();
    let c = &c;
    let (layers, head) = match gpu {
        GpuRequest::Off => return Ok(()),
        GpuRequest::Layers(n) => (n.min(c.n_layer), n >= c.n_layer),
        GpuRequest::Auto => {
            // The planner reserves KV per GPU block itself; the workspace is
            // the session's scratch with the head's logits buffer included.
            let mut input = model_plan_input(info, args, context as u64, free)?;
            input.workspace_bytes = Model::gpu_session_bytes(c, 0, true, context, batch);
            let plan = plan_model(&input).map_err(|e| e.to_string())?;
            (plan.gpu_layers as usize, plan.head_on_gpu)
        }
    };
    let needed = Model::gpu_session_bytes(c, layers, head, context, batch);
    let placed = model
        .offload(std::sync::Arc::clone(&device), layers, head)
        .map_err(|e| e.to_string())?;
    eprintln!(
        "kllm: GPU {}: {} of {} blocks{} | weights {:.0} MiB, KV+scratch {:.0} MiB, {:.0} MiB was free",
        device.name,
        placed.gpu_layers,
        c.n_layer,
        if placed.head { " + head" } else { "" },
        placed.weight_bytes as f64 / MIB,
        needed as f64 / MIB,
        free as f64 / MIB
    );
    Ok(())
}

/// RAM the weights may occupy: `--ram-gib N`, `--ram-gib all` (never stream),
/// or by default what is available now minus the session's own needs and a
/// 1 GiB margin for everything else running.
fn ram_budget(args: &Args, session_bytes: u64) -> Result<Option<u64>, String> {
    match args.value(&["--ram-gib"])? {
        Some("all") => Ok(None),
        Some("auto") | None => {
            Ok(mem_available().map(|a| a.saturating_sub(session_bytes + (1 << 30))))
        }
        Some(v) => {
            let gib: f64 = v
                .parse()
                .map_err(|_| format!("invalid value {v:?} for --ram-gib"))?;
            Ok(Some(bytes_from_gib(gib)?))
        }
    }
}

/// Emit only complete UTF-8 sequences; tokens can split a character.
struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            Err(_) => self.pending.len(), // genuinely invalid: flush lossily
        };
        let text = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
        self.pending.drain(..valid);
        text
    }
}

fn cmd_run(args: &Args) -> Result<(), String> {
    let path = args.positional(1, "GGUF path")?;
    let threads = init_threads(args)?;
    let prompt: String = args
        .value(&["--prompt", "-p"])?
        .ok_or("usage: missing --prompt")?
        .into();
    let n_predict: usize = args.or(&["-n", "--n-predict"], 256)?;
    let info = inspect_file(path).map_err(|e| e.to_string())?;
    let config = Config::from_gguf(&info).map_err(|e| e.to_string())?;
    let context: usize = args.or(&["--context", "-c"], config.train_context.min(4096))?;
    // Tokenize before loading, so the batch can be sized to the prompt that
    // actually has to be processed. A wide batch is what makes a routed
    // model's prefill fast, but it is paid for in logits scratch for the
    // whole session -- and that scratch competes with the page cache the
    // experts live in, so buying it for a twenty-token prompt would slow
    // the generation that follows.
    let tokenizer = Tokenizer::from_gguf(&info).map_err(|e| e.to_string())?;
    let text = if args.has("--chat") {
        let system = args.value(&["--system"])?;
        format_chat(&info, system, &prompt).ok_or("model has no recognisable chat template")?
    } else {
        prompt
    };
    let mut tokens = tokenizer.encode(&text, true, true);
    if tokens.is_empty() {
        return Err("prompt tokenized to nothing".into());
    }
    let batch: usize = args.or(
        &["--batch", "-b"],
        default_batch(&config).min(tokens.len().max(1)),
    )?;
    let model = load_with_plan(args, gpu_request(args)?, path, context, batch, threads)?;
    if tokens.len() >= context {
        return Err(format!(
            "prompt is {} tokens; context is {context}",
            tokens.len()
        ));
    }
    let mut sampler = Sampler::new(SamplerConfig {
        temperature: args.or(&["--temp"], 0.8)?,
        top_k: args.or(&["--top-k"], 40)?,
        top_p: args.or(&["--top-p"], 0.95)?,
        min_p: args.or(&["--min-p"], 0.05)?,
        repeat_penalty: args.or(&["--repeat-penalty"], 1.0)?,
        repeat_last_n: args.or(&["--repeat-last-n"], 64)?,
        seed: args.or(&["--seed", "-s"], 0x6b4c_4c4d)?,
    });
    let ignore_eos = args.has("--ignore-eos");

    let mut session = model
        .new_session(context, batch)
        .map_err(|e| e.to_string())?;
    let mut stdout = io::stdout().lock();
    let mut stream = Utf8Stream {
        pending: Vec::new(),
    };
    if !args.has("--chat") {
        let _ = write!(stdout, "{}", tokenizer.decode(&tokens, false));
        let _ = stdout.flush();
    }

    let start = Instant::now();
    let prompt_len = tokens.len();
    let mut logits = session
        .forward(&tokens, Logits::Last)
        .map_err(|e| e.to_string())?
        .to_vec();
    let prompt_time = start.elapsed();
    let gen_start = Instant::now();
    let mut generated = 0;
    while generated < n_predict && session.position() < context {
        let next = sampler.sample(&logits, &tokens);
        if !ignore_eos && tokenizer.is_eog(next) {
            break;
        }
        tokens.push(next);
        generated += 1;
        let piece = stream.push(&tokenizer.token_bytes(next, false));
        if write!(stdout, "{piece}")
            .and_then(|_| stdout.flush())
            .is_err()
        {
            return Ok(());
        }
        if generated == n_predict || session.position() >= context {
            break;
        }
        logits = session
            .forward(&[next], Logits::Last)
            .map_err(|e| e.to_string())?
            .to_vec();
    }
    let gen_time = gen_start.elapsed();
    let _ = writeln!(stdout, "{}", stream.push(&[]));
    eprintln!(
        "kllm: prompt {prompt_len} tokens in {:.2} s ({:.1} tok/s) | generated {generated} tokens in {:.2} s ({:.1} tok/s)",
        prompt_time.as_secs_f64(),
        prompt_len as f64 / prompt_time.as_secs_f64(),
        gen_time.as_secs_f64(),
        generated.max(1) as f64 / gen_time.as_secs_f64().max(1e-9)
    );
    Ok(())
}

fn cmd_perplexity(args: &Args) -> Result<(), String> {
    let path = args.positional(1, "GGUF path")?;
    let threads = init_threads(args)?;
    let file: String = args
        .value(&["--file", "-f"])?
        .ok_or("usage: missing --file")?
        .into();
    let n_ctx: usize = args.or(&["--context", "-c"], 512)?;
    if n_ctx < 8 {
        return Err("--context must be at least 8".into());
    }
    let model = load_with_plan(args, gpu_request(args)?, path, n_ctx, n_ctx, threads)?;
    let tokenizer = Tokenizer::from_gguf(model.info()).map_err(|e| e.to_string())?;
    let text = std::fs::read_to_string(&file).map_err(|e| format!("{file}: {e}"))?;
    let tokens = tokenizer.encode(&text, true, false);
    let available = tokens.len() / n_ctx;
    let chunks: usize = args.or(&["--chunks"], available)?.min(available);
    if chunks == 0 {
        return Err(format!("{file} has fewer than {n_ctx} tokens"));
    }
    eprintln!(
        "kllm: {} tokens, scoring {chunks} chunks of {n_ctx} (second half of each, like llama-perplexity)",
        tokens.len()
    );

    let start = Instant::now();
    let result = perplexity(
        &model,
        &tokenizer,
        &tokens,
        n_ctx,
        Some(chunks),
        |chunk, ppl| {
            eprint!("[{chunk}]{ppl:.4} ");
        },
    )
    .map_err(|e| e.to_string())?;
    let elapsed = start.elapsed().as_secs_f64();
    eprintln!();
    out!(
        "Final estimate: PPL = {:.4} +/- {:.4} over {} tokens ({} chunks of {n_ctx}), {:.1} tok/s",
        result.ppl,
        result.stderr,
        result.tokens_scored,
        result.chunks,
        (result.chunks * n_ctx) as f64 / elapsed
    );
    Ok(())
}

fn cmd_compare(args: &Args) -> Result<(), String> {
    let reference_path = args.positional(1, "reference GGUF path")?;
    let test_path = args.positional(2, "test GGUF path")?;
    let threads = init_threads(args)?;
    let file: String = args
        .value(&["--file", "-f"])?
        .ok_or("usage: missing --file")?
        .into();
    let n_ctx: usize = args.or(&["--context", "-c"], 512)?;
    if n_ctx < 8 {
        return Err("--context must be at least 8".into());
    }
    // Both models are resident at once, so plan them against half the budget
    // each rather than letting either believe it owns the machine.
    let reference = load_with_plan(args, GpuRequest::Off, reference_path, n_ctx, n_ctx, threads)?;
    let test = load_with_plan(args, GpuRequest::Off, test_path, n_ctx, n_ctx, threads)?;
    let tokenizer = Tokenizer::from_gguf(reference.info()).map_err(|e| e.to_string())?;
    let text = std::fs::read_to_string(&file).map_err(|e| format!("{file}: {e}"))?;
    let tokens = tokenizer.encode(&text, true, false);
    let available = tokens.len() / n_ctx;
    let chunks: usize = args.or(&["--chunks"], available)?.min(available);
    if chunks == 0 {
        return Err(format!("{file} has fewer than {n_ctx} tokens"));
    }
    eprintln!(
        "kllm: {} tokens, {chunks} chunks of {n_ctx}; reference {} ({}), test {} ({})",
        tokens.len(),
        reference_path,
        reference.weight_types().join("/"),
        test_path,
        test.weight_types().join("/"),
    );
    let start = Instant::now();
    let d = kllm::eval::divergence(
        &reference,
        &test,
        &tokenizer,
        &tokens,
        n_ctx,
        Some(chunks),
        |chunk, kl| eprint!("[{chunk}]{kl:.5} "),
    )
    .map_err(|e| e.to_string())?;
    eprintln!();
    out!("reference:      {reference_path}");
    out!("test:           {test_path}");
    out!(
        "tokens scored:  {} ({} chunks of {n_ctx}) in {:.1} s",
        d.tokens_scored,
        d.chunks,
        start.elapsed().as_secs_f64()
    );
    out!(
        "mean KL:        {:.6} +/- {:.6} nats",
        d.kl_mean,
        d.kl_stderr
    );
    out!(
        "KL median/p99:  {:.6} / {:.6}  (max {:.6})",
        d.kl_median,
        d.kl_p99,
        d.kl_max
    );
    out!(
        "top-1 agree:    {:.2}%   top-5 agree: {:.2}%",
        d.top1_agree * 100.0,
        d.top5_agree * 100.0
    );
    out!(
        "PPL:            reference {:.4} -> test {:.4}  ({:+.2}%)",
        d.reference_ppl,
        d.test_ppl,
        (d.test_ppl / d.reference_ppl - 1.0) * 100.0
    );
    Ok(())
}

fn cmd_devices() -> Result<(), String> {
    out!("CPU: {} | kernels {}", cpu_name(), kernels::kernel_name());
    #[cfg(feature = "cuda")]
    match kllm::cuda::default_gpu() {
        Ok(gpu) => {
            let (free, total) = gpu.memory_info().map_err(|e| e.to_string())?;
            out!(
                "GPU 0: {} | compute {}.{} | {} SMs @ {} MHz max | {:.0} MiB free of {:.0} MiB",
                gpu.name,
                gpu.compute_capability.0,
                gpu.compute_capability.1,
                gpu.multiprocessors,
                gpu.clock_mhz,
                free as f64 / MIB,
                total as f64 / MIB
            );
        }
        Err(e) => out!("GPU: none ({e})"),
    }
    #[cfg(not(feature = "cuda"))]
    out!("GPU: support not compiled in (build with --features cuda)");
    Ok(())
}

/// Default prompt batch. A routed model wants a wide one: tokens are
/// grouped by the expert they chose, so each expert's weights are read once
/// per batch rather than once per token that chose it, and a batch of `b`
/// gives each of `n_expert` experts about `b * n_expert_used / n_expert`
/// tokens to work on. Measured on Qwen3-30B-A3B (8 of 128 experts), pp256
/// went 7.1 -> 11.4 -> 16.2 tok/s across batches of 64, 128 and 256 for
/// exactly that reason. Dense models gain nothing from a wider batch and pay
/// for it in logits scratch, so they keep 64.
fn default_batch(config: &Config) -> usize {
    const DENSE: usize = 64;
    let Some(moe) = config.moe else { return DENSE };
    // Aim for ~16 tokens per expert, which is where the measured gain
    // flattened, and cap it so the logits scratch stays sane.
    let wanted = 16 * moe.n_expert / moe.n_expert_used.max(1);
    wanted.clamp(DENSE, 256)
}

fn cmd_bench(args: &Args) -> Result<(), String> {
    let path = args.positional(1, "GGUF path")?;
    let threads = init_threads(args)?;
    let n_prompt: usize = args.or(&["-p", "--n-prompt"], 128)?;
    let n_gen: usize = args.or(&["-n", "--n-gen"], 64)?;
    let repeats: usize = args.or(&["--repeat", "-r"], 3)?.max(1);
    let info = inspect_file(path).map_err(|e| e.to_string())?;
    let batch: usize = args.or(
        &["--batch", "-b"],
        default_batch(&Config::from_gguf(&info).map_err(|e| e.to_string())?),
    )?;
    let context = n_prompt + n_gen + 1;
    let model = load_with_plan(args, gpu_request(args)?, path, context, batch, threads)?;
    let vocab = model.config.n_vocab as u32;
    let prompt: Vec<u32> = (0..n_prompt as u32)
        .map(|i| (i * 7919 + 13) % vocab)
        .collect();

    let mut session = model
        .new_session(context, batch)
        .map_err(|e| e.to_string())?;
    let profile = args.has("--profile");
    // Warm-up: touch every weight page once.
    session
        .forward(&prompt[..n_prompt.min(8)], Logits::Last)
        .map_err(|e| e.to_string())?;
    let (mut pp, mut tg) = (Vec::new(), Vec::new());
    for _ in 0..repeats {
        session.truncate(0);
        let start = Instant::now();
        let logits = session
            .forward(&prompt, Logits::Last)
            .map_err(|e| e.to_string())?;
        let mut next = argmax(logits);
        pp.push(n_prompt as f64 / start.elapsed().as_secs_f64());
        let start = Instant::now();
        for _ in 0..n_gen {
            next = argmax(
                session
                    .forward(&[next], Logits::Last)
                    .map_err(|e| e.to_string())?,
            );
        }
        tg.push(n_gen as f64 / start.elapsed().as_secs_f64());
    }
    let mut breakdown = Vec::new();
    if profile {
        // Extra runs with timing on, so timing cost is not in pp/tg above.
        let mut report = |label: &str, session: &Session, total: f64, tokens: usize| {
            breakdown.push(format!(
                "{label} ({:.3} ms/token):",
                total * 1e3 / tokens as f64
            ));
            let mut accounted = 0.0;
            for (name, t) in kllm::model::PHASES.iter().zip(session.profiler.totals) {
                accounted += t.as_secs_f64();
                breakdown.push(format!(
                    "  {name:>14}: {:5.1}%",
                    100.0 * t.as_secs_f64() / total
                ));
            }
            breakdown.push(format!(
                "  {:>14}: {:5.1}%",
                "other",
                100.0 * (total - accounted) / total
            ));
        };
        session.truncate(0);
        session.set_profiling(true);
        let start = Instant::now();
        let logits = session
            .forward(&prompt, Logits::Last)
            .map_err(|e| e.to_string())?;
        let mut next = argmax(logits);
        report("prefill", &session, start.elapsed().as_secs_f64(), n_prompt);
        session.set_profiling(true);
        let start = Instant::now();
        for _ in 0..n_gen {
            next = argmax(
                session
                    .forward(&[next], Logits::Last)
                    .map_err(|e| e.to_string())?,
            );
        }
        report("decode", &session, start.elapsed().as_secs_f64(), n_gen);
    }
    // The best run, not the mean, is the honest estimate of how fast the code
    // is: anything else sharing the machine can only ever make a run slower,
    // never faster. The mean and spread are still reported, because a wide
    // spread is the signal that the machine was busy and the comparison
    // should be repeated.
    let stats = |v: &[f64]| {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        let best = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (mean, sd, best)
    };
    let (pp_mean, pp_sd, pp_best) = stats(&pp);
    let (tg_mean, tg_sd, tg_best) = stats(&tg);
    let bandwidth = model.weight_bytes() as f64 * tg_best / (1024.0 * MIB);
    out!("model:      {path}");
    out!("types:      {}", model.weight_types().join(", "));
    out!(
        "hardware:   {} | {threads} threads | kernels {}",
        cpu_name(),
        kernels::kernel_name()
    );
    out!("command:    kllm {}", args.0.join(" "));
    out!("pp{n_prompt}:      {pp_best:8.1} tok/s best  (mean {pp_mean:.1} ± {pp_sd:.1})");
    out!(
        "tg{n_gen}:       {tg_best:8.1} tok/s best  (mean {tg_mean:.1} ± {tg_sd:.1}), \
         ~{bandwidth:.1} GiB/s of weights"
    );
    if !breakdown.is_empty() {
        out!("profile:");
        for line in breakdown {
            out!("  {line}");
        }
    }
    Ok(())
}

fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i as u32)
}
