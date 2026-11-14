#!/usr/bin/env python3
"""Record the llama.cpp reference numbers kLLM's eval tests compare against.

usage: make_eval_fixtures.py LLAMA_CPP_BIN_DIR MODELS_DIR [MODEL.gguf ...]

Writes tests/fixtures/perplexity.json and tests/fixtures/divergence.json.

Perplexity is recorded with flash attention both on and off: on quantized
models those two differ by a few tenths of a percent, and that spread is
llama.cpp's own implementation noise, so it is the honest tolerance band for
kLLM rather than a single number.

Divergence is llama.cpp's `--kl-divergence`: the reference model's logits are
saved once, then every other model in the family is scored against them. KL
divergence catches damage that perplexity hides, because a quantization can
leave perplexity alone and still reorder the top of the distribution.
"""
import json
import os
import re
import subprocess
import sys

# Perplexity settings, matching the existing fixture.
PPL_CONTEXT, PPL_CHUNKS = 256, 8
# Divergence uses a longer context: KL per position is noisier than NLL, so
# the extra conditioning is worth the halved chunk count. Four chunks keeps a
# divergence test to a few seconds a model while still scoring ~1000 tokens.
KLD_CONTEXT, KLD_CHUNKS = 512, 4

# Each family's reference model, against which the rest are scored.
REFERENCES = {
    "SmolLM2-135M-Instruct": "SmolLM2-135M-Instruct-f16.gguf",
    "Qwen3-0.6B": "Qwen3-0.6B-Q8_0.gguf",
}


def family(name):
    for prefix in REFERENCES:
        if name.startswith(prefix):
            return prefix
    return None


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout + result.stderr


def build_id(bindir):
    text = run([f"{bindir}/llama-perplexity", "--version"]).strip().splitlines()
    return next((line for line in text if "build" in line), text[-1] if text else "unknown")


def perplexity(bindir, model, text, flash):
    out = run([
        f"{bindir}/llama-perplexity", "-m", model, "-f", text,
        "-c", str(PPL_CONTEXT), "--chunks", str(PPL_CHUNKS), "-t", "4",
        "-fa", "on" if flash else "off",
    ])
    match = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", out)
    if not match:
        raise SystemExit(f"no perplexity in output for {model}:\n{out[-2000:]}")
    return float(match.group(1)), float(match.group(2))


def divergence(bindir, model, text, base_file, save_base=False):
    command = [
        f"{bindir}/llama-perplexity", "-m", model, "-f", text,
        "-c", str(KLD_CONTEXT), "--chunks", str(KLD_CHUNKS), "-t", "4",
        "--kl-divergence-base", base_file,
    ]
    if not save_base:
        command.append("--kl-divergence")
    out = run(command)
    if save_base:
        return None
    def grab(pattern, group=1):
        match = re.search(pattern, out)
        return float(match.group(group)) if match else None
    result = {
        "kl_mean": grab(r"Mean\s+KLD:\s+([\d.]+)"),
        "kl_mean_stderr": grab(r"Mean\s+KLD:\s+[\d.]+\s+±\s+([\d.]+)"),
        "kl_median": grab(r"Median\s+KLD:\s+([\d.]+)"),
        "kl_p99": grab(r"99\.0%\s+KLD:\s+([\d.]+)"),
        "kl_max": grab(r"Maximum KLD:\s+([\d.]+)"),
        "top1_agree": grab(r"Same top p:\s+([\d.]+)"),
        "top1_agree_stderr": grab(r"Same top p:\s+[\d.]+\s+±\s+([\d.]+)"),
        "ppl_test": grab(r"Mean PPL\(Q\)\s+:\s+([\d.]+)"),
        "ppl_reference": grab(r"Mean PPL\(base\)\s+:\s+([\d.]+)"),
    }
    if result["kl_mean"] is None:
        raise SystemExit(f"no KL divergence in output for {model}:\n{out[-3000:]}")
    return result


def main():
    bindir, models_dir = sys.argv[1:3]
    names = sys.argv[3:] or sorted(
        f for f in os.listdir(models_dir) if f.endswith(".gguf")
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fixtures = os.path.join(root, "tests", "fixtures")
    text = os.path.join(fixtures, "wiki.test.head.txt")
    build = build_id(bindir)

    ppl = {}
    for name in names:
        model = os.path.join(models_dir, name)
        on = perplexity(bindir, model, text, True)
        off = perplexity(bindir, model, text, False)
        ppl[name] = {"ppl_fa_on": on[0], "stderr": on[1], "ppl_fa_off": off[0]}
        print(f"{name}: PPL {on[0]:.4f} (fa on) / {off[0]:.4f} (fa off)", flush=True)
    with open(os.path.join(fixtures, "perplexity.json"), "w") as f:
        json.dump({
            "llama_cpp": build,
            "context": PPL_CONTEXT,
            "chunks": PPL_CHUNKS,
            "text": "wiki.test.head.txt",
            "note": "ppl_fa_on is llama.cpp's default (flash attention on). "
                    "The on/off spread is llama.cpp's own implementation noise.",
            "models": ppl,
        }, f, indent=1)
        f.write("\n")

    kld = {}
    for prefix, reference in REFERENCES.items():
        members = [n for n in names if family(n) == prefix and n != reference]
        if not members or not os.path.exists(os.path.join(models_dir, reference)):
            continue
        base_file = f"/tmp/kllm-kld-{prefix}.dat"
        divergence(bindir, os.path.join(models_dir, reference), text, base_file,
                   save_base=True)
        for name in members:
            got = divergence(bindir, os.path.join(models_dir, name), text, base_file)
            got["reference"] = reference
            kld[name] = got
            print(f"{name}: mean KLD {got['kl_mean']:.6f}, "
                  f"top-1 {got['top1_agree']:.2f}%", flush=True)
        os.remove(base_file)
    with open(os.path.join(fixtures, "divergence.json"), "w") as f:
        json.dump({
            "llama_cpp": build,
            "context": KLD_CONTEXT,
            "chunks": KLD_CHUNKS,
            "text": "wiki.test.head.txt",
            "note": "llama-perplexity --kl-divergence against the family's "
                    "reference model, flash attention at its default (on). "
                    "KL divergence is in nats; top1_agree is a percentage.",
            "models": kld,
        }, f, indent=1)
        f.write("\n")
    print(f"wrote perplexity.json ({len(ppl)} models) and "
          f"divergence.json ({len(kld)} models)")


if __name__ == "__main__":
    main()
