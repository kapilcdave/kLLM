#!/usr/bin/env python3
"""Run kLLM's whole eval suite and write one reproducible markdown report.

usage: scripts/eval_report.py [--models DIR] [--llama-cpp DIR] [--out FILE]
                              [--threads N] [--only SUBSTRING] [--quick]

For every model it records, in one table: perplexity, KL divergence from the
family's reference model, top-1 agreement, and prefill/decode speed -- each
beside llama.cpp's number for the same model on the same machine, because a
speed or quality claim against our own past self is not worth much.

Every row carries the model's SHA-256 and the report carries the hardware and
the exact commands, so a number in it can be re-derived later.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time

PPL_CONTEXT, PPL_CHUNKS = 256, 8
KLD_CONTEXT, KLD_CHUNKS = 512, 4
BENCH_PROMPT, BENCH_GEN = 128, 64

REFERENCES = {
    "SmolLM2-135M-Instruct": "SmolLM2-135M-Instruct-f16.gguf",
    "Qwen3-0.6B": "Qwen3-0.6B-Q8_0.gguf",
}


def run(command, timeout=3600):
    started = time.time()
    done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    return done.stdout + done.stderr, time.time() - started


def number(pattern, text, group=1):
    match = re.search(pattern, text)
    return float(match.group(group)) if match else None


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def family(name):
    return next((p for p in REFERENCES if name.startswith(p)), None)


class Kllm:
    def __init__(self, binary, threads):
        self.binary = binary
        self.threads = str(threads)

    def perplexity(self, model, text):
        out, _ = run([self.binary, "perplexity", model, "--file", text,
                      "--context", str(PPL_CONTEXT), "--chunks", str(PPL_CHUNKS),
                      "--threads", self.threads])
        return number(r"PPL = ([\d.]+)", out)

    def compare(self, reference, model, text):
        out, _ = run([self.binary, "compare", reference, model, "--file", text,
                      "--context", str(KLD_CONTEXT), "--chunks", str(KLD_CHUNKS),
                      "--threads", self.threads])
        return {
            "kl_mean": number(r"mean KL:\s+([\d.]+)", out),
            "kl_median": number(r"KL median/p99:\s+([\d.]+)", out),
            "kl_p99": number(r"KL median/p99:\s+[\d.]+ / ([\d.]+)", out),
            "top1": number(r"top-1 agree:\s+([\d.]+)", out),
        }

    def bench(self, model):
        out, _ = run([self.binary, "bench", model, "-p", str(BENCH_PROMPT),
                      "-n", str(BENCH_GEN), "--threads", self.threads, "--repeat", "3"])
        return (number(rf"pp{BENCH_PROMPT}:\s+([\d.]+)", out),
                number(rf"tg{BENCH_GEN}:\s+([\d.]+)", out))

    def devices(self):
        out, _ = run([self.binary, "devices"])
        return out.strip()


class LlamaCpp:
    def __init__(self, bindir, threads):
        self.bindir = bindir
        self.threads = str(threads)

    def perplexity(self, model, text):
        out, _ = run([f"{self.bindir}/llama-perplexity", "-m", model, "-f", text,
                      "-c", str(PPL_CONTEXT), "--chunks", str(PPL_CHUNKS),
                      "-t", self.threads])
        return number(r"Final estimate: PPL = ([\d.]+)", out)

    def compare(self, reference, model, text, base_file):
        run([f"{self.bindir}/llama-perplexity", "-m", reference, "-f", text,
             "-c", str(KLD_CONTEXT), "--chunks", str(KLD_CHUNKS), "-t", self.threads,
             "--kl-divergence-base", base_file])
        out, _ = run([f"{self.bindir}/llama-perplexity", "-m", model, "-f", text,
                      "-c", str(KLD_CONTEXT), "--chunks", str(KLD_CHUNKS),
                      "-t", self.threads, "--kl-divergence",
                      "--kl-divergence-base", base_file])
        return {
            "kl_mean": number(r"Mean\s+KLD:\s+([\d.]+)", out),
            "kl_median": number(r"Median\s+KLD:\s+([\d.]+)", out),
            "kl_p99": number(r"99\.0%\s+KLD:\s+([\d.]+)", out),
            "top1": number(r"Same top p:\s+([\d.]+)", out),
        }

    def bench(self, model):
        out, _ = run([f"{self.bindir}/llama-bench", "-m", model,
                      "-p", str(BENCH_PROMPT), "-n", str(BENCH_GEN),
                      "-t", self.threads, "-r", "3"])
        rows = re.findall(r"\|\s+(pp|tg)\d+\s+\|\s+([\d.]+) ±", out)
        got = {kind: float(value) for kind, value in rows}
        return got.get("pp"), got.get("tg")

    def build(self):
        out, _ = run([f"{self.bindir}/llama-perplexity", "--version"])
        return next((l for l in out.splitlines() if "build" in l), "unknown").strip()


def ratio(ours, theirs):
    if not ours or not theirs:
        return "--"
    return f"{ours / theirs:.2f}x"


def cell(value, digits=4):
    return "--" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--models", default=os.path.join(root, "models"))
    parser.add_argument("--llama-cpp", default=os.environ.get("LLAMA_CPP_BIN"))
    parser.add_argument("--binary", default=os.path.join(root, "target/release/kllm"))
    parser.add_argument("--out")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--only", default="")
    parser.add_argument("--quick", action="store_true",
                        help="speed only: skip perplexity and divergence")
    args = parser.parse_args()

    text = os.path.join(root, "tests/fixtures/wiki.test.head.txt")
    names = sorted(f for f in os.listdir(args.models)
                   if f.endswith(".gguf") and args.only in f)
    if not names:
        raise SystemExit(f"no models matching {args.only!r} in {args.models}")

    ours = Kllm(args.binary, args.threads)
    theirs = LlamaCpp(args.llama_cpp, args.threads) if args.llama_cpp else None
    commit, _ = run(["git", "-C", root, "rev-parse", "--short", "HEAD"])
    dirty, _ = run(["git", "-C", root, "status", "--porcelain"])

    rows = []
    for name in names:
        path = os.path.join(args.models, name)
        print(f"[{name}]", file=sys.stderr, flush=True)
        row = {"name": name, "sha": sha256(path)[:12],
               "mib": os.path.getsize(path) / (1 << 20)}
        row["pp"], row["tg"] = ours.bench(path)
        if theirs:
            row["pp_ref"], row["tg_ref"] = theirs.bench(path)
        if not args.quick:
            row["ppl"] = ours.perplexity(path, text)
            if theirs:
                row["ppl_ref"] = theirs.perplexity(path, text)
            reference = REFERENCES.get(family(name) or "")
            if reference and reference != name:
                reference_path = os.path.join(args.models, reference)
                if os.path.exists(reference_path):
                    row["reference"] = reference
                    row.update(ours.compare(reference_path, path, text))
                    if theirs:
                        got = theirs.compare(reference_path, path, text,
                                             "/tmp/kllm-eval-base.dat")
                        row.update({f"{k}_ref": v for k, v in got.items()})
        rows.append(row)

    report = []
    add = report.append
    add(f"# kLLM eval report -- {time.strftime('%Y-%m-%d')}")
    add("")
    add(f"- commit: `{commit.strip()}`"
        + (" (working tree dirty)" if dirty.strip() else ""))
    add(f"- hardware: {ours.devices()}")
    add(f"- host: {platform.platform()}")
    add(f"- threads: {args.threads}")
    if theirs:
        add(f"- reference: llama.cpp {theirs.build()}")
    add(f"- corpus: `tests/fixtures/wiki.test.head.txt`"
        f" (perplexity {PPL_CHUNKS} chunks of {PPL_CONTEXT},"
        f" divergence {KLD_CHUNKS} chunks of {KLD_CONTEXT})")
    add("")
    add("## Speed")
    add("")
    add("| model | MiB | pp kLLM | pp llama.cpp | ratio | tg kLLM | tg llama.cpp | ratio |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        add(f"| {r['name']} | {r['mib']:.0f} | {cell(r.get('pp'), 1)} | "
            f"{cell(r.get('pp_ref'), 1)} | {ratio(r.get('pp'), r.get('pp_ref'))} | "
            f"{cell(r.get('tg'), 1)} | {cell(r.get('tg_ref'), 1)} | "
            f"{ratio(r.get('tg'), r.get('tg_ref'))} |")
    if not args.quick:
        add("")
        add("## Quality")
        add("")
        add("Perplexity is absolute; KL divergence and top-1 agreement are against "
            "the family's reference model, so they measure what the quantization "
            "changed rather than how good the model is.")
        add("")
        add("| model | PPL kLLM | PPL llama.cpp | vs reference | mean KL | "
            "llama.cpp KL | top-1 | llama.cpp top-1 |")
        add("|---|---:|---:|---|---:|---:|---:|---:|")
        for r in rows:
            top1 = r.get("top1")
            top1_ref = r.get("top1_ref")
            add(f"| {r['name']} | {cell(r.get('ppl'))} | {cell(r.get('ppl_ref'))} | "
                f"{r.get('reference', '--')} | {cell(r.get('kl_mean'), 6)} | "
                f"{cell(r.get('kl_mean_ref'), 6)} | "
                f"{'--' if top1 is None else f'{top1:.2f}%'} | "
                f"{'--' if top1_ref is None else f'{top1_ref:.2f}%'} |")
    add("")
    add("## Model hashes")
    add("")
    add("| model | SHA-256 (first 12) |")
    add("|---|---|")
    for r in rows:
        add(f"| {r['name']} | `{r['sha']}` |")
    add("")
    add("## Commands")
    add("")
    add("```")
    add(f"kllm perplexity MODEL --file tests/fixtures/wiki.test.head.txt "
        f"--context {PPL_CONTEXT} --chunks {PPL_CHUNKS} --threads {args.threads}")
    add(f"kllm compare REFERENCE MODEL --file tests/fixtures/wiki.test.head.txt "
        f"--context {KLD_CONTEXT} --chunks {KLD_CHUNKS} --threads {args.threads}")
    add(f"kllm bench MODEL -p {BENCH_PROMPT} -n {BENCH_GEN} "
        f"--threads {args.threads} --repeat 3")
    if theirs:
        add(f"llama-bench -m MODEL -p {BENCH_PROMPT} -n {BENCH_GEN} "
            f"-t {args.threads} -r 3")
        add(f"llama-perplexity -m MODEL -f tests/fixtures/wiki.test.head.txt "
            f"-c {PPL_CONTEXT} --chunks {PPL_CHUNKS} -t {args.threads}")
    add("```")
    text_out = "\n".join(report) + "\n"
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text_out)
        print(f"wrote {args.out}", file=sys.stderr)
        # Also dump the raw rows so a later report can diff against them.
        with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
            json.dump(rows, f, indent=1)
    else:
        print(text_out)


if __name__ == "__main__":
    main()
