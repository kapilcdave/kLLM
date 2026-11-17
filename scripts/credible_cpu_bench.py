#!/usr/bin/env python3
"""Run an auditable, interleaved kLLM/llama.cpp CPU microbenchmark.

Both engines process the standard llama-bench workload sizes (pp512/tg128),
use the same four physical cores, run three timed repetitions per process,
and alternate process order over five rounds. Results report process-level
means and the median/range across processes. The engines choose their own
internal batching and synthetic token stream, so this is an equivalent-size
engine microbenchmark rather than an identical-token chat-latency test.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_text(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def host_state() -> dict:
    load = os.getloadavg()
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "load_1m": load[0],
        "load_5m": load[1],
        "load_15m": load[2],
        "governor": read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "ac_online": read_text("/sys/class/power_supply/AC/online"),
    }


def parse_kllm(raw: str, prompt: int, generation: int) -> dict:
    metrics = {}
    for name, count in (("pp", prompt), ("tg", generation)):
        match = re.search(
            rf"{name}{count}:.*?mean\s+([\d.]+)\s+±\s+([\d.]+)", raw
        )
        if not match:
            raise ValueError(f"missing kLLM {name}{count} mean")
        metrics[name] = float(match.group(1))
        metrics[f"{name}_within_process_sd"] = float(match.group(2))
    return metrics


def parse_llama(raw: str) -> dict:
    rows = json.loads(raw)
    metrics = {}
    for row in rows:
        name = "pp" if row["n_prompt"] else "tg"
        metrics[name] = float(row["avg_ts"])
        metrics[f"{name}_within_process_sd"] = float(row["stddev_ts"])
    if "pp" not in metrics or "tg" not in metrics:
        raise ValueError("missing llama.cpp pp or tg result")
    return metrics


def summarise(samples: list[dict]) -> dict:
    summary = {}
    for engine in ("kllm", "llama.cpp"):
        good = [s for s in samples if s["engine"] == engine and "error" not in s]
        if not good:
            continue
        summary[engine] = {"successful_rounds": len(good)}
        for metric in ("pp", "tg"):
            values = [s[metric] for s in good]
            summary[engine][metric] = {
                "median_process_mean": statistics.median(values),
                "mean_process_mean": statistics.fmean(values),
                "min_process_mean": min(values),
                "max_process_mean": max(values),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--prompt", type=int, default=512)
    parser.add_argument("--generation", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cpus", default="0-3")
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--llama-repack",
        choices=("0", "1"),
        default="1",
        help="set to 0 when llama.cpp's repacked copy would exceed RAM",
    )
    parser.add_argument(
        "--llama-bench",
        type=Path,
        default=Path("/home/kapil/ai/llama.cpp/cpu/llama-b12099/llama-bench"),
    )
    args = parser.parse_args()
    if args.rounds < 2 or args.repetitions < 2:
        parser.error("use at least two process rounds and two internal repetitions")

    args.out.mkdir(parents=True, exist_ok=True)
    kllm = args.out.resolve() / "kllm-snapshot"
    shutil.copy2(ROOT / "target/release/kllm", kllm)
    model = args.model.resolve()
    llama = args.llama_bench.resolve()
    prefix = ["taskset", "-c", args.cpus]
    commands = {
        "kllm": prefix
        + [
            str(kllm),
            "bench",
            str(model),
            "-p",
            str(args.prompt),
            "-n",
            str(args.generation),
            "-t",
            str(args.threads),
            "-r",
            str(args.repetitions),
            "--gpu-layers",
            "0",
        ],
        "llama.cpp": prefix
        + [
            str(llama),
            "-m",
            str(model),
            "-p",
            str(args.prompt),
            "-n",
            str(args.generation),
            "-t",
            str(args.threads),
            "-r",
            str(args.repetitions),
            "-ngl",
            "0",
            "-dev",
            "none",
            "-o",
            "json",
            "--repack",
            args.llama_repack,
        ],
    }
    data = {
        "schema": 1,
        "model": str(model),
        "model_sha256": digest(model),
        "kllm_binary_sha256": digest(kllm),
        "llama_binary_sha256": digest(llama),
        "kllm_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "llama_version": subprocess.run(
            [str(llama), "--version"], capture_output=True, text=True
        ).stdout.strip(),
        "host": {
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "pinned_cpus": args.cpus,
            "threads": args.threads,
            "initial_state": host_state(),
        },
        "method": {
            "workload": f"pp{args.prompt}/tg{args.generation}",
            "process_rounds": args.rounds,
            "repetitions_per_process": args.repetitions,
            "order": "alternating by round",
            "gpu": "disabled for both engines",
            "reported_statistic": "median of process-level means",
            "limitation": __doc__,
        },
        "commands": commands,
        "samples": [],
        "summary": {},
    }
    output_file = args.out / "results.json"

    for round_id in range(1, args.rounds + 1):
        order = ("kllm", "llama.cpp") if round_id % 2 else ("llama.cpp", "kllm")
        for engine in order:
            if data["samples"]:
                time.sleep(args.delay)
            sample = {
                "engine": engine,
                "round": round_id,
                "host_before": host_state(),
            }
            started = time.monotonic()
            result = subprocess.run(
                commands[engine], capture_output=True, text=True, timeout=args.timeout
            )
            sample["wall_seconds"] = time.monotonic() - started
            sample["exit_code"] = result.returncode
            raw = result.stdout + ("\n" + result.stderr if result.stderr else "")
            log_name = f"{round_id}-{engine}.log"
            (args.out / log_name).write_text(raw)
            sample["raw_log"] = log_name
            try:
                if result.returncode:
                    raise ValueError(f"exit {result.returncode}")
                sample.update(
                    parse_kllm(raw, args.prompt, args.generation)
                    if engine == "kllm"
                    else parse_llama(result.stdout)
                )
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                sample["error"] = str(error)
            data["samples"].append(sample)
            data["summary"] = summarise(data["samples"])
            output_file.write_text(json.dumps(data, indent=2) + "\n")
            print(json.dumps(sample), flush=True)

    if any("error" in sample for sample in data["samples"]):
        raise SystemExit("one or more samples failed; inspect results.json and raw logs")


if __name__ == "__main__":
    main()
