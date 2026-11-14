#!/usr/bin/env python3
"""Record llama.cpp reference outputs that kLLM's tests compare against.

usage: make_fixtures.py LLAMA_CPP_BIN_DIR MODEL.gguf OUT_PREFIX

Writes OUT_PREFIX.tokens.json (tokenizer cases) and OUT_PREFIX.greedy.json
(greedy completions). The llama.cpp build is recorded in each file.
"""
import json, os, subprocess, sys, time, urllib.request

CASES = [
    "Hello world",
    " Hello  world ",
    "Hello world, it's 2026! 12345 tokens.",
    "I'll we've they're you'd she's OK'S",
    "  leading spaces and trailing   ",
    "line one\nline two\n\n\tindented\r\nwindows",
    "naïve café résumé — “quotes” ‘single’ …",
    "日本語のテキストと中文混合",
    "emoji 🚀🔥 and zwj 👩‍💻 family",
    "numbers 3.14159 1,000,000 0x1F 42nd",
    "def f(x):\n    return x**2 + 1  # comment",
    "<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n",
    "a" * 50,
    "Ñandú ΑΒΓ αβγ Привет мир",
    "",
    "   ",
    "\n\n\n",
    "x=1;y=2;;z  =  3",
]

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    "def fibonacci(n):",
]

def main():
    bindir, model, prefix = sys.argv[1:4]
    version = subprocess.run([f"{bindir}/llama-completion", "--version"],
                             capture_output=True, text=True)
    build = (version.stdout + version.stderr).strip().splitlines()[0]
    tokens = []
    for text in CASES:
        out = subprocess.run([f"{bindir}/llama-tokenize", "-m", model, "--stdin", "--ids",
                              ], input=text.encode(), capture_output=True, check=True)
        line = out.stdout.decode().strip().splitlines()[-1]
        tokens.append({"text": text, "ids": json.loads(line)})
    json.dump({"llama_cpp": build, "model": os.path.basename(model), "cases": tokens},
              open(prefix + ".tokens.json", "w"), ensure_ascii=False, indent=1)

    greedy = []
    port = 18000 + os.getpid() % 1000
    server = subprocess.Popen([f"{bindir}/llama-server", "-m", model, "--port", str(port),
                               "-c", "1024", "-t", "4", "--no-warmup", "-np", "1"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(300):
            try:
                if json.load(urllib.request.urlopen(base + "/health"))["status"] == "ok":
                    break
            except Exception:
                time.sleep(0.1)
        for prompt in PROMPTS:
            body = {"prompt": prompt, "n_predict": 32, "temperature": 0, "top_k": 1,
                    "ignore_eos": True, "n_probs": 10, "cache_prompt": False,
                    "return_tokens": True, "post_sampling_probs": False}
            req = urllib.request.Request(base + "/completion", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req))
            steps = [{"id": p["id"], "logprob": p["logprob"],
                      "top": [[t["id"], t["logprob"]] for t in p["top_logprobs"]]}
                     for p in r["completion_probabilities"]]
            greedy.append({"prompt": prompt, "prompt_ids": None, "completion": r["content"],
                           "ids": r["tokens"], "steps": steps})
            tok = urllib.request.Request(base + "/tokenize", json.dumps(
                {"content": prompt, "add_special": True, "parse_special": True}).encode(),
                {"Content-Type": "application/json"})
            greedy[-1]["prompt_ids"] = json.load(urllib.request.urlopen(tok))["tokens"]
    finally:
        server.terminate()
        server.wait()
    json.dump({"llama_cpp": build, "model": os.path.basename(model), "n_predict": 32,
               "cases": greedy}, open(prefix + ".greedy.json", "w"), ensure_ascii=False, indent=1)

main()
