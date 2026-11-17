# Running Claude Code against a local model

What `kllm serve` needs around it to be a thing you actually use rather than a
thing you start by hand. Four files:

| file | goes to | what it is |
|---|---|---|
| `serve.env.example` | `~/.config/kllm/serve.env` | which model, how much context, the shared token |
| `kllm.slice` | `~/.config/systemd/user/kllm.slice` | a memory ceiling so inference cannot wedge the box |
| `kllm-serve.service` | `~/.config/systemd/user/kllm-serve.service` | keeps the server up across reboots |
| `claude-kllm` | `~/.local/bin/claude-kllm` | starts the server if needed, points Claude Code at it |

```sh
mkdir -p ~/.config/kllm ~/.config/systemd/user ~/.local/bin ~/kllm/logs
cp contrib/serve.env.example ~/.config/kllm/serve.env   # then edit KLLM_MODEL
cp contrib/kllm.slice contrib/kllm-serve.service ~/.config/systemd/user/
install -m755 contrib/claude-kllm ~/.local/bin/
systemctl --user daemon-reload
systemctl --user enable --now kllm-serve.service
loginctl enable-linger "$USER"      # survive logout
claude-kllm --kllm-status
claude-kllm                          # Claude Code on the local model
```

`--kllm-status`, `--kllm-warm`, `--kllm-restart` and `--kllm-log` do what they
say; anything else is passed through to `claude`.

## Why systemd and not a background shell

The server holds a multi-gigabyte mapping for as long as it runs. Started from
a shell it gets reaped the moment the box comes under memory pressure, and on
this machine that happens. A unit also means it is back after a reboot, which
matters more than it sounds: a restart throws away the page cache *and* the
prefix cache, and both are expensive to rebuild.

`kllm.slice` exists because of a specific incident -- an interactive job peaked
at 10.6 G RSS on a 15 GiB laptop, the global OOM killer fired twice, and the
swap thrash cost an unrelated recorder 62 of 180 minutes. A cgroup ceiling
contains that. It is set well above `heavy.slice`'s because almost all of
inference's footprint is reclaimable file cache: measured expert locality on a
routed model of this shape is 61% hit at 1.9 GiB of cache, 85% at 4.8 and 92%
at 7.2, so capping at 7 G would sit right on the knee.

## The env vars, and why each one

`claude-kllm` sets these for you. They are listed because when something talks
to the cloud while looking local, the cause is in here.

- `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN` -- where to send requests.
- `ANTHROPIC_MODEL` and the `_SONNET_`/`_HAIKU_`/`_OPUS_` defaults -- Claude
  Code wants a name for its main *and* its background model. They are all the
  same name here, because there is one loaded model.
- `CLAUDE_CODE_MAX_CONTEXT_TOKENS` -- without it Claude Code does not recognise
  the model id, assumes a 1M-token window, never compacts, and eventually
  sends a prompt the server must reject.
- `API_TIMEOUT_MS` (1 hour) -- a cold prefill of a large system prompt takes
  minutes, and the default timeout abandons it.
- `CLAUDE_CODE_MAX_OUTPUT_TOKENS` (2048) -- at single-digit tok/s a 4k answer
  is eight minutes.
- **unset** `ANTHROPIC_API_KEY`, `CLAUDE_CODE_USE_BEDROCK`,
  `CLAUDE_CODE_USE_VERTEX` and friends -- each of these takes precedence over
  `ANTHROPIC_BASE_URL`, so leaving one set sends your traffic to a cloud
  provider while every other sign says local.

## What to expect

Measured on a 4-core laptop with 16 GiB of RAM, no usable GPU:

- **The first request of a session is slow and the rest are not.** Prefill runs
  at 12-18 tok/s, so a large system prompt costs minutes once; after that the
  prefix cache means each turn only reads what is new. Watch
  `prompt N reused + M read` in the log.
- **Warm up before you judge it.** The first answers after a restart read
  experts from NVMe at a few tok/s; it takes two or three before the page cache
  settles. `claude-kllm --kllm-warm`.
- **One session.** Generation serializes, and there is a single prefix cache.
  Two conversations against one server will evict each other and every turn
  becomes a full prefill again.

## The flag that decides whether this is usable daily

`claude-kllm` passes `--exclude-dynamic-system-prompt-sections`, and on this
setup that is not a micro-optimisation.

Claude Code's system prompt and tool definitions come to about **11,000
tokens**, which is a ~13 minute cold prefill here. By default that prompt also
carries per-machine blocks -- working directory, environment info, memory
paths, git status -- and those differ on every run. So the prompt mismatches
near its *start*, the prefix cache misses, and **every new session pays the
full thirteen minutes again**. Measured: a second session with the default
prompt was still prefilling past 600 s.

The flag moves those blocks into the first user message instead. The expensive
part of the prompt becomes byte-identical across sessions, the prefix cache
hits, and only the genuinely new tail is read. Upstream offers it for
cross-user prompt-cache reuse on the API; here it is the difference between
"slow once" and "slow every time".

On a hybrid model the flag is necessary but not sufficient, because a *partial*
prefix match is not cheap there. A dense model's KV cache rewinds for free, but
a recurrence keeps no per-position record, so rewinding to the divergence point
used to replay everything below it: a fresh session reported `8496 reused +
2386 read` and still took 739 s. The server now saves the recurrent state at
that divergence point and resumes from it, which made the same request
40x faster in a controlled run (`682 reused + 10 read`: 44.4 s to 1.1 s). The
cost is one ~66 MiB snapshot, and the *second* session after a restart still
pays the replay once -- that is when the snapshot gets taken.

Two consequences worth keeping in mind:

- **The 13 minutes is per server lifetime, not per session** -- so leave the
  unit running. A restart throws the prefix away and you pay it again. This is
  the main reason it is a systemd unit rather than something you start when you
  need it.
- **Switching models in `serve.env` also throws it away**, on top of the cold
  page cache. Expect a slow first answer after any restart.
