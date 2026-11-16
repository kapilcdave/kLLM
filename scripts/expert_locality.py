#!/usr/bin/env python3
"""Turn a router trace into the number that decides MoE-from-disk.

usage: KLLM_ROUTER_TRACE=/tmp/trace.txt kllm run MODEL --prompt ... -n 500
       expert_locality.py /tmp/trace.txt --expert-mib 1.2 [--layers 48]

An MoE reads only its routed experts per token, so a model far larger than
RAM is viable *if* expert selection repeats enough for a cache to absorb it.
Whether it does is a property of the data. This replays a real decode trace
against an LRU over (layer, expert) pairs and reports the hit rate against
cache size, which is the curve that says whether the idea works.

Read it like this: the useful token rate is roughly

    1 / (active_bytes * (1 - hit_rate) / disk_bandwidth)

so at 1.9 GiB active per token and 2.5 GiB/s of NVMe, a 0% hit rate gives
~1.3 tok/s and a 90% hit rate gives ~13 tok/s. The question is which end of
that range the curve lands on.
"""
import argparse
import collections
import sys


def read_trace(path):
    """Yield (position, layer, [experts]) in order."""
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue
            position, layer, experts = parts
            yield int(position), int(layer), [int(e) for e in experts.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--expert-mib", type=float, default=1.2,
                        help="resident MiB per (layer, expert): gate+up+down")
    parser.add_argument("--active-gib", type=float, default=1.9,
                        help="expert bytes one token reads, for the rate estimate")
    parser.add_argument("--disk-gibs", type=float, default=2.5,
                        help="sequential read bandwidth, GiB/s")
    args = parser.parse_args()

    events = list(read_trace(args.trace))
    if not events:
        raise SystemExit(f"{args.trace}: no usable lines")
    reads = [(layer, e) for _, layer, experts in events for e in experts]
    tokens = len({p for p, _, _ in events})
    layers = len({l for _, l, _ in events})
    distinct = len(set(reads))
    print(f"{len(events)} (token, layer) events over {tokens} positions, "
          f"{layers} layers")
    print(f"{len(reads)} expert reads, {distinct} distinct (layer, expert) pairs")
    print(f"a full residency would be {distinct * args.expert_mib / 1024:.2f} GiB "
          f"at {args.expert_mib} MiB per expert")

    # How skewed is selection? If this is flat, no cache can help.
    per_layer = collections.defaultdict(collections.Counter)
    for layer, e in reads:
        per_layer[layer][e] += 1
    shares = []
    for layer, counter in per_layer.items():
        total = sum(counter.values())
        top = sum(c for _, c in counter.most_common(max(1, len(counter) // 10)))
        shares.append(top / total)
    print(f"the hottest 10% of a layer's experts take "
          f"{100 * sum(shares) / len(shares):.1f}% of its reads "
          f"(10% would mean no skew at all)")

    # The curve. An LRU over (layer, expert) pairs, replayed in trace order.
    print()
    print("| cache | entries | hit rate | est. tok/s |")
    print("|---|---:|---:|---:|")
    budgets = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    for fraction in budgets:
        capacity = max(1, int(distinct * fraction))
        cache = collections.OrderedDict()
        hits = 0
        for key in reads:
            if key in cache:
                cache.move_to_end(key)
                hits += 1
            else:
                if len(cache) >= capacity:
                    cache.popitem(last=False)
                cache[key] = None
        rate = hits / len(reads)
        miss_gib = args.active_gib * (1.0 - rate)
        tok_s = args.disk_gibs / miss_gib if miss_gib > 1e-9 else float("inf")
        print(f"| {capacity * args.expert_mib / 1024:.2f} GiB | {capacity} | "
              f"{100 * rate:.1f}% | {tok_s:.1f} |")

    # Belt and braces: how much of the hit rate is just "the same token's
    # experts across layers"? A per-layer-only cache cannot exploit that.
    repeat_within = sum(
        1 for i in range(1, len(reads)) if reads[i] == reads[i - 1]
    )
    print()
    print(f"immediate repeats (the same (layer, expert) twice in a row): "
          f"{repeat_within} of {len(reads)}")
    print("If the curve is flat, MoE-from-disk is bandwidth-bound and the",
          file=sys.stderr)
    print("expert cache is not worth building; see docs/MOE.md.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
