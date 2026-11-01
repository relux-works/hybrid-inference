"""Per-turn cost model. Rates are tokens/s and bytes/s; times are seconds."""
import argparse
import json
import math
from dataclasses import dataclass


def extend_chunk_limit(kv_bytes=34816, state_bytes=157_000_000):
    # One sequence has 24 bytes of native cell metadata per token; reserve 1 MiB for fixed serialization.
    if (type(kv_bytes) is not int or type(state_bytes) is not int
            or kv_bytes <= 0 or not 0 <= state_bytes < 511 * 1024**2):
        raise ValueError("invalid native delta geometry")
    return min((1024**2 - 4096) // 11,
               (512 * 1024**2 - state_bytes - 1024**2) // (kv_bytes + 24))


@dataclass(frozen=True)
class Policy:
    local_rates: tuple = ((0, 100.0),)
    server_rate: float = 2000.0
    link_rate: float = 12_500_000.0
    fixed: float = 0.05
    kv_bytes: float = 32768.0
    state_bytes: float = 157_000_000.0
    eta: float = 0.85
    margin: float = 0.1
    hysteresis: float = 0.05
    always: bool = False

    def __post_init__(self):
        values = [self.server_rate, self.link_rate, self.fixed, self.kv_bytes, self.state_bytes,
                  self.eta, self.margin, self.hysteresis]
        values.extend(v for row in self.local_rates for v in row)
        if (not all(math.isfinite(v) for v in values)
                or not self.local_rates or any(d < 0 or r <= 0 for d, r in self.local_rates)
                or min(self.server_rate, self.link_rate, self.eta) <= 0
                or min(self.fixed, self.kv_bytes, self.state_bytes, self.hysteresis) < 0
                or not 0 <= self.margin < 1 or self.eta > 1):
            raise ValueError("invalid policy rates or bounds")

    def choose(self, delta, depth, previous=False, link_rate=None, server_factor=1.0):
        if not all(math.isfinite(v) for v in (delta, depth, server_factor)) or delta < 0 or depth < 0 or server_factor <= 0:
            raise ValueError("invalid turn size or server factor")
        rate = sorted(self.local_rates)[0][1]
        for at, value in sorted(self.local_rates):
            if depth >= at:
                rate = value
        link = self.link_rate if link_rate is None else link_rate
        if not math.isfinite(link) or link <= 0:
            raise ValueError("invalid measured link rate")
        local = delta / rate
        remote = self.fixed + delta * server_factor / self.server_rate
        remote += (delta * self.kv_bytes + self.state_bytes) / (link * self.eta)
        threshold = 1 - self.margin + (self.hysteresis if previous else -self.hysteresis)
        return {"offload": delta > 0 and (self.always or remote < local * threshold),
                "local_seconds": local, "offload_seconds": remote}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="JSONL: delta, depth, optional server_factor")
    parser.add_argument("--config", help="JSON Policy fields; local_rates is [[depth, rate], ...]")
    args = parser.parse_args()
    config = json.load(open(args.config)) if args.config else {}
    policy = Policy(**config)
    previous = False
    with open(args.trace) as trace:
        for line in trace:
            row = json.loads(line)
            result = policy.choose(row["delta"], row["depth"], previous,
                                   server_factor=row.get("server_factor", 1))
            print(json.dumps(result))
            previous = result["offload"]


if __name__ == "__main__":
    main()
