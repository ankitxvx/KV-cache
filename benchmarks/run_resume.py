"""Build bench_resume first; run on Linux/WSL to capture reproducible evidence."""
import argparse
import csv
import hashlib
import io
import json
import os
import pathlib
import platform
import statistics
import subprocess
from datetime import datetime, timezone

parser = argparse.ArgumentParser()
parser.add_argument("--binary", default="build-audit/benchmarks/bench_resume")
parser.add_argument("--output", default="benchmarks/results/resume")
args = parser.parse_args()
out = pathlib.Path(args.output)
out.mkdir(parents=True, exist_ok=True)
affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
if affinity:
    os.sched_setaffinity(0, {affinity[0]})
proc = subprocess.run([args.binary], text=True, capture_output=True, check=True)
rows = list(csv.DictReader(io.StringIO(proc.stdout)))
if len(rows) != 84:
    raise RuntimeError(f"Expected 84 measurement rows, got {len(rows)}")
(out / "raw.csv").write_text(proc.stdout)
(out / "stderr.txt").write_text(proc.stderr)
summary = []
for workload in dict.fromkeys(r["workload"] for r in rows):
    for operation in dict.fromkeys(r["operation"] for r in rows):
        group = [r for r in rows if r["workload"] == workload and r["operation"] == operation]
        result = {"workload": workload, "operation": operation, "repetitions": len(group)}
        for field in ("operations_per_second", "payload_GB_per_second", "p50_us", "p95_us", "p99_us"):
            vals = [float(r[field]) for r in group]
            result[field] = {"median": statistics.median(vals), "min": min(vals), "max": max(vals)}
        result["physical_blocks"] = [int(r["physical_blocks"]) for r in group]
        summary.append(result)
cpuinfo = pathlib.Path("/proc/cpuinfo").read_text() if pathlib.Path("/proc/cpuinfo").exists() else ""
cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.splitlines() if line.startswith("model name")), platform.processor())
source_hashes = {}
for folder in ("src", "include", "benchmarks"):
    for path in pathlib.Path(folder).rglob("*"):
        if path.suffix in {".h", ".cpp", ".cu", ".cuh", ".py"} and "results" not in path.parts:
            source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
metadata = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "cpu": cpu, "os": platform.platform(), "python": platform.python_version(),
    "logical_cpus_available": affinity, "pinned_cpu": affinity[0] if affinity else None,
    "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
    "binary_sha256": hashlib.sha256(pathlib.Path(args.binary).read_bytes()).hexdigest(),
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "source_sha256": source_hashes,
    "method": "Release C++17, CPU only, one benchmark thread pinned to one logical CPU; 1 warmup and 7 measured repetitions per workload",
    "geometry": {"layers": 4, "kv_heads": 4, "head_dim": 64, "block_tokens": 16, "dtype": "FP32", "KV_bytes_per_token_all_layers": 8192},
    "scope": "Synthetic cache operations; excludes allocation of the initial pool, input preparation, model computation, GPU transfers and inference generation. Operation latency includes clock instrumentation. GB/s uses decimal bytes.",
    "fork_check": "32 simultaneous forks allocated zero additional physical KV blocks; metadata allocations are not included in that claim.",
    "checks": "Append success, expected physical block count, copied payload equality, fork source length isolation, full prefix attachment and payload equality, zero retained blocks after cleanup",
    "summary": summary,
}
(out / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps({"cpu": cpu, "pinned_cpu": metadata["pinned_cpu"], "summary": summary}, indent=2))
