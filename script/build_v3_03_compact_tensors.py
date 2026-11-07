#!/usr/bin/env python3
"""从已验收的 v1/v2 pair 张量构造 CC-Tensor v3 紧凑缓存。"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "code" / "v3_03"))
from compact_tensor_codec import compact_sample

DEFAULT_ROOT = Path("/data/yiyh/dac2027/dataset/circuit")
DEFAULT_OUT = DEFAULT_ROOT / "circuitnet3_v3_03_compact_tensors"
TASKS = ("A1", "S1", "D1", "D2", "D3", "R1", "P1", "P2")


def memory_budget() -> int:
    """读取可用内存并只允许使用其中一半作为并行预算。"""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024 // 2
    return 4 * 1024**3


def load_meta(path: Path) -> dict:
    meta = json.loads(path.read_text())
    meta["relation_ids"] = {k: int(v) for k, v in meta["relation_ids"].items()}
    return meta


def source_files(source_v1: Path, source_v2: Path, task: str, max_nodes: int) -> list[Path]:
    base = (source_v2 if task == "D1" else source_v1) / task
    files = []
    for path in sorted(base.glob("*.pt")):
        try:
            with torch.serialization.safe_globals([]):
                data = torch.load(path, map_location="cpu", weights_only=False)
            if data["input_nodes"].size(0) <= max_nodes:
                files.append(path)
        except Exception:
            continue
    return files


def build_one(args):
    source, target, meta_path = args
    data = torch.load(source, map_location="cpu", weights_only=False)
    meta = load_meta(meta_path)
    relation_names = [name for name, _ in sorted(meta["relation_ids"].items(), key=lambda item: item[1])]
    result = compact_sample(data, meta, relation_names)
    result["input_node_keys"] = data.get("input_node_keys", [])
    result["output_node_keys"] = data.get("output_node_keys", [])
    result["schema"] = "circuitnet3_v3_03_cc_tensor_v3"
    torch.save(result, target)
    return {"task": data["task"], "design_id": data["design_id"], "status": "ready",
            "file": str(target), "nodes": result["meta"]["num_nodes"],
            "edges": result["meta"]["num_edges"], "targets": result["meta"]["num_targets"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source-v1", type=Path, default=DEFAULT_ROOT / "circuitnet3_v3_03_task_tensors_v1")
    parser.add_argument("--source-v2", type=Path, default=DEFAULT_ROOT / "circuitnet3_v3_03_task_tensors_v2")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--max-nodes", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    jobs, missing = [], []
    for task in args.tasks:
        source_root = args.source_v2 if task == "D1" else args.source_v1
        meta_path = source_root / "meta_index.json"
        for source in source_files(args.source_v1, args.source_v2, task, args.max_nodes):
            target = args.output / task / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not args.force:
                continue
            jobs.append((source, target, meta_path))
        if not meta_path.exists(): missing.append({"task": task, "status": "missing_source_meta"})
    per_worker = max(64 * 1024**2, args.max_nodes * 128 * 1024)
    workers = max(1, min(args.workers, memory_budget() // per_worker))
    print(f"jobs={len(jobs)} workers={workers} memory_budget_gib={memory_budget()/1024**3:.1f}", flush=True)
    results = list(missing)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(build_one, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            source = futures[future][0]
            try: results.append(future.result())
            except Exception as error: results.append({"source": str(source), "status": "failed", "error": repr(error)})
            if index % 100 == 0 or index == len(futures): print(f"progress={index}/{len(futures)}", flush=True)
    results.sort(key=lambda row: (row.get("task", ""), row.get("design_id", "")))
    index = {"schema": "circuitnet3_v3_03_cc_tensor_v3_index", "tasks": args.tasks,
             "max_nodes": args.max_nodes, "results": results,
             "source_versions": {"D1": "v2", "others": "v1"}}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "meta_index.json").write_text(json.dumps(index, indent=2) + "\n")
    from collections import Counter
    print(json.dumps({"results": len(results), "status": dict(Counter(x.get("status") for x in results)),
                      "output": str(args.output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
