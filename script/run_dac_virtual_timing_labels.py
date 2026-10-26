"""在 DAC2027 Nangate45 artifact 上生成 virtual timing labels。"""

import argparse
import json
import os
import re
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

from dataset.dac_virtual_timing import build_virtual_labels


def _run(command: list[str], cwd: Path, log: Path, timeout: int) -> None:
    """运行单个 OpenSTA 任务并保存完整日志。"""
    process = subprocess.Popen(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        log.write_text(output + "\nTIMEOUT\n")
        raise TimeoutError(f"timeout after {timeout}s")
    log.write_text(output)
    if process.returncode:
        raise RuntimeError(f"OpenSTA failed with code {process.returncode}")


def run_one(design_id: str, root: Path, output: Path, liberty: Path,
            max_nodes: int, topk: int, timeout: int) -> dict:
    """生成一个设计的 OpenSTA 路径报告和 torch label artifact。"""
    graph_path = root / "graphs_timing_arc_v1" / f"{design_id}.pt"
    run_dir = root / "ppa_sta_runs" / design_id
    if not graph_path.is_file() or not (run_dir / "mapped_circuit_v2.flat.v").is_file():
        return {"design_id": design_id, "status": "missing_artifact"}
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    if int(graph["n"]) > max_nodes:
        return {"design_id": design_id, "status": "excluded_max_nodes", "nodes": int(graph["n"])}
    label_dir = output / design_id
    label_dir.mkdir(parents=True, exist_ok=True)
    timing_path = label_dir / "timing_paths.json"
    labels_path = label_dir / "labels.pt"
    if not timing_path.is_file() or timing_path.stat().st_size < 50:
        info = json.loads((root / "labels" / f"{design_id}.json").read_text())
        top = info.get("top") or info.get("top_module")
        base_sta = (run_dir / "sta.tcl").read_text(errors="ignore")
        clock = re.search(r"create_clock\s+-name\s+\S+\s+-period\s+([0-9.eE+-]+).*?get_ports\s+\{?([^}\s\]]+)", base_sta)
        period, port = (clock.group(1), clock.group(2)) if clock else ("10.0", "clk")
        sta = label_dir / "virtual_timing.tcl"
        sta.write_text(
            f"read_liberty {liberty}\nread_verilog {run_dir / 'mapped_circuit_v2.flat.v'}\n"
            f"link_design {{{top}}}\ncreate_clock -name ppa_clk -period {period} [get_ports {{{port}}}]\n"
            f"report_checks -path_delay max -group_path_count 100000 -endpoint_path_count 1 "
            f"-format json > {timing_path}\nexit\n")
        _run(["sta", "-no_init", "-exit", str(sta)], label_dir,
             label_dir / "opensta.log", timeout)
    labels, paths = build_virtual_labels(graph_path, timing_path, design_id, topk)
    torch.save(labels, labels_path)
    (label_dir / "paths.json").write_text(json.dumps(paths, indent=2))
    status = "ok" if labels["metadata"]["path_count"] > 0 else "excluded_no_paths"
    manifest = labels["metadata"] | {"status": status, "timing_paths": str(timing_path),
                                      "liberty": str(liberty)}
    (label_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    """按 DAC split 运行可恢复的 OpenSTA label 队列。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/yiyh/graphmath/dac2027/processed/verilog_rtl_corpus"))
    parser.add_argument("--output", type=Path, default=Path("dataset/dac2027_virtual_timing_v1"))
    parser.add_argument("--liberty", type=Path, default=Path("/home/yiyh/tools/ppa-labeler/lib/NangateOpenCellLibrary_typical.lib"))
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-nodes", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.output = args.output.resolve()
    args.liberty = args.liberty.resolve()
    split = json.loads((args.root / "splits/seed42_mapped_circuit_v2.json").read_text())
    ids = split["train"] + split["val"] + split["test"] if args.split == "all" else split[args.split]
    if args.limit:
        ids = ids[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    results, next_log = [], time.monotonic() + 5.0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, design_id, args.root, args.output, args.liberty,
                                args.max_nodes, args.topk, args.timeout) for design_id in ids]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                results.append({"status": "failed", "error": repr(error)})
            if time.monotonic() >= next_log:
                print(f"progress done={len(results)}/{len(ids)}", flush=True)
                next_log = time.monotonic() + 5.0
    (args.output / "manifest.json").write_text(json.dumps({"task": "virtual_timing_v1",
        "designs_requested": len(ids), "results": results}, indent=2))
    print(f"virtual_timing_done={len(results)} output={args.output}")


if __name__ == "__main__":
    main()
