"""批量使用 Yosys/OpenSTA 生成 TimingPredict 的 PPA 标签。"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import re
import subprocess
from pathlib import Path

import py7zr
from ppa_fine import export_fine_labels


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "dataset/raw/timingpredict/timing_raw_design.7z"
DEFAULT_EXTRACT = ROOT / "dataset/raw/timingpredict_extracted"
DEFAULT_OUTPUT = ROOT / "dataset/labeled/timingpredict_ppa"


def run_tool(command: list[str], cwd: Path, log_path: Path) -> str:
    """运行工具并保存 stdout/stderr，保留失败诊断。"""
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    log_path.write_text(result.stdout + "\n" + result.stderr)
    if result.returncode:
        raise RuntimeError("tool failed: " + " ".join(command))
    return result.stdout + result.stderr


def parse_json_log(text: str) -> dict:
    """从 Yosys 混合日志中读取 stat -json 对象。"""
    start = text.find("{")
    if start < 0:
        raise ValueError("Yosys JSON statistics were not found")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    return value


def parse_last_float(path: Path) -> float | None:
    """解析 OpenSTA 摘要文件最后一个浮点数。"""
    values = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
                        path.read_text())
    return float(values[-1]) if values else None


def graph_metadata() -> dict:
    """声明 TimingPredict 的节点、边和稠密关系类型。"""
    return {
        "node_type_count": 1,
        "node_types": ["timing_pin"],
        "node_types_countable": True,
        "edge_type_count": 3,
        "edge_types": ["net_out", "net_in", "cell_out"],
        "edge_types_countable": True,
        "dense_relation_type_count": 4,
        "dense_relation_types": ["r_null", "net_out", "net_in", "cell_out"],
        "null_relation_is_explicit": True,
        "node_label_fields": ["arrival_time", "required_time", "slack", "slew"],
        "edge_label_fields": ["delay", "arrival_time", "slew", "criticality"],
    }


def extract_archive(archive: Path, destination: Path) -> None:
    """只在目标目录不存在时解压原始设计包。"""
    if (destination / "netlists").exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive) as package:
        package.extractall(path=destination)


def label_design(design_dir: Path, output_dir: Path, liberty_min: Path,
                 liberty_max: Path) -> dict:
    """为一个已映射网表生成结构、面积、时序和功耗标签。"""
    name = design_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    netlist = design_dir / f"{name}.synthesis_preroute.v"
    spef = design_dir / f"20-{name}.spef"
    sdc = design_dir / f"{name}.cts_1.sdc"
    yosys_script = output_dir / "stat.ys"
    yosys_script.write_text(
        f"read_verilog {netlist}\nhierarchy -top {name}\n"
        f"stat -liberty {liberty_max} -json\nwrite_json {output_dir / 'netlist.json'}\n"
    )
    yosys_text = run_tool(["yosys", "-s", str(yosys_script)], design_dir,
                          output_dir / "yosys.log")
    stat = parse_json_log(yosys_text)
    sta_script = output_dir / "opensta.tcl"
    sta_script.write_text(
        "set_cmd_units -time ns -capacitance pF -current mA "
        "-voltage V -resistance kOhm -distance um\n"
        f"read_liberty -min {liberty_min}\nread_liberty -max {liberty_max}\n"
        f"read_verilog {netlist}\nlink_design {name}\n"
        f"read_spef {spef}\nread_sdc {sdc}\n"
        f"report_wns > {output_dir / 'wns.txt'}\n"
        f"report_tns > {output_dir / 'tns.txt'}\n"
        f"report_checks -path_delay max -group_path_count 100000 "
        f"-endpoint_path_count 1 "
        f"-format json > {output_dir / 'timing_paths.json'}\n"
        f"report_power -instances [get_cells *] -format json > {output_dir / 'instance_power.json'}\n"
        f"report_power -format json > {output_dir / 'power.json'}\nexit\n"
    )
    run_tool(["sta", "-no_init", "-exit", str(sta_script)], design_dir,
             output_dir / "opensta.log")
    fine = export_fine_labels(
        output_dir / "netlist.json", liberty_max,
        output_dir / "timing_paths.json", output_dir / "instance_power.json",
        output_dir / "fine",
    )
    labels = {
        "design": name,
        "tools": {"yosys": "0.55", "opensta": "3.1.0"},
        "graph_metadata": graph_metadata(),
        "area_and_structure": stat,
        "timing": {
            "wns": parse_last_float(output_dir / "wns.txt"),
            "tns": parse_last_float(output_dir / "tns.txt"),
            "paths": json.loads((output_dir / "timing_paths.json").read_text()),
        },
        "power": json.loads((output_dir / "power.json").read_text()),
        "power_activity": "OpenSTA default activity; no VCD/SAIF supplied",
        "fine_grained": {
            "node_file": str(output_dir / "fine/nodes.json"),
            "relation_file": str(output_dir / "fine/relations.json"),
            "node_count": len(fine["nodes"]),
            "relation_count": len(fine["relations"]),
        },
    }
    (output_dir / "labels.json").write_text(json.dumps(labels, indent=2))
    return {"design": name, "status": "ok", "labels": str(output_dir / "labels.json")}


def parse_args():
    """读取批量标签任务配置。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--extract-root", type=Path, default=DEFAULT_EXTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--designs", nargs="*", help="默认处理全部 21 个设计")
    parser.add_argument("--workers", type=int, default=8,
                        help="并行的独立 Yosys/OpenSTA 进程数")
    return parser.parse_args()


def label_one(task: tuple[str, Path, Path, Path, Path]) -> dict:
    """在独立进程中处理一个设计，避免工具状态互相污染。"""
    name, root, output, liberty_min, liberty_max = task
    try:
        result = label_design(root / name, output / name, liberty_min, liberty_max)
        return result
    except Exception as error:
        return {"design": name, "status": "failed", "error": repr(error)}


def main() -> None:
    """解压并批量生成每个设计的 PPA 标签与总 manifest。"""
    args = parse_args()
    args.archive = args.archive.resolve()
    args.extract_root = args.extract_root.resolve()
    args.output = args.output.resolve()
    extract_archive(args.archive, args.extract_root)
    root = args.extract_root / "netlists"
    liberty_min = root / "techlib/sky130_fd_sc_hd__ff_n40C_1v95.lib"
    liberty_max = root / "techlib/sky130_fd_sc_hd__ss_100C_1v60.lib"
    names = args.designs or sorted(path.name for path in root.iterdir()
                                   if path.is_dir() and path.name != "techlib")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    tasks = [(name, root, args.output, liberty_min, liberty_max) for name in names]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(label_one, task): task[0] for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            result = future.result()
            results.append(result)
            print(f"label_done design={name} progress={index}/{len(names)} "
                  f"status={result['status']}", flush=True)
    results.sort(key=lambda item: item["design"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(results, indent=2))
    print(f"manifest={args.output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
