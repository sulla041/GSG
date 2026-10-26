"""审计并修正 DAC virtual timing label 的空路径状态。"""

import argparse
import json
from pathlib import Path


def main() -> None:
    """把无 OpenSTA path 的样本从可训练状态降级为排除状态。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("dataset/dac2027_virtual_timing_v1"))
    args = parser.parse_args()
    results = []
    for path in sorted(args.root.glob("*/manifest.json")):
        data = json.loads(path.read_text())
        status = "ok" if int(data.get("path_count", 0)) > 0 else "excluded_no_paths"
        data["status"] = status
        path.write_text(json.dumps(data, indent=2) + "\n")
        results.append({"design_id": path.parent.name, "status": status,
                        "path_count": int(data.get("path_count", 0))})
    (args.root / "manifest.json").write_text(json.dumps({
        "task": "virtual_timing_v1", "designs": len(results), "results": results}, indent=2) + "\n")
    counts = {}
    for item in results:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    print(f"virtual_timing_audit designs={len(results)} status={counts}")


if __name__ == "__main__":
    main()
