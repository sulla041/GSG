"""为 virtual timing 任务生成 source-family 分层 design split。"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch


def main() -> None:
    """只使用已生成且有有效 OpenSTA path 的 design。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dac-root", type=Path, default=Path("/home/yiyh/graphmath/dac2027/processed/verilog_rtl_corpus"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/dac2027_virtual_timing_v1"))
    parser.add_argument("--output", type=Path, default=Path("dataset/virtual_timing_split.json"))
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    original = json.loads((args.dac_root / "splits/seed42_mapped_circuit_v2.json").read_text())
    ids = sum((original[part] for part in ("train", "val", "test")), [])
    families = {}
    for design_id in ids:
        label_path = args.labels / design_id / "labels.pt"
        graph_path = args.dac_root / "graphs_timing_arc_v1" / f"{design_id}.pt"
        if not label_path.is_file() or not graph_path.is_file():
            continue
        label = torch.load(label_path, map_location="cpu", weights_only=False)
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        if int(graph["n"]) > 2048 or int(label["metadata"].get("path_count", 0)) == 0:
            continue
        family = design_id.split("__", 1)[0]
        families.setdefault(family, []).append(design_id)
    rng = random.Random(args.seed)
    result = {"train": [], "val": [], "test": []}
    for family, family_ids in sorted(families.items()):
        rng.shuffle(family_ids)
        n = len(family_ids)
        train_end = int(n * 0.8)
        valid_end = train_end + max(1, int(n * 0.1))
        result["train"].extend(family_ids[:train_end])
        result["val"].extend(family_ids[train_end:valid_end])
        result["test"].extend(family_ids[valid_end:])
    for part in result:
        result[part].sort()
    manifest = {"task": "virtual_timing_v1", "seed": args.seed,
                "protocol": "source_family_stratified_80_10_10",
                "family_counts": dict(Counter(d.split("__", 1)[0] for d in sum(result.values(), []))),
                "split_counts": {part: len(value) for part, value in result.items()},
                "splits": result}
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"split_counts": manifest["split_counts"],
                      "family_counts": manifest["family_counts"]}))


if __name__ == "__main__":
    main()
