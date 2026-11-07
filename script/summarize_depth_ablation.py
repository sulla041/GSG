"""CC-Tensor v3 动力学层数消融实验结果汇总与科学天梯分析工具。"""

import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

BASE_DIR = Path("/data/yiyh/dac2027/runs/ablation_depth_cc_tensor")


def parse_last_step(log_path: Path):
    if not log_path.exists():
        return None
    lines = log_path.read_text().splitlines()
    eval_lines = [l for l in lines if l.startswith("step=") and "R2_pool=" in l]
    if not eval_lines:
        return None
    last = eval_lines[-1]
    match = re.search(r"step=\s*(\d+).*?mae=([\d.]+)ns\s+mape=([\d.]+)%\s+r=([\d.]+)\s+R2_pool=([-\d.]+)\s+\(R2_macro=([-\d.]+)\)\s+std_ratio=([\d.]+)", last)
    if not match:
        return None
    return {
        "step": int(match.group(1)), "mae": float(match.group(2)),
        "mape": float(match.group(3)), "r": float(match.group(4)),
        "r2_pool": float(match.group(5)), "r2_macro": float(match.group(6)),
        "std_ratio": float(match.group(7))
    }


def main():
    exp_dirs = sorted([d for d in BASE_DIR.glob("T*") if d.is_dir()], key=lambda x: int(x.name.removeprefix("T")))
    if not exp_dirs:
        print(f"⚠️ 未在 {BASE_DIR} 找到消融实验目录。")
        return

    print("\n" + "=" * 92)
    print("📊 CC-Tensor v3 动力学深度 T 层数消融实验天梯总表 (统一 600 步 Cosine 退火)")
    print("=" * 92)
    print(f"{'深度 T':^8} | {'相关性 r':^10} | {'R2_pool':^10} | {'R2_macro':^10} | {'MAE (ps)':^10} | {'MAPE (%)':^10} | {'std_ratio':^10} | {'状态'}")
    print("-" * 92)

    for d in exp_dirs:
        t_val = d.name
        res = parse_last_step(d / "train.log")
        if res is None:
            print(f"{t_val:^8} | {'-':^10} | {'-':^10} | {'-':^10} | {'-':^10} | {'-':^10} | {'-':^10} | ⏳ 运行中或未开始")
        else:
            mae_ps = res["mae"] * 1000.0
            print(f"{t_val:^8} | {res['r']:^10.4f} | {res['r2_pool']:^10.4f} | {res['r2_macro']:^10.4f} | {mae_ps:^10.1f} | {res['mape']:^10.2f} | {res['std_ratio']:^10.3f} | ✅ 完成 (step={res['step']})")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    main()
