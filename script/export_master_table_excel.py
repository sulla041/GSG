"""根据意涵的高瞻远瞩与最新指示，构建 DAC 2027 顶会级多级表头主实验 Excel 表格。
排序严格按照意涵指定的物理因果层级与任务难易度：
1. 局部 Cell Area (A1) (MAPE 精确修正为 0.12%)
2. DeepGate 信号概率预测 (Signal Prob P(1)) (多跳布尔逻辑)
3. DeepGate 跳变率预测 (Toggle Prob) (多跳布尔逻辑)
4. 网表端到端 Arrival Time 预测 (D1 预布局物理时序)
5. RC 网表 Arrival Time 预测 (D2 后布线寄生 RC 时序)
6. Cell 门弧延迟预测 (D3 单元内门弧微星团)
7. 预布局线网 HPWL 预测 (Net HPWL 空间几何)
8. 图级别芯片总面积预测 (P2 全图宏观分层电表金字塔)
9. 图级别静态漏电功耗预测 (P2 全图宏观分层电表金字塔)
"""

from pathlib import Path
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


def build_master_benchmark_excel(output_path: Path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Grand Master Benchmark"

    # 1. 顶会级配色方案定义 (Navy & Professional Slate Blue)
    c_navy = "1F4E79"       # 顶部总标题深海军蓝
    c_meta = "2B4C7E"       # 任务前导列标题深蓝
    c_base_hdr = "595959"   # Baseline 模型表头稳重深灰
    c_ours_hdr = "1B365D"   # Ours 模型表头至尊深海蓝
    c_sub_base = "F2F2F2"   # Baseline 副指标浅灰底色
    c_sub_ours = "D9E1F2"   # Ours 副指标清爽浅蓝底色
    c_zebra = "F9FAFC"      # 斑马纹交替淡灰
    c_ours_data = "EFF3FA"  # Ours 数据列淡蓝强调底色

    font_title = Font(name="Calibri", size=13, bold=True, color="FFFFFF")
    font_main_hdr = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    font_sub_base = Font(name="Calibri", size=9, bold=True, color="333333")
    font_sub_ours = Font(name="Calibri", size=9, bold=True, color="1F4E79")
    font_data = Font(name="Calibri", size=10, color="000000")
    font_data_bold = Font(name="Calibri", size=10, bold=True, color="1F4E79")

    border_thin = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9")
    )
    border_thick_left = Border(left=Side(style="medium", color="1F4E79"))

    # 2. 模型主列配置
    info_cols = 4  # Task ID, Category, Task Name, Target & Unit
    baselines = ["MLP", "GCN", "GAT", "Graph Transformer", "GPS-R", "PreRoutGNN / SOTA"]
    ours_name = "Our Framework (Ours)"

    total_cols = info_cols + len(baselines) * 4 + 4

    # 3. 第 1 行：总标题
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=total_cols)
    t_cell = ws.cell(row=1, column=1)
    t_cell.value = "DAC 2027 Grand Master Benchmark: Nine Core Physical & Logical Circuit Tasks"
    t_cell.font = font_title
    t_cell.fill = PatternFill("solid", fgColor=c_navy)
    t_cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 34

    # 4. 第 2 行：主列标题
    ws.merge_cells("A2:D2")
    c_info = ws["A2"]
    c_info.value = "Circuit Metric Prediction Tasks (Ordered by Hierarchy & Difficulty)"
    c_info.font = font_main_hdr
    c_info.fill = PatternFill("solid", fgColor=c_meta)
    c_info.alignment = Alignment(horizontal="center", vertical="center")

    curr_col = 5
    for base in baselines:
        ws.merge_cells(start_row=2, start_column=curr_col, end_row=2, end_column=curr_col + 3)
        cell = ws.cell(row=2, column=curr_col)
        cell.value = base
        cell.font = font_main_hdr
        cell.fill = PatternFill("solid", fgColor=c_base_hdr)
        cell.alignment = Alignment(horizontal="center", vertical="center")
        curr_col += 4

    ws.merge_cells(start_row=2, start_column=curr_col, end_row=2, end_column=curr_col + 3)
    c_ours = ws.cell(row=2, column=curr_col)
    c_ours.value = ours_name
    c_ours.font = font_main_hdr
    c_ours.fill = PatternFill("solid", fgColor=c_ours_hdr)
    c_ours.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[2].height = 26

    # 5. 第 3 行：副列标题 (r, R², MAE, MAPE)
    sub_headers = ["Task ID", "Task Category", "Benchmark Task Name", "Target Metric & Unit"]
    for i, h in enumerate(sub_headers, 1):
        cell = ws.cell(row=3, column=i, value=h)
        cell.fill = PatternFill("solid", fgColor="EBF1F5")
        cell.font = Font(name="Calibri", size=9, bold=True, color="1F4E79")
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = border_thin

    metrics = ["r ↑", "R² ↑", "MAE ↓", "MAPE ↓"]
    curr_col = 5
    for _ in baselines:
        for m in metrics:
            cell = ws.cell(row=3, column=curr_col, value=m)
            cell.font = font_sub_base
            cell.fill = PatternFill("solid", fgColor=c_sub_base)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = border_thin
            curr_col += 1

    for m in metrics:
        cell = ws.cell(row=3, column=curr_col, value=m)
        cell.font = font_sub_ours
        cell.fill = PatternFill("solid", fgColor=c_sub_ours)
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = border_thin
        curr_col += 1
    ws.row_dimensions[3].height = 22

    # 6. 数据行定义（严格按意涵指示的 9 大任务排序与难易因果梯队）
    tasks = [
        ("Task 1", "Local Static (0-Hop)", "Local Cell Area (A1)", "Gate Physical Area (μm²)",
         0.9994, 0.9983, 0.0108, 0.0012),
        ("Task 2", "Boolean Logic (Multi-Hop)", "Boolean Signal Prob (DeepGate)", "Logical 1 Prob P(1) ∈ [0, 1]",
         0.9966, 0.9930, 0.0103, 0.0214),
        ("Task 3", "Boolean Logic (Multi-Hop)", "Boolean Toggle Prob (DeepGate)", "Toggle Prob P(toggle) ∈ [0, 1]",
         0.9958, 0.9915, 0.0112, 0.0235),
        ("Task 4", "Causal Timing (Deep Multi-Hop)", "Pre-layout Timing (D1)", "Pin Arrival Time (ps)",
         0.9541, 0.8810, 34.90, 0.1048),
        ("Task 5", "Causal Timing (Parasitic RC)", "Post-route RC Timing (D2)", "Arrival Time with Wire RC (ps)",
         0.9514, 0.8681, 27.30, 0.1043),
        ("Task 6", "Micro Transient Physics", "Cell Arc Delay (D3)", "Std-Cell Arc Delay (ps)",
         0.9630, 0.9207, 4.11, 0.1200),
        ("Task 7", "Spatial Layout Geometry", "Pre-layout Wirelength (HPWL)", "Net Half-Perimeter Wirelength (μm)",
         0.8959, 0.7204, 4.17, 0.4712),
        ("Task 8", "Macro PPA (Hierarchical Meter)", "Global Chip Area (P2)", "Total Cell Physical Area (μm²)",
         0.9754, 0.9497, 95.94, 0.0775),
        ("Task 9", "Macro PPA (Hierarchical Meter)", "Global Leakage Power (P2)", "Total Static Leakage Power (μW)",
         0.9282, 0.8400, 0.42, 0.0958),
    ]

    ours_start_col = 5 + len(baselines) * 4

    for row_idx, task_info in enumerate(tasks, start=4):
        ws.row_dimensions[row_idx].height = 22
        is_even = (row_idx % 2 == 0)
        row_fill = PatternFill("solid", fgColor=c_zebra) if is_even else None

        # 填入任务基本信息 (Task ID, Category, Name, Target)
        for c in range(1, 5):
            cell = ws.cell(row=row_idx, column=c, value=task_info[c - 1])
            cell.font = font_data
            cell.border = border_thin
            if row_fill: cell.fill = row_fill
            cell.alignment = Alignment(horizontal="center" if c in (1, 2) else "left", vertical="center")

        # 填入 Baselines 空白单元格（留空规范供对比）
        for col in range(5, ours_start_col):
            cell = ws.cell(row=row_idx, column=col, value="")
            cell.border = border_thin
            if row_fill: cell.fill = row_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")

        # 填入 Ours 实测数值
        r_val, r2_val, mae_val, mape_val = task_info[4:]
        ours_vals = [r_val, r2_val, mae_val, mape_val]

        for offset, val in enumerate(ours_vals):
            col = ours_start_col + offset
            cell = ws.cell(row=row_idx, column=col, value=val)
            cell.border = border_thin
            cell.font = font_data_bold
            cell.fill = PatternFill("solid", fgColor=c_ours_data)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            if offset in (0, 1):
                cell.number_format = "0.0000"
            elif offset == 2:
                cell.number_format = "0.00" if val >= 1 else "0.0000"
            elif offset == 3:
                cell.number_format = "0.00%"

    # 7. 列宽自适应设置
    ws.column_dimensions["A"].width = 11
    ws.column_dimensions["B"].width = 24
    ws.column_dimensions["C"].width = 30
    ws.column_dimensions["D"].width = 32
    for col in range(5, total_cols + 1):
        col_letter = get_column_letter(col)
        ws.column_dimensions[col_letter].width = 10

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    print(f"Master benchmark table successfully created at: {output_path}")


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "writing" / "Table1_Grand_Master_Benchmark.xlsx"
    build_master_benchmark_excel(target)
