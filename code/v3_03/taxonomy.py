"""电路多阶段图的细粒度节点 taxonomy；只使用图中已有语义字段。"""

EXPECTED_NODE_COUNTS = {"mapped": 58, "place": 68, "route": 83, "rc": 2}


def stage_of(graph):
    """根据 schema 和 metadata 判断当前图阶段。"""
    schema = graph.get("schema", "")
    if schema == "circuitnet3_mapped_cellpin_v1":
        return "mapped"
    if schema == "circuitnet3_place_v1":
        return "place"
    if schema == "circuitnet3_route_v1":
        return "route"
    if schema == "spef_rc_graph_v1":
        return "rc"
    return graph.get("metadata", {}).get("stage", "unknown")


def fine_node_type(node, stage):
    """将粗 kind 展开为可数实体类别；缺字段时保留显式 unknown。"""
    kind = node.get("kind", "unknown")
    if kind == "cell":
        return f"cell:{node.get('cell_master', 'unknown')}"
    if kind == "cell_output_pin":
        return f"pin:{node.get('pin', 'unknown')}"
    if kind in ("physical_input_pin", "physical_output_pin"):
        return f"physical_pin:{kind.removeprefix('physical_')}"
    if kind in ("port_input", "port_output"):
        return f"port:{kind.removeprefix('port_')}"
    if kind in ("net", "physical_net"):
        return f"{kind}:{node.get('sig_type', 'unknown')}"
    if kind == "physical_cell":
        return f"physical_cell:{node.get('cell_master', 'unknown')}"
    if kind == "route_segment":
        return f"route_segment:{node.get('layer', 'unknown')}"
    if kind == "route_via":
        return f"route_via:{node.get('via', 'unknown')}"
    if kind in ("route_net", "route_endpoint", "rc_endpoint"):
        return kind
    return f"{kind}:{node.get('sig_type', 'unknown')}"
