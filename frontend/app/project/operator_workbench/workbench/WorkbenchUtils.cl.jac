"""Workbench utility functions."""

def:pub decode_param(value: Any) -> str {
    text = str(value or "");
    if not text {
        return "";
    }
    try {
        return decodeURIComponent(text);
    } except Exception {
        return text;
    }
}

def:pub canonical_op_param(value: Any) -> str {
    text = decode_param(value).trim();
    if not text {
        return "";
    }
    lowered = text.toLowerCase();
    marker = "/kernels/";
    idx = lowered.indexOf(marker);
    if idx >= 0 {
        text = text.slice(0, idx);
    }
    marker = "\\kernels\\";
    idx = lowered.indexOf(marker);
    if idx >= 0 {
        text = text.slice(0, idx);
    }
    return text.trim();
}

def:pub get_layouted_elements(nodes_data: list, edges_data: list) -> dict {
    levels = {};
    node_map = {};
    children_map = {};

    for n in nodes_data {
        node_map[n["id"]] = n;
        children_map[n["id"]] = [];
    }

    has_parent = {};
    for e in edges_data {
        pid = e["parent_id"];
        cid = e["child_id"];
        # Tree data can be incomplete while generation/optimization is running.
        # If an edge references a node that doesn't exist yet, skip it to avoid UI crashes.
        if not (pid in children_map) {
            continue;
        }
        if not (cid in node_map) {
            continue;
        }
        children_map[pid].push(cid);
        has_parent[cid] = True;
    }

    roots = [];
    for n in nodes_data {
        if not n["id"] in has_parent {
            roots.push(n["id"]);
        }
    }

    queue = [];
    for r in roots {
        queue.push({"id": r, "level": 0});
    }

    layouted_nodes = [];
    visited = {};
    level_counts = {};
    max_level = 0;

    while queue.length > 0 {
        curr = queue.shift();
        cid = curr["id"];
        lvl = curr["level"];

        if lvl > max_level {
            max_level = lvl;
        }

        if cid in visited {
            continue;
        }
        visited[cid] = True;

        if not lvl in level_counts {
            level_counts[lvl] = 0;
        }

        x_pos = level_counts[lvl] * 200;
        y_pos = lvl * 150;
        level_counts[lvl] += 1;

        raw_node = node_map[cid] if cid in node_map else None;
        if not raw_node {
            continue;
        }
        label_val = Number(raw_node["value"] or 0).toFixed(4);

        layouted_nodes.push(
            {
                "id": f"{cid}",
                "position": {"x": x_pos, "y": y_pos},
                "data": {"label": f"Kernel {cid}\n{label_val} ms"},
                "style": {
                    "background": "#18181b",
                    "color": "#e4e4e7",
                    "border": "1px solid #3f3f46",
                    "borderRadius": "8px",
                    "padding": "10px",
                    "fontSize": "12px",
                    "fontFamily": "monospace",
                    "width": 100,
                    "textAlign": "center"
                },
                "type": "default"
            }
        );

        children = children_map[cid] if cid in children_map else [];
        for child_id in children {
            queue.push({"id": child_id, "level": lvl + 1});
        }
    }

    rf_edges = [];
    for e in edges_data {
        if not (e["parent_id"] in node_map) {
            continue;
        }
        if not (e["child_id"] in node_map) {
            continue;
        }
        rf_edges.push(
            {
                "id": f"e{e['parent_id']}-{e['child_id']}",
                "source": f"{e['parent_id']}",
                "target": f"{e['child_id']}",
                "type": "smoothstep",
                "animated": True,
                "style": {"stroke": "#e17055"}
            }
        );
    }

    return {"nodes": layouted_nodes, "edges": rf_edges};
}

def:pub make_tree_signature(nodes_data: list, edges_data: list) -> str {
    node_parts = [];
    for n in nodes_data {
        if not n {
            continue;
        }
        nid = str(n["id"]) if "id" in n else "";
        nval = Number(n["value"]) if "value" in n else 0;
        nts = str(n["timestamp"]) if "timestamp" in n and n["timestamp"] else "";
        ncode = str(n["code"]) if "code" in n and n["code"] else "";
        node_parts.push(f"{nid}:{nval}:{nts}:{ncode.length}");
    }
    node_parts.sort();

    edge_parts = [];
    for e in edges_data {
        if not e {
            continue;
        }
        pid = str(e["parent_id"]) if "parent_id" in e else "";
        cid = str(e["child_id"]) if "child_id" in e else "";
        edge_parts.push(f"{pid}->{cid}");
    }
    edge_parts.sort();

    return node_parts.join("|") + "||" + edge_parts.join("|");
}
