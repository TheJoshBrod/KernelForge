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
    NODE_W = 100;
    H_GAP = 40;
    V_GAP = 170;

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

    # BFS to get traversal order (handles DAG diamonds via visited set)
    bfs_order = [];
    visited = {};
    queue = [];
    for r in roots {
        queue.push(r);
    }
    while queue.length > 0 {
        cid = queue.shift();
        if cid in visited {
            continue;
        }
        visited[cid] = True;
        bfs_order.push(cid);
        for ch in children_map[cid] {
            queue.push(ch);
        }
    }

    # Bottom-up pass: compute subtree pixel width for each node
    subtree_w = {};
    i = bfs_order.length - 1;
    while i >= 0 {
        cid = bfs_order[i];
        children = children_map[cid];
        if children.length == 0 {
            subtree_w[cid] = NODE_W;
        } else {
            total = 0;
            for ch in children {
                total += subtree_w[ch] if ch in subtree_w else NODE_W;
            }
            total += H_GAP * (children.length - 1);
            subtree_w[cid] = total;
        }
        i -= 1;
    }

    # Top-down pass: assign center_x and depth level
    center_x = {};
    node_level = {};

    root_start = 0;
    for r in roots {
        rw = subtree_w[r] if r in subtree_w else NODE_W;
        center_x[r] = root_start + rw / 2;
        node_level[r] = 0;
        root_start += rw + H_GAP;
    }

    for cid in bfs_order {
        children = children_map[cid];
        if children.length == 0 {
            continue;
        }
        cx = center_x[cid];
        total_w = 0;
        for ch in children {
            total_w += subtree_w[ch] if ch in subtree_w else NODE_W;
        }
        total_w += H_GAP * (children.length - 1);
        child_start = cx - total_w / 2;
        for ch in children {
            cw = subtree_w[ch] if ch in subtree_w else NODE_W;
            center_x[ch] = child_start + cw / 2;
            node_level[ch] = node_level[cid] + 1;
            child_start += cw + H_GAP;
        }
    }

    layouted_nodes = [];
    for cid in bfs_order {
        raw_node = node_map[cid] if cid in node_map else None;
        if not raw_node {
            continue;
        }
        label_val = Number(raw_node["value"] or 0).toFixed(4);
        x_pos = (center_x[cid] if cid in center_x else 0) - NODE_W / 2;
        y_pos = (node_level[cid] if cid in node_level else 0) * V_GAP;

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
                    "width": NODE_W,
                    "textAlign": "center"
                },
                "type": "default"
            }
        );
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
                "animated": False,
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
