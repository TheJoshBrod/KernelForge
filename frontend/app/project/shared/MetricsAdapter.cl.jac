"""Shared metrics adapter utilities for dashboard and operator workbench."""

def:pub to_num(value: Any) -> Any {
    if value is None {
        return None;
    }
    try {
        parsed = parseFloat(f"{value}");
        if parsed == parsed {
            return parsed;
        }
    } except Exception {
    # no-op
    }
    return None;
}

def:pub to_list(items: Any) -> list {
    normalized = [];
    if not items {
        return normalized;
    }
    try {
        if Array.isArray(items) {
            return items;
        }
    } except Exception {
    # no-op
    }
    try {
        for item in items {
            normalized.push(item);
        }
        if normalized.length > 0 {
            return normalized;
        }
    } except Exception {
    # no-op
    }
    try {
        vals = Object.values(items);
        if vals and vals.length >= 0 {
            return vals;
        }
    } except Exception {
    # no-op
    }
    return normalized;
}

def:pub list_len(items: Any) -> int {
    return to_list(items).length;
}

def:pub normalize_op_name(value: Any) -> str {
    text = str(value or "").trim().toLowerCase();
    if not text {
        return "";
    }
    prefix = "torch.nn.functional.";
    if text.startsWith(prefix) {
        text = "torch_nn_functional_" + text.replace(prefix, "");
    }
    return text.replace(".", "_").replace("/", "_");
}

def canonical_op_name(value: Any) -> str {
    text = normalize_op_name(value);
    if not text {
        return "";
    }
    prefixes = [
        "torch_nn_functional_",
        "torch_nn_",
        "torch_functional_",
        "torch_",
        "aten_"
    ];
    changed = True;
    while changed and text {
        changed = False;
        for prefix in prefixes {
            if text.startsWith(prefix) and text.length > prefix.length {
                text = text.substring(prefix.length);
                changed = True;
                break;
            }
        }
    }
    return text;
}

def:pub op_matches(candidate: Any, target: Any) -> bool {
    cand = normalize_op_name(candidate);
    wanted = normalize_op_name(target);
    if not cand or not wanted {
        return False;
    }
    if cand == wanted {
        return True;
    }
    if wanted.startsWith(cand + "_") {
        return True;
    }
    if cand.startsWith(wanted + "_") {
        return True;
    }
    cand_canonical = canonical_op_name(candidate);
    wanted_canonical = canonical_op_name(target);
    if cand_canonical and wanted_canonical {
        if cand_canonical == wanted_canonical {
            return True;
        }
        if wanted_canonical.startsWith(cand_canonical + "_") {
            return True;
        }
        if cand_canonical.startsWith(wanted_canonical + "_") {
            return True;
        }
    }
    return False;
}

def pick_row_name(row: Any, fallback_idx: int = 0) -> Any {
    if not row {
        return f"op_{fallback_idx + 1}";
    }
    name = "";
    if "name" in row and row["name"] {
        name = str(row["name"]);
    }
    if not name and "op" in row and row["op"] {
        name = str(row["op"]);
    }
    if not name and "operator" in row and row["operator"] {
        name = str(row["operator"]);
    }
    if not name {
        name = f"op_{fallback_idx + 1}";
    }
    return name;
}

def:pub pick_pytorch_ms(row: Any) -> Any {
    if not row {
        return None;
    }
    pytorch_raw = None;
    if "pytorch" in row {
        pytorch_raw = row["pytorch"];
    }
    if pytorch_raw is None and "baseline" in row {
        pytorch_raw = row["baseline"];
    }
    if pytorch_raw is None and "pytorch_ms" in row {
        pytorch_raw = row["pytorch_ms"];
    }
    return to_num(pytorch_raw);
}

def:pub pick_forged_ms(row: Any) -> Any {
    if not row {
        return None;
    }
    forged_raw = None;
    if "forged" in row {
        forged_raw = row["forged"];
    }
    if forged_raw is None and "optimized" in row {
        forged_raw = row["optimized"];
    }
    if forged_raw is None and "kernel_ms" in row {
        forged_raw = row["kernel_ms"];
    }
    if forged_raw is None and "forged_ms" in row {
        forged_raw = row["forged_ms"];
    }
    return to_num(forged_raw);
}

def:pub normalize_speed_rows(rows: Any) -> list {
    out = [];
    idx = 0;
    for row in to_list(rows) {
        if not row {
            idx += 1;
            continue;
        }

        row_name = pick_row_name(row, idx);
        pytorch_val = pick_pytorch_ms(row);
        forged_val = pick_forged_ms(row);

        if pytorch_val is None and forged_val is None {
            idx += 1;
            continue;
        }

        baseline_val = forged_val;
        if pytorch_val != None {
            baseline_val = pytorch_val;
        }
        if baseline_val is None or baseline_val <= 0 {
            idx += 1;
            continue;
        }

        forged_final = baseline_val;
        if forged_val != None and forged_val > 0 {
            forged_final = forged_val;
        }

        out.push({"name": row_name, "pytorch": baseline_val, "forged": forged_final});
        idx += 1;
    }
    return out;
}

def:pub get_metrics_speed_source(metrics_data: Any) -> list {
    if not metrics_data {
        return [];
    }
    if "speed_comparison" in metrics_data and metrics_data["speed_comparison"] {
        return to_list(metrics_data["speed_comparison"]);
    }
    if "rows" in metrics_data and metrics_data["rows"] {
        return to_list(metrics_data["rows"]);
    }
    if "rows_by_op" in metrics_data and metrics_data["rows_by_op"] {
        return to_list(metrics_data["rows_by_op"]);
    }
    return [];
}

def:pub get_metrics_speed_rows(metrics_data: Any) -> list {
    return normalize_speed_rows(get_metrics_speed_source(metrics_data));
}

def:pub get_metric_row_for_op(metrics_data: Any, op_name: Any) -> Any {
    target = normalize_op_name(op_name);
    if not target or not metrics_data {
        return None;
    }

    rows_by_op = {};
    if "rows_by_op" in metrics_data and metrics_data["rows_by_op"] {
        rows_by_op = metrics_data["rows_by_op"];
    }
    if rows_by_op {
        if target in rows_by_op and rows_by_op[target] {
            return rows_by_op[target];
        }
        for row_key in Object.keys(rows_by_op) {
            if op_matches(row_key, target) {
                return rows_by_op[row_key];
            }
        }
    }

    rows_list = [];
    if "rows" in metrics_data and metrics_data["rows"] {
        rows_list = to_list(metrics_data["rows"]);
    } elif "speed_comparison" in metrics_data and metrics_data["speed_comparison"] {
        rows_list = to_list(metrics_data["speed_comparison"]);
    }
    for row in rows_list {
        if row and op_matches(pick_row_name(row), target) {
            return row;
        }
    }

    return None;
}

def sort_rows_desc_by(rows: Any, key_name: str) -> list {
    sorted_rows = [];
    remaining = [];
    for row in to_list(rows) {
        remaining.push(row);
    }

    while list_len(remaining) > 0 {
        best_idx = 0;
        best_row = remaining[0];
        best_val = to_num(best_row[key_name] if best_row and key_name in best_row else 0);
        if best_val is None {
            best_val = 0;
        }

        scan_idx = 1;
        while scan_idx < list_len(remaining) {
            candidate = remaining[scan_idx];
            candidate_val = to_num(candidate[key_name] if candidate and key_name in candidate else 0);
            if candidate_val is None {
                candidate_val = 0;
            }
            if candidate_val > best_val {
                best_val = candidate_val;
                best_idx = scan_idx;
            }
            scan_idx += 1;
        }

        sorted_rows.push(remaining[best_idx]);
        next_remaining = [];
        next_idx = 0;
        while next_idx < list_len(remaining) {
            if next_idx != best_idx {
                next_remaining.push(remaining[next_idx]);
            }
            next_idx += 1;
        }
        remaining = next_remaining;
    }

    return sorted_rows;
}

def:pub compute_speed_kpis(speed_rows: Any, op_rows: Any) -> dict {
    normalized_rows = normalize_speed_rows(speed_rows);

    op_count_by_name = {};
    for op in to_list(op_rows) {
        if not op or not ("name" in op) or not op["name"] {
            continue;
        }
        op_name = str(op["name"]);
        raw_count = op["count"] if "count" in op else 0;
        count_val = 0;
        if raw_count != None {
            count_val = parseInt(f"{raw_count}");
        }
        if count_val != count_val or count_val <= 0 {
            count_val = 1;
        }
        op_count_by_name[op_name] = count_val;
    }

    best_speedup = 1;
    has_best_speedup = False;
    total_saved_ms = 0;

    for row in normalized_rows {
        if not row {
            continue;
        }
        row_name = pick_row_name(row);
        pytorch_ms = pick_pytorch_ms(row);
        forged_ms = pick_forged_ms(row);

        pytorch_num = to_num(pytorch_ms);
        forged_num = to_num(forged_ms);

        if pytorch_num != None and pytorch_num > 0 and forged_num != None and forged_num > 0 {
            speedup = parseFloat(f"{pytorch_num}") / parseFloat(f"{forged_num}");
            if speedup > best_speedup {
                best_speedup = speedup;
                has_best_speedup = True;
            }
        }

        if pytorch_num != None and forged_num != None {
            count_val = 1;
            if row_name in op_count_by_name {
                count_val = op_count_by_name[row_name];
            }
            delta = (parseFloat(f"{pytorch_num}") - parseFloat(f"{forged_num}")) * count_val;
            if delta > 0 {
                total_saved_ms += delta;
            }
        }
    }

    return {
        "best_speedup": best_speedup,
        "has_best_speedup": has_best_speedup,
        "total_saved_ms": total_saved_ms
    };
}

def:pub build_dashboard_metrics_view(
    speed_rows: Any,
    op_rows: Any,
    op_time_mode: str,
    comparison_view: str,
    speed_limit: int = 8,
    time_limit: int = 12
) -> dict {
    normalized_speed_rows = normalize_speed_rows(speed_rows);

    op_count_by_name = {};
    for op in to_list(op_rows) {
        if not op or not ("name" in op) or not op["name"] {
            continue;
        }
        op_name = str(op["name"]);
        raw_count = op["count"] if "count" in op else 0;
        count_val = 0;
        if raw_count != None {
            count_val = parseInt(f"{raw_count}");
        }
        if count_val != count_val or count_val <= 0 {
            count_val = 1;
        }
        op_count_by_name[op_name] = count_val;
    }

    time_rows = [];
    seen_time_names = {};
    for row in normalized_speed_rows {
        if not row {
            continue;
        }
        row_name = pick_row_name(row);
        pytorch_ms = pick_pytorch_ms(row);
        forged_ms = pick_forged_ms(row);

        selected_ms = None;
        if op_time_mode == "pytorch" {
            selected_ms = pytorch_ms;
        } elif op_time_mode == "forged" {
            selected_ms = forged_ms;
        } else {
            if pytorch_ms != None and forged_ms != None {
                selected_ms = pytorch_ms if pytorch_ms <= forged_ms else forged_ms;
            } elif forged_ms != None {
                selected_ms = forged_ms;
            } else {
                selected_ms = pytorch_ms;
            }
        }

        selected_num = to_num(selected_ms);
        if selected_num is None or selected_num <= 0 {
            continue;
        }
        count_val = 1;
        if row_name in op_count_by_name {
            count_val = op_count_by_name[row_name];
        }
        total_ms = selected_num * count_val;
        if total_ms <= 0 {
            continue;
        }
        time_rows.push({"name": row_name, "count": count_val, "total_time_ms": total_ms});
        seen_time_names[row_name] = True;
    }

    if op_time_mode == "best" {
        for op in to_list(op_rows) {
            if not op or not ("name" in op) or not op["name"] {
                continue;
            }
            row_name = str(op["name"]);
            if row_name in seen_time_names {
                continue;
            }
            fallback_total = to_num(op["total_time_ms"] if "total_time_ms" in op else None);
            if fallback_total is None or fallback_total <= 0 {
                continue;
            }
            count_val = 1;
            if row_name in op_count_by_name {
                count_val = op_count_by_name[row_name];
            }
            time_rows.push({"name": row_name, "count": count_val, "total_time_ms": fallback_total});
            seen_time_names[row_name] = True;
        }
    }

    sorted_time_rows = sort_rows_desc_by(time_rows, "total_time_ms");
    time_list_max = 0;
    if list_len(sorted_time_rows) > 0 {
        max_row = sorted_time_rows[0];
        time_list_max = to_num(max_row["total_time_ms"] if max_row and "total_time_ms" in max_row else 0);
        if time_list_max is None or time_list_max < 0 {
            time_list_max = 0;
        }
    }

    time_list_rows = [];
    time_idx = 0;
    while time_idx < list_len(sorted_time_rows) and time_idx < time_limit {
        row = sorted_time_rows[time_idx];
        row_total = to_num(row["total_time_ms"] if row and "total_time_ms" in row else 0);
        if row_total is None {
            row_total = 0;
        }

        bar_width = 0;
        max_for_width = to_num(time_list_max);
        if max_for_width != None and max_for_width > 0 {
            row_total_num = parseFloat(f"{row_total}");
            max_width_num = parseFloat(f"{max_for_width}");
            bar_width = (row_total_num / max_width_num) * 100;
        }
        if row_total > 0 and bar_width < 2 {
            bar_width = 2;
        }
        if bar_width > 100 {
            bar_width = 100;
        }

        time_list_rows.push(
            {
                "rank": time_idx + 1,
                "name": row["name"] if row and "name" in row else "",
                "value": row_total,
                "count": row["count"] if row and "count" in row else 1,
                "bar_width": f"{bar_width}%"
            }
        );
        time_idx += 1;
    }

    comparison_rows = [];
    for row in normalized_speed_rows {
        if not row {
            continue;
        }
        row_name = pick_row_name(row);
        pytorch_ms = pick_pytorch_ms(row);
        forged_ms = pick_forged_ms(row);

        sort_ms = forged_ms;
        if pytorch_ms != None {
            sort_ms = pytorch_ms;
        }
        if sort_ms is None or sort_ms <= 0 {
            continue;
        }

        ratio_val = None;
        pytorch_num = to_num(pytorch_ms);
        forged_num = to_num(forged_ms);
        if pytorch_num != None and pytorch_num > 0 and forged_num != None {
            forged_float = parseFloat(f"{forged_num}");
            pytorch_float = parseFloat(f"{pytorch_num}");
            ratio_val = forged_float / pytorch_float;
        }

        next_row = {
            "name": row_name,
            "sort_ms": sort_ms,
            "pytorch": 0,
            "forged": 0,
            "ratio": ratio_val
        };
        if pytorch_ms != None {
            next_row["pytorch"] = pytorch_ms;
        }
        if forged_ms != None {
            next_row["forged"] = forged_ms;
        }
        comparison_rows.push(next_row);
    }

    sorted_comparison_rows = sort_rows_desc_by(comparison_rows, "sort_ms");
    top_comparison_rows = [];
    row_idx = 0;
    while row_idx < list_len(sorted_comparison_rows) and row_idx < speed_limit {
        top_comparison_rows.push(sorted_comparison_rows[row_idx]);
        row_idx += 1;
    }

    ratio_chart_rows = [];
    absolute_chart_rows = [];
    for row in top_comparison_rows {
        row_name = str(row["name"]) if row and "name" in row else "";
        ratio_val = to_num(row["ratio"] if row and "ratio" in row else None);
        if ratio_val != None {
            ratio_chart_rows.push({"name": row_name, "ratio": ratio_val});
        }
        absolute_chart_rows.push(
            {
                "name": row_name,
                "forged": row["forged"] if row and "forged" in row else 0,
                "pytorch": row["pytorch"] if row and "pytorch" in row else 0
            }
        );
    }

    effective_view = comparison_view;
    if comparison_view == "ratio" and list_len(ratio_chart_rows) == 0 and list_len(absolute_chart_rows) > 0 {
        effective_view = "absolute";
    }

    speed_chart_data = absolute_chart_rows;
    if effective_view == "ratio" {
        speed_chart_data = ratio_chart_rows;
    }

    return {
        "time_list_rows": time_list_rows,
        "effective_comparison_view": effective_view,
        "speed_chart_data": speed_chart_data,
        "has_speed_chart": list_len(speed_chart_data) > 0
    };
}
