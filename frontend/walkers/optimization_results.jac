"""Walkers for reading and presenting optimization output to the frontend."""

import from json { load, loads, dump, dumps }
import from os { path, listdir }
import from base64 { b64encode }
import from datetime { datetime }
import from zipfile { ZipFile }
import os;
import sys;
import subprocess;
import sqlite3;
import hashlib;

import from .globals { repo_root }
import from .catalog_db {
    _ensure_dir,
    _projects_base_dir,
    _upsert_project_catalog,
    _upsert_project_metric_rows,
    _read_project_metric_rows
}
import from .job_supervisor {
    _project_dir,
    _read_state,
    _apply_llm_env_from_config,
    _refresh_running_job_state,
    _normalize_job_flags,
    _read_log_excerpt,
    _extract_job_error_summary,
    _has_optimize_outputs_after,
    _has_benchmark_outputs_after,
    _modified_after,
    _start_job,
    _find_optimized_dir,
    _normalize_metric_op_name,
    _refresh_project_metrics,
    _collect_project_metrics
}

def _get_node_metric_flags(conn) -> dict {
    flags = {"has_median": False, "has_mean": False};
    try {
        cursor = conn.cursor();
        cursor.execute("PRAGMA table_info(nodes)");
        rows = cursor.fetchall();
        for row in rows {
            if row and len(row) > 1 {
                if row[1] == "median_time_ms" {
                    flags["has_median"] = True;
                } elif row[1] == "mean_time_ms" {
                    flags["has_mean"] = True;
                }
            }
        }
    } except Exception {
        _ = 0;
    }
    return flags;
}

def _tree_node_select_sql(flags: dict) -> str {
    metric_col = "NULL as median_time_ms";
    if flags["has_median"] {
        metric_col = "median_time_ms";
    } elif flags["has_mean"] {
        metric_col = "mean_time_ms as median_time_ms";
    }

    return (
        "SELECT id, visits, value, " + metric_col + ", best_subtree_value, " + "improvement_description, timestamp, code FROM nodes"
    );
}

walker:pub GetDashboardCharts {
    has projectName: str;

    can return_charts with entry {
        payload = _collect_project_metrics(self.projectName);
        report {
            "operator_usage": (
                payload["operator_usage"]
                    if payload and "operator_usage" in payload
                    else []
            ),
            "speed_comparison": (
                payload["speed_comparison"]
                    if payload and "speed_comparison" in payload
                    else []
            ),
            "status": payload["status"] if payload and "status" in payload else "empty",
            "profile_status": (
                payload["profile_status"]
                    if payload and "profile_status" in payload
                    else ""
            ),
            "profile_message": (
                payload["profile_message"]
                    if payload and "profile_message" in payload
                    else ""
            ),
            "errors": payload["errors"] if payload and "errors" in payload else [],
            "metrics_version": (
                payload["metrics_version"]
                    if payload and "metrics_version" in payload
                    else 0
            )
        };
    }
}

walker:pub GetMctsTree {
    has projectName: str,
        opName: str;

    can run with entry {
        project_dir = _project_dir(self.projectName);
        db_path = path.join(project_dir, "trees", self.opName, "nodes.db");

        if not path.exists(db_path) {
            report {
                "success": True,
                "nodes": [],
                "edges": [],
                "message": "No optimization tree yet for this operator."
            };
            disengage;
        }

        mcts_nodes = [];
        edges = [];
        valid_ids = {};

        try {
            # Since Jac can interop with Python, we can use sqlite3 directly
            conn = sqlite3.connect(db_path);
            # Set row factory to dict-like if needed, but manual extraction is safer in Jac logic
            cursor = conn.cursor();
            metric_flags = _get_node_metric_flags(conn);

            # Fetch nodes
            cursor.execute(_tree_node_select_sql(metric_flags));
            rows = cursor.fetchall();
            best_node_id = None;
            best_value_ms = None;
            has_legacy_timing = False;
            for row in rows {
                stored_value = None;
                try {
                    stored_value = float(row[2]) if row[2] is not None else None;
                } except Exception {
                    stored_value = None;
                }
                median_time_ms = None;
                try {
                    median_time_ms = float(row[3]) if row[3] is not None else None;
                } except Exception {
                    median_time_ms = None;
                }
                best_subtree_value = None;
                try {
                    best_subtree_value = float(row[4]) if row[4] is not None else None;
                } except Exception {
                    best_subtree_value = None;
                }
                node_val = median_time_ms;
                timing_metric = "median_time_ms"
                    if metric_flags["has_median"]
                    else "mean_time_ms";
                if node_val is None or node_val <= 0.0 {
                    node_val = stored_value;
                    timing_metric = "legacy_value";
                }
                if node_val is None or node_val <= 0.0 {
                    continue;
                }
                if best_subtree_value is None and node_val is not None {
                    best_subtree_value = node_val;
                }
                if timing_metric == "legacy_value" {
                    has_legacy_timing = True;
                } elif best_value_ms is None or node_val < best_value_ms {
                    best_value_ms = node_val;
                    best_node_id = row[0];
                }
                node_code = row[7];
                code_text = str(node_code).strip() if node_code is not None else "";
                if code_text
                and "\n" not in code_text
                and (
                    code_text.endswith(".cu")
                    or code_text.endswith(".py")
                    or code_text.endswith(".metal")
                    or "/kernels/" in code_text
                ) {
                    resolved_inline_code = False;
                    path_candidates = [];
                    if path.isabs(code_text) {
                        path_candidates.append(code_text);
                    } else {
                        path_candidates.append(
                            path.join(project_dir, "trees", code_text)
                        );
                        path_candidates.append(path.join(project_dir, code_text));
                        path_candidates.append(path.join(repo_root, code_text));
                        path_candidates.append(
                            path.join(
                                repo_root,
                                "kernels",
                                "projects",
                                self.projectName,
                                code_text
                            )
                        );
                        path_candidates.append(
                            path.join(
                                repo_root,
                                "kernels",
                                "projects",
                                self.projectName,
                                "trees",
                                code_text
                            )
                        );
                        path_candidates.append(
                            path.join(
                                project_dir,
                                "kernels",
                                "generated",
                                "individual_op_kernels",
                                code_text
                            )
                        );
                        path_candidates.append(
                            path.join(
                                project_dir,
                                "kernels",
                                "generated",
                                "individual_op_kernels",
                                self.opName,
                                code_text
                            )
                        );
                        path_candidates.append(
                            path.join(
                                project_dir,
                                "kernels",
                                "generated",
                                "individual_op_kernels",
                                self.opName,
                                "kernel",
                                code_text
                            )
                        );
                        path_candidates.append(
                            path.join(
                                project_dir,
                                "kernels",
                                "generated",
                                "individual_op_kernels",
                                self.opName,
                                "kernels",
                                code_text
                            )
                        );
                    }
                    for p in path_candidates {
                        if not path.exists(p) {
                            continue;
                        }
                        try {
                            with open(p, "r", encoding="utf-8", errors="ignore") as f {
                                node_code = f.read();
                            }
                            resolved_inline_code = True;
                            break;
                        } except Exception {
                            continue;
                        }
                    }
                    if not resolved_inline_code {
                        node_code = None;
                    }
                }
                if node_code is None or str(node_code).strip() == "" {
                    kernel_dirs = [
                        path.join(project_dir, "trees", self.opName, "kernels"),
                        path.join(
                            project_dir,
                            "kernels",
                            "generated",
                            "individual_op_kernels",
                            self.opName,
                            "kernel"
                        ),
                        path.join(
                            project_dir,
                            "kernels",
                            "generated",
                            "individual_op_kernels",
                            self.opName,
                            "kernels"
                        ),
                        path.join(
                            project_dir,
                            "kernels",
                            "generated",
                            "individual_op_kernels",
                            self.opName
                        )
                    ];
                    candidates = [
                        f"kernel_{row[0]}.cu",
                        f"kernel_{row[0]}.py",
                        f"kernel_{row[0]}.metal",
                        f"kernel-{row[0]}-g.cu",
                        "kernel_0.cu",
                        "kernel_0.py",
                        "kernel_0.metal",
                        "kernel-0-g.cu",
                        "kernel.cu",
                        "cuda.cu"
                    ];
                    for kernels_dir in kernel_dirs {
                        if not path.exists(kernels_dir) {
                            continue;
                        }
                        for fname in candidates {
                            candidate_path = path.join(kernels_dir, fname);
                            if not path.exists(candidate_path) {
                                continue;
                            }
                            try {
                                with open(
                                    candidate_path,
                                    "r",
                                    encoding="utf-8",
                                    errors="ignore"
                                ) as f {
                                    node_code = f.read();
                                }
                                break;
                            } except Exception {
                                continue;
                            }
                        }
                        if node_code is not None and str(node_code).strip() != "" {
                            break;
                        }
                    }
                }
                valid_ids[row[0]] = True;
                mcts_nodes.append(
                    {
                        "id": row[0],
                        "visits": row[1],
                        "value": node_val,
                        "median_time_ms": median_time_ms,
                        "stored_value": stored_value,
                        "best_subtree_value": best_subtree_value,
                        "improvement_description": row[5],
                        "timestamp": row[6],
                        "timing_metric": timing_metric,
                        "code": node_code
                    }
                );
            }

            # Fetch edges
            cursor.execute("SELECT parent_id, child_id FROM edges");
            rows = cursor.fetchall();
            for row in rows {
                if not (row[0] in valid_ids) or not (row[1] in valid_ids) {
                    continue;
                }
                edges.append({"parent_id": row[0], "child_id": row[1]});
            }

            pytorch_ms = None;
            try {
                metric_payload = _collect_project_metrics(self.projectName);
                op_key = _normalize_metric_op_name(self.opName);
                rows_by_op = (
                    metric_payload["rows_by_op"]
                        if metric_payload
                        and "rows_by_op" in metric_payload
                        and metric_payload["rows_by_op"]
                        else {}
                );
                metric_row = (
                    rows_by_op[op_key] if rows_by_op and op_key in rows_by_op else None
                );
                if not metric_row {
                    metric_rows = (
                        metric_payload["rows"]
                            if metric_payload
                            and "rows" in metric_payload
                            and metric_payload["rows"]
                            else []
                    );
                    for mrow in metric_rows {
                        if not mrow {
                            continue;
                        }
                        row_name = (
                            mrow["name"]
                                if "name" in mrow and mrow["name"]
                                else (
                                    mrow["op"] if "op" in mrow and mrow["op"] else ""
                                )
                        );
                        if _normalize_metric_op_name(row_name) == op_key {
                            metric_row = mrow;
                            break;
                        }
                    }
                }
                if metric_row {
                    raw_pytorch = (
                        metric_row["pytorch_ms"]
                            if "pytorch_ms" in metric_row
                            and metric_row["pytorch_ms"] is not None
                            else (
                                metric_row["pytorch"]
                                    if "pytorch" in metric_row
                                    and metric_row["pytorch"] is not None
                                    else None
                            )
                    );
                    if raw_pytorch is not None {
                        pytorch_ms = float(raw_pytorch);
                    }
                }
            } except Exception {
                _ = 0;
            }

            conn.close();
            report {
                "success": True,
                "nodes": mcts_nodes,
                "edges": edges,
                "pytorch_ms": pytorch_ms,
                "best_node_id": best_node_id,
                "best_value_ms": best_value_ms,
                "has_legacy_timing": has_legacy_timing
            };
        } except Exception as e {
            report {"success": False, "error": str(e)};
        }
    }
}

walker:pub GetProjectMctsSummary {
    has projectName: str;

    can run with entry {
        project_dir = _project_dir(self.projectName);
        trees_dir = path.join(project_dir, "trees");

        total_nodes = 0;
        op_summaries = [];

        if path.exists(trees_dir) {
            for op_name in listdir(trees_dir) {
                db_path = path.join(trees_dir, op_name, "nodes.db");
                if path.exists(db_path) {
                    try {
                        conn = sqlite3.connect(db_path);
                        cursor = conn.cursor();
                        metric_flags = _get_node_metric_flags(conn);

                        cursor.execute(
                            "SELECT COUNT(*) FROM nodes WHERE value IS NOT NULL AND value > 0"
                        );
                        node_count = cursor.fetchone()[0];

                        best_val = None;
                        best_node_id = None;
                        timing_metric = "legacy_value";

                        if metric_flags["has_median"] {
                            cursor.execute(
                                """
                                SELECT id, median_time_ms
                                FROM nodes
                                WHERE median_time_ms IS NOT NULL AND median_time_ms > 0
                                ORDER BY median_time_ms ASC
                                LIMIT 1
                                """
                            );
                            row = cursor.fetchone();
                            if row {
                                best_node_id = int(row[0]);
                                best_val = float(row[1]);
                                timing_metric = "median_time_ms";
                            }
                        }

                        if best_val is None and metric_flags["has_mean"] {
                            cursor.execute(
                                """
                                SELECT id, mean_time_ms
                                FROM nodes
                                WHERE mean_time_ms IS NOT NULL AND mean_time_ms > 0
                                ORDER BY mean_time_ms ASC
                                LIMIT 1
                                """
                            );
                            row = cursor.fetchone();
                            if row {
                                best_node_id = int(row[0]);
                                best_val = float(row[1]);
                                timing_metric = "mean_time_ms";
                            }
                        }

                        total_nodes += node_count;
                        op_summaries.append(
                            {
                                "op": op_name,
                                "nodes": node_count,
                                "best_value": best_val,
                                "best_node_id": best_node_id,
                                "timing_metric": timing_metric
                            }
                        );
                        conn.close();
                    } except Exception as e {
                        print(f"Error summarying {op_name}: {e}");
                    }
                }
            }
        }

        report {"success": True, "total_nodes": total_nodes, "ops": op_summaries};
    }
}

walker:pub GetOpDetails {
    has projectName: str,
        opName: str;

    can run with entry {
        project_dir = _project_dir(self.projectName);
        kernel_timings = _load_kernel_timings(project_dir, self.opName);
        runtime_by_path = (
            kernel_timings["by_path"]
            if kernel_timings and "by_path" in kernel_timings and kernel_timings["by_path"]
            else {}
        );
        root_runtime_ms = (
            kernel_timings["root_runtime_ms"]
            if kernel_timings and "root_runtime_ms" in kernel_timings
            else None
        );
        profile_dir = path.join(project_dir, "io", "individual_ops", self.opName);
        entry_files = [];
        entry_count = 0;
        if path.exists(profile_dir) {
            for fname in listdir(profile_dir) {
                if fname.startswith("entry_") and fname.endswith(".pt") {
                    entry_count = entry_count + 1;
                    if entry_count <= 50 {
                        entry_files.append(_file_info(path.join(profile_dir, fname)));
                    }
                }
            }
        }

        generated_dir = path.join(
            project_dir, "kernels", "generated", "individual_op_kernels", self.opName
        );
        gen_kernel_dir = path.join(generated_dir, "kernel");
        gen_attempt_dir = path.join(generated_dir, "attempts");
        gen_kernels = [];
        for kernel_file in _list_files(gen_kernel_dir, [".cu", ".so"]) {
            gen_kernels.append(_attach_runtime(kernel_file, runtime_by_path));
        }
        gen_kernels.sort(key=_cast_kernel_sort_key);
        gen_logs = _list_files(gen_attempt_dir, [".txt", ".log", ".json"]);
        gen_best_path = path.join(generated_dir, "kernel.cu");
        gen_best = (
            _attach_runtime(_file_info(gen_best_path), runtime_by_path, root_runtime_ms)
            if path.exists(gen_best_path)
            else {}
        );

        opt_base = _find_optimized_dir(self.projectName);
        if not opt_base {
            opt_base = path.join(repo_root, "kernels", "optimized");
        }
        opt_dir = path.join(opt_base, self.opName) if opt_base else "";
        opt_kernel_dir = path.join(opt_dir, "kernels") if opt_dir else "";
        opt_kernel_files = (
            _list_files(opt_kernel_dir, [".cu", ".py", ".metal", ".so"])
                if opt_kernel_dir and path.exists(opt_kernel_dir)
                else _list_files(opt_dir, [".cu", ".py", ".metal", ".so"])
        );
        opt_kernels = [];
        for kernel_file in opt_kernel_files {
            opt_kernels.append(_attach_runtime(kernel_file, runtime_by_path));
        }
        opt_kernels.sort(key=_cast_kernel_sort_key);
        opt_logs = _list_files(opt_dir, [".txt", ".log", ".json"]);

        report {
            "op": self.opName,
            "profile": {
                "count": entry_count,
                "entries": entry_files,
                "dir": path.relpath(profile_dir, repo_root)
                    if path.exists(profile_dir)
                    else ""
            },
            "generated": {
                "dir": path.relpath(generated_dir, repo_root)
                    if path.exists(generated_dir)
                    else "",
                "kernels": gen_kernels,
                "logs": gen_logs,
                "best": gen_best
            },
            "optimized": {
                "dir": path.relpath(opt_dir, repo_root)
                    if opt_dir and path.exists(opt_dir)
                    else "",
                "kernels": opt_kernels,
                "logs": opt_logs
            }
        };
    }
}

walker:pub GetProjectMetrics {
    has projectName: str;

    can run with entry {
        payload = _collect_project_metrics(self.projectName);
        report payload;
    }
}

def _llm_usage_db_path(project_name: str) -> str {
    return path.join(_project_dir(project_name), "llm_usage.db");
}

def _empty_llm_totals() -> dict {
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_cost_usd": 0.0,
        "calls": 0
    };
}

def _read_llm_usage(project_name: str, recent_limit: int) -> dict {
    db_path = _llm_usage_db_path(project_name);
    result = {
        "exists": False,
        "project_totals": _empty_llm_totals(),
        "per_operator": [],
        "recent_calls": []
    };
    if not path.exists(db_path) {
        return result;
    }
    result["exists"] = True;
    try {
        conn = sqlite3.connect(db_path);
        try {
            conn.execute("PRAGMA journal_mode=WAL;");
            cur = conn.cursor();
            cur.execute(
                "SELECT COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), "
                + "COALESCE(SUM(reasoning_tokens),0), COALESCE(SUM(total_cost_usd),0.0), "
                + "COUNT(*) FROM llm_calls"
            );
            row = cur.fetchone();
            if row {
                result["project_totals"] = {
                    "input_tokens": int(row[0] or 0),
                    "output_tokens": int(row[1] or 0),
                    "reasoning_tokens": int(row[2] or 0),
                    "total_cost_usd": float(row[3] or 0.0),
                    "calls": int(row[4] or 0)
                };
            }

            cur.execute(
                "SELECT COALESCE(operator, ''), COALESCE(SUM(input_tokens),0), "
                + "COALESCE(SUM(output_tokens),0), COALESCE(SUM(reasoning_tokens),0), "
                + "COALESCE(SUM(total_cost_usd),0.0), COUNT(*) FROM llm_calls "
                + "GROUP BY operator ORDER BY SUM(total_cost_usd) DESC"
            );
            op_rows = cur.fetchall();
            per_op = [];
            for r in op_rows {
                per_op.append({
                    "operator": r[0],
                    "input_tokens": int(r[1] or 0),
                    "output_tokens": int(r[2] or 0),
                    "reasoning_tokens": int(r[3] or 0),
                    "total_cost_usd": float(r[4] or 0.0),
                    "calls": int(r[5] or 0)
                });
            }
            result["per_operator"] = per_op;

            limit = recent_limit if recent_limit > 0 else 50;
            cur.execute(
                "SELECT ts, COALESCE(operator,''), step_type, iteration, attempt, "
                + "provider, model, input_tokens, output_tokens, reasoning_tokens, "
                + "total_cost_usd FROM llm_calls ORDER BY ts DESC LIMIT ?",
                (limit,)
            );
            recent_rows = cur.fetchall();
            recent = [];
            for r in recent_rows {
                recent.append({
                    "ts": float(r[0] or 0.0),
                    "operator": r[1],
                    "step_type": r[2],
                    "iteration": r[3],
                    "attempt": r[4],
                    "provider": r[5],
                    "model": r[6],
                    "input_tokens": int(r[7] or 0),
                    "output_tokens": int(r[8] or 0),
                    "reasoning_tokens": int(r[9] or 0),
                    "total_cost_usd": float(r[10] or 0.0)
                });
            }
            result["recent_calls"] = recent;
        } finally {
            conn.close();
        }
    } except Exception as e {
        result["error"] = str(e);
    }
    return result;
}

walker:pub GetLlmUsage {
    has projectName: str,
        recentLimit: int = 50;

    can run with entry {
        report _read_llm_usage(self.projectName, self.recentLimit);
    }
}

walker:pub GetOperatorMetrics {
    has projectName: str,
        opName: str;

    can run with entry {
        payload = _collect_project_metrics(self.projectName);
        op_key = _normalize_metric_op_name(self.opName);
        rows_by_op = payload["rows_by_op"]
            if payload and "rows_by_op" in payload
            else {};
        metric = rows_by_op[op_key] if op_key in rows_by_op else None;
        report {
            "success": bool(metric),
            "op": op_key,
            "metric": metric,
            "status": payload["status"] if payload and "status" in payload else "empty",
            "profile_status": (
                payload["profile_status"]
                    if payload and "profile_status" in payload
                    else ""
            ),
            "profile_message": (
                payload["profile_message"]
                    if payload and "profile_message" in payload
                    else ""
            ),
            "errors": payload["errors"] if payload and "errors" in payload else [],
            "metrics_version": (
                payload["metrics_version"]
                    if payload and "metrics_version" in payload
                    else 0
            )
        };
    }
}

walker:pub ImportKernel {
    has projectName: str,
        opName: str,
        source: str,
        backend: str = "CUDA";

    can run with entry {
        project_name = str(self.projectName).strip();
        op_name_raw = str(self.opName).strip();
        source_code = str(self.source);
        backend_raw = str(self.backend).strip().lower();

        if not project_name {
            report {"success": False, "error": "Missing project context."};
            disengage;
        }
        if not op_name_raw {
            report {"success": False, "error": "Missing operator name."};
            disengage;
        }
        if not source_code or str(source_code).strip() == "" {
            report {"success": False, "error": "Kernel source is empty."};
            disengage;
        }

        backend_name = "cuda";
        kernel_file = "kernel.cu";
        success_marker = "success.cuda";

        if backend_raw == "cuda" or backend_raw == "gpu" or backend_raw == "nvidia" {
            backend_name = "cuda";
            kernel_file = "kernel.cu";
            success_marker = "success.cuda";
        } elif backend_raw == "triton" {
            backend_name = "triton";
            kernel_file = "kernel.py";
            success_marker = "success.triton";
        } elif backend_raw == "mps"
        or backend_raw == "metal"
        or backend_raw == "apple" {
            backend_name = "mps";
            kernel_file = "kernel.metal";
            success_marker = "success.mps";
        } elif backend_raw == "cpu" {
            backend_name = "cpu";
            kernel_file = "kernel.py";
            success_marker = "success.cpu";
        }

        if backend_name != "cuda" {
            report {
                "success": False,
                "error": (
                    "Custom upload requires CUDA right now so it can be benchmarked before tree insertion."
                )
            };
            disengage;
        }

        op_name = _normalize_metric_op_name(op_name_raw);
        project_dir = _project_dir(str(project_name));
        generated_dir = path.join(
            project_dir, "kernels", "generated", "individual_op_kernels", op_name
        );
        _ensure_dir(generated_dir);

        kernel_path = path.join(generated_dir, kernel_file);
        stale_markers = ["success.cuda", "success.triton", "success.mps", "success.cpu"];
        for marker_name in stale_markers {
            marker_path = path.join(generated_dir, marker_name);
            if path.exists(marker_path) {
                try {
                    os.remove(marker_path);
                } except Exception as e {
                    _ = str(e);
                }
            }
        }

        try {
            with open(kernel_path, "w", encoding="utf-8", errors="ignore") as f {
                f.write(source_code);
            }
        } except Exception as e {
            report {"success": False, "error": f"Failed to write kernel source: {e}"};
            disengage;
        }

        try {
            with open(path.join(generated_dir, success_marker), "w") as f {
                f.write("ok\n");
            }
        } except Exception as e {
            report {"success": False, "error": f"Failed to write success marker: {e}"};
            disengage;
        }

        benchmark_script = """
import json
import subprocess
import sys
from pathlib import Path

from src.optimizer.tree_store import publish_generated_root

project_dir = Path(sys.argv[1])
project_name = sys.argv[2]
op_name = sys.argv[3]
backend = sys.argv[4]
repo_root = sys.argv[5]

cmd = [
    sys.executable,
    "-m",
    "src.optimizer.benchmarking.benchmark_ops",
    "--project",
    project_name,
    "--ops",
    op_name,
]
proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
if proc.returncode != 0:
    msg = (proc.stderr or proc.stdout or "").strip()
    print(json.dumps({"ok": False, "error": f"Benchmark command failed: {msg or 'unknown error'}"}))
    raise SystemExit(0)

bench_path = project_dir / "benchmarks" / "op_benchmarks.json"
if not bench_path.exists():
    print(json.dumps({"ok": False, "error": "Benchmark output missing."}))
    raise SystemExit(0)

try:
    payload = json.loads(bench_path.read_text(encoding="utf-8"))
except Exception as e:
    print(json.dumps({"ok": False, "error": f"Failed to parse benchmark output: {e}"}))
    raise SystemExit(0)

rows = payload.get("results") if isinstance(payload, dict) else None
if not isinstance(rows, list):
    print(json.dumps({"ok": False, "error": "Invalid benchmark output format."}))
    raise SystemExit(0)

target = None
for row in rows:
    if isinstance(row, dict) and str(row.get("op") or "") == op_name:
        target = row
        break

if target is None:
    print(json.dumps({"ok": False, "error": f"No benchmark row for {op_name}."}))
    raise SystemExit(0)

kernel_status = str(target.get("kernel_status") or "")
kernel_ms = target.get("kernel_ms")
pytorch_ms = target.get("pytorch_ms")
try:
    kernel_ms = float(kernel_ms) if kernel_ms is not None else None
except Exception:
    kernel_ms = None
try:
    pytorch_ms = float(pytorch_ms) if pytorch_ms is not None else None
except Exception:
    pytorch_ms = None

if kernel_status != "ok" or kernel_ms is None or kernel_ms <= 0.0:
    print(
        json.dumps(
            {
                "ok": False,
                "error": "Kernel benchmark did not succeed; tree was not updated.",
                "kernel_status": kernel_status,
                "kernel_ms": kernel_ms,
                "pytorch_ms": pytorch_ms,
            }
        )
    )
    raise SystemExit(0)

published = publish_generated_root(
    project_dir,
    op_name,
    kernel_ms=kernel_ms,
    backend=backend,
    description="Uploaded custom kernel (benchmarked)",
)
if not isinstance(published, dict) or not published.get("ok"):
    reason = published.get("reason") if isinstance(published, dict) else "unknown"
    print(json.dumps({"ok": False, "error": f"Failed to publish root node: {reason}"}))
    raise SystemExit(0)

print(
    json.dumps(
        {
            "ok": True,
            "op": op_name,
            "backend": backend,
            "kernel_ms": kernel_ms,
            "pytorch_ms": pytorch_ms,
            "kernel_status": kernel_status,
        }
    )
)
""";

        benchmark_result = {};
        try {
            out = subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    benchmark_script,
                    project_dir,
                    project_name,
                    op_name,
                    backend_name,
                    repo_root
                ],
                cwd=repo_root
            );
            result_text = out.decode("utf-8").strip();
            benchmark_result = loads(result_text) if result_text else {};
        } except Exception as e {
            report {
                "success": False,
                "error": ("Benchmark execution failed for uploaded kernel. " + str(e))
            };
            disengage;
        }

        if (
            not benchmark_result
            or not ("ok" in benchmark_result)
            or not benchmark_result["ok"]
        ) {
            err_msg = "Uploaded kernel failed benchmark and was not added to tree.";
            if benchmark_result
            and "error" in benchmark_result
            and benchmark_result["error"] {
                err_msg = str(benchmark_result["error"]);
            }
            report {"success": False, "error": err_msg, "details": benchmark_result};
            disengage;
        }

        try {
            _refresh_project_metrics(str(project_name));
        } except Exception as e {
            _ = str(e);
        }

        report {
            "success": True,
            "message": "Kernel benchmarked and added to tree.",
            "op": op_name,
            "kernel_ms": (
                benchmark_result["kernel_ms"]
                    if "kernel_ms" in benchmark_result
                    else None
            ),
            "pytorch_ms": (
                benchmark_result["pytorch_ms"]
                    if "pytorch_ms" in benchmark_result
                    else None
            ),
            "backend": backend_name
        };
    }
}

walker:pub DownloadCast {
    has projectName: str,
        backend: str = "CUDA",
        selectedKernels: dict = {};

    can run with entry {
        project_dir = _project_dir(self.projectName);
        if not path.exists(project_dir) {
            report {
                "success": False,
                "error": f"Project '{self.projectName}' not found."
            };
            disengage;
        }

        # ── Resolve kernel files ──────────────────────────────────────
        kernel_map = {};
        selected_ops = {};
        skipped_ops = {};

        if self.selectedKernels {
            for op_name in self.selectedKernels {
                selected_ops[op_name] = True;
                rel = str(self.selectedKernels[op_name]);
                if rel == "__PYTORCH__" {
                    skipped_ops[op_name] = True;
                    continue;
                }
                full = path.join(repo_root, rel);
                if path.exists(full) {
                    kernel_map[op_name] = full;
                }
            }
        }

        opt_base = _find_optimized_dir(self.projectName);
        if opt_base and path.exists(opt_base) {
            for op_name in listdir(opt_base) {
                op_dir = path.join(opt_base, op_name);
                if (
                    not path.isdir(op_dir)
                    or op_name in kernel_map
                    or op_name in skipped_ops
                    or (self.selectedKernels and op_name in selected_ops)
                ) {
                    continue;
                }
                best_entry = _best_cast_kernel_entry(project_dir, self.projectName, op_name);
                if best_entry and "path" in best_entry and best_entry["path"] {
                    kernel_map[op_name] = path.join(repo_root, best_entry["path"]);
                }
            }
        }

        if not kernel_map {
            report {
                "success": False,
                "error": "No optimized kernels found. Run Generate or Optimize first."
            };
            disengage;
        }

        # ── Resolve weight file ───────────────────────────────────────
        weight_archive_path = "";
        weight_size = 0;
        wdata = b"";
        cfg_path = path.join(project_dir, "config.json");
        if path.exists(cfg_path) {
            try {
                with open(cfg_path, "r") as f {
                    cfg = load(f);
                }
                arts = (
                    cfg["artifacts"]["weights"]
                        if cfg
                        and "artifacts" in cfg
                        and cfg["artifacts"]
                        and "weights" in cfg["artifacts"]
                        else []
                );
                for art in arts {
                    rel = art["relpath"] if art and "relpath" in art else "";
                    if rel {
                        wp = path.join(project_dir, rel);
                        if not path.exists(wp) {
                            wp = path.join(project_dir, "weights.pt");
                        }
                        if path.exists(wp) {
                            with open(wp, "rb") as f {
                                wdata = f.read();
                            }
                            sha = hashlib.sha256(wdata).hexdigest();
                            weight_archive_path = f"weights/{sha}.pt";
                            weight_size = len(wdata);
                            break;
                        }
                    }
                }
            } except Exception as e {
                print(f"Error reading project config for weights: {e}");
            }
        }

        if not weight_archive_path {
            for wname in ["weights.pt", "model.pt"] {
                wp = path.join(project_dir, wname);
                if path.exists(wp) {
                    with open(wp, "rb") as f {
                        wdata = f.read();
                    }
                    sha = hashlib.sha256(wdata).hexdigest();
                    weight_archive_path = f"weights/{sha}.pt";
                    weight_size = len(wdata);
                    break;
                }
            }
        }

        # ── Build file map {archive_path -> bytes} ────────────────────
        file_map = {};

        model_py = path.join(project_dir, "model.py");
        model_class_name = "";
        if path.exists(model_py) {
            with open(model_py, "rb") as f {
                model_py_bytes = f.read();
            }
            file_map["model.py"] = model_py_bytes;
            # Extract the first class name defined in model.py so the cast
            # runner can instantiate it without guessing.
            import ast as _ast;
            try {
                _tree = _ast.parse(model_py_bytes.decode("utf-8"));
                for _node in _ast.walk(_tree) {
                    if isinstance(_node, _ast.ClassDef) {
                        model_class_name = _node.name;
                        break;
                    }
                }
            } except Exception {
                model_class_name = "";
            }
        }

        # Pack model_config.json for HuggingFace models so the cast runner can
        # instantiate the model without needing to infer the architecture.
        # Priority: explicit file in project dir > auto-generated from config_class.
        model_cfg_src = path.join(project_dir, "model_config.json");
        if path.exists(model_cfg_src) {
            with open(model_cfg_src, "rb") as f {
                file_map["model_config.json"] = f.read();
            }
        } elif model_class_name and "model.py" in file_map {
            # Auto-generate: import the model class and call config_class() for defaults.
            import importlib.util as _ilu;
            import sys as _sys;
            import json as _json;
            import tempfile as _tmp;
            import os as _os;
            try {
                _tf = _tmp.NamedTemporaryFile(suffix=".py", delete=False, mode="wb");
                _tf.write(file_map["model.py"]);
                _tf.close();
                _spec = _ilu.spec_from_file_location("_kf_export_model", _tf.name);
                _mod = _ilu.module_from_spec(_spec);
                _sys.modules["_kf_export_model"] = _mod;
                _spec.loader.exec_module(_mod);
                _ModelCls = getattr(_mod, model_class_name, None);
                if _ModelCls and hasattr(_ModelCls, "config_class") {
                    _cfg = _ModelCls.config_class();
                    _cfg_dict = _cfg.to_dict();

                    # Patch shape-dependent fields from the actual state dict so the
                    # config matches the weights (e.g. num_labels for classifiers).
                    if wdata {
                        import io as _io;
                        import torch as _tw;
                        _sd = _tw.load(
                            _io.BytesIO(wdata), map_location="cpu", weights_only=True
                        );
                        # Common head weight key patterns, checked in priority order.
                        _HEAD_KEYS = [
                            "classifier.1.weight",  # HF ResNet
                            "classifier.weight",  # HF ViT, BERT, etc.
                            "fc.weight",  # torchvision ResNet
                            "head.weight",  # DeiT
                            "heads.head.weight",  # ViT-B

                        ];
                        for _hk in _HEAD_KEYS {
                            if _hk in _sd and len(_sd[_hk].shape) == 2 {
                                _cfg_dict["num_labels"] = int(_sd[_hk].shape[0]);
                                break;
                            }
                        }
                    }

                    file_map["model_config.json"] = _json.dumps(
                        _cfg_dict, indent=2
                    ).encode(
                        "utf-8"
                    );
                }
                _sys.modules.pop("_kf_export_model", None);
                _os.unlink(_tf.name);
            } except Exception as _e {
                print(f"[WARN] Could not auto-generate model_config.json: {_e}");
                _sys.modules.pop("_kf_export_model", None);
            }
        }

        if weight_archive_path {
            file_map[weight_archive_path] = wdata;
        }

        ops_manifest = [];
        # Detect the current GPU's SM version for tagging precompiled binaries.
        import torch as _torch;
        sm_version = "";
        if _torch.cuda.is_available() {
            _cap = _torch.cuda.get_device_capability();
            sm_version = f"sm_{_cap[0]}{_cap[1]}";
        }

        for op_name in kernel_map {
            kf = kernel_map[op_name];
            with open(kf, "rb") as f {
                kdata = f.read();
            }
            cu_path = f"kernels/{op_name}/kernel.cu";
            wrapper_path = f"kernels/{op_name}/wrapper.py";
            file_map[cu_path] = kdata;
            wrapper_src = f"# Cast dispatch wrapper for {op_name}\n# Generated by KernelForge\n";
            file_map[wrapper_path] = wrapper_src.encode("utf-8");

            # Bundle the precompiled .so if it exists alongside kernel.cu.
            op_precompiled = {};
            if sm_version {
                so_file = path.join(path.dirname(kf), f"{op_name}.so");
                if path.exists(so_file) {
                    with open(so_file, "rb") as f {
                        file_map[f"compiled/{sm_version}/{op_name}.so"] = f.read();
                    }
                    op_precompiled[sm_version] = f"compiled/{sm_version}/{op_name}.so";
                }
            }

            ops_manifest.append(
                {
                    "name": op_name,
                    "kernel_dir": f"kernels/{op_name}/",
                    "cuda_source": cu_path,
                    "wrapper": wrapper_path,
                    "precompiled": op_precompiled
                }
            );
        }

        all_precompiled_sms = list(
            {sm for op in ops_manifest for sm in op["precompiled"].keys()}
        );

        loader_stub = b"# Cast vendored runtime loader\n# pip install cast for the full runtime\n";
        file_map["loader.py"] = loader_stub;

        manifest_obj = {
            "project_name": self.projectName,
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "model_class": model_class_name,
            "model_init_args": {},
            "weight_file": weight_archive_path,
            "ops": ops_manifest
        };
        file_map["manifest.json"] = dumps(manifest_obj, indent=2).encode("utf-8");

        # ── Checksums ─────────────────────────────────────────────────
        checksum_lines = [];
        for fpath in file_map {
            hexhash = hashlib.sha256(file_map[fpath]).hexdigest();
            checksum_lines.append(f"{hexhash}  {fpath}");
        }
        checksums_bytes = "\n".join(checksum_lines).encode("utf-8");
        archive_checksum = hashlib.sha256(checksums_bytes).hexdigest();
        file_map["checksums.sha256"] = checksums_bytes;

        # ── HEADER.json ───────────────────────────────────────────────
        header_obj = {
            "format_version": "1.0",
            "file_type": "kernelforge_inference",
            "project_name": self.projectName,
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "kernelforge_version": "0.1.0",
            "runtime": {
                "min_cast_version": "0.1",
                "min_torch_version": "2.1.0",
                "min_cuda_version": "12.0",
                "target_sm_versions": []
            },
            "contents": {
                "optimized_op_count": len(ops_manifest),
                "total_op_count": len(ops_manifest),
                "has_precompiled": len(all_precompiled_sms) > 0,
                "precompiled_sm_versions": all_precompiled_sms,
                "weight_size_bytes": weight_size
            },
            "archive_checksum": archive_checksum
        };
        header_bytes = dumps(header_obj, indent=2).encode("utf-8");

        # ── Write ZIP (HEADER.json must be first entry) ───────────────
        export_dir = path.join(project_dir, "exports");
        _ensure_dir(export_dir);
        cast_path = path.join(export_dir, self.projectName + ".cast");

        try {
            with ZipFile(cast_path, "w") as zipf {
                zipf.writestr("HEADER.json", header_bytes);
                for fpath in file_map {
                    zipf.writestr(fpath, file_map[fpath]);
                }
            }
            file_size = path.getsize(cast_path);
            report {
                "success": True,
                "name": self.projectName + ".cast",
                "size": file_size,
                "export_path": cast_path
            };
        } except Exception as e {
            print(f"DownloadCast error: {e}");
            report {"success": False, "error": f"Failed to build cast file: {e}"};
        }
    }
}

walker:pub GetCastChunk {
    has export_path: str,
        offset: int = 0,
        chunkSize: int = 5242880;

    can run with entry {
        base = path.realpath(_projects_base_dir());
        real = path.realpath(self.export_path);
        if not (real.startswith(base + os.sep) or real == base) {
            report {"success": False, "error": "Access denied."};
            disengage;
        }
        if not path.exists(real) {
            report {"success": False, "error": "Export file not found."};
            disengage;
        }
        total_size = path.getsize(real);
        try {
            with open(real, "rb") as f {
                f.seek(self.offset);
                chunk = f.read(self.chunkSize);
            }
            b64 = b64encode(chunk).decode("utf-8");
            report {
                "success": True,
                "data": b64,
                "offset": self.offset,
                "size": len(chunk),
                "total_size": total_size,
                "done": (self.offset + len(chunk)) >= total_size
            };
        } except Exception as e {
            print(f"GetCastChunk error: {e}");
            report {"success": False, "error": f"Failed to read chunk: {e}"};
        }
    }
}

# --- Private helpers used by GetOpDetails ---
def _file_kind(fname: str) -> str {
    if fname.endswith(".cu")
    or fname.endswith(".txt")
    or fname.endswith(".log")
    or fname.endswith(".json") {
        return "text";
    }
    return "binary";
}

def _file_info(full_path: str) -> dict {
    return {
        "name": path.basename(full_path),
        "path": path.relpath(full_path, repo_root),
        "size": path.getsize(full_path),
        "mtime": path.getmtime(full_path),
        "kind": _file_kind(path.basename(full_path))
    };
}

def _positive_float(value: Any) -> float {
    try {
        parsed = float(value);
        if parsed > 0.0 {
            return parsed;
        }
    } except Exception {
        _ = 0;
    }
    return None;
}

def _resolve_tree_kernel_path(project_dir: str, op_name: str, node_id: int, code_ref: str) -> str {
    candidates = [];

    code_text = str(code_ref).strip() if code_ref is not None else "";
    if code_text {
        if path.isabs(code_text) {
            candidates.append(code_text);
        } else {
            candidates.append(path.join(project_dir, "trees", code_text));
            candidates.append(path.join(project_dir, code_text));
            candidates.append(path.join(repo_root, code_text));
        }
    }

    kernel_dir = path.join(project_dir, "trees", op_name, "kernels");
    candidates.append(path.join(kernel_dir, f"kernel_{node_id}.cu"));
    candidates.append(path.join(kernel_dir, f"kernel_{node_id}.py"));
    candidates.append(path.join(kernel_dir, f"kernel_{node_id}.metal"));

    for candidate in candidates {
        if candidate and path.exists(candidate) and path.isfile(candidate) {
            return candidate;
        }
    }
    return "";
}

def _load_kernel_timings(project_dir: str, op_name: str) -> dict {
    db_path = path.join(project_dir, "trees", op_name, "nodes.db");
    by_path = {};
    root_runtime_ms = None;

    if not path.exists(db_path) {
        return {"by_path": by_path, "root_runtime_ms": root_runtime_ms};
    }

    conn = None;
    try {
        conn = sqlite3.connect(db_path);
        cursor = conn.cursor();
        cursor.execute(
            "SELECT id, value, code FROM nodes WHERE value IS NOT NULL AND value > 0"
        );
        rows = cursor.fetchall();
        for row in rows {
            runtime_ms = _positive_float(row[1]);
            if runtime_ms is None {
                continue;
            }

            resolved_path = _resolve_tree_kernel_path(
                project_dir, op_name, int(row[0]), row[2]
            );
            if resolved_path {
                by_path[path.relpath(resolved_path, repo_root)] = runtime_ms;
            }
            if int(row[0]) == 0 and root_runtime_ms is None {
                root_runtime_ms = runtime_ms;
            }
        }
    } except Exception {
        _ = 0;
    } finally {
        if conn {
            try {
                conn.close();
            } except Exception {
                _ = 0;
            }
        }
    }

    return {"by_path": by_path, "root_runtime_ms": root_runtime_ms};
}

def _attach_runtime(file_entry: dict, runtime_by_path: dict, fallback_runtime: float = None) -> dict {
    if not file_entry {
        return file_entry;
    }

    enriched = {**file_entry};
    rel_path = str(file_entry["path"]) if "path" in file_entry and file_entry["path"] else "";
    if rel_path and runtime_by_path and rel_path in runtime_by_path {
        runtime_ms = _positive_float(runtime_by_path[rel_path]);
        if runtime_ms is not None {
            enriched["runtime_ms"] = runtime_ms;
            return enriched;
        }
    }

    if fallback_runtime is not None {
        runtime_ms = _positive_float(fallback_runtime);
        if runtime_ms is not None {
            enriched["runtime_ms"] = runtime_ms;
        }
    }
    return enriched;
}

def _cast_kernel_label_rank(label: str) -> int {
    if label == "Best Generated" {
        return 0;
    }
    if label == "Optimized" {
        return 1;
    }
    if label == "Generated" {
        return 2;
    }
    return 3;
}

def _cast_kernel_sort_key(item: dict) -> list {
    runtime_ms = (
        _positive_float(item["runtime_ms"])
        if item and "runtime_ms" in item
        else None
    );
    runtime_sort = runtime_ms if runtime_ms is not None else 1e300;
    label = str(item["label"]) if item and "label" in item and item["label"] else "";
    name = str(item["name"]) if item and "name" in item and item["name"] else "";
    return [
        1 if runtime_ms is None else 0,
        runtime_sort,
        _cast_kernel_label_rank(label),
        name
    ];
}

def _collect_cast_kernel_candidates(project_dir: str, project_name: str, op_name: str) -> list {
    kernel_timings = _load_kernel_timings(project_dir, op_name);
    runtime_by_path = (
        kernel_timings["by_path"]
        if kernel_timings and "by_path" in kernel_timings and kernel_timings["by_path"]
        else {}
    );
    root_runtime_ms = (
        kernel_timings["root_runtime_ms"]
        if kernel_timings and "root_runtime_ms" in kernel_timings
        else None
    );

    candidates = [];

    generated_dir = path.join(
        project_dir, "kernels", "generated", "individual_op_kernels", op_name
    );
    gen_best_path = path.join(generated_dir, "kernel.cu");
    if path.exists(gen_best_path) {
        candidates.append(
            {
                **_attach_runtime(
                    _file_info(gen_best_path),
                    runtime_by_path,
                    root_runtime_ms
                ),
                "label": "Best Generated"
            }
        );
    }

    opt_base = _find_optimized_dir(project_name);
    if not opt_base {
        opt_base = path.join(repo_root, "kernels", "optimized");
    }
    opt_dir = path.join(opt_base, op_name) if opt_base else "";
    opt_kernel_dir = path.join(opt_dir, "kernels") if opt_dir else "";
    opt_kernel_files = (
        _list_files(opt_kernel_dir, [".cu", ".py", ".metal", ".so"])
        if opt_kernel_dir and path.exists(opt_kernel_dir)
        else _list_files(opt_dir, [".cu", ".py", ".metal", ".so"])
    );
    for kernel_file in opt_kernel_files {
        candidates.append(
            {**_attach_runtime(kernel_file, runtime_by_path), "label": "Optimized"}
        );
    }

    gen_kernel_dir = path.join(generated_dir, "kernel");
    best_path = (
        path.relpath(gen_best_path, repo_root)
        if path.exists(gen_best_path)
        else ""
    );
    for kernel_file in _list_files(gen_kernel_dir, [".cu", ".so"]) {
        rel_path = (
            str(kernel_file["path"])
            if "path" in kernel_file and kernel_file["path"]
            else ""
        );
        if best_path and rel_path == best_path {
            continue;
        }
        candidates.append(
            {**_attach_runtime(kernel_file, runtime_by_path), "label": "Generated"}
        );
    }

    candidates.sort(key=_cast_kernel_sort_key);
    return candidates;
}

def _best_cast_kernel_entry(project_dir: str, project_name: str, op_name: str) -> dict {
    candidates = _collect_cast_kernel_candidates(project_dir, project_name, op_name);
    if candidates and len(candidates) > 0 {
        return candidates[0];
    }
    return {};
}

def _list_files(dir_path: str, exts: list = None, max_files: int = 200) -> list {
    files = [];
    if not path.exists(dir_path) {
        return files;
    }
    count = 0;
    for fname in listdir(dir_path) {
        full_path = path.join(dir_path, fname);
        if path.isdir(full_path) {
            continue;
        }
        if exts {
            ok = False;
            for ext in exts {
                if fname.endswith(ext) {
                    ok = True;
                }
            }
            if not ok {
                continue;
            }
        }
        files.append(_file_info(full_path));
        count = count + 1;
        if count >= max_files {
            break;
        }
    }
    return files;
}
