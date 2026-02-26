"""SQLite catalog and metrics DB helpers."""

import from json { loads, dumps }
import from os { path, makedirs }
import from datetime { datetime }
import os;
import sqlite3;

import from .globals { repo_root }

def:pub _ensure_dir(dir_path: str) -> None {
    if not path.exists(dir_path) {
        makedirs(dir_path);
    }
}

def:pub _projects_base_dir() -> str {
    data_dir = (
        os.environ["KFORGE_DATA_DIR"]
        if "KFORGE_DATA_DIR" in os.environ and os.environ["KFORGE_DATA_DIR"]
        else path.join(repo_root, "kernels")
    );
    projects_dir = path.join(data_dir, "projects");
    _ensure_dir(projects_dir);
    return projects_dir;
}

def:pub _catalog_db_path() -> str {
    return path.join(_projects_base_dir(), "catalog.db");
}

def:pub _ensure_catalog_schema() -> None {
    db_path = _catalog_db_path();
    conn = sqlite3.connect(db_path);
    cursor = conn.cursor();
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            name TEXT PRIMARY KEY,
            project_path TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    );
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS operator_metrics (
            project_name TEXT NOT NULL,
            op_name TEXT NOT NULL,
            op_count INTEGER NOT NULL DEFAULT 0,
            pytorch_ms REAL,
            forged_ms REAL,
            winner TEXT,
            total_time_ms REAL NOT NULL DEFAULT 0,
            updated_at TEXT,
            PRIMARY KEY (project_name, op_name)
        )
        """
    );
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS project_metric_meta (
            project_name TEXT PRIMARY KEY,
            status TEXT,
            profile_status TEXT,
            profile_message TEXT,
            errors_json TEXT,
            metrics_version INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT
        )
        """
    );
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS run_state (
            project_name TEXT NOT NULL,
            job_key TEXT NOT NULL,
            status TEXT,
            control TEXT,
            active INTEGER NOT NULL DEFAULT 0,
            phase TEXT,
            message TEXT,
            progress_percent REAL,
            started_at TEXT,
            updated_at TEXT,
            finished_at TEXT,
            log TEXT,
            command_json TEXT,
            run_id TEXT,
            pid INTEGER,
            progress_json TEXT,
            extras_json TEXT,
            PRIMARY KEY (project_name, job_key)
        )
        """
    );
    try {
        cursor.execute("PRAGMA table_info(project_metric_meta)");
        meta_cols = {};
        for col in cursor.fetchall() {
            if col and len(col) > 1 and col[1] {
                meta_cols[str(col[1])] = True;
            }
        }
        if not ("metrics_version" in meta_cols) {
            cursor.execute(
                "ALTER TABLE project_metric_meta ADD COLUMN metrics_version INTEGER NOT NULL DEFAULT 0"
            );
        }
    } except Exception as e {
        print(f"Warning: failed to migrate project_metric_meta schema: {e}");
    }
    conn.commit();
    conn.close();
}

def:pub _json_load_safe(raw: str, default_value: Any) -> Any {
    if not raw {
        return default_value;
    }
    try {
        return loads(raw);
    } except Exception {
        return default_value;
    }
}

def:pub _upsert_project_catalog(project_name: str, project_path: str = "") -> None {
    if not project_name {
        return;
    }
    _ensure_catalog_schema();
    now = datetime.now().isoformat();
    db_path = _catalog_db_path();
    conn = sqlite3.connect(db_path);
    cursor = conn.cursor();
    created_at = now;
    cursor.execute(
        "SELECT created_at FROM projects WHERE name = ?",
        (project_name,)
    );
    existing = cursor.fetchone();
    if existing and len(existing) > 0 and existing[0] {
        created_at = str(existing[0]);
    }
    cursor.execute(
        """
        INSERT OR REPLACE INTO projects(name, project_path, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        (project_name, project_path, created_at, now)
    );
    conn.commit();
    conn.close();
}

def:pub _load_project_catalog() -> list {
    _ensure_catalog_schema();
    db_path = _catalog_db_path();
    conn = sqlite3.connect(db_path);
    cursor = conn.cursor();
    cursor.execute(
        "SELECT name, project_path, created_at, updated_at FROM projects ORDER BY lower(name)"
    );
    rows = cursor.fetchall();
    conn.close();
    out = [];
    for row in rows {
        if not row or len(row) < 1 or not row[0] {
            continue;
        }
        out.append(
            {
                "name": str(row[0]),
                "project_path": str(row[1]) if len(row) > 1 and row[1] else "",
                "created_at": str(row[2]) if len(row) > 2 and row[2] else "",
                "updated_at": str(row[3]) if len(row) > 3 and row[3] else ""
            }
        );
    }
    return out;
}

def:pub _write_job_state_db(project_name: str, job_key: str, job_state: dict) -> None {
    if not project_name or not job_key or not job_state {
        return;
    }
    _ensure_catalog_schema();
    now = datetime.now().isoformat();
    status = str(job_state["status"]) if "status" in job_state and job_state["status"] else "";
    control = str(job_state["control"]) if "control" in job_state and job_state["control"] else "";
    active = 1 if ("active" in job_state and job_state["active"]) else 0;
    phase = str(job_state["phase"]) if "phase" in job_state and job_state["phase"] else "";
    message = str(job_state["message"]) if "message" in job_state and job_state["message"] else "";
    progress_percent = None;
    if "progress_percent" in job_state and job_state["progress_percent"] is not None {
        try {
            progress_percent = float(job_state["progress_percent"]);
        } except Exception {
            progress_percent = None;
        }
    }
    started_at = str(job_state["started_at"]) if "started_at" in job_state and job_state["started_at"] else "";
    updated_at = str(job_state["updated_at"]) if "updated_at" in job_state and job_state["updated_at"] else now;
    finished_at = str(job_state["finished_at"]) if "finished_at" in job_state and job_state["finished_at"] else "";
    log_path = str(job_state["log"]) if "log" in job_state and job_state["log"] else "";
    run_id = str(job_state["run_id"]) if "run_id" in job_state and job_state["run_id"] else "";
    pid_val = None;
    if "pid" in job_state and job_state["pid"] is not None and str(job_state["pid"]) != "" {
        try {
            pid_val = int(job_state["pid"]);
        } except Exception {
            pid_val = None;
        }
    }
    command_json = "[]";
    if "command" in job_state and job_state["command"] is not None {
        try {
            command_json = dumps(job_state["command"]);
        } except Exception {
            command_json = "[]";
        }
    }
    progress_json = "{}";
    if "progress" in job_state and job_state["progress"] is not None {
        try {
            progress_json = dumps(job_state["progress"]);
        } except Exception {
            progress_json = "{}";
        }
    }
    known_keys = [
        "status",
        "control",
        "active",
        "phase",
        "message",
        "progress_percent",
        "started_at",
        "updated_at",
        "finished_at",
        "log",
        "command",
        "run_id",
        "pid",
        "progress"
    ];
    extras = {};
    for key in job_state {
        if key in known_keys {
            continue;
        }
        extras[key] = job_state[key];
    }
    extras_json = "{}";
    try {
        extras_json = dumps(extras);
    } except Exception {
        extras_json = "{}";
    }

    db_path = _catalog_db_path();
    conn = sqlite3.connect(db_path);
    cursor = conn.cursor();
    cursor.execute(
        """
        INSERT OR REPLACE INTO run_state(
            project_name, job_key, status, control, active, phase, message,
            progress_percent, started_at, updated_at, finished_at, log,
            command_json, run_id, pid, progress_json, extras_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            project_name,
            job_key,
            status,
            control,
            active,
            phase,
            message,
            progress_percent,
            started_at,
            updated_at,
            finished_at,
            log_path,
            command_json,
            run_id,
            pid_val,
            progress_json,
            extras_json
        )
    );
    conn.commit();
    conn.close();
}

def:pub _write_project_run_state_db(project_name: str, state: dict) -> None {
    if not project_name or not state or len(state) == 0 {
        return;
    }
    _ensure_catalog_schema();
    db_path = _catalog_db_path();
    conn = sqlite3.connect(db_path);
    cursor = conn.cursor();
    cursor.execute("DELETE FROM run_state WHERE project_name = ?", (project_name,));
    conn.commit();
    conn.close();
    for job_key in state {
        job_state = state[job_key];
        if not job_state or not ("status" in job_state) {
            continue;
        }
        _write_job_state_db(project_name, str(job_key), job_state);
    }
}

def:pub _read_project_run_state_db(project_name: str) -> dict {
    if not project_name {
        return {};
    }
    _ensure_catalog_schema();
    db_path = _catalog_db_path();
    conn = sqlite3.connect(db_path);
    cursor = conn.cursor();
    cursor.execute(
        """
        SELECT
            job_key, status, control, active, phase, message, progress_percent,
            started_at, updated_at, finished_at, log, command_json, run_id, pid,
            progress_json, extras_json
        FROM run_state
        WHERE project_name = ?
        ORDER BY job_key
        """,
        (project_name,)
    );
    rows = cursor.fetchall();
    conn.close();

    state = {};
    for row in rows {
        if not row or len(row) < 16 or not row[0] {
            continue;
        }
        job_key = str(row[0]);
        job = {};
        job["status"] = str(row[1]) if row[1] else "";
        job["control"] = str(row[2]) if row[2] else "";
        job["active"] = bool(row[3]) if row[3] is not None else False;
        job["phase"] = str(row[4]) if row[4] else "";
        job["message"] = str(row[5]) if row[5] else "";
        if row[6] is not None {
            try {
                job["progress_percent"] = float(row[6]);
            } except Exception as e {
                _ = str(e);
            }
        }
        if row[7] {
            job["started_at"] = str(row[7]);
        }
        if row[8] {
            job["updated_at"] = str(row[8]);
        }
        if row[9] {
            job["finished_at"] = str(row[9]);
        }
        if row[10] {
            job["log"] = str(row[10]);
        }
        if row[11] {
            job["command"] = _json_load_safe(str(row[11]), []);
        }
        if row[12] {
            job["run_id"] = str(row[12]);
        }
        if row[13] is not None {
            try {
                job["pid"] = int(row[13]);
            } except Exception as e {
                _ = str(e);
            }
        }
        if row[14] {
            job["progress"] = _json_load_safe(str(row[14]), {});
        }
        extras = _json_load_safe(str(row[15]) if row[15] else "{}", {});
        if extras and type(extras) == dict {
            for extra_key in extras {
                if not (extra_key in job) {
                    job[extra_key] = extras[extra_key];
                }
            }
        }
        state[job_key] = job;
    }
    return state;
}

def:pub _upsert_project_metric_rows(project_name: str, payload: dict) -> None {
    if not project_name {
        return;
    }
    _ensure_catalog_schema();
    now = datetime.now().isoformat();
    rows = payload["rows"] if payload and "rows" in payload and payload["rows"] else [];
    status = str(payload["status"]) if payload and "status" in payload and payload["status"] else "empty";
    profile_status = (
        str(payload["profile_status"])
        if payload and "profile_status" in payload and payload["profile_status"]
        else ""
    );
    profile_message = (
        str(payload["profile_message"])
        if payload and "profile_message" in payload and payload["profile_message"]
        else ""
    );
    errors_json = "[]";
    if payload and "errors" in payload and payload["errors"] {
        try {
            errors_json = dumps(payload["errors"]);
        } except Exception {
            errors_json = "[]";
        }
    }

    db_path = _catalog_db_path();
    conn = sqlite3.connect(db_path);
    cursor = conn.cursor();
    metrics_version = 1;
    cursor.execute(
        "SELECT metrics_version FROM project_metric_meta WHERE project_name = ?",
        (project_name,)
    );
    existing_meta = cursor.fetchone();
    if existing_meta and len(existing_meta) > 0 and existing_meta[0] is not None {
        try {
            metrics_version = int(existing_meta[0]) + 1;
        } except Exception {
            metrics_version = 1;
        }
    }

    cursor.execute(
        "DELETE FROM operator_metrics WHERE project_name = ?",
        (project_name,)
    );
    for row in rows {
        if not row {
            continue;
        }
        op_name = str(row["name"]) if "name" in row and row["name"] else "";
        if not op_name {
            continue;
        }
        op_count = 0;
        try {
            op_count = int(row["count"]) if "count" in row and row["count"] else 0;
        } except Exception {
            op_count = 0;
        }
        pytorch_ms = None;
        if "pytorch_ms" in row and row["pytorch_ms"] is not None {
            try {
                pytorch_ms = float(row["pytorch_ms"]);
            } except Exception {
                pytorch_ms = None;
            }
        }
        forged_ms = None;
        if "forged_ms" in row and row["forged_ms"] is not None {
            try {
                forged_ms = float(row["forged_ms"]);
            } except Exception {
                forged_ms = None;
            }
        }
        total_time_ms = 0.0;
        try {
            total_time_ms = (
                float(row["total_time_ms"])
                if "total_time_ms" in row and row["total_time_ms"] is not None
                else 0.0
            );
        } except Exception {
            total_time_ms = 0.0;
        }
        winner = str(row["winner"]) if "winner" in row and row["winner"] else "";
        cursor.execute(
            """
            INSERT OR REPLACE INTO operator_metrics(
                project_name, op_name, op_count, pytorch_ms, forged_ms, winner,
                total_time_ms, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_name,
                op_name,
                op_count,
                pytorch_ms,
                forged_ms,
                winner,
                total_time_ms,
                now
            )
        );
    }

    cursor.execute(
        """
        INSERT OR REPLACE INTO project_metric_meta(
            project_name, status, profile_status, profile_message, errors_json, metrics_version, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            project_name,
            status,
            profile_status,
            profile_message,
            errors_json,
            metrics_version,
            now
        )
    );
    conn.commit();
    conn.close();
}

def:pub _read_project_metric_rows(project_name: str) -> dict {
    _ensure_catalog_schema();
    db_path = _catalog_db_path();
    conn = sqlite3.connect(db_path);
    cursor = conn.cursor();
    cursor.execute(
        """
        SELECT op_name, op_count, pytorch_ms, forged_ms, winner, total_time_ms
        FROM operator_metrics
        WHERE project_name = ?
        ORDER BY op_name
        """,
        (project_name,)
    );
    metric_rows = cursor.fetchall();

    cursor.execute(
        """
        SELECT status, profile_status, profile_message, errors_json, metrics_version, updated_at
        FROM project_metric_meta
        WHERE project_name = ?
        """,
        (project_name,)
    );
    meta = cursor.fetchone();
    conn.close();

    rows = [];
    rows_by_op = {};
    speed_comparison = [];
    operator_usage = [];

    for m in metric_rows {
        if not m or len(m) < 6 or not m[0] {
            continue;
        }
        op_name = str(m[0]);
        count = 0;
        try {
            count = int(m[1]) if m[1] is not None else 0;
        } except Exception {
            count = 0;
        }
        pytorch_ms = None;
        if m[2] is not None {
            try {
                parsed = float(m[2]);
                if parsed > 0.0 {
                    pytorch_ms = parsed;
                }
            } except Exception {
                pytorch_ms = None;
            }
        }
        forged_ms = None;
        if m[3] is not None {
            try {
                parsed = float(m[3]);
                if parsed > 0.0 {
                    forged_ms = parsed;
                }
            } except Exception {
                forged_ms = None;
            }
        }
        winner = str(m[4]) if m[4] else "unknown";
        total_time_ms = 0.0;
        try {
            total_time_ms = float(m[5]) if m[5] is not None else 0.0;
        } except Exception {
            total_time_ms = 0.0;
        }

        row = {
            "name": op_name,
            "count": count,
            "winner": winner,
            "total_time_ms": total_time_ms
        };
        if pytorch_ms is not None {
            row["pytorch_ms"] = pytorch_ms;
        }
        if forged_ms is not None {
            row["forged_ms"] = forged_ms;
        }

        rows.append(row);
        rows_by_op[op_name] = row;

        if pytorch_ms is not None {
            srow = {"name": op_name, "pytorch": pytorch_ms};
            if forged_ms is not None {
                srow["forged"] = forged_ms;
                srow["optimized"] = forged_ms;
            }
            speed_comparison.append(srow);
        }
        if total_time_ms > 0.0 {
            operator_usage.append(
                {"name": op_name, "count": count, "total_time_ms": total_time_ms}
            );
        }
    }

    status = "empty";
    profile_status = "";
    profile_message = "";
    errors = [];
    metrics_version = 0;
    metrics_updated_at = "";
    meta_vals = [];
    if meta is not None {
        try {
            meta_vals = list(meta);
        } except Exception {
            meta_vals = [];
        }
    }
    if len(meta_vals) > 0 and meta_vals[0] {
        status = str(meta_vals[0]);
    }
    if len(meta_vals) > 1 and meta_vals[1] {
        profile_status = str(meta_vals[1]);
    }
    if len(meta_vals) > 2 and meta_vals[2] {
        profile_message = str(meta_vals[2]);
    }
    if len(meta_vals) > 3 and meta_vals[3] {
        try {
            parsed_errors = loads(str(meta_vals[3]));
            if parsed_errors {
                errors = parsed_errors;
            }
        } except Exception {
            errors = [];
        }
    }
    if len(meta_vals) > 4 and meta_vals[4] is not None {
        try {
            metrics_version = int(meta_vals[4]);
        } except Exception {
            metrics_version = 0;
        }
    }
    if len(meta_vals) > 5 and meta_vals[5] {
        metrics_updated_at = str(meta_vals[5]);
    }

    return {
        "rows": rows,
        "rows_by_op": rows_by_op,
        "operator_usage": operator_usage,
        "speed_comparison": speed_comparison,
        "status": status,
        "profile_status": profile_status,
        "profile_message": profile_message,
        "errors": errors,
        "metrics_version": metrics_version,
        "updated_at": metrics_updated_at
    };
}
