# KernelForge Roadmap

Single source of truth for open issues and completed work. Design specs (`ui-features.md`, `queue_design_spec.md.resolved`) are kept separately.

---

## Done

- **kernel_job_runner.jac:L68-82** - Backend designation now validates correctly; nonsensical CUDA-for-ROCm fallback removed. Error thrown on unrecognized backend.
- **kernel_job_runner.jac:L179-194** - `StartOptimize` no longer assumes CUDA; backend is resolved dynamically in the pipeline per kernel.
- **pipeline.py:L744-755** - Parallel mode exposed in the frontend; worker count capped to node count (no over-spawning).
- **pipeline.py:L851-888** - Cleanup logic extracted to its own function/file.
- **queue_design_spec.md** - "Validating" vs "Verifying" terminology corrected throughout the spec.
- **settings.py / loader.py** - CUDA path hardcoded to 12.1. Fixed: 5-step auto-detection in place; 12.1 is last-resort fallback only.
- **backends/cuda/loader.py** - Compiled `.so` cache never invalidated on kernel source change. Fixed: `profiler.py:get_module` writes a `.source_hash` file and invalidates stale `.so` on mismatch.
- **backends/cuda/profiler.py** - `TORCH_CUDA_ARCH_LIST` set unconditionally, overwriting user-configured value. Fixed: changed to `os.environ.setdefault`.
- **backends/cuda/profiler.py** - `get_gpu_specs()` could return zeroed critical fields silently. Fixed: warns to stderr listing affected fields; walkers surface `gpu_warning` in response; modals fire `toast.warning`.
- **pipeline.py / queue state** - Race condition in `update_queue_state()`. Fixed: full read-modify-write cycle covered by `file_lock`; atomic write via `.pid.tmp` + `replace()`.
- **pipeline.py:~L38** - Bare `except:` catching `KeyboardInterrupt`/`SystemExit`. Fixed: all handlers use `except Exception:` or specific types throughout.
- **core/mcts.py** - SQLite connections not closed in exception paths. Fixed: all connections use `with sqlite3.connect(...) as conn:` context managers throughout.
- **core/mcts.py:~L104-109** - Speedup inf/NaN breaks UCT node selection. Not an issue: `child_val > 0` guard prevents division, defaults to 1.0; `speedup_vs_parent` is not used in UCT scoring.
- **backends/cuda/verifier.py:~L374** - Subprocess crash caused full-timeout wait before liveness check. Fixed: timeout loop polls `_WORKER_PROCESS.is_alive()` every 0.5s; crash detected promptly.

---

## P0 - Blockers (must fix before v1.0)

- **backends/metal.py** - Metal backend is a near-empty stub. Will fail cryptically when selected. Either complete it or gate behind a clear "not supported" error at the point of selection.

- **dashboard.jac** - No GPU is associated with a project at creation time. MCTS scores are meaningless if the profiling GPU changes between sessions. On new project creation, prompt the user to select the attached GPU. Attach GPU name to all kernels and PyTorch op profiles. On open with a different GPU, warn and re-profile everything before optimizing.

---

## P1 - High Priority (visible bugs and broken UX, fix before v1.0)

- **project_admin.jac:L626-637** - `GetProjectStatus` filters out tasks with `current_step == "Starting"`. Subprocess takes 2-5 s to start, so the queue panel shows empty right after submit. Either pass "Starting" through to the frontend or have the initial write skip it and use the first real step.

- **Dashboard.cl.jac:L593-657** - Polling `useEffect` depends on `[name, forgeBusy, any_job_active]`. When `forgeBusy` flips False after `StartGenerate`, `any_job_active` may also be False (empty queue due to the "Starting" filter). Poll interval jumps to 15 s. Compound with the blank queue: users can wait up to 15 s before seeing any task.

- **WorkQueuePanel.cl.jac:L28-38** - `completedHistory` deduplication uses `str(t["id"]) == str(task["id"])`. Re-running generate for the same operator without clearing history blocks the new entry. Fix: only replace an existing entry if the incoming task is Done or Failed.

- **Dashboard.cl.jac:L1089-1091** - `active_tasks_dict` objects mutated in place (`t["id"] = k` where `t` is a direct reference into status state). Violates React immutability. Shallow-copy before mutating: `t = {**active_tasks_dict[k]}; t["id"] = k;`

- **WorkQueuePanel.cl.jac:L105-120** - The `sub_info` else branch shows "queued" for any step that is not "Failed", "Generating", or "Verifying". "Monitoring", "Benchmarking", "Validating", and "Profiling" all display "queued". Each active step needs its own label.

- **job_supervisor.jac:L1170-1207** - `_write_initial_queue_state` for `[OPT]` creates a single `"seq_opt"` entry with `op_name` set to `ops_list[0]` only. Verify the pipeline updates `op_name` as it moves to each subsequent operator; add the update if missing.

- **WorkQueuePanel.cl.jac** - Direct property access on dynamic backend fields (e.g. `current_operator`) with no null guard. Will crash if the backend sends an incomplete state object. Add defensive checks throughout.

- **parallel_worker.py** - When `parent_node.code` is missing, worker silently falls back to a default kernel path, optimizing the wrong kernel. Should raise an explicit error instead.

- **parallel_worker.py** - `llm.chat()` has no timeout. One hung API call stalls the worker indefinitely. Add a timeout with retry and backoff.

- **backends/triton/remote_worker.py** - Leftover DEBUG print statements throughout; Triton backend appears untested end-to-end. Clean up before shipping.

---

## P2 - Polish (noticeable gaps, not outright broken)

- **WorkQueuePanel.cl.jac:L97-103** - Progress bar width never set for "Monitoring" step; `bar_w` stays "0%" - identical to a task that hasn't started. Add `if c_step == "Monitoring" { bar_w = "15%"; }`.

- **job_supervisor.jac:L1192** - `attempt_max` hardcoded to 8 for `[GEN]` tasks in `_write_initial_queue_state`. If `max_attempts` is overridden via `config.json`, the queue panel shows the wrong max until the first `update_queue_state` call.

- **WorkQueuePanel** - No confirmation dialog before clearing the queue. A single misclick discards in-progress jobs.

- **WorkQueuePanel** - Failed tasks show a red dot but fire no toast or alert. Easy to miss with many operators running simultaneously.

- **pipeline.py / tree_store.py** - No disk space check before starting a job. Users get a cryptic `OSError` mid-run. Check `shutil.disk_usage()` at job start and warn if free space is low.

- **backends/cuda/verifier.py** - `MAX_LLM_ANALYSIS = 1`: only the first failed IO entry gets LLM feedback. If the kernel fails on entry 2+, the user gets no explanation. Raise the limit or pick the most informative failure.

- **benchmarking/benchmark_ops.py:~L137-139** - `torch.load()` called on potentially large `.pt` files with no timeout. A slow or corrupted file hangs the job indefinitely.

- **All backends** - No dependency version compatibility check at startup. torch/triton/pycuda version mismatches produce cryptic import errors with no user guidance.

- **backends/cuda/profiler.py vs loader.py** - Device detection logic duplicated and divergent. "mps" vs "metal" naming also inconsistent. Centralize to a single source of truth.

- **core/generator.py:~L223** - Only failed prompts saved to `garbage_dump/`. Successful prompts never logged. Add logging for all generations so the best kernels can be audited.

- **Logging** - Inconsistent prefixes across modules (`[workflow]`, `[Generator]`, etc.). Standardize so logs can be reliably parsed and grepped.

- **workflow.py:~L446-457** - If operator N fails mid-generation, operators 0..N-1 are already committed to the DB with no rollback. Add a way to retry only the failed operators.

---

## P3 - Post v1.0

- **core/ssh_client.py** - No SSH keepalive or reconnect logic. Remote optimization hangs silently on network interruption.

- **settings.py** - Batch size, retry limit, and ancestry depth not user-configurable without editing source. Expose as CLI flags or a config file.

- **pipeline.py / tree_store.py** - No kernel cache size limit. The `kernels/` directory grows unboundedly. Add a configurable max with LRU eviction.

- **pipeline.py** - No per-iteration timing or ETA. Progress bar shows an absolute count only. Track time-per-iteration and display estimated completion.

- **Frontend** - No side-by-side kernel diff viewer. Users cannot see what changed between iterations to understand why a kernel improved or regressed.

- **Export logic** - No schema version field in exports. Future tool upgrades will silently misread old export files. Add a version field and a migration layer.

- **workflow.py** - `run_generate` scans the IO directory once at startup. Operators added mid-run are invisible until restart.

---

## Desktop (Tauri) - Open Issues

- **globals.jac** - `repo_root: str = path.abspath("..")` is evaluated at module load time using Python's CWD. If the sidecar inherits a CWD that doesn't resolve correctly, all project data paths break. Reliable fix: read `KFORGE_DATA_DIR` env var first; fall back to the relative path only if unset. Set `KFORGE_DATA_DIR` to an absolute path in `src-tauri/binaries/jac-sidecar.bat`.

- **Tauri window / sidecar startup race** - Existing projects sometimes don't appear on first load. The Tauri window likely fires the initial `GetProjects` request before the JAC sidecar is ready to serve. Add a retry/poll loop on the frontend for the initial data fetch, or confirm that the existing `JAC_SIDECAR_PORT` gate in `main.rs` reliably prevents the window from showing before the server is ready.
