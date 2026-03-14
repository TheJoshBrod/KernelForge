"""
Main pipeline for CUDA kernel generation and validation.

Walks through each operation in a benchmark to:
1. Monitor kernel and ATen calls
2. Generate CUDA kernel code
3. Validate correctness through iterative refinement
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from src.config import ensure_llm_config, load_project_config

ensure_llm_config()

import torch
from tqdm import tqdm

from src.llm_tools import GenModel
import src.generator.generator as generator
import src.generator.monitor as monitor
import src.generator.prompts.prompts as prompts
import src.generator.templates as templates
from src.optimizer.backends.cuda import CUDABackend
from src.optimizer.backends.triton import TritonBackend
from src.optimizer.pipeline import update_queue_state
from src.progress import update_job_progress, wait_if_paused, check_cancelled
try:
    from src import codex_utils
except ImportError:
    codex_utils = None  # codex is optional


def _is_triton() -> bool:
    return os.environ.get("KFORGE_TARGET_DEVICE", "").strip().lower() == "triton"


def _kernel_ext() -> str:
    return ".py" if _is_triton() else ".cu"


def _kernel_filename() -> str:
    return f"kernel{_kernel_ext()}"


def _success_filename() -> str:
    """Backend-specific success marker, e.g. 'success.cuda', 'success.triton'."""
    device = os.environ.get("KFORGE_TARGET_DEVICE", "cuda").strip().lower()
    return f"success.{device}"


def _validate_kernel(cu_code, entry_file, log_file_loc, tmpdir, ssh_config=None):
    """Route to the unified backend verifier."""
    # Derive io_dir from entry_file path
    io_dir = Path(entry_file).parent
    paths = {
        "io_dir": io_dir,
        "tmp_dir": Path(tmpdir),
    }

    if _is_triton():
        backend = TritonBackend()
    else:
        # Default to CUDA
        backend = CUDABackend()

    # Backend validates all files in io_dir
    success, log_msg = backend.validate_kernel(cu_code, paths, ssh_config=ssh_config)

    # Decode success flags from log message for compatibility with generator logic
    # Generator expects: (call_success, exec_success, feedback)
    call_success = True
    if "[Compilation Failed]" in log_msg or "[Import/Compilation Failed]" in log_msg:
        call_success = False
    
    exec_success = success
    if not success and call_success:
        # If compilation passed but Overall success is False, then it's a runtime/mismatch error
        exec_success = False
        
    # Write log file manually as we didn't pass it to validate_kernel (it just returns string)
    try:
        with open(log_file_loc, "w") as f:
            f.write(log_msg)
    except Exception:
        pass

    return call_success, exec_success, log_msg

# Configuration
MAX_ATTEMPTS = int(os.environ.get("KFORGE_MAX_ATTEMPTS", "8"))
OUTPUT_BASE_DIR = Path("kernels/generated")


def _normalize_op_name(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


def _find_project_dir(io_dir: Path) -> Path | None:
    for parent in [io_dir] + list(io_dir.parents):
        if (parent / "config.json").exists() and (parent / "model.py").exists():
            return parent
    return None


def _load_generator_config(project_dir: Path | None) -> dict:
    if not project_dir:
        return {}
    config_path = project_dir / "config.json"
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    cfg = data.get("generator") if isinstance(data, dict) else None
    return cfg if isinstance(cfg, dict) else {}


def _op_set(names) -> set[str]:
    out: set[str] = set()
    if not names:
        return out
    for name in names:
        if not name:
            continue
        out.add(_normalize_op_name(str(name)))
    return out


def _bool_env(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _codex_model() -> str | None:
    return os.environ.get("KFORGE_CODEX_MODEL") or os.environ.get("OPENAI_MODEL")


def _codex_sandbox() -> str:
    return os.environ.get("KFORGE_CODEX_SANDBOX", "workspace-write")


def _codex_attempts(default: int = 3) -> int:
    try:
        return int(os.environ.get("KFORGE_CODEX_MAX_ATTEMPTS", str(default)))
    except Exception:
        return default


def _build_codex_prompt(
    base_prompt: str,
    function_name: str,
    op_key: str,
    project_dir: Path | None,
    *,
    feedback: str | None = None,
    mode: str = "generate",
) -> str:
    kf = _kernel_filename()
    if _is_triton():
        sig_hint = "Keep the launch(...) signature exactly unchanged."
    else:
        sig_hint = "Keep the torch::Tensor launch(...) signature exactly unchanged."
    parts = [
        f"Mode: {mode}",
        f"Edit {kf} only.",
        sig_hint,
    ]
    if project_dir:
        parts.append(
            "After edits, run: "
            f"python -m src.optimizer.pipeline kernels/projects/{project_dir.name}/io/individual_ops {project_dir.name} --op {op_key}"
        )
    if feedback:
        parts.append("Previous validation error:\n" + feedback)
    parts.append(
        f"Specification (use as the source of truth; do NOT output tags, just edit {kf}):\n"
        + base_prompt
    )
    return "\n\n".join(parts)


def _load_op_counts(io_dir: Path) -> dict:
    summary_path = io_dir.parent / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    counts = data.get("op_counts") if isinstance(data, dict) else None
    if not isinstance(counts, dict):
        return {}
    return {_normalize_op_name(k): int(v) for k, v in counts.items()}


def _write_failure_report(
    op_dir: Path,
    stage: str,
    message: str,
    extra: dict | None = None,
) -> None:
    report = {
        "stage": stage,
        "message": message,
    }
    if extra:
        report.update(extra)
    attempts_dir = op_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    report_path = attempts_dir / "failure.json"
    try:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception:
        pass


def _validate_static_kernel(
    cu_code: str, entry_files: list[str], op_dir: Path, tag: str, ssh_config=None
) -> tuple[bool, str]:
    kernel_dir = op_dir / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)

    for i, entry_file in enumerate(entry_files):
        if not wait_if_paused():
            return False, "Generation cancelled."
        if check_cancelled():
            return False, "Generation cancelled."

        tmpdir = tempfile.mkdtemp(prefix="gins_verifier_")
        log_file_loc = op_dir / "attempts" / f"log-{tag}-{i}.txt"
        os.makedirs(log_file_loc.parent, exist_ok=True)

        call_success, exec_success, feedback = _validate_kernel(
            cu_code, entry_file, log_file_loc, tmpdir, ssh_config=ssh_config
        )

        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)

        ext = _kernel_ext()
        if not (call_success and exec_success):
            with open(kernel_dir / f"kernel-{tag}-{i}{ext}", "w") as f:
                f.write(cu_code)
            with open(op_dir / "attempts" / f"feedback-{tag}-{i}.txt", "w") as f:
                f.write(feedback)
            return False, feedback

    with open(op_dir / _kernel_filename(), "w", encoding="utf-8") as f:
        f.write(cu_code)
    with open(kernel_dir / f"kernel-{tag}-g{ext}", "w") as f:
        f.write(cu_code)
    return True, ""


def validate_with_retries(
    output_dir: Path,
    entry_files: list[str],
    gen_model: GenModel,
    initial_prompt: str,
    function_name: str,
    *,
    op_key: str,
    project_dir: Path | None,
    template: str | None,
    ssh_config=None,
    _proj_base_dir: Path | None = None,
    _task_key: str | None = None,
) -> tuple[bool, str, str]:
    """
    Attempt to validate and fix kernel code up to MAX_ATTEMPTS times.

    Args:
        output_dir: Directory to save kernel outputs
        entry_files: List of paths to entry_*.pt files containing inputs/outputs
        gen_model: GenModel instance with system prompt and history
        initial_prompt: The initial prompt for generation

    Returns:
        bool: is the final kernel successful
    """

    # Iterative kernel folder
    kernel_dir = output_dir / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)

    # Try n times to go through entire test suite
    try:
        max_attempts = int(os.environ.get("KFORGE_MAX_ATTEMPTS", str(MAX_ATTEMPTS)))
    except Exception:
        max_attempts = MAX_ATTEMPTS
    last_feedback = ""
    last_stage = "llm"

    use_codex_generate = codex_utils.op_enabled(
        "KFORGE_CODEX_GENERATE", "KFORGE_CODEX_GENERATE_OPS", op_key
    ) if codex_utils else False
    use_codex_repair = codex_utils.op_enabled(
        "KFORGE_CODEX_REPAIR", "KFORGE_CODEX_REPAIR_OPS", op_key
    ) if codex_utils else False
    codex_work_dir = output_dir / "work"
    codex_seed = template or ("# TODO: implement kernel.py\n" if _is_triton() else "// TODO: implement kernel.cu\n")
    codex_max_attempts = _codex_attempts()
    codex_model = _codex_model()
    codex_sandbox = _codex_sandbox()

    # Get model name from config for normal LLM generation
    llm_provider = ensure_llm_config()
    llm_model = ""
    if llm_provider == "openai":
        llm_model = os.environ.get("OPENAI_MODEL", "gpt-5")
    elif llm_provider == "anthropic":
        llm_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    elif llm_provider == "google" or llm_provider == "gemini":
        llm_model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    
    attempts_total = codex_max_attempts if use_codex_generate else max_attempts
    task_meta = {"tag": "[GEN]", "op_name": op_key}

    def _run_codex_repair(feedback: str, attempt_idx: int) -> tuple[bool, str, str]:
        # Note: This still uses base_prompt logic which isn't carried here perfectly
        # For now, we assume codex uses its own prompt construction.
        # But we need base_prompt for Codex...
        # Let's extract base_prompt from initial_prompt for now
        base_prompt = initial_prompt 
        
        repair_prompt = _build_codex_prompt(
            base_prompt,
            function_name,
            op_key,
            project_dir,
            feedback=feedback,
            mode="repair",
        )
        codex_utils.prepare_work_dir(codex_work_dir, cu_code, task=repair_prompt, filename=_kernel_filename())
        ok, err = codex_utils.run_codex(
            codex_work_dir, repair_prompt, model=codex_model, sandbox=codex_sandbox
        )
        if ok:
            try:
                repaired = (codex_work_dir / _kernel_filename()).read_text(encoding="utf-8")
            except Exception as exc:
                repaired = ""
                err = f"Codex repair read failed: {exc}"
            if repaired:
                # Re-validate repaired kernel from scratch
                repair_tmpdir = tempfile.mkdtemp(prefix="gins_verifier_")
                repaired_ok = True
                for j, entry_file in enumerate(tqdm(entry_files, desc="Repair Tests")):
                    log_file_loc = output_dir / "attempts" / f"log-repair-{attempt_idx}-{j}.txt"
                    os.makedirs(log_file_loc.parent, exist_ok=True)
                    call_success, exec_success, repair_feedback = _validate_kernel(
                        repaired, entry_file, log_file_loc, repair_tmpdir, ssh_config=ssh_config
                    )
                    repaired_ok = repaired_ok and call_success and exec_success
                    if not repaired_ok:
                        with open(kernel_dir / f"kernel-repair-{attempt_idx}-{j}{_kernel_ext()}", "w") as f:
                            f.write(repaired)
                        return False, repair_feedback, "codex_repair"
                if os.path.exists(repair_tmpdir):
                    shutil.rmtree(repair_tmpdir)
                if repaired_ok:
                    with open(kernel_dir / f"kernel-repair-{attempt_idx}-g{_kernel_ext()}", "w") as f:
                        f.write(repaired)
                    with open(output_dir / _kernel_filename(), "w", encoding="utf-8") as f:
                        f.write(repaired)
                    return True, "", "success"
        return False, err or feedback, "codex_repair"

    for attempt in range(attempts_total):
        if not wait_if_paused():
            return False, "Generation paused/cancelled.", "control"
        if check_cancelled():
            return False, "Generation cancelled.", "control"

        if _proj_base_dir and _task_key:
            update_queue_state(_proj_base_dir, {"active_tasks": {_task_key: {
                **task_meta,
                "current_step": "Generating",
                "attempt_current": attempt + 1,
                "attempt_max": max_attempts,
                "op_name": op_key,
                "tag": "[GEN]",
            }}})

        # Generate kernel
        if use_codex_generate:
            prompt = _build_codex_prompt(
                initial_prompt, # Use initial prompt as base
                function_name,
                op_key,
                project_dir,
                feedback=last_feedback if attempt > 0 else None,
                mode="generate",
            )
            codex_utils.prepare_work_dir(codex_work_dir, codex_seed, task=prompt, filename=_kernel_filename())
            ok, err = codex_utils.run_codex(
                codex_work_dir, prompt, model=codex_model, sandbox=codex_sandbox
            )
            if not ok:
                last_feedback = err or "Codex generation failed."
                last_stage = "codex_generate"
                continue
            try:
                cu_code = (codex_work_dir / _kernel_filename()).read_text(encoding="utf-8")
                codex_seed = cu_code
            except Exception as e:
                last_feedback = f"Codex output read failed: {e}"
                last_stage = "codex_generate"
                continue
        else:
            if not llm_model:
                return False, f"Missing configuration for LLM provider {llm_provider}.", "llm_api"
            
            try:
                # FIRST ATTEMPT: Use initial_prompt
                if attempt == 0:
                    current_prompt = initial_prompt
                # SUBSEQUENT ATTEMPTS: Use repair prompt (handled in loop end) which is added to history
                # But here we just call generate which uses history
                # However, generator.generate takes a msg. 
                # If we already added to history, we might pass empty msg? 
                # Let's adjust logic: 
                # On attempt 0, we pass initial_prompt.
                # On attempt > 0, we've already appended USER message with feedback to history (see end of loop).
                # But generator.generate takes a prompt and calls chat().
                # GenModel.chat() appends user message.
                # So we should pass the prompt we want to send.
                # BUT wait, the loop logic below appends to `conversation_history` list in the OLD code.
                # In the NEW code, we need to pass the prompt to `generator.generate`.
                
                # Logic:
                # If attempt 0: prompt = initial_prompt
                # If attempt > 0: prompt has been determined at end of previous loop (repair prompt)
                
                prompt_to_send = initial_prompt
                if attempt > 0:
                    # Previous loop failure logic should have set prompt_to_send (repair msg)
                    # Use a variable `next_prompt`?
                    # Let's refactor the loop slightly.
                    pass
                
                # Actually, simpler:
                # We need a `current_prompt` variable.
                if attempt == 0:
                     msg = initial_prompt
                else:
                    # In previous iteration failure block, we generated the repair prompt
                    # We need to make sure we use it here.
                    # See `repair` variable below.
                    msg = last_repair_prompt
                
                cu_code = generator.generate(gen_model, msg, llm_model)

            except Exception as e:
                print(f"Failed on attempt {attempt}\n{e}")
                last_feedback = f"LLM generation failed: {e}"
                last_stage = "llm_api"
                return False, last_feedback, last_stage

        tmpdir = tempfile.mkdtemp(prefix="gins_verifier_")

        # Output newest version of kernel
        with open(output_dir / _kernel_filename(), "w", encoding="utf-8") as f:
            f.write(cu_code)

        # For each generated kernel validate ALL input/output
        # Unified Validation (Batch)
        log_file_loc = output_dir / "attempts" / f"log-{attempt}.txt"
        os.makedirs(log_file_loc.parent, exist_ok=True)

        if not entry_files:
             return False, "No entry files", "setup"

        update_job_progress(attempt, max_attempts, f"Validating {op_key} (attempt {attempt + 1}/{max_attempts}){' — remote' if ssh_config else ''}")

        if _proj_base_dir and _task_key:
            update_queue_state(_proj_base_dir, {"active_tasks": {_task_key: {
                **task_meta,
                "current_step": "Validating",
                "op_name": op_key,
                "tag": "[GEN]",
            }}})

        # Validate all inputs at once using backend
        call_success, exec_success, feedback = _validate_kernel(
            cu_code, entry_files[0], log_file_loc, tmpdir, ssh_config=ssh_config
        )

        print(feedback)

        is_valid = call_success and exec_success
        if not is_valid:
            # Save failed kernel
            with open(kernel_dir / f"kernel-{attempt}-failed{_kernel_ext()}", "w") as f:
                f.write(cu_code)

            last_feedback = feedback
            last_stage = "llm_validate"

            # Optional Codex/Repair logic
            if use_codex_repair:
                try:
                    repaired_ok, repair_feedback, repair_stage = _run_codex_repair(
                        feedback, attempt
                    )
                    if repaired_ok:
                        return True, "", "success"
                    last_feedback = repair_feedback
                    last_stage = repair_stage
                except NameError:
                    pass

            update_job_progress(attempt, max_attempts, f"Repairing {op_key} (attempt {attempt + 1}/{max_attempts})")
            last_repair_prompt = prompts.get_repair_prompt(
                function_name=function_name,
                attempt=attempt,
                feedback=last_feedback,
            )

        # Delete tmp directory before next generation
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)

        # If all testcases passed, escape
        if is_valid:
            print(f"SUCCESSFUL on {attempt + 1}")
            # Save kernel
            with open(kernel_dir / f"kernel-{attempt}-g{_kernel_ext()}", "w") as f:
                f.write(cu_code)

            if _proj_base_dir and _task_key:
                update_queue_state(_proj_base_dir, {"active_tasks": {_task_key: {
                    **task_meta,
                    "current_step": "Done",
                    "status": "Done",
                    "op_name": op_key,
                    "tag": "[GEN]",
                }}})
            return True, "", "success"

    if _proj_base_dir and _task_key:
        result_msg = f"{last_stage}: {last_feedback[:120]}" if last_feedback else last_stage
        update_queue_state(_proj_base_dir, {"active_tasks": {_task_key: {
            **task_meta,
            "current_step": "Failed",
            "status": "Failed",
            "result": result_msg,
            "op_name": op_key,
            "tag": "[GEN]",
        }}})
    return False, last_feedback, last_stage


def process_function(
    directory_name: str,
    entry_files: list[str],
    op_dir: Path,
    *,
    use_baseline: bool = True,
    baseline_as_template: bool = True,
    project_dir: Path | None = None,
    ssh_config=None,
):
    """
    Process all profiled calls for a given function.

    Args:
        directory_name: Name of the directory that is based on the PyTorch API function (e.g. "torch-nn-functional-relu" -> "torch.nn.functional.relu")
        entry_files: List of paths to entry_*.pt files
        op_dir: Output directory for this operation
    """

    # Load first call to set up context for profiling
    first_call = torch.load(
        entry_files[0], map_location='cpu', weights_only=False)
    first_args = first_call.get("args", [])
    first_kwargs = first_call.get("kwargs", {})

    # Extract function name out of directory name
    function_name = first_call.get("function_name")
    if not function_name:
        print(f"Skipping {directory_name}: no function_name stored")
        return False

    context = {
        "torch": torch,
        "F": torch.nn.functional,
        "args": first_args,
        "kwargs": first_kwargs,
    }
    print(function_name)
    exec_str = f"{function_name}(*args, **kwargs)"

    # Set up GenModel
    sys_prompt = prompts.get_system_prompt()
    gen_model = GenModel(sys_prompt)

    # Profile operation
    try:
        op_details = monitor.profile_single_op(context, exec_str)
    except Exception as e:
        print(e)
        return False

    # Load all calls for prompt generation
    call_list = []
    for entry_file in entry_files:
        try:
            entry = torch.load(
                entry_file, map_location='cpu', weights_only=False)
            call_list.append(entry)
        except Exception as e:
            print(f"Error loading {entry_file}: {e}")
            continue

    if not call_list:
        print(f"Failed to load any entries for {function_name}")
        return False

    template = None
    if baseline_as_template:
        template = templates.template_for_prompt(function_name)
    prompt = prompts.generate_full_llm_prompt(
        call_list, function_name, op_details, template=template
    )

    call_list.clear()

    op_key = _normalize_op_name(function_name)
    task_key = "gen_" + op_key
    proj_base_dir = project_dir if project_dir else None

    # Validate loop - pass entry files directly
    success, failure_msg, failure_stage = validate_with_retries(
        op_dir,
        entry_files,
        gen_model,
        prompt,
        function_name,
        op_key=op_key,
        project_dir=project_dir,
        template=template,
        ssh_config=ssh_config,
        _proj_base_dir=proj_base_dir,
        _task_key=task_key,
    )

    # Track performance
    if success:
        success_file = op_dir / _success_filename()
        with open(success_file, "w") as f:
            f.write("passed")
    else:
        _write_failure_report(
            op_dir,
            failure_stage,
            failure_msg or "Kernel validation failed.",
            {"function": function_name},
        )

    return success


def main():
    """Main entry point: load benchmarks and process each one."""
    parser = argparse.ArgumentParser(description="Generate kernels from profiled ops.")
    parser.add_argument("io_dir", nargs="?", help="Path to profiled ops directory")
    parser.add_argument("--io-dir", dest="io_dir_opt", default=None, help="Path to profiled ops directory")
    parser.add_argument("--out-dir", dest="out_dir", default=None, help="Base output directory for kernels")
    parser.add_argument(
        "--only-ops",
        dest="only_ops",
        default=None,
        help="Comma-separated op names to generate (overrides config only_ops)",
    )
    parser.add_argument(
        "--skip-ops",
        dest="skip_ops",
        default=None,
        help="Comma-separated op names to skip (extends config skip_ops)",
    )
    parser.add_argument(
        "--remote",
        default="",
        help="Path to SSH config JSON for remote validation/profiling",
    )
    args = parser.parse_args()

    io_dir = args.io_dir_opt or args.io_dir
    if not io_dir:
        print("Usage: python -m src.generator.main <io_dir> [--out-dir <dir>]")
        sys.exit(1)

    ssh_config = None
    if args.remote:
        try:
            with open(args.remote, "r") as f:
                remote_cfg = json.load(f)
            connections = remote_cfg.get("ssh_connections", [])
            active_idx = remote_cfg.get("active_ssh_index", -1)
            if 0 <= active_idx < len(connections):
                ssh_config = connections[active_idx]
                print(f"Remote mode: Using SSH connection to {ssh_config.get('host')}")
            else:
                print(f"Warning: invalid active_ssh_index {active_idx} in {args.remote}")
        except Exception as e:
            print(f"Warning: failed to load SSH config from {args.remote}: {e}")

    global OUTPUT_BASE_DIR
    if args.out_dir:
        OUTPUT_BASE_DIR = Path(args.out_dir)

    project_dir = _find_project_dir(Path(io_dir))
    
    # Use src.config to handle LLM configuration from project config
    if project_dir:
        config_path = project_dir / "config.json"
        if config_path.exists():
            os.environ["KFORGE_CONFIG_PATH"] = str(config_path)
            # Reload config to apply project-specific settings
            ensure_llm_config()

    # Load generator specific config manually for other settings (skip_ops, etc)
    project_cfg = load_project_config(project_dir)
    gen_cfg = project_cfg.get("generator", {})

    target_device = os.environ.get("KFORGE_TARGET_DEVICE", "").strip().lower()
    if not target_device:
        cfg_device = str(gen_cfg.get("target_device", "")).strip().lower() if isinstance(gen_cfg, dict) else ""
        target_device = cfg_device
    if target_device in {"gpu", "cuda"}:
        target_device = "cuda"
    elif target_device == "triton":
        target_device = "triton"
    elif target_device == "mps":
        target_device = "mps"
    elif target_device == "cpu":
        target_device = "cpu"
    else:
        target_device = "cuda"
    if target_device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU target for generation.")
        target_device = "cpu"
    if target_device == "triton" and not torch.cuda.is_available():
        print("CUDA not available (needed for Triton); falling back to CPU target.")
        target_device = "cpu"
    if target_device == "mps":
        if not (hasattr(torch, "backends") and torch.backends.mps.is_available()):
            print("MPS not available; falling back to CPU target for generation.")
            target_device = "cpu"
    os.environ["KFORGE_TARGET_DEVICE"] = target_device
    skip_ops = _op_set(gen_cfg.get("skip_ops"))
    only_ops = _op_set(gen_cfg.get("only_ops"))
    if args.only_ops:
        only_ops = _op_set(str(args.only_ops).split(","))
    if args.skip_ops:
        skip_ops = skip_ops.union(_op_set(str(args.skip_ops).split(",")))
    max_ops = gen_cfg.get("max_ops")
    use_baseline = bool(gen_cfg.get("use_baseline_kernels", True))
    baseline_as_template = bool(gen_cfg.get("use_baseline_as_template", True))
    env_use_baseline = _bool_env("KFORGE_USE_BASELINE_KERNELS")
    if env_use_baseline is not None:
        use_baseline = env_use_baseline
    env_baseline_template = _bool_env("KFORGE_USE_BASELINE_TEMPLATE")
    if env_baseline_template is not None:
        baseline_as_template = env_baseline_template
    if target_device in {"cpu", "mps"}:
        use_baseline = False
        baseline_as_template = False
    if "extra_validation_cases" in gen_cfg:
        os.environ["KFORGE_EXTRA_VALIDATION_CASES"] = str(
            gen_cfg.get("extra_validation_cases")
        )
    if "max_attempts" in gen_cfg:
        os.environ["KFORGE_MAX_ATTEMPTS"] = str(gen_cfg.get("max_attempts"))

    # Loop over all function directories
    function_dirs = sorted(glob.glob(os.path.join(io_dir, "*")))
    jobs: list[tuple[str, list[str], str, str]] = []

    for func_dir in function_dirs:
        if not os.path.isdir(func_dir):
            continue

        function_name = os.path.basename(func_dir).replace("_", ".")
        op_key = _normalize_op_name(function_name)
        if skip_ops and op_key in skip_ops:
            continue
        if only_ops and op_key not in only_ops:
            continue
        entry_files = sorted(glob.glob(os.path.join(func_dir, "entry_*.pt")))
        if not entry_files:
            continue
        jobs.append((func_dir, entry_files, function_name, op_key))

    if max_ops:
        try:
            max_ops = int(max_ops)
        except Exception:
            max_ops = None
    if max_ops:
        op_counts = _load_op_counts(Path(io_dir))
        jobs.sort(key=lambda j: op_counts.get(j[3], 0), reverse=True)
        jobs = jobs[:max_ops]

    total_jobs = len(jobs)
    update_job_progress(0, total_jobs, "Starting generation")

    completed = 0
    # Track failed task keys locally to remove them at the start of the next
    # op iteration — avoids reading queue.json without a lock (M5).
    failed_task_keys: list[str] = []
    for job_idx, (func_dir, entry_files, function_name, op_key) in enumerate(tqdm(
        jobs, desc="Processing functions"
    )):
        if not wait_if_paused():
            print("Generation cancelled.")
            return
        if check_cancelled():
            print("Generation cancelled.")
            return
        update_job_progress(completed, total_jobs, f"{function_name} ({completed + 1}/{total_jobs})")
        print(function_name)

        op_dir = OUTPUT_BASE_DIR / "individual_op_kernels" / op_key
        op_dir.mkdir(parents=True, exist_ok=True)

        if project_dir:
            # Remove stale "Failed" entries from the previous op (M5 — no lock-free read).
            queue_update: dict = {}
            if failed_task_keys:
                update_queue_state(project_dir, {"remove_tasks": failed_task_keys})
                failed_task_keys = []
            # Advance current_operator; update pending_operators only in multi-op
            # standalone mode so we don't overwrite workflow.py's correct state when
            # main.py is called as a single-op subprocess (C3/C4).
            if len(jobs) > 1:
                remaining_keys = [j[3] for j in jobs[job_idx + 1:]]
                update_queue_state(project_dir, {
                    "current_operator": op_key,
                    "pending_operators": remaining_keys,
                })
            else:
                update_queue_state(project_dir, {"current_operator": op_key})

        performance_file = op_dir / _success_filename()
        if performance_file.exists():
            completed += 1
            update_job_progress(completed, total_jobs, function_name)
            # Already done in a prior run — remove from active_tasks silently (M3).
            if project_dir:
                update_queue_state(project_dir, {"remove_tasks": ["gen_" + op_key]})
            continue

        if skip_ops and op_key in skip_ops:
            skip_file = op_dir / "skip.json"
            skip_file.write_text(
                json.dumps(
                    {"op": function_name, "reason": "skip_ops list"},
                    indent=2,
                ),
                encoding="utf-8",
            )
            completed += 1
            update_job_progress(completed, total_jobs, function_name)
            if project_dir:
                update_queue_state(project_dir, {"remove_tasks": ["gen_" + op_key]})
            continue

        baseline_code = None
        baseline_ok = False
        if use_baseline and templates.has_baseline_kernel(function_name):
            baseline_path = templates.baseline_kernel_path(function_name)
            # Skip baselines that don't match the target backend's extension
            if baseline_path.suffix != _kernel_ext():
                print(f"  Skipping {baseline_path.suffix} baseline (target is {_kernel_ext()}): {function_name}")
            else:
                baseline_code = templates.load_baseline_kernel(function_name)
                if baseline_code:
                    baseline_ok, baseline_error = _validate_static_kernel(
                        baseline_code, entry_files, op_dir, "baseline", ssh_config=ssh_config
                    )
                    if not baseline_ok:
                        _write_failure_report(
                            op_dir,
                            "baseline",
                            baseline_error or "Baseline kernel failed validation.",
                            {"function": function_name},
                        )

        codex_enabled = codex_utils.op_enabled(
            "KFORGE_CODEX_GENERATE", "KFORGE_CODEX_GENERATE_OPS", op_key
        ) if codex_utils else False

        if baseline_ok and not codex_enabled:
            success_file = op_dir / _success_filename()
            with open(success_file, "w") as f:
                f.write("passed")
            completed += 1
            update_job_progress(completed, total_jobs, function_name)
            if project_dir:
                update_queue_state(project_dir, {"remove_tasks": ["gen_" + op_key]})
            continue

        success = process_function(
            function_name,
            entry_files,
            op_dir,
            use_baseline=use_baseline,
            baseline_as_template=baseline_as_template,
            project_dir=project_dir,
            ssh_config=ssh_config,
        )
        if not success and baseline_ok:
            if baseline_code:
                with open(op_dir / _kernel_filename(), "w", encoding="utf-8") as f:
                    f.write(baseline_code)
            success_file = op_dir / _success_filename()
            with open(success_file, "w") as f:
                f.write("passed")
            try:
                fallback_note = op_dir / "attempts" / "fallback.json"
                fallback_note.write_text(
                    json.dumps(
                        {
                            "function": function_name,
                            "reason": "codex_failed_fallback_to_baseline",
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass
        elif not success:
            # Track for cleanup at the start of the next op (M5).
            failed_task_keys.append("gen_" + op_key)
        if check_cancelled():
            print("Generation cancelled.")
            return
        completed += 1
        update_job_progress(completed, total_jobs, function_name)

    if project_dir:
        # Remove any remaining failed tasks and reset operator tracking.
        if failed_task_keys:
            update_queue_state(project_dir, {"remove_tasks": failed_task_keys})
        update_queue_state(project_dir, {
            "pending_operators": [],
            "current_operator": "",
        })


if __name__ == "__main__":
    main()
