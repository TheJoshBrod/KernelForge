"""
src/generator/monitor.py
Monitors and Preprocesses PyTorch Aten and CUDA Kernel Abstraction Layer Calls for Generator.
"""
import os
import torch

aten_output: str = ""
kernel_output: str = ""
_HAS_CUDA = torch.cuda.is_available()


def handle_trace(prof):
    """
    Display profiled events grouped by high-level ATen operators, 
    with the associated CUDA kernels underneath, preserving order.
    """
    global kernel_output
    global aten_output

    current_op = None
    for event in prof.events():
        if event.self_device_time_total == 0 and event.self_cpu_time_total == 0:
            continue  # skip events with no device time (profiler noise)

        if event.key.startswith("aten::"):
            # Start a new high-level op
            current_op = event.key
            aten_output += f"       [Op: {current_op}]\n"

        elif "ProfilerStep" not in event.key:
            # Low-level kernel; associate with current op if exists
            if current_op:
                kernel_output += f"     [Kernel: {event.key}]\n"
            else:
                # Kernel not associated with any high-level op
                kernel_output += f"     [Kernel: {event.key}]\n"


def profile_single_op(context: dict, full_exec_string: str) -> str:
    """
    Profiles a *single line* of PyTorch code
    and the formatted op_details string.

    Args:
        context (dict): The current execution context, for our purposes usually just imports pytorch.
        full_exec_string (str): The single line of code to execute (e.g., "c = torch.matmul(a, b)").

    Output:
        op_details (str): Formatted string of Aten/kernel info for the LLM.
    """
    target_device = os.environ.get("CGINS_TARGET_DEVICE", "").strip().lower()
    if not _HAS_CUDA or target_device == "cpu":
        return ""

    # 1. Reset global profiler strings for this specific op
    global aten_output, kernel_output
    aten_output = ""
    kernel_output = ""

    # 2. Set deterministic state
    torch.cuda.manual_seed(100)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 3. We'll run the profiler for just a few steps on this one op
    # (wait, warmup, active)
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1)

    # 4. Create a *copy* of the context to safely execute the line
    # This ensures we don't modify the main context if exec fails
    # (though main.py will update its own context upon success)
    temp_context = context.copy()

    # 6. Run the profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        on_trace_ready=handle_trace,
        record_shapes=True
    ) as p:
        for _ in range(3):  # Total steps: wait + warmup + active
            exec(full_exec_string, temp_context)
            p.step()

    # 8. Format op_details string for the LLM
    op_details = f"aten output:\n{aten_output}\n\n\nkernel output:\n{kernel_output}"
    return op_details
