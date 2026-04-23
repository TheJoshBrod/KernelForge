Kernel Forge Work Queue UI/UX Design
This document describes how Kernel Forge presents the progress of long-running GPU kernel generation and optimization jobs to the user. It covers the two main UI surfaces on the Dashboard: the Work Queue Panel and the Operator Results Table.

1. The Work Queue Panel
The Work Queue panel resides on the Dashboard and acts as the live "command center" for tracking background jobs.

1.1 Active Jobs (In Progress)
When a user launches a generation or optimization job (e.g., clicking "Generate" on an operator), the panel populates with live tasks.

Granular Status Bars: Each executing task shows a progress bar indicating its current phase.
Generation phases: [Monitoring] → [Generating] → [Verifying] → [Benchmarking]
Optimization phases: [Generating] → [Validating] → [Profiling]
Retry Counters: If the LLM generates a kernel that fails to compile, the UI updates to show the repair attempt (e.g., "attempt 2/8"). This assures the user the system is automatically trying to fix the issue.
GPU Lock Indicator: Benchmarking and profiling require exclusive GPU access. The header displays "GPU Slot: <operator>" to let the user know exactly which task is currently locking the hardware.
1.2 Completed Jobs History
As backend workers finish tasks (either succeeding or failing), they are removed from the active view and appended to the "Completed" section at the bottom of the panel.

Operator Grouping: Since MCTS optimization might test 50+ kernels for a single operator, the completed list collapses them by operator name to prevent UI clutter.
Best Result Highlighting: Next to each operator in the completed list, the UI extracts the best performing kernel from that run (e.g., "Kernel 14 · 0.12ms").
Success/Failure Indicators: The left border of the row changes color so the user can gauge outcomes at a glance:
Green: The best kernel outperformed the PyTorch baseline.
Yellow: Kernels were generated/benchmarked successfully, but none beat PyTorch.
Red: All generation/optimization attempts failed.
2. Operator Results Table
While the Work Queue focuses on live execution, the Operator Results Table provides a holistic, static ledger of the state of every operator discovered in the user's project. The Status column is the most important field here:

Priority Cascade for the "Status" Column
Live Activity Display: If the Work Queue is currently processing an operator, the table's status column mirrors the queue. It will display the active step (e.g., "Generating", "Profiling", or "Validating") so the user immediately knows the operator is being worked on.
"Forged ✓" (Green): The ultimate goal for an operator. This status appears if a custom kernel has been successfully generated, benchmarked, and the resulting speedup strictly beats the PyTorch baseline (speedup > 1.001x).
"Benchmarked" (Yellow): Appears if the system has generated and profiled a kernel, but the kernel was slower than (or identical to) the PyTorch baseline. This acts as a signal to the user: The pipeline works, but you probably need to run an Optimization job on this operator to get a speedup.
"Not started" (Gray text): The default state when an operator is discovered by the profiler but no generation job has been triggered.
Actions
The action button on the far right of the table changes dynamically based on the operator's progress:

If the operator is "Not started", the button offers "Generate".
If it has already been "Benchmarked" or "Forged ✓", the button seamlessly switches to "Optimize" to encourage further MCTS tuning.
