"""Dashboard header KPI bar component."""

import from react { useState }
import from ....components.ui.Icons { CpuIcon }

def:pub DashboardHeader(props: any) -> Any {
    name = props.name;
    forgedCount = props.forgedCount;
    opsCount = props.opsCount;
    modelSpeedupRatio = props.modelSpeedupRatio;
    hardwareStatus = props.hardwareStatus;
    projectBackend = props.projectBackend;
    forgedTooltipPos = props.forgedTooltipPos;
    setForgedTooltipPos = props.setForgedTooltipPos;

    speedup_tip_state = useState(None);
    speedupTooltipPos = speedup_tip_state[0];
    setSpeedupTooltipPos = speedup_tip_state[1];

    return
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-[var(--surface-0)] border border-[var(--border)] rounded-lg p-4 shadow-sm">
            <div className="flex items-center gap-4">
                <div className="flex items-center justify-center w-10 h-10 rounded-md bg-zinc-900 border border-[var(--border)]">
                    <CpuIcon className="w-5 h-5 text-[var(--accent-orange)]"/>
                </div>
                <div>
                    <h2 className="text-xl font-semibold tracking-tight text-[var(--text-strong)]">
                        Project Dashboard
                    </h2>
                    <p className="text-[10px] uppercase tracking-wide text-[var(--text-muted)] mt-0.5">
                        Forge command center · {name}
                    </p>
                </div>
            </div>

            <div className="flex flex-wrap items-center gap-6 md:gap-8 pb-2 md:pb-0">
                <div
                    className="flex flex-col cursor-help"
                    onMouseMove={lambda e: any -> None { setForgedTooltipPos({"x": e.clientX, "y": e.clientY}); }}
                    onMouseLeave={lambda -> None { setForgedTooltipPos(None); }}
                >
                    <span className="text-[9px] uppercase tracking-[0.16em] text-zinc-500 mb-1">Coverage</span>
                    <div className="flex items-baseline">
                        <span className="text-xs font-bold text-zinc-200">{forgedCount}</span>
                        <span className="text-xs text-zinc-500">/ {opsCount} forged</span>
                    </div>
                    {<div
                        style={{"position": "fixed", "left": forgedTooltipPos["x"] + 14, "top": forgedTooltipPos["y"] + 14, "zIndex": 9999, "pointerEvents": "none"}}
                        className="bg-zinc-900 border border-zinc-700 rounded-md px-3 py-2 text-xs text-zinc-300 shadow-xl max-w-[220px] leading-relaxed"
                    >
                        Operators with a custom kernel that outperforms PyTorch's built-in implementation.
                    </div>
                    if forgedTooltipPos
                    else <span/>}
                </div>
                <div className="w-px h-8 bg-[var(--border)] hidden sm:block"></div>
                <div
                    className="flex flex-col cursor-help"
                    onMouseMove={lambda e: any -> None { setSpeedupTooltipPos({"x": e.clientX, "y": e.clientY}); }}
                    onMouseLeave={lambda -> None { setSpeedupTooltipPos(None); }}
                >
                    <span className="text-[9px] uppercase tracking-[0.16em] text-zinc-500 mb-1">Model Speedup</span>
                    {<span className="text-sm font-bold text-emerald-400">{modelSpeedupRatio.toFixed(2)}×</span>
                    if modelSpeedupRatio > 1.0
                    else <span className="text-sm font-bold text-zinc-500">—</span>}
                    {<div
                        style={{"position": "fixed", "left": speedupTooltipPos["x"] + 14, "top": speedupTooltipPos["y"] + 14, "zIndex": 9999, "pointerEvents": "none"}}
                        className="bg-zinc-900 border border-zinc-700 rounded-md px-3 py-2 text-xs text-zinc-300 shadow-xl max-w-[260px] leading-relaxed"
                    >
                        <div className="font-semibold text-zinc-100 mb-1">Model Speedup Ratio</div>
                        <div className="text-zinc-400 mb-2">Total op time with PyTorch ÷ estimated op time with forged kernels.</div>
                        <div className="font-mono text-[10px] text-zinc-400 bg-zinc-800 rounded px-2 py-1">total_pytorch_ms / (total_pytorch_ms − savings_ms)</div>
                        <div className="text-zinc-500 mt-2 text-[10px]">Only operators where the forged kernel outperforms PyTorch contribute to savings. All profiled ops are included in the denominator.</div>
                    </div>
                    if speedupTooltipPos
                    else <span/>}
                </div>
                <div className="w-px h-8 bg-[var(--border)] hidden sm:block"></div>
                <div className="flex flex-col">
                    <span className="text-[9px] uppercase tracking-[0.16em] text-zinc-500 mb-1">
                        Target & Backend
                    </span>
                    <span className="text-xs font-semibold text-zinc-300" title={hardwareStatus}>
                        {hardwareStatus} <span className="text-zinc-500 mx-1">|</span> <span className="text-indigo-400 capitalize">{projectBackend if projectBackend else "CUDA"}</span>
                    </span>
                </div>
            </div>
        </div>;
}
