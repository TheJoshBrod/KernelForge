"""Dashboard header KPI bar component."""

import from ....components.ui.Icons { CpuIcon }

def:pub DashboardHeader(props: any) -> Any {
    name = props.name;
    forgedCount = props.forgedCount;
    opsCount = props.opsCount;
    modelSpeedupPct = props.modelSpeedupPct;
    hardwareStatus = props.hardwareStatus;
    forgedTooltipPos = props.forgedTooltipPos;
    setForgedTooltipPos = props.setForgedTooltipPos;

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
                    <div className="flex items-baseline gap-1.5">
                        <span className="text-sm font-bold text-zinc-200">{forgedCount}</span>
                        <span className="text-xs text-zinc-500">/ {opsCount} forged</span>
                    </div>
                </div>
                <div className="w-px h-8 bg-[var(--border)] hidden sm:block"></div>
                <div className="flex flex-col">
                    <span className="text-[9px] uppercase tracking-[0.16em] text-zinc-500 mb-1">Model Speedup</span>
                    {<span className="text-sm font-bold text-emerald-400">+{modelSpeedupPct.toFixed(1)}% faster</span>
                    if modelSpeedupPct > 0
                    else <span className="text-sm font-bold text-zinc-500">—</span>}
                </div>
                <div className="w-px h-8 bg-[var(--border)] hidden sm:block"></div>
                <div className="flex flex-col">
                    <span className="text-[9px] uppercase tracking-[0.16em] text-zinc-500 mb-1">
                        Selected GPU
                    </span>
                    <span className="text-xs font-semibold text-zinc-300" title={hardwareStatus}>{hardwareStatus}</span>
                </div>
            </div>

            {<div
                style={{"position": "fixed", "left": forgedTooltipPos["x"] + 14, "top": forgedTooltipPos["y"] + 14, "zIndex": 9999, "pointerEvents": "none"}}
                className="bg-zinc-900 border border-zinc-700 rounded-md px-3 py-2 text-xs text-zinc-300 shadow-xl max-w-[220px] leading-relaxed"
            >
                Operators with a custom kernel that outperforms PyTorch's built-in implementation.
            </div>
            if forgedTooltipPos
            else <span/>}
        </div>;
}
