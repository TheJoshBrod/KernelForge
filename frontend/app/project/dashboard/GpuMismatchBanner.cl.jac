"""GPU mismatch warning banner component."""

def:pub GpuMismatchBanner(
    benchmarkedGpuName: str,
    currentGpuName: str,
    onClone: any,
    onDismiss: any,
    cloneBusy: bool
) -> Any {
    return
        <div className="flex items-center gap-3 rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
            <svg
                className="h-4 w-4 shrink-0 text-amber-400"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
            >
                <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
                />
            </svg>
            <span className="flex-1">
                {f"Metrics were measured on {benchmarkedGpuName}. Current GPU is {currentGpuName}."}
            </span>
            <button
                onClick={onClone}
                disabled={cloneBusy}
                className="shrink-0 rounded border border-amber-500/50 bg-amber-500/20 px-3 py-1 text-xs font-medium text-amber-100 hover:bg-amber-500/30 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
                {<span className="flex items-center gap-1.5">
                    {<svg
                        className="h-3 w-3 animate-spin"
                        fill="none"
                        viewBox="0 0 24 24"
                    >
                        <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="4"
                        />
                        <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                        />
                    </svg> if cloneBusy else <span/>}
                    {f"Clone for {currentGpuName}"}
                </span>}
            </button>
            <button
                onClick={onDismiss}
                className="shrink-0 text-xs text-amber-400/70 hover:text-amber-300 transition-colors"
            >
                Dismiss
            </button>
        </div>;
}
