"""Benchmark log viewer modal component."""

import from ....components.ui.Modal { Modal }

def:pub BenchmarkLogModal(props: any) -> Any {
    isOpen = props.isOpen;
    name = props.name;
    onClose = props.onClose;
    benchLog = props.benchLog;
    benchLogLoading = props.benchLogLoading;
    benchLogCopied = props.benchLogCopied;
    onRefresh = props.onRefresh;
    onCopy = props.onCopy;

    return
        <Modal
            isOpen={isOpen}
            onClose={onClose}
            title="Benchmark log"
        >
            <div className="p-4 space-y-3">
                <div className="flex items-center justify-between gap-3">
                    <div className="text-[10px] uppercase tracking-wider font-bold text-zinc-500">
                        projects/{name}/logs/benchmark.log
                    </div>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={onRefresh}
                            className="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 rounded-md text-xs text-zinc-300 border border-zinc-700 transition-colors"
                        >
                            Refresh
                        </button>
                        <button
                            onClick={onCopy}
                            className="px-3 py-1.5 bg-[var(--accent-orange)] hover:bg-[var(--molten-gold-dark)] rounded-md text-xs text-zinc-950 border border-[var(--accent-orange)]/40 transition-colors"
                        >
                            {"Copied" if benchLogCopied else "Copy"}
                        </button>
                    </div>
                </div>
                {<div className="text-xs text-zinc-400">
                    Loading...
                </div>
                if benchLogLoading
                else <pre
                    className="text-xs font-mono text-zinc-200 whitespace-pre-wrap break-words bg-zinc-950 border border-zinc-800 rounded-lg p-3"
                >
                    {benchLog or "No log yet."}
                </pre>}
            </div>
        </Modal>;
}
