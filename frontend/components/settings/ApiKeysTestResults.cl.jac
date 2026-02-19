import from react { useState }

def:pub ApiKeysTestResults(props: any) -> Any {
    testResults = props.testResults;
    onClose = props.onClose;

    if not testResults {
        return None;
    }

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animation-fade-in text-white">
            <div className="bg-[var(--bg-panel)] border border-[var(--border)] rounded-sm shadow-2xl w-full max-w-xl max-h-[80vh] flex flex-col clip-anvil relative">
                <div className="p-4 border-b border-[var(--border)] pr-12 relative flex items-center justify-between shrink-0">
                    <div className="absolute top-0 left-0 right-0 heat-divider"></div>
                    <h3 className="text-sm font-bold text-white industrial-font-display uppercase tracking-wider">
                        Test Results
                    </h3>
                    <button
                        onClick={onClose}
                        className="text-[var(--text-muted)] hover:text-white transition-colors p-1"
                    >
                        ✕
                    </button>
                </div>
                <div className="p-4 bg-[var(--bg-darker)]/30 overflow-y-auto charcoal-scrollbar">
                    <div
                        className={"p-4 rounded-sm text-sm " + (
                            "bg-green-900/20 border border-green-800/50 text-green-300 shadow-[inset_0_0_15px_rgba(34,197,94,0.1)]"
                            if testResults["success"]
                            else "bg-red-900/20 border border-red-800/50 text-red-300 shadow-[inset_0_0_15px_rgba(239,68,68,0.1)]"
                        )}
                    >
                        <div className="flex items-center gap-2 mb-3 pb-2 border-b border-[var(--border)]/50">
                            <span
                                className={(
                                    "text-green-500"
                                    if testResults["success"]
                                    else "text-red-500"
                                )}
                            >
                                {"●" if testResults["success"] else "✖"}
                            </span>
                            <span className="font-bold tracking-wide uppercase text-xs">
                                {"All Tests Passed"
                                if testResults["success"]
                                else "Tests Failed"}
                            </span>
                        </div>
                        {<div className="space-y-2">
                            {Object.entries(testResults["results"]).map(
                                lambda entry: any  -> any { key = entry[0];val = entry[
                                    1
                                ];return (
                                    <div
                                        key={key}
                                        className="flex flex-col gap-1 text-xs p-3 bg-[var(--bg-darkest)] border border-[var(--border)] rounded-sm"
                                    >
                                        <div className="flex justify-between items-center">
                                            <span className="capitalize text-[var(--text-muted)] font-mono tracking-wider">
                                                {key}
                                            </span>
                                            <span
                                                className={"font-mono " + (
                                                    "text-green-400"
                                                    if val["success"]
                                                    else "text-red-400 font-bold"
                                                )}
                                            >
                                                {val["latency_ms"] + "ms" if val["success"] else "Failed"}
                                            </span>
                                        </div>
                                        {<div className="mt-1 pb-1 font-mono text-[10px] text-red-400/80 break-words whitespace-pre-wrap max-h-32 overflow-y-auto charcoal-scrollbar">
                                            {val["error"]}
                                        </div> if not val["success"] else None}
                                    </div>
                                ); }
                            )}
                        </div>
                        if "results" in testResults and testResults["results"]
                        else <div className="text-xs text-[var(--text-muted)] font-mono flex-wrap">
                            {testResults["error"] if "error" in testResults else ""}
                        </div>}
                    </div>
                </div>
            </div>
        </div>
    );
}
