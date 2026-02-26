"""Workbench Controls Component"""

def:pub WorkbenchControls -> Any {
    has viewMode: str,
        setViewMode: Any,
        liveUpdates: bool,
        setLiveUpdates: Any,
        fetchData: Any;

    return
        <div
            className="absolute top-2 left-2 z-10 bg-zinc-900/90 rounded-lg border border-zinc-800 p-1 flex gap-1 shadow-lg"
        >
            <button
                className={"px-3 py-1.5 text-xs font-bold rounded transition-all " + (
                    "bg-zinc-700 text-white shadow-sm"
                    if viewMode == "graph"
                    else "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/50"
                )}
                onClick={lambda : setViewMode("graph")}
            >
                Graph
            </button>
            <button
                className={"px-3 py-1.5 text-xs font-bold rounded transition-all " + (
                    "bg-zinc-700 text-white shadow-sm"
                    if viewMode == "list"
                    else "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/50"
                )}
                onClick={lambda : setViewMode("list")}
            >
                List
            </button>
            <button
                className="px-3 py-1.5 text-xs font-bold rounded transition-all text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50"
                onClick={lambda -> None { fetchData(False);}}
            >
                Refresh
            </button>
            <button
                className={"px-3 py-1.5 text-xs font-bold rounded transition-all " + (
                    "bg-teal-700/80 text-white shadow-sm"
                    if liveUpdates
                    else "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/50"
                )}
                onClick={lambda -> None { setLiveUpdates(not liveUpdates);}}
            >
                {"Live On" if liveUpdates else "Live Off"}
            </button>
        </div>;
}
