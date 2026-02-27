"""Workbench Controls Component"""

def:pub WorkbenchControls(props: any) -> Any {
    viewMode = props.viewMode;
    onViewModeChange = props.onViewModeChange;
    liveUpdates = props.liveUpdates;
    onLiveUpdatesChange = props.onLiveUpdatesChange;
    fetchData = props.fetchData;

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
                onClick={lambda : onViewModeChange("graph")}
            >
                Graph
            </button>
            <button
                className={"px-3 py-1.5 text-xs font-bold rounded transition-all " + (
                    "bg-zinc-700 text-white shadow-sm"
                    if viewMode == "list"
                    else "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/50"
                )}
                onClick={lambda : onViewModeChange("list")}
            >
                List
            </button>
        </div>;
}
