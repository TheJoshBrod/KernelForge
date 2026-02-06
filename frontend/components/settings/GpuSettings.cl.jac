"""GPU Settings Component"""

import from ...assets.ChevronDownIcon { ChevronDownIcon }

def:pub GpuSettings(config: any, systemInfo: any, handleGpuSelect: any, handleSave: any, saving: bool) -> any {
    return <div className="space-y-4">
        <div className="space-y-2">
                <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Select Active GPU</label>
                <div className="relative">
                <select
                    value={config["selected_gpu_index"] if "selected_gpu_index" in config else 0}
                    onChange={lambda e: any -> None { handleGpuSelect(parseInt(e.target.value)); }}
                    className="w-full h-12 appearance-none bg-zinc-900 border border-zinc-800 hover:border-zinc-700 rounded-lg px-4 text-zinc-200 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all cursor-pointer"
                    disabled={not systemInfo or not "gpus" in systemInfo}
                >
                    {(
                        systemInfo["gpus"].map(lambda gpu: any, i: any -> any:
                            <option value={i} key={gpu["name"]}>{f"{i}: {gpu['name']}"}</option>
                        )
                        if systemInfo and "gpus" in systemInfo
                        else <option>Loading GPUs...</option>
                    )}
                </select>
                <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-zinc-500">
                    <ChevronDownIcon />
                </div>
                </div>
        </div>

        {(
            <div className="p-3 bg-zinc-900/50 rounded-lg border border-zinc-800 space-y-1">
                <h3 className="text-sm font-medium text-zinc-300">Selected GPU Stats</h3>
                <div className="grid grid-cols-[auto_1fr] gap-4 text-sm">
                    <div className="text-zinc-500">Name</div>
                    <div className="text-zinc-200 text-right whitespace-nowrap overflow-hidden text-ellipsis pl-2" title={systemInfo["gpus"][config["selected_gpu_index"]]["name"]}>
                        {systemInfo["gpus"][config["selected_gpu_index"]]["name"]}
                    </div>
                    <div className="text-zinc-500">Memory</div>
                    <div className="text-zinc-200 text-right">{systemInfo["gpus"][config["selected_gpu_index"]]["memory"]}</div>
                    <div className="text-zinc-500">Architecture Version</div>
                    <div className="text-zinc-200 text-right">{systemInfo["gpus"][config["selected_gpu_index"]]["compute_capability"]}</div>
                    <div className="text-zinc-500">SM Count</div>
                    <div className="text-zinc-200 text-right">{systemInfo["gpus"][config["selected_gpu_index"]]["sm_count"]}</div>
                </div>
            </div>
            if systemInfo and "gpus" in systemInfo and config["selected_gpu_index"] < systemInfo["gpus"].length
            else null
        )}
        <button
            onClick={lambda -> None { print("Connect to SSH"); }}
            className="w-full h-12 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 font-bold rounded-lg transition-all shadow-lg hover:shadow-zinc-900/50 border border-zinc-700"
        >
            Connect to SSH
        </button>
        <button
            onClick={handleSave}
            disabled={saving}
            className="w-full h-12 bg-zinc-100 hover:bg-white text-zinc-950 font-bold rounded-lg transition-all shadow-lg hover:shadow-zinc-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
        >
            {("Saving..." if saving else "Save Selection")}
        </button>
    </div>;
}
