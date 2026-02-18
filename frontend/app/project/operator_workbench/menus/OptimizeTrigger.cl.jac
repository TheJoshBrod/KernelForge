"""Optimize Trigger Component"""

import from .....components.ui.Icons { ZapIcon }

def:pub OptimizeTrigger(props: any) {
    onOptimize = props.onOptimize;

    def handleClick -> None {
        if onOptimize {
            onOptimize();
        }
    }

    return
        <button
            className="w-full p-3 text-left text-sm text-zinc-300 hover:bg-zinc-800 hover:text-white flex items-center gap-3 transition-colors border-t border-zinc-800"
            onClick={handleClick}
        >
            <ZapIcon className="w-4 h-4 text-amber-400" />
            <div>
                <div className="font-semibold">
                    Optimize Existing Tree
                </div>
                <div className="text-[10px] text-zinc-500">
                    Continue optimizing current best
                </div>
            </div>
        </button>;
}
