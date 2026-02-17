"""Optimize Menu Component"""

import from "../../../../components/ui/Icons.cl.jac" { ZapIcon }

def:pub OptimizeMenu(props: any) {
    onComplete = props.onComplete;
    def handleOptimize -> None {
        alert("Optimize Existing Tree: Coming soon!");
        if onComplete { onComplete(); }
    }

    return (
        <button 
            className="p-3 text-left text-sm text-zinc-300 hover:bg-zinc-800 hover:text-white flex items-center gap-3 transition-colors border-t border-zinc-800"
            onClick={handleOptimize}
        >
            <ZapIcon className="w-4 h-4 text-amber-400" />
            <div>
                <div className="font-semibold">Optimize Existing Tree</div>
                <div className="text-[10px] text-zinc-500">Continue optimizing current best</div>
            </div>
        </button>
    );
}
