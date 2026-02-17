"""Generate Menu Component"""

import from "../../../../components/ui/Icons.cl.jac" { PlusIcon }

def:pub GenerateMenu(onComplete: any) {
    console.log("GenerateMenu: Render", onComplete);

    def handleGenerate -> None {
        alert("Generate New Root: Coming soon!");
        if onComplete { onComplete(); }
    }

    return 
        <button 
            className="p-3 text-left text-sm text-zinc-300 hover:bg-zinc-800 hover:text-white flex items-center gap-3 transition-colors border-t border-zinc-800"
            onClick={handleGenerate}
        >
            <PlusIcon className="w-4 h-4 text-emerald-400" />
            <div>
                <div className="font-semibold">Generate New Root</div>
                <div className="text-[10px] text-zinc-500">AI generates a new starting kernel</div>
            </div>
        </button>;
}
