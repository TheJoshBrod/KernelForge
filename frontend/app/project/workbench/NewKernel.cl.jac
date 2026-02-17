"""New Kernel Component"""

import from react { useState, useRef, useEffect }
import from "../../../components/ui/Icons.cl.jac" {
    CpuIcon,
    UploadIcon,
    PlusIcon,
    ZapIcon
}
import from "@jac/runtime" { jacSpawn }

def:pub NewKernel {
    has projectName: str;

    is_open_state = useState(False);
    isOpen = is_open_state[0];
    setIsOpen = is_open_state[1];

    menu_ref = useRef(None);

    # Close menu on outside click
    useEffect(lambda -> None {
        def handle_click_outside(event: any) {
            if menu_ref.current and not menu_ref.current.contains(event.target) {
                setIsOpen(False);
            }
        }
        document.addEventListener("mousedown", handle_click_outside);
        return lambda -> None {
            document.removeEventListener("mousedown", handle_click_outside);
        };
    }, []);

    async def handleUpload -> None {
        try {
            console.log("Uploading custom kernel...");
            resp = await jacSpawn("ImportKernel", "", {});
            console.log("Upload result:", resp);
            alert("Custom kernel uploaded!"); 
            setIsOpen(False);
        } except (e) {
            console.error("Upload failed", e);
        }
    }

    def handleGenerate -> None {
        alert("Generate New Root: Coming soon!");
        setIsOpen(False);
    }

    def handleOptimize -> None {
        alert("Optimize Existing Tree: Coming soon!");
        setIsOpen(False);
    }

    return 
        <div className="relative" ref={menu_ref}>
            <button 
                className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-3 px-4 rounded-lg transition-all shadow-lg hover:shadow-indigo-500/20 flex items-center justify-center gap-2"
                onClick={lambda -> None { setIsOpen(not isOpen); }}
            >
                <CpuIcon className="w-5 h-5" />
                Create New Kernel Node
            </button>
            <div className="text-center mt-2 text-[10px] text-zinc-500">
                Create a new starting point for optimization.
            </div>

            {
                <div className="absolute bottom-full left-0 w-full mb-2 bg-zinc-900 border border-zinc-800 rounded-lg shadow-xl overflow-hidden z-20 flex flex-col">
                    <button 
                        className="p-3 text-left text-sm text-zinc-300 hover:bg-zinc-800 hover:text-white flex items-center gap-3 transition-colors"
                        onClick={handleUpload}
                    >
                        <UploadIcon className="w-4 h-4 text-indigo-400" />
                        <div>
                            <div className="font-semibold">Upload Custom Kernel</div>
                            <div className="text-[10px] text-zinc-500">Import a manually written kernel</div>
                        </div>
                    </button>
                    <button 
                        className="p-3 text-left text-sm text-zinc-300 hover:bg-zinc-800 hover:text-white flex items-center gap-3 transition-colors border-t border-zinc-800"
                        onClick={handleGenerate}
                    >
                        <PlusIcon className="w-4 h-4 text-emerald-400" />
                        <div>
                            <div className="font-semibold">Generate New Root</div>
                            <div className="text-[10px] text-zinc-500">AI generates a new starting kernel</div>
                        </div>
                    </button>
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
                </div>
                if isOpen
                else <span></span>
            }
        </div>;
}
