"""Generate Menu Component"""

import from react { useState }
import from .....components.ui.Icons { PlusIcon }

def:pub GenerateMenu(props: any) {
    onComplete = props.onComplete;
    onStarted = props.onStarted;
    projectName = props.projectName;
    opName = props.opName;

    is_working_state = useState(False);
    isWorking = is_working_state[0];
    setIsWorking = is_working_state[1];

    async def handleGenerate -> None {
        if not projectName or not opName {
            alert("Missing project or operator.");
            if onComplete {
                onComplete();
            }
            return;
        }
        setIsWorking(True);
        try {
            resp = await __jacSpawn(
                "StartGenerateAdvanced",
                "",
                {
                    "projectName": projectName,
                    "ops": [opName],
                    "optimize": False,
                    "benchmark": False
                }
            );
            data = resp.reports[0] if resp and resp.reports else {};
            if data and "success" in data and not data["success"] {
                err = data["error"]
                if "error" in data
                else "Failed to queue generation.";
                alert(err);
            } else {
                if onStarted {
                    onStarted(data);
                }
            }
        } except Exception as e {
            console.error("Failed to start generation:", e);
            alert("Failed to start generation.");
        } finally {
            setIsWorking(False);
            if onComplete {
                onComplete();
            }
        }
    }

    return (
        <button
            className={"p-3 text-left text-sm text-zinc-300 hover:bg-zinc-800 hover:text-white flex items-center gap-3 transition-colors border-t border-zinc-800 " + (
                "opacity-60 cursor-not-allowed" if isWorking else ""
            )}
            onClick={handleGenerate}
            disabled={isWorking}
        >
            <PlusIcon className="w-4 h-4 text-emerald-400" />
            <div>
                <div className="font-semibold">
                    {"Starting..." if isWorking else "Generate New Root"}
                </div>
                <div className="text-[10px] text-zinc-500">
                    AI generates a new starting kernel
                </div>
            </div>
        </button>
    );
}
