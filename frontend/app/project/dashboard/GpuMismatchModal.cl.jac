"""GPU Mismatch Modal component."""

import from ....components.ui.Modal { Modal }
import from "react-router-dom" { useNavigate }

def:pub GpuMismatchModal(props: any) -> Any {
    isOpen = props.isOpen if props and "isOpen" in props else False;
    onClose = props.onClose if props and "onClose" in props else None;
    benchmarkedGpuName = props.benchmarkedGpuName if props and "benchmarkedGpuName" in props else "";
    currentGpuName = props.currentGpuName if props and "currentGpuName" in props else "";
    onClone = props.onClone if props and "onClone" in props else None;
    onReturnToProject = props.onReturnToProject if props and "onReturnToProject" in props else None;
    cloneBusy = props.cloneBusy if props and "cloneBusy" in props else False;

    navigate = useNavigate();

    return
        <Modal isOpen={isOpen} onClose={onClose} title="GPU Mismatch">
            <div className="p-4 flex flex-col gap-4">
                <div className="flex items-start gap-3">
                    <svg
                        className="h-5 w-5 shrink-0 text-amber-400 mt-0.5"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        strokeWidth={2}
                    >
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
                        />
                    </svg>
                    <p className="text-sm text-zinc-300 leading-relaxed">
                        {f"Metrics were measured on {benchmarkedGpuName}. Running optimization or generation on {currentGpuName} would mix incompatible timing data and corrupt the MCTS tree."}
                    </p>
                </div>
                <div className="border-t border-zinc-800 flex justify-end gap-3 pt-4">
                    <button
                        onClick={lambda -> None { navigate("/");}}
                        className="text-zinc-500 hover:text-zinc-300 text-sm transition-colors"
                    >
                        Exit to Home
                    </button>
                    <button
                        onClick={onReturnToProject}
                        className="text-zinc-400 hover:text-white text-sm transition-colors"
                    >
                        Return to Project
                    </button>
                    <button
                        onClick={onClone}
                        disabled={cloneBusy}
                        className="bg-amber-500/20 border border-amber-500/50 text-amber-100 hover:bg-amber-500/30 disabled:opacity-50 disabled:cursor-not-allowed px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
                    >
                        {<svg
                            className="h-3.5 w-3.5 animate-spin"
                            fill="none"
                            viewBox="0 0 24 24"
                        >
                            <circle
                                className="opacity-25"
                                cx="12"
                                cy="12"
                                r="10"
                                stroke="currentColor"
                                strokeWidth="4"
                            />
                            <path
                                className="opacity-75"
                                fill="currentColor"
                                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                            />
                        </svg> if cloneBusy else <span/>}
                        {f"Clone for {currentGpuName}"}
                    </button>
                </div>
            </div>
        </Modal>;
}
