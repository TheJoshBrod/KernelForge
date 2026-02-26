"""Workbench Editor Panel Component"""

import from .....components.editor.ForgePythonEditor { ForgePythonEditor }
import from .....components.ui.Icons { CodeIcon }

def:pub WorkbenchEditor -> Any {
    has selectedNodeId: Any,
        selectedNodeData: dict;

    return
        <div className="flex-1 flex flex-col min-h-0 border-b border-zinc-800">
            <div
                className="h-10 bg-zinc-900/50 border-b border-zinc-800 flex items-center px-4 justify-between"
            >
                <span
                    className="text-xs font-bold text-zinc-500 uppercase flex items-center gap-2"
                >
                    <CodeIcon className="w-4 h-4"/>
                    Kernel Code
                </span>
                <span className="text-xs font-mono text-orange-300">
                    Kernel {" " + (selectedNodeId or "?")}
                </span>
            </div>
            <div className="flex-1 min-h-0 relative bg-[#0d0a08] border-l-2 border-[var(--accent)]/30">
                <ForgePythonEditor
                    value={selectedNodeData["code"] if selectedNodeData and "code" in selectedNodeData else ""}
                    readOnly={True}
                    language="cpp"
                    placeholder="// Select a kernel to view code"
                    fontSize={12}
                />
            </div>
        </div>;
}
