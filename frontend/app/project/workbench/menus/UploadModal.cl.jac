"""Upload Menu Component"""

import from react { useState, useRef }
import from "../../../../components/ui/Icons.cl.jac" { UploadIcon, CodeIcon, SettingsIcon }
import from "../../../../components/ui/Modal.cl.jac" { Modal }
import from "@jac/runtime" { jacSpawn }

def:pub UploadModal(props: any) {
    isOpen = props.isOpen;
    onClose = props.onClose;
    onSuccess = props.onSuccess;

    # State
    code_state = useState("");
    code = code_state[0];
    setCode = code_state[1];

    backend_state = useState("CUDA");
    backend = backend_state[0];
    setBackend = backend_state[1];

    is_loading_state = useState(False);
    isLoading = is_loading_state[0];
    setIsLoading = is_loading_state[1];

    file_input_ref = useRef(None);

    # Handlers
    def handleFileSelect(e: any) {
        file = e.target.files[0];
        if file {
            reader = Reflect.construct(FileReader, []);
            def on_load(evt: any) {
                setCode(evt.target.result);
            }
            reader.onload = on_load;
            reader.readAsText(file);
        }
    }

    async def handleCreate -> None {
        if not code {
             alert("Please enter some code or upload a file.");
             return;
        }

        setIsLoading(True);
        try {
            console.log("Importing kernel...", {backend, code});
            
            # Call the ImportKernel walker with the code and backend
            resp = await jacSpawn("ImportKernel", "", {
                "source": code,
                "backend": backend
            });
            
            console.log("Upload result:", resp);
            alert("Kernel created successfully!"); 
            
            onClose();
            if onSuccess { onSuccess(); }
            
            # Reset state
            setCode("");
            setBackend("CUDA");
        } except (e) {
            console.error("Upload failed", e);
            alert("Failed to create kernel: " + e.toString());
        } finally {
            setIsLoading(False);
        }
    }

    if not isOpen {
        return null;
    }

    return (
        <Modal 
            isOpen={isOpen} 
            onClose={onClose}
            title="Upload Custom Kernel"
        >
            <div className="p-6 space-y-6">
                # {/* Code Editor Section */}
                <div className="space-y-4">
                    <div className="flex items-center justify-between">
                        <label className="text-sm font-medium text-zinc-300 flex items-center gap-2">
                            <CodeIcon className="w-4 h-4 text-indigo-400" />
                            Kernel Source Code
                        </label>
                        
                        # {/* File Upload Trigger */}
                        <div>
                            <input 
                                type="file" 
                                ref={file_input_ref}
                                className="hidden"
                                onChange={lambda e: any -> None { handleFileSelect(e); }}
                                accept=".cu,.cpp,.c,.py,.jac"
                            />
                            <button 
                                onClick={lambda -> None { file_input_ref.current.click(); }}
                                className="text-xs flex items-center gap-2 px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 hover:text-white rounded-md text-zinc-400 transition-colors border border-zinc-700"
                            >
                                <UploadIcon className="w-3 h-3" />
                                Load from File
                            </button>
                        </div>
                    </div>
                    
                    <div className="relative group">
                        <textarea 
                            value={code}
                            onChange={lambda e: any -> None { setCode(e.target.value); }}
                            className="w-full h-80 bg-zinc-950 border border-zinc-800 rounded-lg p-4 font-mono text-xs text-zinc-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 resize-none leading-relaxed"
                            placeholder="// Enter your kernel code here (e.g. CUDA C++)..."
                            spellCheck={False}
                        />
                        <div className="absolute bottom-4 right-4 text-[10px] text-zinc-600 pointer-events-none">
                            {code.length} chars
                        </div>
                    </div>
                </div>

                # {/* Settings Section */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-zinc-300 flex items-center gap-2">
                            <SettingsIcon className="w-4 h-4 text-amber-400" />
                            Target Backend
                        </label>
                        <div className="relative">
                            <select 
                                value={backend} 
                                onChange={lambda e: any -> None { setBackend(e.target.value); }}
                                className="w-full bg-zinc-900 border border-zinc-700 rounded-lg p-2.5 text-sm text-zinc-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 appearance-none"
                            >
                                <option value="CUDA">CUDA</option>
                                <option value="Triton">Triton (Python)</option>
                            </select>
                            <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none">
                                <svg className="w-4 h-4 text-zinc-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                                </svg>
                            </div>
                        </div>
                        <p className="text-[10px] text-zinc-500 px-1">
                            Select the backend compatible with your kernel code.
                        </p>
                    </div>
                </div>

                # {/* Actions */}
                <div className="pt-6 flex justify-end gap-3 border-t border-zinc-800">
                    <button 
                        onClick={onClose}
                        className="px-4 py-2 text-sm text-zinc-400 hover:text-white transition-colors"
                        disabled={isLoading}
                    >
                        Cancel
                    </button>
                    <button 
                        onClick={handleCreate}
                        disabled={isLoading}
                        className={"px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition-all shadow-lg hover:shadow-indigo-500/20 flex items-center gap-2 " + ("opacity-50 cursor-not-allowed" if isLoading else "")}
                    >
                        {"Creating..." if isLoading else "Create Kernel"}
                        {<UploadIcon className="w-4 h-4" /> if not isLoading else null}
                    </button>
                </div>
            </div>
        </Modal>
    );
}

