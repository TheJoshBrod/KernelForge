"""New Kernel Component"""

import from react { useState, useRef, useEffect }
import from "react-router-dom" { useParams }
import from .....components.ui.Icons { CpuIcon }
import from .menus.UploadModal { UploadModal }
import from .menus.UploadTrigger { UploadTrigger }
import from .menus.GenerateModal { GenerateModal }
import from .menus.GenerateTrigger { GenerateTrigger }
import from .menus.OptimizeModal { OptimizeModal }
import from .menus.OptimizeTrigger { OptimizeTrigger }

def:pub NewKernel(props: any) -> Any {
    projectName = props.projectName;
    opName = props.opName;
    hasTree = props.hasTree if props and "hasTree" in props else False;
    gpuMismatch = props.gpuMismatch if props and "gpuMismatch" in props else False;
    onGpuMismatch = props.onGpuMismatch if props and "onGpuMismatch" in props else None;
    onUploadSuccess = props.onUploadSuccess if props and "onUploadSuccess" in props else None;

    params = useParams();
    resolvedProjectName = projectName or params.projectName;
    resolvedOpName = opName or params.opName;
    resolvedHasTree = hasTree;
    if not resolvedHasTree and resolvedProjectName and resolvedOpName {
        resolvedHasTree = True;
    }

    is_open_state = useState(False);
    isOpen = is_open_state[0];
    setIsOpen = is_open_state[1];

    # Modal States
    upload_modal_open_state = useState(False);
    uploadModalOpen = upload_modal_open_state[0];
    setUploadModalOpen = upload_modal_open_state[1];

    generate_modal_open_state = useState(False);
    generateModalOpen = generate_modal_open_state[0];
    setGenerateModalOpen = generate_modal_open_state[1];
    generate_modal_session_state = useState(0);
    generateModalSession = generate_modal_session_state[0];
    setGenerateModalSession = generate_modal_session_state[1];
    generate_modal_session_ref = useRef(0);

    optimize_modal_open_state = useState(False);
    optimizeModalOpen = optimize_modal_open_state[0];
    setOptimizeModalOpen = optimize_modal_open_state[1];
    optimize_modal_session_state = useState(0);
    optimizeModalSession = optimize_modal_session_state[0];
    setOptimizeModalSession = optimize_modal_session_state[1];
    optimize_modal_session_ref = useRef(0);
    generate_modal_session_ref.current = generateModalSession;
    optimize_modal_session_ref.current = optimizeModalSession;

    menu_ref = useRef(None);

    missingProject = False;
    if not resolvedProjectName {
        missingProject = True;
    }

    # Close menu on outside click
    useEffect(
        lambda -> Any { def handle_click_outside(event: any) {
            if menu_ref.current and not menu_ref.current.contains(event.target) {
                setIsOpen(False);
            }
        } document.addEventListener("mousedown", handle_click_outside);return lambda -> None { document.removeEventListener(
            "mousedown", handle_click_outside
        );}; },
        []
    );

    on_complete = lambda data: anyNone  -> None {
        if data and "_queued" in data and data["_queued"] and "_message" in data and data["_message"] {
            alert(str(data["_message"]));
        }
        setIsOpen(False);
    };

    def closeGenerateModal(session_id: any = None) -> None {
        if session_id is not None and session_id != "" {
            try {
                if int(session_id) != int(generate_modal_session_ref.current) {
                    return;
                }
            } except Exception {
            # no-op
            }
        }
        setGenerateModalOpen(False);
    }

    def closeOptimizeModal(session_id: any = None) -> None {
        if session_id is not None and session_id != "" {
            try {
                if int(session_id) != int(optimize_modal_session_ref.current) {
                    return;
                }
            } except Exception {
            # no-op
            }
        }
        setOptimizeModalOpen(False);
    }

    def closeGenerateModalAndMenu(session_id: any = None) -> None {
        closeGenerateModal(session_id);
        setIsOpen(False);
    }

    def closeOptimizeModalAndMenu(session_id: any = None) -> None {
        closeOptimizeModal(session_id);
        setIsOpen(False);
    }

    # Trigger Handlers
    def handle_upload_click -> None {
        if missingProject {
            setIsOpen(False);
            return;
        }
        if gpuMismatch {
            setIsOpen(False);
            if onGpuMismatch { onGpuMismatch(); }
            return;
        }
        setUploadModalOpen(True);
        setIsOpen(False);
    }

    def handle_generate_click -> None {
        if missingProject {
            setIsOpen(False);
            return;
        }
        if gpuMismatch {
            setIsOpen(False);
            if onGpuMismatch { onGpuMismatch(); }
            return;
        }
        setGenerateModalSession(lambda p: any -> any { return p + 1; });
        setGenerateModalOpen(True);
        setIsOpen(False);
    }

    def handle_optimize_click -> None {
        if missingProject {
            setIsOpen(False);
            return;
        }
        if gpuMismatch {
            setIsOpen(False);
            if onGpuMismatch { onGpuMismatch(); }
            return;
        }
        setOptimizeModalSession(lambda p: any -> any { return p + 1; });
        setOptimizeModalOpen(True);
        setIsOpen(False);
    }

    def toggle_menu -> None {
        if missingProject {
            return;
        }
        setIsOpen(not isOpen);
    }

    return
        <div className="relative" ref={menu_ref}>
            <button
                className={"w-full font-bold py-3 px-4 rounded-lg transition-all shadow-lg flex items-center justify-center gap-2 " + (
                    "bg-zinc-700 text-zinc-300 cursor-not-allowed opacity-70"
                    if missingProject
                    else "bg-[var(--accent-orange)] hover:bg-[var(--molten-gold-dark)] text-zinc-950 shadow-[0_10px_24px_rgba(225,112,85,0.28)] hover:shadow-[0_12px_28px_rgba(232,168,56,0.26)]"
                )}
                onClick={toggle_menu}
                disabled={missingProject}
            >
                <CpuIcon className="w-5 h-5"/>
                Create New Kernel Node
            </button>
            {<div
                className="absolute bottom-full left-0 w-full mb-2 bg-zinc-900 border border-zinc-800 rounded-lg shadow-xl overflow-hidden z-20 flex flex-col"
            >
                <UploadTrigger onUpload={handle_upload_click}/>
                <GenerateTrigger onGenerate={handle_generate_click}/>
                <OptimizeTrigger onOptimize={handle_optimize_click} hasTree={resolvedHasTree}/>
            </div>
            if isOpen
            else <span/>}
            <UploadModal
                isOpen={uploadModalOpen}
                onClose={lambda -> None { setUploadModalOpen(False);}}
                onSuccess={lambda -> None { setIsOpen(False); if onUploadSuccess { onUploadSuccess(); }}}
                projectName={resolvedProjectName}
                opName={resolvedOpName}
            />
            {<GenerateModal
                key={"new-kernel-generate-modal-" + str(generateModalSession)}
                isOpen={generateModalOpen}
                modalSession={generateModalSession}
                onClose={closeGenerateModal}
                onComplete={closeGenerateModalAndMenu}
                onStarted={on_complete}
                projectName={resolvedProjectName}
                opName={resolvedOpName}
            />
            if generateModalOpen
            else <span/>}
            {<OptimizeModal
                key={"new-kernel-optimize-modal-" + str(optimizeModalSession)}
                isOpen={optimizeModalOpen}
                modalSession={optimizeModalSession}
                onClose={closeOptimizeModal}
                onComplete={closeOptimizeModalAndMenu}
                onStarted={on_complete}
                projectName={resolvedProjectName}
                opName={resolvedOpName}
                hasTree={resolvedHasTree}
            />
            if optimizeModalOpen
            else <span/>}
        </div>;
}
