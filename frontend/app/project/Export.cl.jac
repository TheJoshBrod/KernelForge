"""Export Page"""

import from react { useState }
import from ...components.ui.Icons { UploadIcon, ZapIcon }
import from ...components.ui.GlowButton { GlowButton }
import from "react-router-dom" { useParams }
import from .CastExportModal { CastExportModal }

def:pub Export -> Any {
    params = useParams();
    name = params.projectName;

    cast_modal_state = useState(False);
    castModalOpen = cast_modal_state[0];
    setCastModalOpen = cast_modal_state[1];

    return
        <div
            className="h-full bg-[var(--bg-charcoal)] text-zinc-100 flex flex-col p-10 gap-8 font-sans"
        >
            <div>
                <h2 className="text-2xl font-bold mb-1 text-[var(--text-light)]">
                    Export
                </h2>
                <p className="text-zinc-400 text-sm">
                    Package and distribute your project or deploy an optimized model.
                </p>
            </div>
            <div className="flex gap-6 flex-wrap">
                <div
                    className="flex flex-col gap-4 bg-[var(--bg-panel)] border border-[var(--border)] rounded-xl p-6 w-80 shadow-sm"
                >
                    <div className="flex items-center gap-3">
                        <div
                            className="p-2 bg-[var(--surface-0)] rounded-lg border border-[var(--border)]"
                        >
                            <UploadIcon className="w-5 h-5 text-[var(--text-muted)]"/>
                        </div>
                        <h3 className="text-base font-semibold text-[var(--text-light)]">
                            Export Project as Anvil File
                        </h3>
                    </div>
                    <p className="text-zinc-400 text-sm leading-relaxed flex-1">
                        Export the {" "} <span className="text-[var(--text-light)] font-medium">entire project</span> including model source, weights, profiling data, generated kernels, and optimization history into a portable {" "} <span className="font-mono text-xs bg-zinc-800 px-1 py-0.5 rounded">.anvil</span> archive. Load it into any Kernel Forge instance to restore and continue your work exactly where you left off.
                    </p>
                    <GlowButton
                        label="Coming soon..."
                        onClick={lambda -> None {}}
                        disabled={True}
                        className="w-full"
                    />
                </div>
                <div
                    className="flex flex-col gap-4 bg-[var(--bg-panel)] border border-[var(--border)] rounded-xl p-6 w-80 shadow-sm"
                >
                    <div className="flex items-center gap-3">
                        <div
                            className="p-2 bg-[var(--surface-0)] rounded-lg border border-[var(--border)]"
                        >
                            <ZapIcon className="w-5 h-5 text-[var(--accent)]"/>
                        </div>
                        <h3 className="text-base font-semibold text-[var(--text-light)]">
                            Export Model as Cast File
                        </h3>
                    </div>
                    <p className="text-zinc-400 text-sm leading-relaxed flex-1">
                        Bundle your model weights and the KernelForge-optimized CUDA kernels into a self-contained {" "} <span className="font-mono text-xs bg-zinc-800 px-1 py-0.5 rounded">.cast</span> inference package. Anyone with the {" "} <span className="text-[var(--text-light)] font-medium">cast runtime</span> can load it and run inference immediately. No Kernel Forge installation required.
                    </p>
                    <GlowButton
                        label="Download Cast"
                        onClick={lambda -> None { setCastModalOpen(True);}}
                        className="w-full"
                    />
                </div>
            </div>
            <CastExportModal
                isOpen={castModalOpen}
                onClose={lambda -> None { setCastModalOpen(False);}}
                projectName={name}
            />
        </div>;
}
