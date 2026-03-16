"""About Settings Page"""

import from "react-router-dom" { useNavigate }
import from ...components.backdrops.EmberBackground { EmberBackground }
import from ...components.ui.Icons { GithubIcon, DiscordIcon }

def:pub AboutSettings(props: any) -> Any {
    navigate = useNavigate();

    def handleReturn -> None {
        navigate("/settings");
    }

    return
        <main className="relative h-screen w-screen overflow-hidden bg-[var(--bg-charcoal)] flex flex-col items-center justify-center font-sans text-white">
            <div className="absolute inset-0 z-0 industrial-grid-bg opacity-30"></div>
            <div className="noise-overlay opacity-30"></div>
            <EmberBackground className="absolute inset-0 z-0"/>
            <div className="absolute inset-0 bg-gradient-to-t from-black/50 via-transparent to-black/50 z-10 pointer-events-none"></div>

            <div className="relative z-20 flex flex-col items-center justify-center mb-4 animation-fade-in-up">
                <div className="flex items-center gap-4 mb-2">
                    <img src="/KernelForge.svg" alt="Kernel Forge Logo" className="w-12 h-12" />
                    <h1 className="text-3xl font-bold text-white tracking-wider uppercase industrial-font-display" style={{"textShadow": "0 0 20px rgba(225, 112, 85, 0.4)"}}>
                        Kernel Forge
                    </h1>
                </div>
            </div>

            <div className="relative z-10 w-full max-w-4xl h-[70vh] bg-[var(--bg-panel)]/90 backdrop-blur-md border border-[var(--border)] rounded-sm shadow-2xl flex flex-col animation-fade-in-up delay-1 clip-hex">
                <div className="p-6 border-b border-[var(--border)] space-y-4 shrink-0 relative overflow-hidden">
                    <div className="absolute top-0 left-0 right-0 heat-divider"></div>
                    <div className="flex justify-between items-start">
                        <div>
                            <h1 className="text-2xl font-bold text-white tracking-wide uppercase industrial-font-display">
                                About Kernel Forge
                            </h1>
                            <p className="text-xs text-[var(--accent)] font-mono uppercase tracking-widest mt-1">
                                System Information & Resources
                            </p>
                        </div>
                        <div className="flex gap-4">
                            <a
                                href="https://discord.gg/"
                                target="_blank"
                                rel="noreferrer"
                                className="text-white/50 hover:text-[#5865F2] transition-all duration-300 hover:scale-110 hover:-translate-y-1 hover:drop-shadow-[0_0_12px_rgba(88,101,242,0.6)]"
                                title="Discord Server"
                            >
                                <DiscordIcon className="w-6 h-6"/>
                            </a>
                            <a
                                href="https://github.com/TheJoshBrod/CGinS"
                                target="_blank"
                                rel="noreferrer"
                                className="text-white/50 hover:text-white transition-all duration-300 hover:scale-110 hover:-translate-y-1 hover:drop-shadow-[0_0_12px_rgba(255,255,255,0.6)]"
                                title="GitHub Repository"
                            >
                                <GithubIcon className="w-6 h-6"/>
                            </a>
                        </div>
                    </div>
                </div>
                
                <div className="flex-1 overflow-y-auto p-6 space-y-6 min-h-0 charcoal-scrollbar bg-[var(--bg-darker)]/30">
                    <div className="space-y-4">
                        <div className="bg-[var(--surface-0)] border border-[var(--border)] p-5 rounded-sm clip-anvil relative group hover:border-[var(--accent)] transition-colors">
                            <h2 className="text-lg font-bold text-white mb-2 uppercase font-mono tracking-wider flex items-center gap-2">
                                <span className="w-2 h-2 rounded-full bg-[var(--accent)]"></span>
                                Kernel Forge
                            </h2>
                            <p className="text-[var(--text-light)] text-sm leading-relaxed mb-4">
                                Developer tool to autonomously generate and optimize customized kernels for PyTorch models. 
                                Kernel Forge relies on Large Language Models to iterative generate, benchmark, and refine kernels, 
                                bringing expert-level performance to standard deployment pipelines.
                            </p>
                            
                            <div className="grid grid-cols-2 gap-4 mt-6">
                                <div className="bg-[var(--bg-darker)] p-3 rounded border border-[var(--border)]">
                                    <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1 font-mono">Version</div>
                                    <div className="text-sm font-medium text-white">v1.0.0-beta</div>
                                </div>
                                <div className="bg-[var(--bg-darker)] p-3 rounded border border-[var(--border)]">
                                    <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1 font-mono">License</div>
                                    <div className="text-sm font-medium text-white">MIT</div>
                                </div>
                            </div>
                        </div>

                        <div className="bg-[var(--surface-0)] border border-[var(--border)] p-5 rounded-sm clip-anvil relative group hover:border-[var(--accent)] transition-colors">
                            <h2 className="text-lg font-bold text-white mb-4 uppercase font-mono tracking-wider flex items-center gap-2">
                                <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                                Core Technologies
                            </h2>
                            <ul className="space-y-2 text-sm text-[var(--text-light)]">
                                <li className="flex items-center gap-3">
                                    <span className="text-[var(--accent)] font-mono text-xs">01</span>
                                    <span><strong>Model Profiling:</strong> Hooks into PyTorch execution via `__torch_dispatch__`.</span>
                                </li>
                                <li className="flex items-center gap-3">
                                    <span className="text-[var(--accent)] font-mono text-xs">02</span>
                                    <span><strong>LLM CodeGen:</strong> Powered by Anthropic Claude, OpenAI, and Google Gemini.</span>
                                </li>
                                <li className="flex items-center gap-3">
                                    <span className="text-[var(--accent)] font-mono text-xs">03</span>
                                    <span><strong>Validation Engine:</strong> Automatic compilation and numerical equality checking against PyTorch standard outputs.</span>
                                </li>
                                <li className="flex items-center gap-3">
                                    <span className="text-[var(--accent)] font-mono text-xs">04</span>
                                    <span><strong>Frontend:</strong> Written in Jac-lang (a Python superset to compile directly to React / Web).</span>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>
                
                <div className="p-6 border-t border-[var(--border)] bg-[var(--bg-panel)] shrink-0 flex justify-end items-center gap-4">
                    <button
                        onClick={handleReturn}
                        className="app-button"
                    >
                        Back
                    </button>
                </div>
            </div>
        </main>;
}
