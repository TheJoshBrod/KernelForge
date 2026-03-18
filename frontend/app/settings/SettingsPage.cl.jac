"""Settings Page"""

import from react { useEffect }
import from "react-router-dom" { useNavigate, useLocation }
import from ...components.backdrops.EmberBackground { EmberBackground }
import from ...components.settings.SettingsMenu { SettingsMenu }
import from ...components.ui.DesktopRuntime { isDesktopBuild }
import from ...components.ui.LoadingScreen { LoadingScreen }
import from ...context.ConfigContext { useConfig }

def:pub SettingsPage -> Any {
    navigate = useNavigate();
    location = useLocation();
    config_ctx = useConfig();
    loading = config_ctx.loading;
    desktopBuild = isDesktopBuild();

    has returnTo: str = "/";
    panelClass = "relative z-10 w-full max-w-4xl h-[70vh] border border-[var(--border)] rounded-sm shadow-2xl flex flex-col bg-[var(--bg-panel)]";
    if not desktopBuild {
        panelClass += " backdrop-blur-md animation-fade-in-up delay-1 clip-hex";
    }
    headingClass = "relative z-20 flex flex-col items-center justify-center mb-8";
    if not desktopBuild {
        headingClass += " animation-fade-in-up";
    }

    useEffect(
        lambda -> None { try {
            candidate = "";
            state = location.state;
            if state and "return_to" in state and state["return_to"] {
                candidate = state["return_to"];
            }

            if candidate
            and str(candidate).startsWith("/")
            and not str(candidate).startsWith("/settings") {
                sessionStorage.setItem("settings_return_to", str(candidate));
                returnTo = str(candidate);
            } else {
                stored = sessionStorage.getItem("settings_return_to");
                if stored
                and str(stored).startsWith("/")
                and not str(stored).startsWith("/settings") {
                    returnTo = str(stored);
                } else {
                    returnTo = "/";
                }
            }
        } except Exception {
            returnTo = "/";
        }},
        []
    );

    useEffect(
        lambda -> None {
            document.title = "Kernel Forge";
        },
        []
    );

    def handleReturn -> None {
        navigate(returnTo or "/", {"replace": True});
    }

    if loading {
        return <LoadingScreen/>;
    }

    return
        <main
            className="relative h-screen w-screen overflow-hidden bg-[var(--bg-charcoal)] flex flex-col items-center justify-center font-sans text-white"
        >
            <div className="absolute inset-0 z-0 industrial-grid-bg opacity-30"></div>
            {<div className="noise-overlay opacity-30"></div> if not desktopBuild else None}
            {<EmberBackground className="absolute inset-0 z-0"/> if not desktopBuild else None}
            <div
                className="absolute inset-0 bg-gradient-to-t from-black/50 via-transparent to-black/50 z-10 pointer-events-none"
            ></div>
            <div className={headingClass}>
                <div className="flex items-center gap-6 mb-2">
                    <img src="/KernelForge.svg" alt="Kernel Forge Logo" className="w-16 h-16" />
                    <div className="flex flex-col">
                        <h1 className="text-5xl font-bold text-white tracking-wider uppercase industrial-font-display" style={{"textShadow": "0 0 20px rgba(225, 112, 85, 0.4)"}}>
                            Kernel Forge
                        </h1>
                    </div>
                </div>
            </div>
            <div className={panelClass}>
                <div
                    className="p-6 border-b border-[var(--border)] space-y-4 shrink-0 relative overflow-hidden"
                >
                    {<div
                        className="absolute top-0 left-0 right-0 heat-divider"
                    ></div> if not desktopBuild else None}
                    <h1
                        className="text-2xl font-bold text-white tracking-wide uppercase industrial-font-display"
                    >
                        Settings
                    </h1>
                    <p
                        className="text-xs text-[var(--accent)] font-mono uppercase tracking-widest"
                    >
                        Configure your application
                    </p>
                </div>
                <div className="flex-1 overflow-y-auto p-4 space-y-2 min-h-0 charcoal-scrollbar bg-[var(--bg-darker)]/30">
                    <SettingsMenu/>
                </div>
                <div className="p-6 border-t border-[var(--border)] bg-[var(--bg-panel)] shrink-0 flex justify-end">
                    <button
                        onClick={handleReturn}
                        className="app-button"
                    >
                        Return to Forge
                    </button>
                </div>
            </div>
        </main>;
}
