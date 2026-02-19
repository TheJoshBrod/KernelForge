"""Settings Page"""

import from react { useEffect }
import from "react-router-dom" { useNavigate, useLocation }
import from ...components.backdrops.GameOfLifeBackground { GameOfLifeBackground }
import from ...components.settings.SettingsMenu { SettingsMenu }
import from ...context.ConfigContext { useConfig }

def:pub SettingsPage -> Any {
    navigate = useNavigate();
    location = useLocation();
    config_ctx = useConfig();
    loading = config_ctx.loading;

    has returnTo: str = "/";

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

    def handleReturn -> None {
        navigate(returnTo or "/", {"replace": True});
    }

    if loading {
        return
            <div className="app-shell app-shell--center settings-shell">
                <GameOfLifeBackground/>
                <div className="app-backdrop"></div>
                <div className="app-content-layer">
                    <p>
                        Loading settings...
                    </p>
                </div>
            </div>;
    }

    return
        <main
            className="relative h-screen w-screen overflow-hidden bg-black flex items-center justify-center font-sans text-white"
        >
            <GameOfLifeBackground className="absolute inset-0 z-0"/>
            <div
                className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-black/50 z-10 pointer-events-none"
            ></div>
            <div
                className="relative z-20 w-full max-w-4xl h-[80vh] bg-zinc-900/80 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl flex flex-col p-8 overflow-y-auto custom-scrollbar"
            >
                <div className="settings-header">
                    <h1 className="settings-title">
                        Settings
                    </h1>
                    <p className="settings-subtitle">
                        Configure your application
                    </p>
                </div>
                <SettingsMenu/>
                <div className="settings-actions-block">
                    <div className="settings-actions">
                        <button
                            onClick={handleReturn}
                            className="app-button app-button--ghost app-button--wide"
                        >
                            Return
                        </button>
                    </div>
                </div>
            </div>
        </main>;
}
