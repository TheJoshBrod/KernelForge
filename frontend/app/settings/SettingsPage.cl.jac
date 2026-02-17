"""Settings Page"""

import from react { useEffect }
import from "react-router-dom" { useNavigate, useLocation }
import from ...components.backdrops.GameOfLifeBackground { GameOfLifeBackground }
import from ...components.settings.SettingsMenu { SettingsMenu }
import from ...context.ConfigContext { useConfig }

def:pub SettingsPage {
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
            and str(candidate).startswith("/")
            and not str(candidate).startswith("/settings") {
                sessionStorage.setItem("settings_return_to", str(candidate));
                returnTo = str(candidate);
            } else {
                stored = sessionStorage.getItem("settings_return_to");
                if stored
                and str(stored).startswith("/")
                and not str(stored).startswith("/settings") {
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
                <GameOfLifeBackground />
                <div className="app-backdrop"></div>
                <div className="app-content-layer">
                    <p>
                        Loading settings...
                    </p>
                </div>
            </div>;
    }

    return
        <main className="app-shell app-shell--center settings-shell">
            <GameOfLifeBackground />
            <div className="app-backdrop"></div>
            <div className="settings-card panel app-content-layer">
                <div className="settings-header">
                    <h1 className="settings-title">
                        Settings
                    </h1>
                    <p className="settings-subtitle">
                        Configure your application
                    </p>
                </div>
                <SettingsMenu />
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
