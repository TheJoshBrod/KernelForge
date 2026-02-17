"""Settings Page"""

import from react { useEffect }
import from "react-router-dom" { useNavigate }
import from ...components.backdrops.GameOfLifeBackground { GameOfLifeBackground }
import from ...components.settings.SettingsMenu { SettingsMenu }
import from ...context.ConfigContext { useConfig }

def:pub SettingsPage {
    navigate = useNavigate();
    config_ctx = useConfig();
    loading = config_ctx.loading;

    def handleReturn -> None {
        navigate("/");
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
