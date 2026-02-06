"""Splash Screen Component"""

import from "backdrops/CrashingWaveBackground" { CrashingWaveBackground }
import from .Mascot { Mascot }

def:pub SplashScreen(exiting: bool) -> any {
    splashClass = "splash-screen";
    if exiting {
        splashClass = "splash-screen is-exiting";
    }

    return
        <div className={splashClass}>
            <CrashingWaveBackground />
            <div className="app-backdrop app-backdrop--splash"></div>
            <div className="splash-content">
                <h1 className="splash-title">
                    CGinS
                </h1>
                <p className="splash-subtitle">
                    Agentic PyTorch Model Optimizer
                </p>
                <div className="splash-mascot">
                    <Mascot
                        variant="hero"
                        ambient={True}
                        className="splash-mascot-img"
                    />
                </div>
                <div className="splash-loading">
                    Loading resources...
                </div>
            </div>
        </div>;
}
