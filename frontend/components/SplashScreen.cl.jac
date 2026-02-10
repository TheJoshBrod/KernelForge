"""Splash Screen Component"""

import from "./backdrops/GameOfLifeBackground.cl.jac" { GameOfLifeBackground }
import from "./ui/Icons.cl.jac" { CGinSLogo }

def:pub SplashScreen(exiting: bool) -> Any {
    splashClass = "splash-screen";
    if exiting {
        splashClass = "splash-screen is-exiting";
    }

    return
        <div className={splashClass}>
            <GameOfLifeBackground className="app-life-canvas--splash" />
            <div className="app-backdrop app-backdrop--splash"></div>
            <div className="splash-content">
                <h1 className="splash-title">
                    CGinS
                </h1>
                <p className="splash-subtitle">
                    Agentic PyTorch Model Optimizer
                </p>
                <div className="splash-mascot">
                    <CGinSLogo className="mascot mascot--hero splash-mascot-img" />
                </div>
                <div className="splash-loading">
                    Loading resources...
                </div>
            </div>
        </div>;
}
