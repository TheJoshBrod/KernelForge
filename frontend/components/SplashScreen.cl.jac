"""Splash Screen Component"""

import from .backdrops.EmberBackground { EmberBackground }
import from .ui.Icons { KernelForgeLogo }

def:pub SplashScreen(exiting: bool) -> Any {
    splashClass = "splash-screen";
    if exiting {
        splashClass = "splash-screen is-exiting";
    }

    return
        <div className={splashClass}>
            <div className="absolute inset-0 z-0 industrial-grid-bg opacity-40"></div>
            <div className="noise-overlay opacity-40"></div>
            <EmberBackground className="absolute inset-0 z-0"/>
            <div
                className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-black/50 z-10 pointer-events-none"
            ></div>
            <div className="splash-content">
                <div className="splash-mascot bg-[var(--bg-panel)]/50 backdrop-blur-sm border border-[var(--border)] rounded-sm p-8 shadow-[0_0_30px_rgba(225,112,85,0.2)]">
                    <KernelForgeLogo className="w-32 h-32 text-[var(--accent)] splash-mascot-img"/>
                </div>
                <div className="flex flex-col items-center mt-6">
                    <h1 className="text-6xl font-bold text-white tracking-widest uppercase industrial-font-display mb-2 drop-shadow-[0_0_15px_rgba(225,112,85,0.5)]">
                        KERNEL FORGE
                    </h1>
                    <p className="text-sm font-mono text-[var(--accent)] tracking-[0.3em] uppercase">
                        Autonomous CUDA Optimization
                    </p>
                </div>
                <div className="splash-loading mt-12 flex items-center gap-3">
                    <div className="w-2 h-2 bg-[var(--accent)] rounded-full animate-ping"></div>
                    <span className="font-mono text-zinc-400 tracking-widest">IGNITING FORGE...</span>
                </div>
            </div>
        </div>;
}
