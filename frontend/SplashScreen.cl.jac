"""Splash Screen Component"""

import from "backdrops/CrashingWaveBackground" { CrashingWaveBackground }
import from ..assets.CGinSLogo { CGinSLogo }

def:pub SplashScreen(exiting: bool) -> any {
    return
        <div
            className={f"fixed inset-0 z-50 flex flex-col items-center justify-center bg-zinc-950 transition-all duration-1000 ease-in-out {(
                'opacity-0 pointer-events-none' if exiting else 'opacity-100'
            )}"}
        >
            <CrashingWaveBackground />
            <div
                className={f"relative z-10 transition-all duration-1000 ease-in-out transform flex flex-col items-center {(
                    'scale-75 -translate-y-24' if exiting else 'scale-100'
                )}"}
            >
                <h1
                    className="text-8xl font-bold tracking-tighter bg-gradient-to-b from-white to-zinc-400 bg-clip-text text-transparent"
                >
                    CGinS
                </h1>
                <div
                    className={f"text-center space-y-8 mt-4 transition-opacity duration-500 {(
                        'opacity-0' if exiting else 'opacity-100'
                    )}"}
                >
                    <p
                        className="text-zinc-500 font-medium tracking-wide uppercase text-lg"
                    >
                        Agentic PyTorch Model Optimizer
                    </p>
                    <div
                        className="w-full h-16 flex items-center justify-center relative"
                    >
                        <CGinSLogo className="w-32 h-32 text-zinc-200 animate-pulse" />
                        <div className="text-zinc-500 font-mono text-sm animate-pulse">
                            Loading resources...
                        </div>
                    </div>
                </div>
            </div>
        </div>;
}
