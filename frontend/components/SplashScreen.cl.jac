"""Splash Screen Component"""

def:pub SplashScreen(exiting: bool) -> any {
    return
        <div
            className={f"fixed inset-0 z-50 flex flex-col items-center justify-center bg-zinc-950 transition-all duration-1000 ease-in-out {'opacity-0 pointer-events-none' if exiting else 'opacity-100'}"}
        >
            <div
                className={f"transition-all duration-1000 ease-in-out transform flex flex-col items-center {'scale-75 -translate-y-24' if exiting else 'scale-100'}"}
            >
                <h1
                    className="text-8xl font-bold tracking-tighter bg-gradient-to-b from-white to-zinc-400 bg-clip-text text-transparent"
                >
                    CGinS
                </h1>
                <div
                    className={f"text-center space-y-8 mt-4 transition-opacity duration-500 {'opacity-0' if exiting else 'opacity-100'}"}
                >
                    <p
                        className="text-zinc-500 font-medium tracking-wide uppercase text-lg"
                    >
                        Agentic PyTorch Model Optimizer
                    </p>
                    <div
                        className="w-full h-64 flex items-center justify-center relative"
                    >
                        <svg
                            viewBox="0 0 120 120"
                            className="w-32 h-32 text-zinc-200 animate-pulse"
                            fill="currentColor"
                        >
                            <path
                                d="M 60 10 C 30 10 10 40 10 70 L 10 100 C 10 110 30 110 30 100 C 30 110 50 110 50 100 C 50 110 70 110 70 100 C 70 110 90 110 90 100 C 90 110 110 110 110 100 L 110 70 C 110 40 90 10 60 10 Z"
                                className="drop-shadow-lg"
                            />
                            <circle cx="45" cy="50" r="5" fill="#18181b" />
                            <circle cx="75" cy="50" r="5" fill="#18181b" />
                            <ellipse cx="60" cy="70" rx="4" ry="3" fill="#18181b" opacity="0.2" />
                        </svg>
                        <div
                            className="absolute -bottom-8 text-zinc-600 text-xs font-mono"
                        >
                            (Loading...)
                        </div>
                    </div>
                </div>
            </div>
            <div className="absolute top-2 right-6">
                <div
                    className="p-2 rounded-full text-zinc-400"
                >
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    >
                        <path
                            d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.1a2 2 0 0 1-1-1.74v-.51a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"
                        />
                        <circle cx="12" cy="12" r="3" />
                    </svg>
                </div>
            </div>
        </div>;
}
