"""Reusable Loading Screen Component"""

def:pub LoadingScreen(props: any) -> Any {
    message = props["message"] if props and "message" in props else "loading...";
    return
        <div className="flex items-center justify-center w-screen h-screen bg-[var(--bg-charcoal)] text-[var(--accent)] font-mono text-sm tracking-widest uppercase">
            {message}
        </div>;
}
