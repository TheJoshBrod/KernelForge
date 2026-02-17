"""Generic Modal Component"""

import from "react-dom" { createPortal }
import from react { useEffect }
import from .Icons { XIcon }

def:pub Modal(props: any) {
    isOpen = props.isOpen;
    onClose = props.onClose;
    children = props.children;
    title = props.title or "";

    # Close on Escape key
    useEffect(
        lambda -> None { def handle_keydown(e: any) {
            if e.key == "Escape" {
                onClose();
            }
        } if isOpen {
            window.addEventListener("keydown", handle_keydown);
        }return lambda -> None { window.removeEventListener("keydown", handle_keydown);}; },
        [isOpen, onClose]
    );

    if not isOpen {
        return null;
    }

    return createPortal(
        <div
            className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-in fade-in duration-200"
        >
            <div
                className="bg-zinc-900 border border-zinc-800 rounded-xl shadow-2xl w-full max-w-2xl max-h-[90vh] flex flex-col overflow-hidden animate-in zoom-in-95 duration-200"
                onClick={lambda e: any  -> None { e.stopPropagation();}}
            >
                <div
                    className="flex items-center justify-between p-4 border-b border-zinc-800 bg-zinc-900/50"
                >
                    <h3 className="text-lg font-semibold text-zinc-100">
                        {title}
                    </h3>
                    <button
                        onClick={onClose}
                        className="p-1 hover:bg-zinc-800 rounded-lg text-zinc-400 hover:text-white transition-colors"
                    >
                        <XIcon className="w-5 h-5" />
                    </button>
                </div>
                <div className="flex-1 overflow-y-auto custom-scrollbar">
                    {children}
                </div>
            </div>
            <div className="absolute inset-0 -z-10" onClick={onClose} />
        </div>,
        document.body
    );
}
