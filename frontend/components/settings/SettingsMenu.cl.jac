"""Settings Menu Component"""

import from "react-router-dom" { useNavigate }
import from ...assets.ArrowRightIcon { ArrowRightIcon }

def:pub SettingsMenu() -> any {
    navigate = useNavigate();
    return
        <div className="space-y-4">
            <button
                onClick={lambda -> None { navigate("/settings/api-keys");}}
                className="w-full h-14 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 hover:border-zinc-600 rounded-lg px-6 flex items-center justify-between group transition-all"
            >
                <span className="font-semibold text-zinc-300 group-hover:text-white">
                    API Keys
                </span>
                <ArrowRightIcon
                    className="text-zinc-500 group-hover:text-white transition-colors"
                />
            </button>
            <button
                onClick={lambda -> None { navigate("/settings/gpu");}}
                className="w-full h-14 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 hover:border-zinc-600 rounded-lg px-6 flex items-center justify-between group transition-all"
            >
                <span className="font-semibold text-zinc-300 group-hover:text-white">
                    GPU Settings
                </span>
                <ArrowRightIcon
                    className="text-zinc-500 group-hover:text-white transition-colors"
                />
            </button>
            <button
                onClick={lambda -> None { navigate("/settings/ssh");}}
                className="w-full h-14 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 hover:border-zinc-600 rounded-lg px-6 flex items-center justify-between group transition-all"
            >
                <span className="font-semibold text-zinc-300 group-hover:text-white">
                    SSH Settings
                </span>
                <ArrowRightIcon
                    className="text-zinc-500 group-hover:text-white transition-colors"
                />
            </button>
        </div>;
}
