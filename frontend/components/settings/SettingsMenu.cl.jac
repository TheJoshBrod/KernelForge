"""Settings Menu Component"""

import from ...assets.ArrowRightIcon { ArrowRightIcon }

def:pub SettingsMenu(setView: any) -> any {
    return <div className="space-y-4">
        <button
            onClick={lambda -> None { setView("api_keys"); }}
            className="w-full h-14 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 hover:border-zinc-600 rounded-lg px-6 flex items-center justify-between group transition-all"
        >
            <span className="font-semibold text-zinc-300 group-hover:text-white">API Keys</span>
            <ArrowRightIcon className="text-zinc-500 group-hover:text-white transition-colors" />
        </button>
        <button
            onClick={lambda -> None { setView("gpu"); }}
            className="w-full h-14 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 hover:border-zinc-600 rounded-lg px-6 flex items-center justify-between group transition-all"
        >
            <span className="font-semibold text-zinc-300 group-hover:text-white">GPU Settings</span>
            <ArrowRightIcon className="text-zinc-500 group-hover:text-white transition-colors" />
        </button>
        <button
            onClick={lambda -> None { setView("ssh"); }}
            className="w-full h-14 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 hover:border-zinc-600 rounded-lg px-6 flex items-center justify-between group transition-all"
        >
            <span className="font-semibold text-zinc-300 group-hover:text-white">SSH Settings</span>
            <ArrowRightIcon className="text-zinc-500 group-hover:text-white transition-colors" />
        </button>
    </div>;
}
