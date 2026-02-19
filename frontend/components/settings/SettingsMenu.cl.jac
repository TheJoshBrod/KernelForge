"""Settings Menu Component"""

import from "react-router-dom" { useNavigate }
import from ..ui.Icons { ArrowRightIcon, CodeIcon, CpuIcon, ShieldCheckIcon }

def:pub SettingsMenu -> Any {
    navigate = useNavigate();

    items = [
        {
            "href": "/settings/api-keys",
            "title": "API Keys",
            "desc": "Store OpenAI, Anthropic, and Google keys.",
            "Icon": ShieldCheckIcon
        },
        {
            "href": "/settings/gpu",
            "title": "GPU",
            "desc": "Select device and backend for local runs.",
            "Icon": CpuIcon
        },
        {
            "href": "/settings/ssh",
            "title": "SSH",
            "desc": "Connect to remote machines and query GPUs.",
            "Icon": CodeIcon
        }
    ];

    rows = [];
    for item in items {
        Icon = item["Icon"];
        rows.push(
            <button
                key={item["href"]}
                type="button"
                className="w-full text-left p-4 mb-2 bg-[var(--surface)] border-l-2 border-transparent hover:border-[var(--accent)] hover:bg-[var(--surface-0)] transition-all group flex flex-col gap-2 clip-anvil"
                onClick={lambda -> None { navigate(item['href']);}}
            >
                <div className="flex items-center justify-between w-full">
                    <span
                        className="font-medium text-[var(--text-light)] group-hover:text-white text-lg flex items-center gap-3 font-sans uppercase tracking-wide"
                    >
                        <Icon className="w-5 h-5 text-[var(--accent)]"/>
                        {item["title"]}
                    </span>
                    <span className="text-[var(--text-muted)] group-hover:text-[var(--accent)] text-sm shrink-0 flex items-center gap-1 transition-transform group-hover:translate-x-1 font-mono uppercase tracking-wider text-xs">
                        Configure
                        <ArrowRightIcon className="w-3 h-3"/>
                    </span>
                </div>
                <div className="text-xs text-[var(--text-muted)] group-hover:text-[var(--text-light)] font-mono tracking-widest uppercase ml-8 mt-1">
                    {item["desc"]}
                </div>
            </button>
        );
    }

    return
        <div className="form-stack">
            {rows}
        </div>;
}
