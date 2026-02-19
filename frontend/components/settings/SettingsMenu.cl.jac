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
                className="launcher-link w-full"
                onClick={lambda -> None { navigate(item['href']);}}
            >
                <div className="launcher-meta-row">
                    <span
                        className="font-semibold text-zinc-200 flex items-center gap-2"
                    >
                        <Icon className="w-4 h-4 text-emerald-400"/>
                        {item["title"]}
                    </span>
                    <ArrowRightIcon className="w-4 h-4 text-zinc-500"/>
                </div>
                <div className="text-xs text-zinc-500 mt-1">
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
