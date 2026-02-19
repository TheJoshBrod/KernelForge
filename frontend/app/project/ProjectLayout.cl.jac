"""Project Layout Component"""

import from "react-router-dom" { Outlet, Link, useLocation, useParams }
import from ...components.ui.Icons { HomeIcon, ArrowRightIcon, CodeIcon, SettingsIcon }

def:pub ProjectLayout -> Any {
    location = useLocation();
    params = useParams();
    projectName = params.projectName;

    navItems = [
        {
            "name": "Dashboard",
            "href": f"/project/{projectName}",
            "icon": HomeIcon,
            "exact": True
        },
        {
            "name": "Operator Workbench",
            "href": f"/project/{projectName}/operator-workbench",
            "icon": CodeIcon,
            "exact": False
        }
    ];

    def is_active(item: dict) -> bool {
        if item["exact"] {
            return location.pathname == item["href"];
        }
        return location.pathname.startsWith(item["href"]);
    }

    nav_links = [];
    for item in navItems {
        Icon = item["icon"];
        nav_links.push(
            <Link
                key={item["name"]}
                to={item["href"]}
                className={"flex items-center gap-3 px-4 py-3 mb-2 rounded-sm transition-all border border-transparent font-mono text-xs uppercase tracking-widest clip-anvil " + (
                    "bg-[var(--bg-light)] text-[var(--accent)] !border-l-2 !border-l-[var(--accent)] shadow-[inset_4px_0_0_var(--accent)]"
                    if is_active(item)
                    else "text-[var(--text-muted)] hover:text-[var(--text-light)] hover:bg-[var(--surface-0)] hover:border-[var(--border)]"
                )}
            >
                <Icon className="w-4 h-4"/>
                <span>
                    {item["name"]}
                </span>
            </Link>
        );
    }

    return
        <div
            className="flex h-screen bg-[var(--bg-charcoal)] text-[var(--text-light)] overflow-hidden font-sans"
        >
            <div
                className="w-64 h-full bg-[var(--bg-panel)] border-r border-[var(--border)] flex flex-col flex-shrink-0 relative z-20 shadow-[4px_0_24px_rgba(0,0,0,0.5)]"
            >
                <div className="p-6 border-b border-[var(--border)] relative bg-[var(--bg-darker)]">
                    <div className="absolute top-0 left-0 right-0 heat-divider"></div>
                    <Link to="/" className="flex items-center gap-3 mb-1 hover:opacity-80 transition-opacity">
                        <img src="/KernelForge.svg" alt="Logo" className="w-8 h-8" />
                        <h1
                            className="text-xl font-bold text-[var(--text-light)] tracking-wider uppercase industrial-font-display"
                            style={{"textShadow": "0 0 10px rgba(225, 112, 85, 0.3)"}}
                        >
                            Kernel Forge
                        </h1>
                    </Link>
                    <div className="text-[10px] text-[var(--accent)] font-mono uppercase tracking-widest truncate mt-2">
                        Project: {projectName}
                    </div>
                </div>
                <nav className="flex-1 px-4 py-4 space-y-2 overflow-y-auto charcoal-scrollbar">
                    {nav_links}
                </nav>
                <div className="p-4 border-t border-[var(--border)] space-y-2 bg-[var(--bg-darker)]/30">
                    <Link
                        to="/settings"
                        state={{"return_to": location.pathname}}
                        className="flex items-center gap-3 px-4 py-3 text-[var(--text-muted)] hover:text-[var(--text-light)] w-full rounded-sm hover:bg-[var(--surface-0)] hover:border-[var(--border)] transition-all font-mono text-xs uppercase tracking-widest border border-transparent clip-anvil"
                    >
                        <SettingsIcon className="w-4 h-4"/>
                        <span>
                            Settings
                        </span>
                    </Link>
                    <Link
                        to="/"
                        className="flex items-center gap-3 px-4 py-3 text-[var(--text-muted)] hover:text-[var(--text-light)] w-full rounded-sm hover:bg-[var(--surface-0)] hover:border-[var(--border)] transition-all font-mono text-xs uppercase tracking-widest border border-transparent clip-anvil"
                    >
                        <ArrowRightIcon className="w-4 h-4"/>
                        <span>
                            Back to Projects
                        </span>
                    </Link>
                </div>
            </div>
            <div className="flex-1 overflow-auto bg-[var(--bg-charcoal)]">
                <Outlet/>
            </div>
        </div>;
}
