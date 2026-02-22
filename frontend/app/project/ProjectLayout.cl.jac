"""Project Layout Component"""

import from react { useState, useEffect }
import from "react-router-dom" { Outlet, Link, useLocation, useParams }
import from ...components.ui.Icons { HomeIcon, ArrowRightIcon, CodeIcon, UploadIcon, SettingsIcon, MenuIcon, ChevronLeftIcon, ChevronRightIcon }

def:pub ProjectLayout -> Any {
    location = useLocation();
    params = useParams();
    projectName = params.projectName;

    useEffect(
        lambda -> None {
            if projectName {
                document.title = f"{projectName} - Kernel Forge";
            } else {
                document.title = "Kernel Forge";
            }
            return lambda -> None {
                document.title = "Kernel Forge";
            };
        },
        [projectName]
    );

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
        },
        {
            "name": "Export",
            "href": f"/project/{projectName}/export",
            "icon": UploadIcon,
            "exact": False
        }
    ];

    collapsed_state = useState(False);
    isCollapsed = collapsed_state[0];
    setIsCollapsed = collapsed_state[1];

    def toggleCollapse -> None {
        setIsCollapsed(not isCollapsed);
    }

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
                    "justify-center px-0 " if isCollapsed else ""
                ) + (
                    "bg-[var(--bg-light)] text-[var(--accent)] !border-l-2 !border-l-[var(--accent)] shadow-[inset_4px_0_0_var(--accent)]"
                    if is_active(item)
                    else "text-[var(--text-muted)] hover:text-[var(--text-light)] hover:bg-[var(--surface-0)] hover:border-[var(--border)]"
                )}
                title={(item["name"] if isCollapsed else "")}
            >
                <Icon className={"w-4 h-4 " + ("shrink-0" if isCollapsed else "")}/>
                <span className={"truncate " + ("hidden" if isCollapsed else "block")}>
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
                className={"h-full bg-[var(--bg-panel)] border-r border-[var(--border)] flex flex-col flex-shrink-0 relative z-20 shadow-[4px_0_24px_rgba(0,0,0,0.5)] transition-all duration-300 ease-in-out " + (
                    "w-16" if isCollapsed else "w-64"
                )}
            >
                <div className={"p-6 border-b border-[var(--border)] relative bg-[var(--bg-darker)] " + ("flex justify-center px-2 py-4" if isCollapsed else "")}>
                    <div className="absolute top-0 left-0 right-0 heat-divider"></div>

                    <div className={"flex items-center justify-between mb-1 " + ("hidden" if isCollapsed else "flex")}>
                        <Link to="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
                            <img src="/KernelForge.svg" alt="Logo" className="w-8 h-8 shrink-0" />
                            <h1
                                className="text-xl font-bold text-[var(--text-light)] tracking-wider uppercase industrial-font-display truncate"
                                style={{"textShadow": "0 0 10px rgba(225, 112, 85, 0.3)"}}
                            >
                                Kernel Forge
                            </h1>
                        </Link>
                        <button
                            onClick={toggleCollapse}
                            className="p-1 hover:bg-[var(--surface-0)] rounded text-[var(--text-muted)] hover:text-[var(--text-light)] transition-colors border border-transparent hover:border-[var(--border)] shrink-0"
                            title="Collapse Sidebar"
                        >
                            <ChevronLeftIcon className="w-4 h-4"/>
                        </button>
                    </div>
                    <div className={"text-[10px] text-[var(--accent)] font-mono uppercase tracking-widest truncate mt-2 " + ("hidden" if isCollapsed else "block")}>
                        Project: {projectName}
                    </div>

                    <div className={"flex justify-center " + ("hidden" if not isCollapsed else "flex")}>
                        <button
                            onClick={toggleCollapse}
                            className="group relative flex items-center justify-center w-10 h-10 rounded-md hover:bg-[var(--surface-0)] border border-transparent hover:border-[var(--border)] transition-all"
                            title="Expand Sidebar"
                        >
                            <img src="/KernelForge.svg" alt="Logo" className="w-8 h-8 group-hover:hidden" />
                            <ChevronRightIcon className="w-6 h-6 hidden group-hover:block text-[var(--text-muted)] group-hover:text-[var(--accent)]" />
                        </button>
                    </div>
                </div>
                <nav className={"flex-1 py-4 space-y-2 overflow-y-auto charcoal-scrollbar " + ("px-2" if isCollapsed else "px-4")}>
                    {nav_links}
                </nav>
                <div className={"p-4 border-t border-[var(--border)] space-y-2 bg-[var(--bg-darker)]/30 " + ("px-2" if isCollapsed else "")}>
                    <Link
                        to="/settings"
                        state={{"return_to": location.pathname}}
                        className={"flex items-center gap-3 px-4 py-3 text-[var(--text-muted)] hover:text-[var(--text-light)] w-full rounded-sm hover:bg-[var(--surface-0)] hover:border-[var(--border)] transition-all font-mono text-xs uppercase tracking-widest border border-transparent clip-anvil " + (
                            "justify-center px-0" if isCollapsed else ""
                        )}
                        title={( "Settings" if isCollapsed else "" )}
                    >
                        <SettingsIcon className={"w-4 h-4 " + ("shrink-0" if isCollapsed else "")}/>
                        <span className={"truncate " + ("hidden" if isCollapsed else "block")}>
                            Settings
                        </span>
                    </Link>
                    <Link
                        to="/"
                        className={"flex items-center gap-3 px-4 py-3 text-[var(--text-muted)] hover:text-[var(--text-light)] w-full rounded-sm hover:bg-[var(--surface-0)] hover:border-[var(--border)] transition-all font-mono text-xs uppercase tracking-widest border border-transparent clip-anvil " + (
                            "justify-center px-0" if isCollapsed else ""
                        )}
                        title={( "Back to Projects" if isCollapsed else "" )}
                    >
                        <ArrowRightIcon className={"w-4 h-4 " + ("shrink-0" if isCollapsed else "")}/>
                        <span className={"truncate " + ("hidden" if isCollapsed else "block")}>
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
