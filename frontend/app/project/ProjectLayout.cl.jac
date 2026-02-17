"""Project Layout Component"""

import from "react-router-dom" { Outlet, Link, useLocation, useParams }
import from ...components.ui.Icons { HomeIcon, ArrowRightIcon, CodeIcon, SettingsIcon }

def:pub ProjectLayout {
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
            "name": "Optimization Log",
            "href": f"/project/{projectName}/workbench",
            "icon": CodeIcon,
            "exact": False
        }
    ];

    def is_active(item: dict) -> bool {
        if item["exact"] {
            return location.pathname == item["href"];
        }
        return location.pathname.startswith(item["href"]);
    }

    nav_links = [];
    for item in navItems {
        Icon = item["icon"];
        nav_links.push(
            <Link
                key={item["name"]}
                to={item["href"]}
                className={"flex items-center gap-3 px-4 py-3 rounded-lg transition-all " + (
                    "bg-zinc-900 text-white"
                    if is_active(item)
                    else "text-zinc-400 hover:text-white hover:bg-zinc-900"
                )}
            >
                <Icon className="w-[18px] h-[18px]" />
                <span className="text-sm font-medium">
                    {item["name"]}
                </span>
            </Link>
        );
    }

    return
        <div
            className="flex h-screen bg-zinc-950 text-zinc-100 overflow-hidden font-sans"
        >
            <div
                className="w-64 h-full bg-zinc-950 border-r border-zinc-800 flex flex-col flex-shrink-0"
            >
                <div className="p-6">
                    <h1
                        className="text-xl font-bold text-white tracking-tight flex items-center gap-2"
                    >
                        <span>
                            Kernel
                        </span>
                        <span className="text-indigo-500">
                            Forge
                        </span>
                    </h1>
                    <div className="text-xs text-zinc-500 mt-1 truncate">
                        {projectName}
                    </div>
                </div>
                <nav className="flex-1 px-4 space-y-2 overflow-y-auto">
                    {nav_links}
                </nav>
                <div className="p-4 border-t border-zinc-800">
                    <Link
                        to="/settings"
                        className="flex items-center gap-3 px-4 py-3 text-zinc-400 hover:text-white w-full rounded-lg hover:bg-zinc-900 transition-all"
                    >
                        <SettingsIcon className="w-[18px] h-[18px]" />
                        <span className="text-sm">
                            Settings
                        </span>
                    </Link>
                    <Link
                        to="/"
                        className="flex items-center gap-3 px-4 py-3 text-zinc-400 hover:text-white w-full rounded-lg hover:bg-zinc-900 transition-all"
                    >
                        <ArrowRightIcon className="w-[18px] h-[18px]" />
                        <span className="text-sm">
                            Back to Projects
                        </span>
                    </Link>
                </div>
            </div>
            <div className="flex-1 overflow-auto bg-zinc-900">
                <Outlet />
            </div>
        </div>;
}
