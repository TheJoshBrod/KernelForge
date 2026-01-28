"""Home Page Component"""

import from react { useEffect, useState }

def:pub Home(onNavigate: any, getProjectsWalker: any) -> any {
    has projects: list = [];
    has loading: bool = True;
    has isDropdownOpen: bool = False;
    has searchQuery: str = "";

    useEffect(
        lambda -> None { async def fetchProjects -> None {
            try {
                if getProjectsWalker {
                    data = await getProjectsWalker();
                    projects = data["projects"] if data and "projects" in data else [];
                }
            } except Exception as e {
                console.error("Failed to fetch projects", e);
            } finally {
                loading = False;
            }
        } fetchProjects();},
        []
    );

    def toggleDropdown() -> None {
        isDropdownOpen = not isDropdownOpen;
    }

    def selectProject(projectName: str) -> None {
        isDropdownOpen = False;
        searchQuery = "";
        onNavigate(f"/project/{projectName}");
    }

    def handleSearchChange(e: any) -> None {
        searchQuery = e.target.value;
        isDropdownOpen = True;
    }

    recentProjects = projects[:3];
    filteredProjects = [
        p
        for p in projects
        if p.name.lower().includes(searchQuery.lower())
    ];

    return
        <main
            className="flex min-h-screen flex-col items-center justify-center p-6 relative bg-zinc-950 text-zinc-100 selection:bg-zinc-800"
        >
            <div className="absolute top-2 right-6">
                <a
                    onClick={lambda -> None { onNavigate("/settings");}}
                    className="flex items-center justify-center p-2 rounded-full hover:bg-zinc-800 transition-colors text-zinc-400 hover:text-white cursor-pointer"
                    aria-label="Settings"
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
                </a>
            </div>
            <div className="flex flex-col items-center w-full max-w-md space-y-12">
                <div className="text-center space-y-2">
                    <h1
                        className="text-6xl font-bold tracking-tighter bg-gradient-to-b from-white to-zinc-400 bg-clip-text text-transparent"
                    >
                        CGinS
                    </h1>
                    <p
                        className="text-zinc-500 font-medium tracking-wide uppercase text-sm"
                    >
                        Agentic PyTorch Model Optimizer
                    </p>
                </div>
                <div className="flex flex-col sm:flex-row gap-4 w-full">
                    <button
                        className="flex-1 h-12 bg-zinc-950 hover:bg-zinc-900 border border-zinc-800 text-zinc-100 font-medium rounded-lg transition-all duration-200 ease-in-out shadow-[0_0_20px_rgba(255,255,255,0.1)] hover:shadow-[0_0_30px_rgba(255,255,255,0.2)]"
                    >
                        Upload Full Model
                    </button>
                    <a
                        onClick={lambda -> None { onNavigate("/new-project");}}
                        className="flex-1 cursor-pointer"
                    >
                        <button
                            className="w-full h-12 flex items-center justify-center text-zinc-300 font-medium rounded-lg transition-all duration-200 rainbow-shadow"
                        >
                            Upload Weights
                        </button>
                    </a>
                </div>
                <div className="w-full space-y-2 relative group">
                    <div
                        className="text-xs font-semibold text-zinc-500 uppercase tracking-wider ml-1"
                    >
                        Load Existing
                    </div>
                    <div className="relative">
                        <div
                            className="w-full h-12 flex items-center justify-between bg-zinc-900/50 border border-zinc-800 hover:border-zinc-700 hover:bg-zinc-900/80 rounded-lg px-4 text-zinc-300 transition-all cursor-text group"
                        >
                            <input
                                type="text"
                                placeholder="Select or search a project..."
                                value={searchQuery}
                                onChange={handleSearchChange}
                                onClick={lambda -> None { isDropdownOpen = True;}}
                                className="bg-transparent border-none outline-none w-full text-zinc-300 placeholder-zinc-500 h-full"
                            />
                            <button
                                onClick={lambda -> None { toggleDropdown();}}
                                className="ml-2 focus:outline-none"
                            >
                                <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    width="16"
                                    height="16"
                                    viewBox="0 0 24 24"
                                    fill="none"
                                    stroke="currentColor"
                                    strokeWidth="2"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    className={"text-zinc-500 transition-transform duration-200 " + (
                                        "rotate-180" if isDropdownOpen else ""
                                    )}
                                >
                                    <path d="m6 9 6 6 6-6" />
                                </svg>
                            </button>
                        </div>
                        {(
                            <div
                                className="absolute top-full left-0 w-full mt-2 bg-zinc-950 border border-zinc-800 rounded-lg shadow-[0_4px_20px_rgba(0,0,0,0.5)] overflow-hidden z-50 animate-in fade-in zoom-in-95 duration-100"
                            >
                                <div className="max-h-60 overflow-y-auto py-1">
                                    {(
                                        <div
                                            className="px-4 py-3 text-sm text-zinc-500 text-center italic"
                                        >
                                            No projects found
                                        </div>
                                    )
                                    if filteredProjects.length == 0
                                    else (
                                        [
                                            <div
                                                key={project.name}
                                                onClick={lambda -> None { selectProject(
                                                    project.name
                                                );}}
                                                className="w-full text-left px-4 py-2.5 hover:bg-zinc-800 text-zinc-400 hover:text-zinc-100 transition-colors cursor-pointer text-sm flex items-center justify-between group"
                                            >
                                                <span>
                                                    {project.name}
                                                </span>
                                                <svg
                                                    xmlns="http://www.w3.org/2000/svg"
                                                    width="14"
                                                    height="14"
                                                    viewBox="0 0 24 24"
                                                    fill="none"
                                                    stroke="currentColor"
                                                    strokeWidth="2"
                                                    strokeLinecap="round"
                                                    strokeLinejoin="round"
                                                    className="opacity-0 group-hover:opacity-100 transition-opacity text-zinc-500"
                                                >
                                                    <path d="M5 12h14" />
                                                    <path d="m12 5 7 7-7 7" />
                                                </svg>
                                            </div> for project in filteredProjects
                                        ]
                                    )}
                                </div>
                            </div>
                        )
                        if isDropdownOpen
                        else null}
                    </div>
                </div>
                <div className="w-full flex flex-col items-center space-y-3 pt-4">
                    <h3 className="text-sm font-medium text-zinc-400">
                        Recent Projects
                    </h3>
                    <div className="flex flex-col gap-2 text-center">
                        {(
                            [
                                <a
                                    key={project.name}
                                    onClick={lambda -> None { onNavigate(
                                        f"/project/{project.name}"
                                    );}}
                                    className="text-blue-400 hover:text-blue-300 transition-colors text-sm hover:underline underline-offset-4 cursor-pointer"
                                >
                                    {project.name}
                                </a> for project in recentProjects
                            ]
                        )
                        if recentProjects.length > 0
                        else <p className="text-zinc-600 text-xs italic">
                            No recently accessed projects
                        </p>}
                    </div>
                </div>
            </div>
        </main>;
}
