"""Home Page Component"""

import from react { useEffect, useState, useRef, useLayoutEffect }
import from .SplashScreen { SplashScreen }
import from ..assets.SettingsIcon { SettingsIcon }
import from ..assets.ChevronDownIcon { ChevronDownIcon }
import from ..assets.ArrowRightIcon { ArrowRightIcon }

def:pub Home(onNavigate: any, getProjectsWalker: any) -> any {
    has projects: list = [];
    has loading: bool = True;
    has showSplash: bool = True;
    has exitingSplash: bool = False;
    has isDropdownOpen: bool = False;
    has searchQuery: str = "";
    has dropdownRef: any = useRef(None);

    useLayoutEffect(lambda -> None {
        if window.sessionStorage.getItem("splashShown") {
            showSplash = False;
        }
    }, []);

    useEffect(
        lambda -> None { def handleClickOutside(event: any) -> None {
            if dropdownRef.current and not dropdownRef.current.contains(event.target) {
                isDropdownOpen = False;
            }
        } document.addEventListener("mousedown", handleClickOutside);return lambda -> None { document.removeEventListener(
            "mousedown", handleClickOutside
        );}; },
        [dropdownRef]
    );

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
                # Only run animation if we haven't shown it yet this session
                if not window.sessionStorage.getItem("splashShown") {
                     loading = False;
                     setTimeout(lambda -> None {
                         exitingSplash = True;
                         setTimeout(lambda -> None {
                             showSplash = False;
                             window.sessionStorage.setItem("splashShown", "true");
                         }, 1000);
                     }, 2500);
                } else {
                    loading = False;
                }
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
            <div className={"absolute top-2 right-6 " + ("hidden" if showSplash else "")}>
                <a
                    onClick={lambda -> None { onNavigate("/settings");}}
                    className="flex items-center justify-center p-2 rounded-full hover:bg-zinc-800 transition-colors text-zinc-400 hover:text-white cursor-pointer"
                    aria-label="Settings"
                >
                    <SettingsIcon />
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
                    <div className="relative" ref={dropdownRef}>
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
                                <ChevronDownIcon
                                    className={"text-zinc-500 transition-transform duration-200 " + (
                                        "rotate-180" if isDropdownOpen else ""
                                    )}
                                />
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
                                                <ArrowRightIcon
                                                    className="opacity-0 group-hover:opacity-100 transition-opacity text-zinc-500"
                                                />
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
            { <SplashScreen exiting={exitingSplash} /> if showSplash else null }
        </main>;
}
