"""Home Page Component"""

import from react { useEffect, useState, useRef, useLayoutEffect }
import from .SplashScreen { SplashScreen }
import from ..assets.SettingsIcon { SettingsIcon }
import from ..assets.ArrowRightIcon { ArrowRightIcon }
import from ..assets.CGinSLogo { CGinSLogo }


def:pub Home(onNavigate: any, getProjectsWalker: any, getSystemInfoWalker: any) -> any {
    has projects: list = [];
    has loading: bool = True;
    has showSplash: bool = True;
    has exitingSplash: bool = False;
    has isDropdownOpen: bool = False;
    has searchQuery: str = "";
    has dropdownRef: any = useRef(None);
    has systemInfo: dict = {};

    useLayoutEffect(lambda -> None {
        if window.sessionStorage.getItem("splashShown") {
            showSplash = False;
        }
    }, []);

    useEffect(
        lambda -> None {
            async def fetchSystemInfo -> None {
                if getSystemInfoWalker {
                    info = await getSystemInfoWalker();
                    systemInfo = info;
                }
            }
            fetchSystemInfo();
        }, []
    );

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
            <div className="absolute top-2 right-6">
                <a
                    onClick={lambda -> None { onNavigate("/settings");}}
                    className="flex items-center justify-center p-2 rounded-full hover:bg-zinc-800 transition-colors text-zinc-400 hover:text-white cursor-pointer"
                    aria-label="Settings"
                >
                    <SettingsIcon />
                </a>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-16 w-full max-w-6xl items-start">
                <div className="flex flex-col space-y-12 pt-8">
                    <div className="text-left space-y-2">
                        <div className="flex items-center gap-2">
                            <h1
                                className="text-6xl font-bold tracking-tighter bg-gradient-to-b from-white to-zinc-400 bg-clip-text text-transparent"
                            >
                                CGinS
                            </h1>
                            <CGinSLogo className="w-24 h-24" />
                        </div>
                        <p
                            className="text-zinc-500 font-medium tracking-wide uppercase text-sm"
                        >
                            Agentic PyTorch Model Optimizer
                        </p>
                        {
                            <div className="flex flex-col gap-1 pt-1">
                                {(
                                    [
                                        <div className="flex items-center gap-2 text-xs text-zinc-600 font-mono" key={index}>
                                            <span className="w-2 h-2 rounded-full bg-green-500/50 block shadow-[0_0_8px_rgba(34,197,94,0.4)] animate-pulse"></span>
                                            <span>{gpu["name"]}</span>
                                            <span className="text-zinc-700">|</span>
                                            <span>{gpu["memory"]} VRAM</span>
                                        </div>
                                        for gpu in systemInfo["gpus"]
                                    ]
                                )
                                if systemInfo and "available" in systemInfo and systemInfo["available"] and "gpus" in systemInfo
                                else (
                                    <div className="flex items-center gap-2 text-xs text-zinc-600 font-mono">
                                        <span className="w-2 h-2 rounded-full bg-red-500/50 block shadow-[0_0_8px_rgba(239,68,68,0.4)] animate-pulse"></span>
                                        <span>NVIDIA GPU not found</span>
                                    </div>
                                )}
                            </div>
                            if systemInfo
                            else null
                        }
                    </div>
                    
                    <div className="flex flex-col gap-4 w-full max-w-md">
                        <button
                            className="w-full h-14 bg-zinc-950 hover:bg-zinc-900 border border-zinc-800 text-zinc-100 font-medium rounded-lg transition-all duration-200 ease-in-out shadow-[0_0_20px_rgba(255,255,255,0.1)] hover:shadow-[0_0_30px_rgba(255,255,255,0.2)] text-lg"
                        >
                            Upload Full Model
                        </button>
                        <a
                            onClick={lambda -> None { onNavigate("/new-project");}}
                            className="w-full cursor-pointer"
                        >
                            <button
                                className="w-full h-14 flex items-center justify-center text-zinc-300 font-medium rounded-lg transition-all duration-200 rainbow-shadow text-lg"
                            >
                                Upload Weights
                            </button>
                        </a>
                    </div>
                </div>

                <div className="flex flex-col space-y-4 w-full max-w-md ml-auto">
                    <div className="w-full space-y-2">
                        <div className="text-xs font-semibold text-zinc-500 uppercase tracking-wider ml-1">
                            Load Project
                        </div>
                        
                        <div className="w-full bg-zinc-900/30 border border-zinc-800 rounded-lg overflow-hidden flex flex-col">
                            <div className="p-2 border-b border-zinc-900">
                                <div className="flex items-center bg-zinc-900/50 rounded-md px-3 h-10 border border-zinc-800/50 focus-within:border-zinc-700 transition-colors">
                                    <input
                                        type="text"
                                        placeholder="Search projects..."
                                        value={searchQuery}
                                        onChange={handleSearchChange}
                                        className="bg-transparent border-none outline-none w-full text-zinc-300 placeholder-zinc-600 text-sm h-full"
                                    />
                                </div>
                            </div>

                            
                            <div className="h-[400px] overflow-y-auto custom-scrollbar flex flex-col">
                                {(
                                    <div className="flex-1 flex items-center justify-center text-sm text-zinc-600 italic">
                                        No projects found
                                    </div>
                                ) 
                                if filteredProjects.length == 0
                                else (
                                    [
                                        <div
                                            key={project.name}
                                            onClick={lambda -> None { selectProject(project.name);}}
                                            className="w-full text-left px-4 py-3 hover:bg-zinc-900/50 text-zinc-400 hover:text-zinc-100 transition-colors cursor-pointer text-sm flex items-center justify-between group border-b border-zinc-900 last:border-0 shrink-0"
                                        >
                                            <span>{project.name}</span>
                                            <ArrowRightIcon className="opacity-0 group-hover:opacity-100 transition-opacity text-zinc-600 h-4 w-4" />
                                        </div> 
                                        for project in filteredProjects[:10]
                                    ]
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            { <SplashScreen exiting={exitingSplash} /> if showSplash else null }
        </main>;
}
