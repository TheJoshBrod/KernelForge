"""Home Page Component"""

import from react { useEffect }
import from "react-router-dom" { useNavigate }
import from ...components.backdrops.GameOfLifeBackground { GameOfLifeBackground }
import from ...components.ui.GlowButton { GlowButton }
import from ...components.ui.Icons { CGinSLogo, SettingsIcon, ArrowRightIcon }
import from ...components.SplashScreen { SplashScreen }



def:pub Home -> Any {
    navigate = useNavigate();
    has projects: list = [],
        loading: bool = True,
        splashDone: bool = (sessionStorage.getItem("splash_shown") == "true"),
        searchQuery: str = "";

    async def fetchProjects -> None {
        start: float = Date.now();
        try {
            resp = await __jacSpawn("GetProjects", "", {});
            
            data = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else None;
            
            if data and data["projects"] {
                projects = data["projects"];
            } else {
                projects = [];
            }
        } except Exception as e {
            console.error("Failed to fetch projects", e);
        } finally {
            elapsed = Date.now() - start;
            if not splashDone and elapsed < 3000 {
                def sleeper(resolve: any, reject: any) -> None {
                    setTimeout(resolve, 3000 - elapsed);
                }
                await Reflect.construct(Promise, [sleeper]);
                sessionStorage.setItem("splash_shown", "true");
            }
            loading = False;
            splashDone = True;
        }
    }

    useEffect(lambda -> None { fetchProjects();}, []);

    # Filter Projects
    filteredProjects = [];
    for p in projects {
        try {
            if searchQuery == "" or p["name"].toLowerCase().includes(searchQuery.toLowerCase()) {
                filteredProjects.append(p);
            }
        } except Exception as e {
            console.error("Error filtering project:", p, e);
        }
    }

    # Generate Project List Items
    projectList = [];
    if loading {
        projectList.append(
            <div className="text-zinc-500 text-center py-4 italic">
                Loading projects...
            </div>
        );
    } elif filteredProjects.length == 0 {
        projectList.append(
             <div className="text-zinc-500 text-center py-4">
                { "No projects found." if projects.length == 0 else "No matching projects." }
            </div>
        );
    } else {
        for innerProj in filteredProjects {
            if innerProj {
                projName = innerProj["name"];
                clickHandler = lambda ignore: any -> None { navigate("/project/" + projName); };
                
                projectList.append(
                    <button
                        key={projName}
                        onClick={clickHandler}
                        className="w-full text-left p-4 rounded-lg hover:bg-zinc-800/50 transition-colors border border-transparent hover:border-zinc-700 group flex items-center justify-between gap-4"
                    >
                        <span className="font-medium text-zinc-200 group-hover:text-white text-lg truncate min-w-0 flex-1">
                            {projName}
                        </span>
                        <span className="text-zinc-600 group-hover:text-zinc-400 text-sm shrink-0 flex items-center gap-1 transition-transform group-hover:translate-x-1">
                            Open <ArrowRightIcon className="w-4 h-4" />
                        </span>
                    </button>
                );
            }
        }
    }



    return
        <main className="relative h-screen w-screen overflow-hidden bg-black flex items-center justify-center font-sans text-white">
            <SplashScreen exiting={splashDone} />
            <GameOfLifeBackground className="absolute inset-0 z-0" />
            
            <div className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-black/50 z-10 pointer-events-none"></div>

            <div className="absolute top-6 right-6 z-50">
                <button
                    onClick={lambda -> None { navigate("/settings", {"state": {"return_to": "/"}}); }}
                    className="p-2 rounded-full hover:bg-white/10 text-white/70 hover:text-white transition-all duration-200"
                    title="Settings"
                >
                    <SettingsIcon className="w-6 h-6" />
                </button>
            </div>

            <div className="relative z-10 w-full max-w-4xl h-[80vh] bg-zinc-900/80 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl flex flex-col animation-fade-in-up">
                
                <div className="p-6 border-b border-white/5 space-y-4 shrink-0">
                    <div className="flex items-center gap-4">
                        <div className="p-2 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg shadow-lg">
                            <CGinSLogo className="w-8 h-8 text-white" />
                        </div>
                        <div>
                             <h1 className="text-xl font-bold text-white tracking-tight">
                                Select a Project
                            </h1>
                            <p className="text-xs text-zinc-400 font-medium uppercase tracking-wider">
                                CGinS Console
                            </p>
                        </div>
                    </div>
                    
                    <div className="relative">
                        <input
                            type="text"
                            placeholder="Search projects..."
                            value={searchQuery}
                            onChange={lambda e: any -> None { searchQuery = e.target.value; }}
                            className="w-full bg-black/40 border border-white/10 rounded-lg px-4 py-2 text-sm text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/20 transition-all"
                        />
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto p-4 space-y-2 min-h-0 custom-scrollbar">
                    {projectList}
                </div>

                <div className="p-6 border-t border-white/5 bg-zinc-900/50 shrink-0">
                    <GlowButton
                        label="+ New Project"
                        onClick={lambda -> None { navigate("/new-project"); }}
                        className="w-full py-3 text-base justify-center font-semibold"
                    />
                </div>
            </div>
        </main>;
}
