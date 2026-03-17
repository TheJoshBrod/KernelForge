"""Home Page Component"""

import from react { useEffect, useState }
import from "react-router-dom" { useNavigate }
import from ...components.backdrops.EmberBackground { EmberBackground }
import from ...components.ui.GlowButton { GlowButton }
import from ...components.ui.Icons { KernelForgeLogo, SettingsIcon, ArrowRightIcon, GithubIcon, DiscordIcon, TrashIcon }
import from ...components.SplashScreen { SplashScreen }

def:pub Home -> Any {
    navigate = useNavigate();
    desktop_build = window["__KFORGE_DESKTOP_BUILD__"]
    if "__KFORGE_DESKTOP_BUILD__" in window and window["__KFORGE_DESKTOP_BUILD__"]
    else None;
    min_splash_ms = 700
    if desktop_build and "desktop" in desktop_build and desktop_build["desktop"]
    else 3000;

    projects_state = useState([]);
    projects = projects_state[0];
    setProjects = projects_state[1];

    loading_state = useState(True);
    loading = loading_state[0];
    setLoading = loading_state[1];

    splash_done_state = useState(sessionStorage.getItem("splash_shown") == "true");
    splashDone = splash_done_state[0];
    setSplashDone = splash_done_state[1];

    search_query_state = useState("");
    searchQuery = search_query_state[0];
    setSearchQuery = search_query_state[1];

    project_to_delete_state = useState("");
    projectToDelete = project_to_delete_state[0];
    setProjectToDelete = project_to_delete_state[1];

    async def confirmDelete -> None {
        if not projectToDelete { return; }
        deletingName = projectToDelete;
        setProjectToDelete("");
        setProjects(projects.filter(lambda p: any -> bool { return p["name"] != deletingName; }));
        try {
            await __jacSpawn("DeleteProject", "", {"projectName": deletingName});
        } except Exception as e {
            console.error("Failed to delete project:", e);
            await fetchProjects();
        }
    }

    async def fetchProjects -> None {
        start: float = Date.now();
        try {
            resp = await __jacSpawn("GetProjects", "", {});

            data = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else None;

            if data and data["projects"] {
                setProjects(data["projects"]);
            } else {
                setProjects([]);
            }
        } except Exception as e {
            console.error("Failed to fetch projects", e);
        } finally {
            elapsed = Date.now() - start;
            if not splashDone and elapsed < min_splash_ms {
                def sleeper(resolve: any, reject: any) -> None {
                    setTimeout(resolve, min_splash_ms - elapsed);
                }
                await Reflect.construct(Promise, [sleeper]);
                sessionStorage.setItem("splash_shown", "true");
            }
            setLoading(False);
            setSplashDone(True);
        }
    }

    useEffect(lambda -> None {
        fetchProjects();
        document.title = "Kernel Forge";
    }, []);

    # Filter Projects
    filteredProjects = [];
    for p in projects {
        try {
            if searchQuery == ""
            or p["name"].toLowerCase().includes(searchQuery.toLowerCase()) {
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
            <div className="text-zinc-500 text-center py-4 italic font-mono">
                INITIALIZING...
            </div>
        );
    } elif filteredProjects.length == 0 {
        projectList.append(
            <div
                className="text-zinc-500 text-center py-4 font-mono uppercase tracking-widest text-xs"
            >
                {"NO PROJECTS FOUND"
                if projects.length == 0
                else "NO MATCHING PROJECTS"}
            </div>
        );
    } else {
        for innerProj in filteredProjects {
            if innerProj {
                projName = innerProj["name"];
                clickHandler = lambda ignore: any  -> None { navigate(
                    "/project/" + ignore.currentTarget.getAttribute("data-projname")
                );};
                deleteHandler = lambda e: any -> None {
                    e.stopPropagation();
                    setProjectToDelete(e.currentTarget.getAttribute("data-projname"));
                };

                projectList.append(
                    <div key={projName} className="flex w-full mb-2 bg-[var(--surface)] border-l-2 border-transparent hover:border-[var(--accent)] transition-all clip-anvil overflow-hidden">
                        <button
                            data-projname={projName}
                            onClick={clickHandler}
                            className="group flex-1 text-left p-4 flex items-center justify-between gap-4 hover:bg-[var(--surface-0)] transition-colors"
                        >
                            <span
                                className="font-medium text-[var(--text-light)] group-hover:text-white text-lg truncate min-w-0 flex-1 font-sans"
                            >
                                {projName}
                            </span>
                            <span
                                className="text-[var(--text-muted)] group-hover:text-[var(--accent)] text-sm shrink-0 flex items-center gap-1 transition-transform group-hover:translate-x-1 font-mono uppercase tracking-wider text-xs"
                            >
                                Load
                                <ArrowRightIcon className="w-3 h-3"/>
                            </span>
                        </button>
                        <button
                            data-projname={projName}
                            onClick={deleteHandler}
                            className="p-4 px-6 bg-red-500/10 hover:bg-red-500/20 text-red-500/70 hover:text-red-500 transition-all border-l border-red-500/20"
                            title="Delete Project"
                        >
                            <TrashIcon className="w-5 h-5"/>
                        </button>
                    </div>
                );
            }
        }
    }

    deleteModal = None;
    if projectToDelete != "" {
        deleteModal = (
            <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm animation-fade-in">
                <div className="bg-[var(--bg-panel)] border border-[var(--border)] p-8 max-w-md w-full relative overflow-hidden clip-anvil">
                    <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-red-600 to-orange-500"></div>
                    <h2 className="text-xl font-bold text-white mb-4 industrial-font-display uppercase tracking-wider">
                        Delete Project
                    </h2>
                    <p className="text-[var(--text-light)] mb-8 font-sans">
                        Are you sure you want to delete <span className="text-white font-mono break-all">{projectToDelete}</span>? This action cannot be undone and will destroy all generated kernels and benchmarks.
                    </p>
                    <div className="flex justify-end gap-4">
                        <button
                            onClick={lambda -> None { setProjectToDelete(""); }}
                            className="px-4 py-2 border border-[var(--border)] text-[var(--text-muted)] hover:text-white hover:border-[var(--text-muted)] transition-colors font-mono uppercase tracking-widest text-sm"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={confirmDelete}
                            className="px-4 py-2 bg-red-600/20 border border-red-500/50 text-red-500 hover:bg-red-600 hover:text-white hover:border-red-600 transition-colors font-mono uppercase tracking-widest text-sm clip-anvil"
                        >
                            Delete
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    return
        <main
            className="relative h-screen w-screen overflow-hidden bg-[var(--bg-charcoal)] flex flex-col items-center justify-center font-sans text-white"
        >
            <SplashScreen exiting={splashDone}/>
            <div className="absolute inset-0 z-0 industrial-grid-bg opacity-30"></div>
            <div className="noise-overlay opacity-30"></div>
            <EmberBackground className="absolute inset-0 z-0"/>
            <div
                className="absolute inset-0 bg-gradient-to-t from-black/30 via-transparent to-black/50 z-10 pointer-events-none"
            ></div>
            {(
                <div className="absolute top-6 right-8 z-50 flex items-center gap-6">
                    <a
                        href="https://discord.gg/"
                        target="_blank"
                        rel="noreferrer"
                        className="text-white/50 hover:text-[#5865F2] transition-all duration-300 hover:scale-110 hover:-translate-y-1 hover:drop-shadow-[0_0_12px_rgba(88,101,242,0.6)]"
                        title="Discord Server"
                    >
                        <DiscordIcon className="w-7 h-7"/>
                    </a>
                    <a
                        href="https://github.com/TheJoshBrod/CGinS"
                        target="_blank"
                        rel="noreferrer"
                        className="text-white/50 hover:text-white transition-all duration-300 hover:scale-110 hover:-translate-y-1 hover:drop-shadow-[0_0_12px_rgba(255,255,255,0.6)]"
                        title="GitHub Repository"
                    >
                        <GithubIcon className="w-7 h-7"/>
                    </a>
                </div>
            ) if splashDone else None}
            <div className="relative z-20 flex flex-col items-center justify-center mb-8 animation-fade-in-up">
                <div className="flex items-center gap-6 mb-2">
                    <img src="/KernelForge.svg" alt="Kernel Forge Logo" className="w-16 h-16" />
                    <div className="flex flex-col">
                        <h1 className="text-5xl font-bold text-white tracking-wider uppercase industrial-font-display" style={{"textShadow": "0 0 20px rgba(225, 112, 85, 0.4)"}}>
                            Kernel Forge
                        </h1>
                    </div>
                </div>
            </div>
            <div
                className="relative z-10 w-full max-w-4xl h-[70vh] bg-[var(--bg-panel)]/90 backdrop-blur-md border border-[var(--border)] rounded-sm shadow-2xl flex flex-col animation-fade-in-up delay-1 clip-hex"
            >
                <div
                    className="p-6 border-b border-[var(--border)] space-y-4 shrink-0 relative overflow-hidden"
                >
                    <div
                        className="absolute top-0 left-0 right-0 heat-divider"
                    ></div>
                    <div className="relative flex items-center gap-4">
                        <input
                            type="text"
                            placeholder="SEARCH PROJECTS..."
                            value={searchQuery}
                            onChange={lambda e: any  -> None { setSearchQuery(e.target.value);}}
                            className="w-full bg-[var(--bg-darker)] border border-[var(--border)] rounded-sm px-4 py-2 text-sm text-[var(--text-light)] placeholder-[var(--text-muted)] focus:outline-none focus:border-[var(--accent)] focus:ring-1 focus:ring-[var(--accent)]/20 transition-all font-mono uppercase"
                        />
                        <button
                            onClick={lambda -> None { navigate(
                                "/settings", {"state": {"return_to": "/"}}
                            );}}
                            className="p-2 rounded-sm bg-[var(--bg-darker)] border border-[var(--border)] hover:border-[var(--accent)] text-[var(--text-muted)] hover:text-[var(--accent)] transition-all shrink-0"
                            title="Settings"
                        >
                            <SettingsIcon className="w-5 h-5"/>
                        </button>
                    </div>
                </div>
                <div
                    className="flex-1 overflow-y-auto p-4 space-y-2 min-h-0 charcoal-scrollbar bg-[var(--bg-darker)]/30"
                >
                    {projectList}
                </div>
                <div
                    className="p-6 border-t border-[var(--border)] bg-[var(--bg-panel)] shrink-0"
                >
                    <GlowButton
                        label="+ INITIATE NEW PROJECT"
                        onClick={lambda -> None { navigate("/new-project");}}
                        className="w-full py-3 text-base justify-center font-bold tracking-widest clip-anvil"
                    />
                </div>
            </div>

            {deleteModal}
        </main>;
}
