"""Home Page Component"""

import from react { useEffect, useState }

def:pub Home(onNavigate: any, getProjectsWalker: any) -> any {
    has projects: list = [];
    has loading: bool = True;

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

    def handleProjectSelect(e: any) -> None {
        projectId = e.target.value;
        if projectId {
            onNavigate(f"/project/{projectId}");
        }
    }

    recentProjects = projects[:3];

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
                            className="w-full h-12 flex items-center justify-center bg-zinc-950 hover:bg-zinc-900 border border-zinc-800 hover:border-zinc-500 text-zinc-300 font-medium rounded-lg transition-all duration-200 shadow-[0_0_20px_rgba(255,255,255,0.1)] hover:shadow-[0_0_30px_rgba(255,255,255,0.2)]"
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
                        <select
                            defaultValue=""
                            onChange={handleProjectSelect}
                            className="w-full h-12 appearance-none bg-zinc-900/50 border border-zinc-800 hover:border-zinc-700 rounded-lg px-4 text-zinc-300 focus:outline-none focus:ring-1 focus:ring-zinc-600 focus:border-zinc-600 transition-all cursor-pointer"
                        >
                            <option value="" disabled>
                                Select a project to load...
                            </option>
                            {[
                                <option key={project.name} value={project.name}>
                                    {project.name}
                                </option> for project in projects
                            ]}
                        </select>
                        <div
                            className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-zinc-500"
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
                            >
                                <path d="m6 9 6 6 6-6" />
                            </svg>
                        </div>
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
