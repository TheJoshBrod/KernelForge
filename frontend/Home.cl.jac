"""Home Page Component"""

import from react { useEffect }
import from "@jac/runtime" { jacSpawn }
import from "react-router-dom" { useNavigate }
import from "backdrops/GameOfLifeBackground" { GameOfLifeBackground }
import from "SplashScreen" { SplashScreen }
import from .Mascot { Mascot }

def:pub Home -> any {
    navigate = useNavigate();
    has projects: list = [],
        loading: bool = True,
        splashDone: bool = False;

    async def fetchProjects -> None {
        try {
            resp = await jacSpawn("GetProjects", "", {});
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
            loading = False;
            splashDone = True;
        }
    }

    useEffect(lambda -> None { fetchProjects();}, []);

    def handleProjectSelect(e: any) -> None {
        projectId = e.target.value;
        if projectId {
            navigate("/project/" + projectId);
        }
    }

    recentProjects = [];
    recentLinks = [];
    projectOptions = [];
    hasRecent = False;
    idx = 0;
    for project in projects {
        projectOptions = projectOptions.concat(
            [
                <option key={project["name"]} value={project["name"]}>
                    {project["name"]}
                </option>
            ]
        );
        if idx < 4 {
            recentProjects = recentProjects.concat([project]);
            hasRecent = True;
            idx = idx + 1;
        }
    }
    for project in recentProjects {
        recentLinks = recentLinks.concat(
            [
                <button
                    key={project["name"]}
                    onClick={lambda -> None { navigate("/project/" + project["name"]);}}
                    className="launcher-link"
                >
                    {project["name"]}
                </button>
            ]
        );
    }

    return
        <main className="app-shell launcher-shell">
            <SplashScreen exiting={splashDone} />
            <GameOfLifeBackground className="app-life-canvas--subtle" />
            <div className="app-backdrop app-backdrop--natural"></div>
            <div className="launcher-container app-content-layer">
                <header className="launcher-header">
                    <div className="launcher-brand">
                        <Mascot
                            variant="logo"
                            ambient={True}
                            className="launcher-logo"
                        />
                        <div>
                            <div className="launcher-title">
                                CGinS
                            </div>
                            <div className="launcher-subtitle">
                                Project Console
                            </div>
                        </div>
                    </div>
                </header>
                <section className="launcher-grid">
                    <div className="panel launcher-panel">
                        <span className="launcher-section-title">
                            Actions
                        </span>
                        <button
                            className="app-button app-button--primary app-button--full"
                            onClick={lambda -> None { navigate("/new-project");}}
                        >
                            New Project
                        </button>
                        <div className="form-field">
                            <label className="input-label">
                                Load Existing Project
                            </label>
                            <select
                                defaultValue=""
                                onChange={handleProjectSelect}
                                className="launcher-select"
                            >
                                <option value="" disabled>
                                    Select a project...
                                </option>
                                {projectOptions}
                            </select>
                        </div>
                        <button
                            className="app-button app-button--ghost app-button--full"
                            onClick={lambda -> None { navigate("/settings");}}
                        >
                            Settings
                        </button>
                    </div>
                    <div className="panel launcher-panel">
                        <div className="launcher-meta-row">
                            <span className="panel-title">
                                Recent Projects
                            </span>
                            <span className="table-muted">
                                {projects.length}total
                            </span>
                        </div>
                        <div className="launcher-list">
                            {recentLinks
                            if hasRecent
                            else <span className="launcher-muted">
                                {"Loading projects..."
                                if loading
                                else "No recent projects yet."}
                            </span>}
                        </div>
                    </div>
                </section>
            </div>
        </main>;
}
