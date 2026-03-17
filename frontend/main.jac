"""Main entry point for frontend."""
import from walkers.globals { repo_root, config_path, config_store }
import from walkers.config { GetConfig, SaveConfig }
import from walkers.system {
    GetSystemInfo,
    CheckSshConnection,
    TestApiKey,
    TestAllApiKeys,
    TestLLMConnection,
    DetectHardware
}
import from walkers.llm { GetOpenLLMModels }
import from walkers.kernel_job_runners {
    StartProfile,
    StartGenerate,
    StartOptimize,
    StartBenchmark,
    SetJobControl,
    ClearProjectQueue,
    GetJobLog
}
import from walkers.project_admin {
    CreateProject,
    DeleteProject,
    GetProjects,
    GetProjectOps,
    GetProjectStatus,
    GetProjectBenchmarks,
    ExportProject,
    DownloadExport,
    ReadRepoFile,
    CloneProjectForGpu
}
import from walkers.optimization_results {
    GetDashboardCharts,
    GetMctsTree,
    GetProjectMctsSummary,
    GetOpDetails,
    GetProjectMetrics,
    GetOperatorMetrics,
    ImportKernel,
    DownloadCast,
    GetCastChunk
}

cl import from react { useEffect }
cl import from sonner { Toaster, toast }
cl import from "react-router-dom" { BrowserRouter as Router, Routes, Route, Navigate }
cl import from .context.ConfigContext { ConfigProvider }
cl import from .app.home.Home { Home }
cl import from .app.settings.SettingsPage { SettingsPage }
cl import from .app.settings.ApiKeysPage { ApiKeysPage }
cl import from .app.settings.GpuPage { GpuPage }
cl import from .app.settings.SshPage { SshPage }
cl import from .app.settings.AboutSettings { AboutSettings }
cl import from .app.create.NewProject { NewProject }
cl import from .app.project.ProjectLayout { ProjectLayout }
cl import from .app.project.dashboard.Dashboard { Dashboard }
cl import from .app.project.operator_workbench.OperatorWorkbench { OperatorWorkbench }
cl import from .app.project.export.Export { Export }
cl import "./styles.css";

cl {
    async def maybeCheckForDesktopUpdate -> None {
        build_info = (
            window["__KFORGE_DESKTOP_BUILD__"]
            if "__KFORGE_DESKTOP_BUILD__" in window and window["__KFORGE_DESKTOP_BUILD__"]
            else None
        );
        if not build_info or not ("desktop" in build_info and build_info["desktop"]) {
            return;
        }

        repo = (
            str(build_info["github_repo"])
            if "github_repo" in build_info and build_info["github_repo"]
            else ""
        );
        releases_url = (
            str(build_info["releases_url"])
            if "releases_url" in build_info and build_info["releases_url"]
            else f"https://github.com/{repo}/releases"
        );
        current_sha = (
            str(build_info["commit"])
            if "commit" in build_info and build_info["commit"]
            else ""
        );
        if not repo or not current_sha {
            return;
        }

        try {
            branch_resp = await fetch(
                f"https://api.github.com/repos/{repo}/branches/main",
                {"headers": {"Accept": "application/vnd.github+json"}}
            );
            if not branch_resp or not branch_resp.ok {
                return;
            }
            payload = await branch_resp.json();
            latest_sha = (
                str(payload["commit"]["sha"])
                if payload and "commit" in payload and payload["commit"] and "sha" in payload["commit"]
                else ""
            );
            if not latest_sha or latest_sha == current_sha {
                return;
            }

            cache_key = "kf_desktop_update_sha";
            last_sha = localStorage.getItem(cache_key);
            if last_sha == latest_sha {
                return;
            }
            localStorage.setItem(cache_key, latest_sha);
            toast(
                "Kernel Forge desktop update available",
                {
                    "description": f"GitHub main is at {latest_sha[0:7]} and this build is {current_sha[0:7]}. Install the latest desktop release from {releases_url}."
                }
            );
        } except Exception as e {
            console.warn("Desktop update check failed:", e);
        }
    }

    def NotFound {
        return
            <div className="flex min-h-screen flex-col items-center justify-center">
                <h1>
                    404 Not Found
                </h1>
                <p className="text-zinc-500 mt-2">
                    The page you are looking for does not exist.
                </p>
            </div>;
    }

    def:pub app {
        useEffect(
            lambda -> None {
                window.Exception = Error;
                document.title = "Kernel Forge";
                
                # Dynamic Favicon
                link = document.querySelector("link[rel~='icon']");
                if not link {
                    link = document.createElement("link");
                    link.rel = "icon";
                    document.head.appendChild(link);
                }
                link.href = "/KernelForge.svg";
                maybeCheckForDesktopUpdate();
            }, []
        );
        return
            <Router>
                <ConfigProvider>
                    <div className="min-h-screen font-sans antialiased">
                        <Routes>
                            <Route path="/" element={<Home/>}/>
                            <Route path="/settings" element={<SettingsPage/>}/>
                            <Route path="/settings/api-keys" element={<ApiKeysPage/>}/>
                            <Route path="/settings/gpu" element={<GpuPage/>}/>
                            <Route path="/settings/ssh" element={<SshPage/>}/>
                            <Route path="/settings/about" element={<AboutSettings/>}/>
                            <Route path="/new-project" element={<NewProject/>}/>
                            <Route
                                path="/project/:projectName"
                                element={<ProjectLayout/>}
                            >
                                <Route index element={<Dashboard/>}/>
                                <Route
                                    path="settings/*"
                                    element={<Navigate to="/settings" replace={True}/>}
                                />
                                <Route
                                    path="operator-workbench"
                                    element={<OperatorWorkbench/>}
                                />
                                <Route
                                    path="operator-workbench/:opName"
                                    element={<OperatorWorkbench/>}
                                />
                                <Route path="export" element={<Export/>}/>
                            </Route>
                            <Route path="*" element={<NotFound/>}/>
                        </Routes>
                        <Toaster richColors={True} theme="dark"/>
                    </div>
                </ConfigProvider>
            </Router>;
    }

}
