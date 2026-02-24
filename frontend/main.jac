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
import from walkers.project {
    CreateProject,
    GetProjects,
    GetProjectOps,
    GetProjectStatus,
    SetJobControl,
    StartProfile,
    StartGenerate,
    StartOptimize,
    GetDashboardCharts,
    StartBenchmark,
    ExportProject,
    GetOpDetails,
    ReadRepoFile,
    DownloadExport,
    GetProjectBenchmarks,
    GetMctsTree,
    ImportKernel,
    GetProjectMctsSummary,
    GetProjectMetrics,
    GetOperatorMetrics,
    DeleteProject,
    DownloadCast,
    GetCastChunk
}

cl import from react { useEffect }
cl import from sonner { Toaster }
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
cl import from .app.project.Dashboard { Dashboard }
cl import from .app.project.operator_workbench.OperatorWorkbench { OperatorWorkbench }
cl import from .app.project.Export { Export }
cl import "./styles.css";

cl {
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
