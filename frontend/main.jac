"""Main entry point for frontend."""
import from walkers.globals { repo_root, config_path, config_store }
import from walkers.config { GetConfig, SaveConfig }
import from walkers.system { GetSystemInfo, CheckSshConnection, TestLLMConnection, DetectHardware }
import from walkers.project {
    CreateProject,
    GetProjects,
    GetProjectOps,
    GetProjectStatus,
    SetJobControl,
    StartProfile,
    StartGenerate,
    StartGenerateAdvanced,
    StartOptimize,
    StartOptimizeAdvanced,
    GetDashboardCharts,
    StartBenchmark,
    ExportProject,
    GetOpDetails,
    ReadRepoFile,
    DownloadExport,
    GetProjectBenchmarks,
    GetMctsTree,
    GetProjectMctsSummary
}

cl import from react { useEffect }
cl import from sonner { Toaster }
cl import from "react-router-dom" { BrowserRouter as Router, Routes, Route }
cl import from "./context/ConfigContext.cl.jac" { ConfigProvider }
cl import from "./app/home/Home.cl.jac" { Home }
cl import from "./app/settings/SettingsPage.cl.jac" { SettingsPage }
cl import from "./app/settings/ApiKeysPage.cl.jac" { ApiKeysPage }
cl import from "./app/settings/GpuPage.cl.jac" { GpuPage }
cl import from "./app/settings/SshPage.cl.jac" { SshPage }
cl import from "./app/create/NewProject.cl.jac" { NewProject }
cl import from "./app/project/ProjectLayout.cl.jac" { ProjectLayout }
cl import from "./app/project/Dashboard.cl.jac" { Dashboard }
cl import from "./app/project/Profiling.cl.jac" { Profiling }
cl import from "./app/project/Workbench.cl.jac" { Workbench }
cl import from "./app/project/Arena.cl.jac" { Arena }
cl import from "./app/project/Verification.cl.jac" { Verification }
cl import from "./app/project/Export.cl.jac" { Export }
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
        useEffect(lambda -> None { document.title = "CGinS";}, []);
        return
            <Router>
                <ConfigProvider>
                    <div className="min-h-screen font-sans antialiased">
                        <Routes>
                            <Route path="/" element={<Home />} />
                            <Route path="/settings" element={<SettingsPage />} />
                            <Route
                                path="/settings/api-keys"
                                element={<ApiKeysPage />}
                            />
                            <Route path="/settings/gpu" element={<GpuPage />} />
                            <Route path="/settings/ssh" element={<SshPage />} />
                            <Route path="/new-project" element={<NewProject />} />
                            <Route path="/project/:projectName" element={<ProjectLayout />}>
                                <Route index element={<Dashboard />} />
                                <Route path="profiling" element={<Profiling />} />
                                <Route path="workbench" element={<Workbench />} />
                                <Route path="workbench/:opName" element={<Workbench />} />
                                <Route path="arena" element={<Arena />} />


                                <Route path="verification" element={<Verification />} />
                                <Route path="export" element={<Export />} />
                            </Route>
                            <Route path="*" element={<NotFound />} />
                        </Routes>
                        <Toaster richColors={True} theme="dark" />
                    </div>
                </ConfigProvider>
            </Router>;
    }
}
