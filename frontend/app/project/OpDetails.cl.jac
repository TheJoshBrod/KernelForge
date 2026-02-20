"""Op Details Component"""

import from react { useEffect }
import from "react-router-dom" { useNavigate, useParams }
import from ...components.backdrops.GameOfLifeBackground { GameOfLifeBackground }
import from ...components.ui.Icons { KernelForgeLogo }

def:pub OpDetails -> Any {
    navigate = useNavigate();
    params = useParams();
    projectName: str = params.projectName or "";
    opName: str = params.opName or "";

    has details: dict = {},
        loading: bool = True,
        errorMsg: str = "",
        selectedFile: dict = {},
        fileContent: str = "";

    async def fetchDetails -> None {
        loading = True;
        errorMsg = "";
        if not projectName or not opName {
            errorMsg = "Missing project or op name.";
            loading = False;
            return;
        }
        try {
            resp = await __jacSpawn(
                "GetOpDetails", "", {"projectName": projectName, "opName": opName}
            );
            data = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else None;
            if data {
                details = data;
            } else {
                errorMsg = "No data returned.";
            }
        } except Exception as e {
            console.error("Failed to load op details:", e);
            errorMsg = "Failed to load op details.";
        } finally {
            loading = False;
        }
    }

    async def openFile(file: dict) -> None {
        selectedFile = file;
        fileContent = "";
        if not file {
            return;
        }
        if file["kind"] != "text" {
            fileContent = "Binary file preview not available.";
            return;
        }
        try {
            resp = await __jacSpawn(
                "ReadRepoFile",
                "",
                {"projectName": projectName, "relPath": file["path"]}
            );
            data = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else None;
            if data and data["success"] {
                fileContent = data["content"];
            } else {
                fileContent = data["error"] if data else "Failed to read file.";
            }
        } except Exception as e {
            console.error("Failed to read file:", e);
            fileContent = "Failed to read file.";
        }
    }

    def handleKeyDown(event: any) -> None {
        if event.key == "Escape" {
            navigate("/project/" + projectName);
        }
    }

    useEffect(
        lambda -> Any { window.addEventListener("keydown", handleKeyDown);return lambda -> None { window.removeEventListener(
            "keydown", handleKeyDown
        );}; },
        [projectName]
    );

    useEffect(lambda -> None { fetchDetails();}, [projectName, opName]);

    profile = details["profile"] if details and "profile" in details else {};
    generated = details["generated"] if details and "generated" in details else {};
    optimized = details["optimized"] if details and "optimized" in details else {};

    gen_kernels = generated["kernels"] if generated and "kernels" in generated else [];
    gen_logs = generated["logs"] if generated and "logs" in generated else [];
    opt_kernels = optimized["kernels"] if optimized and "kernels" in optimized else [];
    opt_logs = optimized["logs"] if optimized and "logs" in optimized else [];
    files_label = f"{gen_kernels.length} gen / {opt_kernels.length} opt";

    genKernelButtons = [];
    for f in gen_kernels {
        genKernelButtons = genKernelButtons.concat(
            [
                <button
                    key={f["path"]}
                    onClick={lambda -> None { openFile(f);}}
                    className="file-button"
                >
                    {f["name"]}
                </button>
            ]
        );
    }

    genLogButtons = [];
    for f in gen_logs {
        genLogButtons = genLogButtons.concat(
            [
                <button
                    key={f["path"]}
                    onClick={lambda -> None { openFile(f);}}
                    className="file-button"
                >
                    {f["name"]}
                </button>
            ]
        );
    }

    optKernelButtons = [];
    for f in opt_kernels {
        optKernelButtons = optKernelButtons.concat(
            [
                <button
                    key={f["path"]}
                    onClick={lambda -> None { openFile(f);}}
                    className="file-button"
                >
                    {f["name"]}
                </button>
            ]
        );
    }

    optLogButtons = [];
    for f in opt_logs {
        optLogButtons = optLogButtons.concat(
            [
                <button
                    key={f["path"]}
                    onClick={lambda -> None { openFile(f);}}
                    className="file-button"
                >
                    {f["name"]}
                </button>
            ]
        );
    }

    hasSelected = selectedFile and "name" in selectedFile;

    skeletonFiles = [];
    for i in [0, 1, 2, 3, 4] {
        skeletonFiles = skeletonFiles.concat(
            [<div key={i} className="skeleton skeleton-line"></div>]
        );
    }

    if loading {
        return
            <div className="app-shell">
                <GameOfLifeBackground/>
                <div className="app-backdrop"></div>
                <div className="app-layout app-content-layer">
                    <aside className="sidebar">
                        <div className="panel sidebar-panel">
                            <div className="panel-header">
                                <span className="panel-title">
                                    Loading op details
                                </span>
                            </div>
                            <div className="panel-body">
                                {skeletonFiles}
                            </div>
                        </div>
                    </aside>
                    <main className="content">
                        <div className="panel">
                            <div className="panel-header">
                                <span className="panel-title">
                                    File preview
                                </span>
                            </div>
                            <div className="panel-body">
                                <div className="skeleton skeleton-block"></div>
                            </div>
                        </div>
                    </main>
                </div>
            </div>;
    }

    if errorMsg {
        return
            <div className="app-shell">
                <GameOfLifeBackground/>
                <div className="app-backdrop"></div>
                <div className="app-layout app-content-layer">
                    <main className="content">
                        <div className="panel">
                            <div className="panel-header">
                                <span className="panel-title">
                                    Error
                                </span>
                            </div>
                            <div className="panel-body">
                                <p className="table-muted">
                                    {errorMsg}
                                </p>
                                <button
                                    onClick={lambda -> None { navigate(
                                        "/project/" + projectName
                                    );}}
                                    className="ghost-button"
                                >
                                    Back to Project
                                </button>
                            </div>
                        </div>
                    </main>
                </div>
            </div>;
    }

    return
        <div className="app-shell">
            <GameOfLifeBackground/>
            <div className="app-backdrop"></div>
            <div className="app-layout app-content-layer">
                <aside className="sidebar">
                    <div className="sidebar-header">
                        <div className="brand">
                            <KernelForgeLogoclassName="mascot mascot--logo"/>
                            <div className="brand-meta">
                                <span>
                                    Op Details
                                </span>
                                <span className="brand-tag">
                                    {opName}
                                </span>
                            </div>
                        </div>
                        <div className="sidebar-actions">
                            <button
                                onClick={lambda -> None { navigate(
                                    "/project/" + projectName
                                );}}
                                className="ghost-button"
                            >
                                Back
                            </button>
                            <button
                                onClick={lambda -> None { navigate("/settings");}}
                                className="ghost-button"
                            >
                                Settings
                            </button>
                        </div>
                    </div>
                    <div className="panel sidebar-panel">
                        <div className="panel-header">
                            <span className="panel-title">
                                Profile summary
                            </span>
                            <span className="table-muted">
                                Calls:{profile["count"]
                                if profile and "count" in profile
                                else 0}
                            </span>
                        </div>
                        <div className="panel-body">
                            <p className="table-muted">
                                Select a kernel or log to inspect below.
                            </p>
                        </div>
                    </div>
                    <div className="panel sidebar-panel">
                        <div className="panel-header">
                            <span className="panel-title">
                                Generated kernels
                            </span>
                        </div>
                        <div className="panel-body">
                            {genKernelButtons
                            if genKernelButtons.length > 0
                            else <p className="table-muted">
                                No generated kernels yet.
                            </p>}
                        </div>
                    </div>
                    <div className="panel sidebar-panel">
                        <div className="panel-header">
                            <span className="panel-title">
                                Generation logs
                            </span>
                        </div>
                        <div className="panel-body">
                            {genLogButtons
                            if genLogButtons.length > 0
                            else <p className="table-muted">
                                No generation logs yet.
                            </p>}
                        </div>
                    </div>
                    <div className="panel sidebar-panel">
                        <div className="panel-header">
                            <span className="panel-title">
                                Optimized kernels
                            </span>
                        </div>
                        <div className="panel-body">
                            {optKernelButtons
                            if optKernelButtons.length > 0
                            else <p className="table-muted">
                                No optimized kernels yet.
                            </p>}
                        </div>
                    </div>
                    <div className="panel sidebar-panel">
                        <div className="panel-header">
                            <span className="panel-title">
                                Optimization logs
                            </span>
                        </div>
                        <div className="panel-body">
                            {optLogButtons
                            if optLogButtons.length > 0
                            else <p className="table-muted">
                                No optimization logs yet.
                            </p>}
                        </div>
                    </div>
                </aside>
                <main className="content">
                    <div className="page-header">
                        <div>
                            <h1 className="page-title">
                                {opName}
                            </h1>
                            <div className="badge-row">
                                <span className="badge">
                                    Project:{projectName}
                                </span>
                                <span className="badge">
                                    Files:{files_label}
                                </span>
                            </div>
                        </div>
                        <KernelForgeLogoclassName="mascot mascot--cta"/>
                    </div>
                    <div className="panel">
                        <div className="panel-header">
                            <span className="panel-title">
                                File preview
                            </span>
                            <span className="table-muted">
                                {selectedFile["name"]
                                if hasSelected
                                else "Select a kernel or log"}
                            </span>
                        </div>
                        <div className="panel-body">
                            {<pre className="code-block">
                                {fileContent}
                            </pre>
                            if hasSelected
                            else <p className="table-muted">
                                No file selected.
                            </p>}
                        </div>
                    </div>
                </main>
            </div>
        </div>;
}
