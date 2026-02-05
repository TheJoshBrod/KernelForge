"""Op Details Component"""

import from react { useEffect }
import from "@jac/runtime" { jacSpawn }
import from "react-router-dom" { useNavigate, useParams }

def:pub OpDetails -> any {
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
            resp = await jacSpawn(
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
        } except (error) {
            console.error("Failed to load op details:", error);
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
            resp = await jacSpawn(
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
        } except (error) {
            console.error("Failed to read file:", error);
            fileContent = "Failed to read file.";
        }
    }

    def handleKeyDown(event: any) -> None {
        if event.key == "Escape" {
            navigate("/project/" + projectName);
        }
    }

    useEffect(
        lambda -> None { window.addEventListener("keydown", handleKeyDown);return lambda -> None { window.removeEventListener(
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

    genKernelButtons = [];
    for f in gen_kernels {
        genKernelButtons = genKernelButtons.concat(
            [
                <button
                    key={f["path"]}
                    onClick={lambda -> None { openFile(f);}}
                    className="w-full text-left border border-zinc-800/60 bg-zinc-900/40 hover:bg-zinc-800 text-xs font-mono px-3 py-2 rounded transition-all"
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
                    className="w-full text-left border border-zinc-800/60 bg-zinc-900/40 hover:bg-zinc-800 text-xs font-mono px-3 py-2 rounded transition-all"
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
                    className="w-full text-left border border-zinc-800/60 bg-zinc-900/40 hover:bg-zinc-800 text-xs font-mono px-3 py-2 rounded transition-all"
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
                    className="w-full text-left border border-zinc-800/60 bg-zinc-900/40 hover:bg-zinc-800 text-xs font-mono px-3 py-2 rounded transition-all"
                >
                    {f["name"]}
                </button>
            ]
        );
    }

    hasSelected = selectedFile and "name" in selectedFile;

    if loading {
        return
            <div
                className="flex min-h-screen items-center justify-center bg-zinc-950 text-zinc-100"
            >
                <p>
                    Loading op details...
                </p>
            </div>;
    }

    if errorMsg {
        return
            <div
                className="flex min-h-screen flex-col items-center justify-center bg-zinc-950 text-zinc-100 gap-4"
            >
                <p className="text-sm text-zinc-400">
                    {errorMsg}
                </p>
                <button
                    onClick={lambda -> None { navigate("/project/" + projectName);}}
                    className="px-4 py-2 rounded bg-zinc-800 hover:bg-zinc-700 text-sm"
                >
                    Back to Project
                </button>
            </div>;
    }

    return
        <div className="flex min-h-screen bg-zinc-950 text-zinc-100 font-sans">
            <div className="absolute top-2 right-6">
                <a
                    onClick={lambda -> None { navigate("/settings");}}
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
            <div className="absolute top-2 right-16 z-10">
                <a
                    onClick={lambda -> None { navigate("/project/" + projectName);}}
                    className="flex items-center justify-center p-2 rounded-full hover:bg-zinc-800 transition-colors text-zinc-400 hover:text-white bg-zinc-950/50 backdrop-blur-sm cursor-pointer"
                    aria-label="Go Back"
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
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="15" y1="9" x2="9" y2="15"></line>
                        <line x1="9" y1="9" x2="15" y2="15"></line>
                    </svg>
                </a>
            </div>
            <aside
                className="w-1/3 border-r border-zinc-900 flex flex-col p-6 space-y-8"
            >
                <div>
                    <h2 className="text-xl font-bold text-zinc-100">
                        {opName}
                    </h2>
                    <p className="text-xs text-zinc-500 mt-1">
                        Profiled calls: {profile["count"]
                        if profile and "count" in profile
                        else 0}
                    </p>
                </div>
                <div className="space-y-3">
                    <h3 className="text-sm uppercase tracking-wider text-zinc-500">
                        Generated Kernels
                    </h3>
                    <div className="space-y-2">
                        {genKernelButtons
                        if genKernelButtons.length > 0
                        else <p className="text-xs text-zinc-600">
                            No generated kernels yet.
                        </p>}
                    </div>
                </div>
                <div className="space-y-3">
                    <h3 className="text-sm uppercase tracking-wider text-zinc-500">
                        Generation Logs
                    </h3>
                    <div className="space-y-2">
                        {genLogButtons
                        if genLogButtons.length > 0
                        else <p className="text-xs text-zinc-600">
                            No generation logs yet.
                        </p>}
                    </div>
                </div>
                <div className="space-y-3">
                    <h3 className="text-sm uppercase tracking-wider text-zinc-500">
                        Optimized Kernels
                    </h3>
                    <div className="space-y-2">
                        {optKernelButtons
                        if optKernelButtons.length > 0
                        else <p className="text-xs text-zinc-600">
                            No optimized kernels yet.
                        </p>}
                    </div>
                </div>
                <div className="space-y-3">
                    <h3 className="text-sm uppercase tracking-wider text-zinc-500">
                        Optimization Logs
                    </h3>
                    <div className="space-y-2">
                        {optLogButtons
                        if optLogButtons.length > 0
                        else <p className="text-xs text-zinc-600">
                            No optimization logs yet.
                        </p>}
                    </div>
                </div>
            </aside>
            <main className="w-2/3 p-8 overflow-y-auto">
                <div className="mb-6">
                    <h3 className="text-lg font-semibold">
                        File Preview
                    </h3>
                    <p className="text-xs text-zinc-500">
                        {selectedFile["name"]
                        if hasSelected
                        else "Select a kernel or log to preview."}
                    </p>
                </div>
                <div
                    className="bg-zinc-900/40 border border-zinc-800 rounded-xl p-4 min-h-[400px]"
                >
                    {<pre
                        className="whitespace-pre-wrap text-xs text-zinc-200 font-mono"
                    >
                        {fileContent}
                    </pre>
                    if hasSelected
                    else <p className="text-sm text-zinc-500">
                        No file selected.
                    </p>}
                </div>
            </main>
        </div>;
}
