"""Export Page"""

import from ...components.ui.Icons { UploadIcon, CodeIcon, ShieldCheckIcon }
import from ...components.ui.GlowButton { GlowButton }
import from "react-router-dom" { useParams }
import from react { useState }

def:pub Export -> Any {
    params = useParams();
    name = params.projectName;

    downloading = False;

    async def handleDownloadExport -> None {
        if not name {
            return;
        }

        # downloading = True; (State needs signal impl in Jac later)
        try {
            # Mock export ops for now - should come from selection
            export_ops = [];

            resp = await __jacSpawn(
                "DownloadExport", "", {"projectName": name, "ops": export_ops}
            );
            data = resp.reports[0] if resp and resp.reports else None;

            if data and data["success"] {
                file_name = data["name"] if data["name"] else f"{name}.kforge.zip";
                data_uri = f"data:application/zip;base64,{data['data']}";
                blob_resp = await fetch(data_uri);
                blob = await blob_resp.blob();
                url = URL.createObjectURL(blob);
                link = document.createElement("a");
                link.href = url;
                link.download = file_name;
                link.click();
                URL.revokeObjectURL(url);
            }
        } except Exception as e {
            console.error("Failed to download export:", e);
        } finally {
        # downloading = False;
        }
    }

    targets = [
        {
            "name": "PyTorch Extension (cpp)",
            "desc": "Standard setup.py integration",
            "checked": True
        },
        {
            "name": "ONNX Runtime",
            "desc": "Export graph with custom ops",
            "checked": False
        },
        {"name": "TensorRT Plugin", "desc": "High-perf inference", "checked": False},
        {"name": "Standalone C++", "desc": "No Python dependency", "checked": False}
    ];

    target_elements = [];
    for t in targets {
        is_checked = t["checked"];
        target_elements.push(
            <label
                key={t["name"]}
                className={"flex items-center gap-3 p-3 border rounded-lg cursor-pointer transition-all " + (
                    "bg-indigo-900/30 border-indigo-500/50"
                    if is_checked
                    else "bg-zinc-900 border-zinc-800 hover:border-zinc-700"
                )}
            >
                <input
                    type="checkbox"
                    defaultChecked={is_checked}
                    className="w-4 h-4 text-indigo-500 rounded focus:ring-indigo-500 bg-zinc-800 border-zinc-600"
                />
                <div>
                    <div className="text-sm font-medium text-zinc-200">
                        {t["name"]}
                    </div>
                    <div className="text-xs text-zinc-500">
                        {t["desc"]}
                    </div>
                </div>
            </label>
        );
    }

    return
        <div
            className="h-full bg-zinc-900 text-zinc-100 flex p-8 gap-12 justify-center font-sans"
        >
            <div className="w-1/3 max-w-sm space-y-6">
                <div>
                    <h2 className="text-2xl font-bold mb-2">
                        Export Package
                    </h2>
                    <p className="text-zinc-400 text-sm">
                        Bundle selected kernels into a deployable library.
                    </p>
                </div>
                <div className="space-y-3">
                    <label
                        className="text-xs text-zinc-500 uppercase tracking-wider font-semibold"
                    >
                        Targets
                    </label>
                    <div className="space-y-2">
                        {target_elements}
                    </div>
                </div>
                <div className="pt-4 border-t border-zinc-800">
                    <GlowButton onClick={handleDownloadExport} className="w-full">
                        <UploadIcon className="w-4 h-4"/>
                        Download Artifact
                    </GlowButton>
                    <p className="text-xs text-zinc-500 text-center mt-3">
                        Includes: .cu source, headers, setup.py, tests
                    </p>
                </div>
            </div>
            <div
                className="w-1/3 max-w-sm bg-zinc-950 border border-zinc-800 rounded-xl p-6 shadow-sm flex flex-col"
            >
                <h3
                    className="text-sm font-semibold text-zinc-400 mb-4 flex items-center gap-2"
                >
                    <CodeIcon className="w-4 h-4"/>
                    Package Structure
                </h3>
                <div
                    className="font-mono text-sm text-zinc-300 space-y-1 overflow-y-auto flex-1 text-xs"
                >
                    <div className="text-indigo-400">
                        📦 generated_kernels/
                    </div>
                    <div className="pl-4 border-l border-zinc-800 ml-1.5">
                        <div className="flex items-center gap-2 py-0.5">
                            📄 setup.py
                        </div>
                        <div className="flex items-center gap-2 py-0.5">
                            📄 README.md
                        </div>
                        <div className="text-emerald-400 mt-1">
                            📂 src/
                        </div>
                        <div className="pl-4 border-l border-zinc-800 ml-1.5">
                            <div className="flex items-center gap-2 py-0.5">
                                <span className="text-yellow-200">
                                    C
                                </span>
                                fused_conv_bn_relu.cu
                            </div>
                            <div className="flex items-center gap-2 py-0.5">
                                <span className="text-yellow-200">
                                    H
                                </span>
                                kernel_utils.h
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>;
}
