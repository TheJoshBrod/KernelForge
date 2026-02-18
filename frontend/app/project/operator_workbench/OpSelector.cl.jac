"""Operator Selector Component"""

import from react { useState, useEffect }
import from "react-router-dom" { useNavigate, useParams }

def:pub OpSelector {
    has projectName: str;

    navigate = useNavigate();
    params = useParams();
    resolvedProjectName = projectName or params.projectName;

    ops_state = useState([]);
    ops = ops_state[0];
    setOps = ops_state[1];

    loading_ops_state = useState(True);
    loadingOps = loading_ops_state[0];
    setLoadingOps = loading_ops_state[1];

    # Format Op Name
    def format_op(raw: str) -> str {
        raw_text = raw or "";
        prefix = "torch_nn_functional_";
        if raw_text.startsWith(prefix) {
            # Keep labels compact in the Workbench list.
            return raw_text.replace(prefix, "");
        }
        return raw_text;
    }

    async def fetchOps -> None {
        if not resolvedProjectName {
            console.error("No project name found in OpSelector");
            return;
        }
        setLoadingOps(True);
        try {
            # console.log("Fetching ops for project:", resolvedProjectName);
            ops_resp = await __jacSpawn(
                "GetProjectOps", "", {"projectName": resolvedProjectName}
            );
            ops_data = ops_resp.reports[0] if ops_resp and ops_resp.reports else None;
            op_list = ops_data["ops"] if ops_data and "ops" in ops_data else [];
            setOps(op_list);
        } except Exception as e {
            console.error("Failed to fetch ops:", e);
        } finally {
            setLoadingOps(False);
        }
    }

    useEffect(lambda -> None { fetchOps();}, [resolvedProjectName]);

    op_items = [];
    for op in ops {
        op_items.push(
            <div
                key={op["name"]}
                className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg hover:border-indigo-500/50 cursor-pointer transition-all flex justify-between items-center group"
                onClick={lambda -> None { navigate(
                    f"/project/{resolvedProjectName}/operator-workbench/{op['name']}"
                );}}
            >
                <span
                    className="font-mono text-sm text-zinc-300 group-hover:text-white"
                >
                    {format_op(op["name"])}
                </span>
                <span className="text-xs text-zinc-600 bg-zinc-950 px-2 py-1 rounded">
                    {op["count"]} {" calls"}
                </span>
            </div>
        );
    }

    return
        <div
            className="flex-1 flex flex-col p-8 bg-zinc-900 text-zinc-100 h-screen overflow-hidden"
        >
            <div className="mb-8">
                <h2 className="text-2xl font-bold text-white mb-2">
                    Operator Workbench
                </h2>
                <p className="text-zinc-400">
                    Select an operator to explore.
                </p>
            </div>
            <div
                className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 overflow-y-auto pb-20"
            >
                {op_items}{<div
                    className="col-span-full text-center text-zinc-500 py-12 italic"
                >
                    Loading operators...
                </div>
                if loadingOps
                else <div
                    className="col-span-full text-center text-zinc-500 py-12 italic"
                >
                    No operators found.
                </div>
                if ops.length == 0
                else <span></span>}
            </div>
        </div>;
}
