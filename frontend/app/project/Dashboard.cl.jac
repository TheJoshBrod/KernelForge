"""Dashboard Page"""

import from react { useEffect, useState }
import from "@jac/runtime" { jacSpawn }
import from "react-router-dom" { useNavigate, useParams }
import from "../../components/ui/Icons.cl.jac" {
    ActivityIcon,
    CpuIcon,
    CodeIcon,
    ShieldCheckIcon,
    ArrowRightIcon
}
import from "../../components/ui/GlowButton.cl.jac" { GlowButton }

def:pub Dashboard {
    navigate = useNavigate();
    params = useParams();
    name = params.projectName;

    has ops: list = [],
        loading: bool = True,
        status: dict = {},
        hardwareInfo: dict = {},
        hardwareLoading: bool = False;

    def format_op(raw: str) -> str {
        raw_text = raw or "";
        prefix = "torch_nn_functional_";
        if raw_text.startswith(prefix) {
            return "torch.nn.functional." + raw_text.replace(prefix, "");
        }
        return raw_text.replace("_", ".");
    }

    async def refreshData -> None {
        loading = True;
        try {
            if name {
                # Fetch Ops
                ops_resp = await jacSpawn("GetProjectOps", "", {"projectName": name});
                ops_data = ops_resp.reports[0] if ops_resp and ops_resp.reports else None;
                ops = ops_data["ops"] if ops_data and "ops" in ops_data else [];

                # Fetch Status
                status_resp = await jacSpawn("GetProjectStatus", "", {"projectName": name});
                status_data = status_resp.reports[0] if status_resp and status_resp.reports else {};
                status = status_data or {};

                # Detect Hardware
                hw_resp = await jacSpawn("DetectHardware", "", {});
                hw_data = hw_resp.reports[0] if hw_resp and hw_resp.reports else {};
                hardwareInfo = hw_data or {};
            }
        } except (error) {
            console.error("Failed to fetch dashboard data:", error);
        } finally {
            loading = False;
        }
    }

    useEffect(lambda -> None { refreshData(); }, [name]);
    
    # Auto-refresh status every 5 seconds
    useEffect(
        lambda -> None { 
            timer = setInterval(lambda -> None { 
                jacSpawn("GetProjectStatus", "", {"projectName": name}).then(lambda resp: any -> None {
                    data = resp.reports[0] if resp and resp.reports else {};
                    status = data or {};
                });
            }, 5000);
            return lambda -> None { clearInterval(timer); }; 
        },
        [name]
    );

    # Helper for hardware status
    hardwareStatus = "Unknown";
    if hardwareInfo and "mps_available" in hardwareInfo and hardwareInfo["mps_available"] {
        hardwareStatus = "MPS: Apple Silicon";
    } elif hardwareInfo and "cuda_available" in hardwareInfo {
        if hardwareInfo["cuda_available"] {
             gpu_name = hardwareInfo["device_name"] or "GPU";
             count = hardwareInfo["device_count"] or 1;
             suffix = f" x{count}" if count > 1 else "";
             hardwareStatus = f"CUDA: {gpu_name}{suffix}";
        } else {
             hardwareStatus = "CUDA Not Detected";
        }
    } else {
        hardwareStatus = "Checking Hardware...";
    }


    # Helper for Status Cards
    status_cards = [];
    
    # Kernel Count Card
    status_cards.push(
        <div key="kernels" className="bg-zinc-900 border border-zinc-800 p-4 rounded-lg">
            <div className="flex justify-between items-start mb-2">
                <div className="p-2 bg-indigo-500/10 rounded-lg">
                    <CodeIcon className="text-indigo-400 w-5 h-5" />
                </div>
                <span className="text-xs text-zinc-500 bg-zinc-800 px-2 py-1 rounded">Total</span>
            </div>
            <div className="text-2xl font-bold text-white mb-1">
                {ops.length}
            </div>
            <div className="text-xs text-zinc-400">
                Detected Kernels
            </div>
        </div>
    );

    # Hardware Card
    status_cards.push(
        <div key="hardware" className="bg-zinc-900 border border-zinc-800 p-4 rounded-lg">
             <div className="flex justify-between items-start mb-2">
                <div className="p-2 bg-emerald-500/10 rounded-lg">
                    <CpuIcon className="text-emerald-400 w-5 h-5" />
                </div>
                <span className="text-xs text-zinc-500 bg-zinc-800 px-2 py-1 rounded">Device</span>
            </div>
            <div className="text-lg font-bold text-white mb-1 truncate" title={hardwareStatus}>
                {hardwareStatus}
            </div>
             <div className="text-xs text-zinc-400">
                Active Backend
            </div>
        </div>
    );

    # Job Status List
    job_rows = [];
    jobs = [
        {"key": "prepare", "label": "Analysis"},
        {"key": "generate", "label": "Generation"},
        {"key": "optimize", "label": "Optimization"},
        {"key": "benchmark", "label": "Benchmark"}
    ];

    for job in jobs {
        job_data = status[job["key"]] if status and job["key"] in status else {};
        job_status = job_data["status"] if job_data and "status" in job_data else "idle";
        
        status_color = "text-zinc-500";
        if job_status == "running" { status_color = "text-indigo-400 animate-pulse"; }
        elif job_status == "completed" { status_color = "text-emerald-400"; }
        elif job_status == "failed" { status_color = "text-red-400"; }

        job_rows.push(
            <div key={job["key"]} className="flex justify-between items-center py-2 border-b border-zinc-800/50 last:border-0">
                <span className="text-sm text-zinc-300">{job["label"]}</span>
                <span className={"text-xs font-mono capitalize " + status_color}>
                    {job_status}
                </span>
            </div>
        );
    }
    
    # Operators List
    op_list_items = [];
    for op in ops {
        op_path = f"/project/{name}/{op['name']}";
        op_list_items.push(
            <div key={op["name"]} className="group flex items-center justify-between p-3 hovering:bg-zinc-800/50 rounded-lg border border-transparent hover:border-zinc-800 transition-all cursor-pointer" onClick={lambda -> None { navigate(op_path); }}>
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded bg-zinc-800 flex items-center justify-center text-zinc-500 group-hover:bg-indigo-500/20 group-hover:text-indigo-400 transition-colors">
                        <ActivityIcon className="w-4 h-4" />
                    </div>
                    <div>
                        <div className="text-sm font-medium text-zinc-200 group-hover:text-white">
                            {format_op(op["name"])}
                        </div>
                        <div className="text-xs text-zinc-500">
                            {op["count"]} calls
                        </div>
                    </div>
                </div>
                 <ArrowRightIcon className="w-4 h-4 text-zinc-600 group-hover:text-zinc-400" />
            </div>
        );
    }

    return
        <div className="p-8 max-w-7xl mx-auto space-y-8">
            
            <div>
                 <h2 className="text-2xl font-bold text-white mb-2">Project Dashboard</h2>
                 <p className="text-zinc-400">Overview of {name}</p>
            </div>

            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {status_cards}
                
                
                 <div className="col-span-1 md:col-span-2 bg-zinc-900 border border-zinc-800 p-4 rounded-lg">
                     <div className="flex justify-between items-center mb-4">
                        <h3 className="text-sm font-semibold text-zinc-200">Pipeline Status</h3>
                        <div className="text-xs text-zinc-500">Auto-refresh: 5s</div>
                     </div>
                     <div className="space-y-1">
                        {job_rows}
                     </div>
                </div>
            </div>

            
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg overflow-hidden">
                <div className="p-4 border-b border-zinc-800 flex justify-between items-center bg-zinc-950/50">
                    <h3 className="font-semibold text-zinc-200">Detected Kernels</h3>
                    <span className="text-xs text-zinc-500">{ops.length} operators found</span>
                </div>
                <div className="p-2 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                    {
                        op_list_items if ops.length > 0 
                        else 
                        <div className="col-span-full p-8 text-center text-zinc-500">
                            No kernels detected yet. Run fetching to begin.
                        </div>
                    }
                </div>
            </div>
        </div>;
}
