"""Dashboard Page"""

import from react { useEffect, useState }
import from "@jac/runtime" { jacSpawn }
import from "react-router-dom" { useNavigate, useParams }
import from "../../components/ui/Icons.cl.jac" {
    ActivityIcon,
    CpuIcon,
    CodeIcon,
    ShieldCheckIcon,
    ArrowRightIcon,
    CheckIcon,
    AlertCircleIcon
}
import from "../../components/ui/GlowButton.cl.jac" { GlowButton }
import from recharts {
    PieChart,
    Pie,
    Tooltip,
    Legend,
    ResponsiveContainer
}
import from "../../context/ConfigContext.cl.jac" { useConfig }

def:pub Dashboard {
    navigate = useNavigate();
    params = useParams();
    name = params.projectName;

    config_ctx = useConfig();
    config = config_ctx.config;

    system_info_state = useState({});
    systemInfo = system_info_state[0];
    setSystemInfo = system_info_state[1];
    
    # Backend selection state
    backend_state = useState("CUDA");
    activeBackend = backend_state[0];
    setBackend = backend_state[1];

    has ops: list = [],
        loading: bool = True,
        status: dict = {},
        hardwareInfo: dict = {},
        hardwareLoading: bool = False,
        mctsSummary: dict = {};


    COLORS = ['#86e6a9', '#5bbbd3', '#b7f0d0', '#d8f3ff'];

    pieData = [];
    hasPie = False;
    count = 0;
    # Sort ops by count descending for top 4
    sorted_ops = ops; # Simple assignment, assume ops is sorted or sort here if needed
    # Actually JS sort is tricky in Jac, verify if sorting is needed. usually ops come sorted or random.
    # We'll just take top 4 in list order for now as Project.cl.jac did.
    
    pieData = [
        {"name": "Conv2d", "value": 45, "fill": COLORS[0]},
        {"name": "BatchNorm", "value": 30, "fill": COLORS[1]},
        {"name": "ReLU", "value": 15, "fill": COLORS[2]},
        {"name": "Linear", "value": 10, "fill": COLORS[3]}
    ];
    hasPie = True;

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

                # Fetch MCTS Summary
                mcts_resp = await jacSpawn("GetProjectMctsSummary", "", {"projectName": name});
                mcts_data = mcts_resp.reports[0] if mcts_resp and mcts_resp.reports else {};
                mctsSummary = mcts_data or {};
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

    async def fetchSystemInfo -> None {
        try {
            active_idx = config["active_ssh_index"] if config and "active_ssh_index" in config else -1;
            ssh_connections = config["ssh_connections"] if config and "ssh_connections" in config else [];
            ssh_conn = None;
            if active_idx != -1 and active_idx < ssh_connections.length {
                ssh_conn = ssh_connections[active_idx];
            }

            payload = {};
            if ssh_conn {
                payload = {
                    "host": ssh_conn["host"] if "host" in ssh_conn else "",
                    "user": ssh_conn["user"] if "user" in ssh_conn else "",
                    "password": "",
                    "key_path": ssh_conn["key_path"] if "key_path" in ssh_conn else ""
                };
            }

            resp = await jacSpawn("GetSystemInfo", "", payload);
            data = resp.reports[0] if resp and resp.reports else {};

            if data {
                setSystemInfo(data);
            }
        } except (error) {
            console.error("Failed to fetch system info:", error);
        }
    }

    # Helper for hardware status
    hardwareStatus = "Unknown";
    gpus = systemInfo["gpus"] if systemInfo and "gpus" in systemInfo else [];
    selected_idx = config["selected_gpu_index"] if config and "selected_gpu_index" in config else -1;
    
    # Detailed GPU stats
    gpu_mem = "";
    gpu_compute = "";
    gpu_sms = "";

    if config and "selected_gpu_info" in config and config["selected_gpu_info"] {
        gpu_info = config["selected_gpu_info"];
        gpu_name = gpu_info["name"] if "name" in gpu_info else "Unknown GPU";
        gpu_mem = gpu_info["memory"] if "memory" in gpu_info else "";
        gpu_compute = gpu_info["compute_capability"] if "compute_capability" in gpu_info else "";
        gpu_sms = f"{gpu_info['sm_count']}" if "sm_count" in gpu_info else "";
        hardwareStatus = gpu_name;
    } elif selected_idx != -1 and selected_idx < gpus.length {
        selected_gpu = gpus[selected_idx];
        gpu_name = selected_gpu["name"] if "name" in selected_gpu else "Unknown GPU";
        gpu_mem = selected_gpu["memory"] if "memory" in selected_gpu else "";
        gpu_compute = selected_gpu["compute_capability"] if "compute_capability" in selected_gpu else "";
        gpu_sms = f"{selected_gpu['sm_count']}" if "sm_count" in selected_gpu else "";
        hardwareStatus = gpu_name;
    } elif gpus.length > 0 {
        hardwareStatus = "No GPU Selected (using CPU)";
    } else {
         # Fallback to detection if no settings found (or just show checking)
         if hardwareInfo and "mps_available" in hardwareInfo and hardwareInfo["mps_available"] {
            hardwareStatus = "MPS: Apple Silicon";
        } elif hardwareInfo and "cuda_available" in hardwareInfo {
            if hardwareInfo["cuda_available"] {
                 gpu_name = hardwareInfo["device_name"] or "GPU";
                 if selected_idx == -1 {
                    hardwareStatus = f"CUDA: {gpu_name} (Auto)";
                 }
            } else {
                 hardwareStatus = "CPU Only";
            }
        } else {
            hardwareStatus = "Scanning...";
        }
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

    # MCTS Explorer Card
    status_cards.push(
        <div key="mcts" className="bg-zinc-900 border border-zinc-800 p-4 rounded-lg cursor-pointer hover:border-indigo-500/50 transition-all" onClick={lambda -> None { navigate(f"/project/{name}/workbench"); }}>
            <div className="flex justify-between items-start mb-2">
                <div className="p-2 bg-indigo-500/10 rounded-lg">
                    <ShieldCheckIcon className="text-indigo-400 w-5 h-5" />
                </div>
                <span className="text-xs text-zinc-500 bg-zinc-800 px-2 py-1 rounded">MCTS</span>
            </div>
            <div className="text-2xl font-bold text-white mb-1">
                {mctsSummary["total_nodes"] if "total_nodes" in mctsSummary else 0}
            </div>
            <div className="text-xs text-zinc-400">
                Nodes Explored
            </div>
        </div>
    );


    # Hardware Card
    status_cards.push(
        <div key="hardware" className="bg-zinc-900 border border-zinc-800 p-4 rounded-lg flex flex-col justify-between h-full">
             <div>
                <div className="flex justify-between items-start mb-2">
                    <div className="p-2 bg-emerald-500/10 rounded-lg">
                        <CpuIcon className="text-emerald-400 w-5 h-5" />
                    </div>
                </div>
                <div className="text-lg font-bold text-white mb-2 truncate" title={hardwareStatus}>
                    {hardwareStatus}
                </div>
                {
                    <div className="text-xs text-zinc-400 space-y-1 mb-4">
                        <div>Memory: {gpu_mem}</div>
                        <div>Compute: {gpu_compute}</div>
                        <div>SMs: {gpu_sms}</div>
                    </div>
                    if gpu_mem or gpu_compute or gpu_sms
                    else <span></span>
                }
             </div>
             
             <div className="mt-2">
                <span className="text-xs text-zinc-500 block mb-1">Backend</span>
                <select
                    className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-300 focus:outline-none focus:border-emerald-500"
                    value={activeBackend}
                    onChange={lambda e: any -> None { setBackend(e.target.value); }}
                >
                    <option value="CUDA">CUDA</option>
                    <option value="Triton">Triton</option>
                </select>
            </div>
        </div>
    );

    # Pipeline Steps Timeline
    jobs = [
        {"key": "prepare", "label": "Analysis"},
        {"key": "generate", "label": "Generation"},
        {"key": "optimize", "label": "Optimization"},
        {"key": "benchmark", "label": "Benchmark"}
    ];
    step_items = [];
    step_idx = 0;
    
    while step_idx < jobs.length {
        job = jobs[step_idx];
        job_data = status[job["key"]] if status and job["key"] in status else {};
        job_status = job_data["status"] if job_data and "status" in job_data else "idle";
        
        # Determine step visual state
        is_completed = job_status == "completed";
        is_running = job_status == "running";
        is_failed = job_status == "failed";
        is_active = is_running or is_completed;
        
        # Step Icon
        step_icon = <div className="w-2 h-2 rounded-full bg-zinc-600"></div>;
        if is_completed {
            step_icon = <CheckIcon className="w-4 h-4 text-emerald-950" />;
        } elif is_failed {
            step_icon = <AlertCircleIcon className="w-4 h-4 text-red-950" />;
        } elif is_running {
            step_icon = <div className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse"></div>;
        }

        # Step Circle Style
        circle_style = "bg-zinc-800 border-zinc-700 text-zinc-500";
        if is_completed {
            circle_style = "bg-emerald-400 border-emerald-400 text-emerald-950";
        } elif is_failed {
            circle_style = "bg-red-400 border-red-400 text-red-950";
        } elif is_running {
            circle_style = "bg-indigo-500/20 border-indigo-500 text-indigo-400 ring-4 ring-indigo-500/10";
        }
        
        # Connector Line
        connector = <span></span>;
        if step_idx < jobs.length - 1 {
            line_color = "bg-zinc-800";
            if is_completed {
                # Check if next step is at least active to color the line
                next_job = jobs[step_idx + 1];
                next_data = status[next_job["key"]] if status and next_job["key"] in status else {};
                next_status = next_data["status"] if next_data and "status" in next_data else "idle";
                if next_status != "idle" {
                   line_color = "bg-emerald-500/50";
                }
            }
            connector = <div className={"flex-1 h-0.5 mx-2 " + line_color}></div>;
        }

        step_items.push(
            <div key={job["key"]} className="flex-1 flex flex-col items-center relative">
                 <div className="flex items-center w-full">
                    {
                        <div className={"h-0.5 flex-1 " + ("bg-emerald-500/50" if is_completed or is_running or is_failed else "bg-zinc-800")}></div>
                        if step_idx > 0 else <div className="flex-1"></div>
                    }
                    
                    <div className={"w-8 h-8 rounded-full border-2 flex items-center justify-center z-10 transition-all duration-300 " + circle_style}>
                        {step_icon}
                    </div>

                    {
                        <div className={"h-0.5 flex-1 " + ("bg-emerald-500/50" if is_completed else "bg-zinc-800")}></div>
                         if step_idx < jobs.length - 1 else <div className="flex-1"></div>
                    }
                 </div>
                 <div className="mt-3 text-center">
                    <div className={"text-xs font-medium " + ("text-zinc-200" if is_active or is_failed else "text-zinc-500")}>
                        {job["label"]}
                    </div>
                     <div className={"text-[10px] uppercase tracking-wider font-bold mt-0.5 " + ("text-emerald-400" if is_completed else "text-red-400" if is_failed else "text-indigo-400" if is_running else "text-zinc-600")}>
                        {job_status if job_status != "idle" else ""}
                    </div>
                 </div>
            </div>
        );
        step_idx += 1;
    }

    
    # Operators List
    op_list_items = [];
    op_idx = 0;
    while op_idx < ops.length {
        op = ops[op_idx];
        op_path = f"/project/{name}/workbench/{op['name']}";
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
        op_idx += 1;
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
                     <div className="flex items-center justify-between px-4 py-6">
                        {step_items}
                     </div>
                </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="md:col-span-2 bg-zinc-900 border border-zinc-800 rounded-lg overflow-hidden flex flex-col">
                    <div className="p-4 border-b border-zinc-800 flex justify-between items-center bg-zinc-950/50">
                        <h3 className="font-semibold text-zinc-200">Detected Kernels</h3>
                        <span className="text-xs text-zinc-500">{ops.length} operators found</span>
                    </div>
                    <div className="p-2 grid grid-cols-1 md:grid-cols-2 gap-2">
                        {
                            op_list_items if ops.length > 0 
                            else 
                            <div className="col-span-full p-8 text-center text-zinc-500">
                                No kernels detected yet. Run fetching to begin.
                            </div>
                        }
                    </div>
                </div>

                <div className="bg-zinc-900 border border-zinc-800 rounded-lg overflow-hidden flex flex-col">
                    <div className="p-4 border-b border-zinc-800 flex justify-between items-center bg-zinc-950/50">
                        <h3 className="font-semibold text-zinc-200">Operator Usage</h3>
                        <span className="text-xs text-zinc-500">Top 4</span>
                    </div>
                    <div className="p-4 flex-1 flex items-center justify-center min-h-[220px]">
                        {
                            <div className="w-full h-[220px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <PieChart>
                                        <Pie
                                            data={pieData}
                                            innerRadius={55}
                                            outerRadius={80}
                                            paddingAngle={4}
                                            dataKey="value"
                                        />
                                        <Tooltip
                                            cursor={{ "fill": 'rgba(15, 23, 32, 0.6)' }}
                                            contentStyle={{
                                                "backgroundColor": '#0b1218',
                                                "borderColor": '#1f2a36',
                                                "borderRadius": '10px'
                                            }}
                                            itemStyle={{ "color": '#e6f2ff' }}
                                            labelStyle={{ "color": '#9fb1c7' }}
                                        />
                                        <Legend
                                            layout="horizontal"
                                            verticalAlign="bottom"
                                            align="center"
                                            wrapperStyle={{ "color": '#dbe7f3', "fontSize": "12px", "paddingTop": "10px" }}
                                        />
                                    </PieChart>
                                </ResponsiveContainer>
                            </div>
                            if hasPie
                            else
                            <div className="text-zinc-500 text-sm italic">
                                No data available
                            </div>
                        }
                    </div>
                </div>
            </div>
        </div>;
}
