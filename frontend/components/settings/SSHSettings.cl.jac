"""SSH Settings Component"""

import from react { useState, useEffect }
import from ...assets.ChevronDownIcon { ChevronDownIcon }

def:pub SSHSettings(config: any, handleSaveSshConnection: any, handleRemoveSshConnection: any, handleSetActiveSshConnection: any, handleUpdateSshConnection: any, saving: bool, checkSshConnectionWalker: any) -> any {
    ssh_connections = config["ssh_connections"] if "ssh_connections" in config else [];
    active_idx = config["active_ssh_index"] if "active_ssh_index" in config else -1;

    # View state: "list" or "add"
    view_state = useState("list");
    view = view_state[0];
    setView = view_state[1];

    # Form State for Add/Edit
    form_state = useState({"host": "", "user": "", "key_path": "", "password": ""});
    formData = form_state[0];
    setFormData = form_state[1];

    # Status map for connection health: {index: "pending" | "success" | "failed"}
    status_map_state = useState({});
    statusMap = status_map_state[0];
    setStatusMap = status_map_state[1];

    def updateStatus(idx: int, status: str) -> None {
        setStatusMap(lambda prev: any -> any {
            new_map = {**prev};
            # JS object keys are strings anyway
            new_map[idx] = status;
            return new_map;
        });
    }

    # Effect to check the active connection on mount or change
    useEffect(
        lambda -> None {
            if active_idx != -1 and active_idx < ssh_connections.length {
                # Check the active connection
                async def check_active() -> None {
                    conn = ssh_connections[active_idx];
                    updateStatus(active_idx, "pending");
                    
                    pwd = conn["password"] if "password" in conn else "";
                    k_path = conn["key_path"] if "key_path" in conn else "";
                    
                    res = await checkSshConnectionWalker(conn["host"], conn["user"], pwd, k_path);
                    success = res["success"] if "success" in res else False;

                    if success {
                        updateStatus(active_idx, "success");
                        if "timestamp" in res {
                             handleUpdateSshConnection(active_idx, {"last_connected": res["timestamp"]});
                        }
                    } else {
                        updateStatus(active_idx, "failed");
                    }
                }
                check_active();
            }
        }, [active_idx]
    );

    def handleConnectToggle(idx: int) -> None {
        if active_idx == idx {
            # Disconnect
            handleSetActiveSshConnection(-1);
            updateStatus(idx, "unknown");
        } else {
            # Connect
            handleSetActiveSshConnection(idx);
            # The useEffect will trigger the check
        }
    }

    def handleChange(field: str, value: str) -> None {
        setFormData(lambda prev: any -> any { return {**prev, [field]: value}; });
    }

    def onSave() -> None {
        handleSaveSshConnection(formData);
        setView("list");
        setFormData({"host": "", "user": "", "key_path": "", "password": ""});
    }

    if view == "add" {
        return <div className="space-y-4">
            <h2 className="text-xl font-bold">Add New Connection</h2>
            <div className="space-y-3 p-4 bg-zinc-900/50 rounded-lg border border-zinc-800">
                <div className="space-y-1">
                    <label className="text-xs font-semibold text-zinc-400 uppercase">Host IP</label>
                    <input type="text" value={formData["host"]} onChange={lambda e: any -> None { handleChange("host", e.target.value); }} 
                        className="w-full h-10 bg-zinc-900 border border-zinc-800 rounded px-3 text-zinc-200" placeholder="192.168.1.100" />
                </div>
                <div className="space-y-1">
                    <label className="text-xs font-semibold text-zinc-400 uppercase">User</label>
                    <input type="text" value={formData["user"]} onChange={lambda e: any -> None { handleChange("user", e.target.value); }}
                        className="w-full h-10 bg-zinc-900 border border-zinc-800 rounded px-3 text-zinc-200" placeholder="user" />
                </div>
                <div className="space-y-1">
                    <label className="text-xs font-semibold text-zinc-400 uppercase">Password</label>
                    <input type="password" value={formData["password"]} onChange={lambda e: any -> None { handleChange("password", e.target.value); }}
                        className="w-full h-10 bg-zinc-900 border border-zinc-800 rounded px-3 text-zinc-200" placeholder="••••••" />
                </div>
                <div className="space-y-1">
                    <label className="text-xs font-semibold text-zinc-400 uppercase">Key Path</label>
                    <input type="text" value={formData["key_path"]} onChange={lambda e: any -> None { handleChange("key_path", e.target.value); }}
                        className="w-full h-10 bg-zinc-900 border border-zinc-800 rounded px-3 text-zinc-200" placeholder="~/.ssh/id_rsa" />
                </div>
            </div>
            <div className="flex gap-3">
                <button onClick={onSave} disabled={saving} className="flex-1 h-10 bg-zinc-100 hover:bg-white text-zinc-950 font-bold rounded-lg transition-all shadow-lg hover:shadow-zinc-500/20 disabled:opacity-50 disabled:cursor-not-allowed">
                    {("Saving..." if saving else "Save")}
                </button>
                <button onClick={lambda -> None { setView("list"); }} className="flex-1 h-10 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 font-bold rounded-lg transition-all border border-zinc-700">
                    Cancel
                </button>
            </div>
        </div>;
    }

    # LIST VIEW
    return <div className="space-y-4">
        <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold">SSH Connections</h2>
        </div>

        <div className="bg-zinc-900/50 rounded-lg p-4 border border-zinc-800 min-h-[300px]">
            <div className="flex items-center gap-2 font-medium border-b border-zinc-800 pb-2 mb-2 text-zinc-400 uppercase text-xs tracking-wider px-2">
                <div className="flex-1 min-w-0">user</div>
                <div className="flex-[2] min-w-0">host</div>
                <div className="w-40">last connected</div>
                <div className="w-16 text-center">status</div>
                <div className="w-24 pl-2">toggle</div>
                <div className="w-16 text-right"></div>
            </div>
            
            <div className="space-y-2">
                {ssh_connections.map(lambda conn: any, i: any -> any:
                    <div key={i} className="flex items-center gap-2 text-sm py-2 border-b border-zinc-800/50 last:border-0 hover:bg-zinc-900/40 rounded transition-colors px-2">
                        <div className="flex-1 min-w-0 truncate text-zinc-300" title={conn["user"]}>{conn["user"]}</div>
                        <div className="flex-[2] min-w-0 truncate text-zinc-300 font-mono" title={conn["host"]}>{conn["host"]}</div>
                        <div className="w-40 text-zinc-500 text-xs">{(conn["last_connected"] if "last_connected" in conn else "N/A")}</div>
                        <div className="w-16 flex justify-center">
                            <div className={f"w-2.5 h-2.5 rounded-full {(
                                'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]' if i in statusMap and statusMap[i] == 'success'
                                else 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]' if i in statusMap and statusMap[i] == 'failed'
                                else 'bg-yellow-500 animate-pulse shadow-[0_0_8px_rgba(234,179,8,0.5)]' if i in statusMap and statusMap[i] == 'pending'
                                else 'bg-zinc-700'
                            )}"} />
                        </div>
                        <div className="w-24 flex justify-end">
                            <button
                                onClick={lambda -> None { handleConnectToggle(i); }}
                                className={f"px-3 py-1.5 rounded text-xs font-bold w-full transition-all border {(
                                    'bg-red-500/10 text-red-500 border-red-500/50 hover:bg-red-500/20' if active_idx == i
                                    else 'bg-zinc-800 text-zinc-300 border-zinc-700 hover:bg-zinc-700 hover:text-white'
                                )}"}
                            >
                                {("Disconnect" if active_idx == i else "Connect")}
                            </button>
                        </div>
                        <div className="w-16 pl-2 flex justify-end">
                             <button
                                onClick={lambda -> None { handleRemoveSshConnection(i); }}
                                className="p-1.5 rounded text-zinc-500 hover:text-red-400 hover:bg-red-400/10 transition-colors"
                                title="Delete Connection"
                            >
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>
                            </button>
                        </div>
                    </div>
                )}
                {(
                    <div className="text-zinc-500 text-center py-12 flex flex-col items-center gap-2">
                        <span className="text-4xl opacity-20">🔌</span>
                        <span className="italic">No connections saved</span>
                    </div>
                    if ssh_connections.length == 0
                    else null
                )}
            </div>
        </div>

        <div className="flex gap-3">
             <button
                onClick={lambda -> None { setView("add"); }}
                className="flex-1 h-12 bg-zinc-100 hover:bg-white text-zinc-950 font-bold rounded-lg transition-all shadow-lg hover:shadow-zinc-500/20"
            >
                Add New SSH
            </button>
            # Return button handled by parent
        </div>
    </div>;
}
