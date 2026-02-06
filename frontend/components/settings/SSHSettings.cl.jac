"""SSH Settings Component"""

import from react { useState, useEffect }
import from ...assets.ChevronDownIcon { ChevronDownIcon }

def:pub SSHSettings(config: any, handleSaveSshConnection: any, handleRemoveSshConnection: any, handleSetActiveSshConnection: any, saving: bool, checkSshConnectionWalker: any) -> any {
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
                    updateStatus(active_idx, "success" if res else "failed");
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
                <button onClick={onSave} disabled={saving} className="flex-1 h-10 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-lg transition-all">
                    {("Saving..." if saving else "Save")}
                </button>
                <button onClick={lambda -> None { setView("list"); }} className="flex-1 h-10 bg-zinc-700 hover:bg-zinc-600 text-zinc-200 font-bold rounded-lg transition-all">
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

        <div className="bg-zinc-300 rounded-lg p-4 text-zinc-900 min-h-[300px]">
            <div className="grid grid-cols-[1fr_2fr_2fr_1fr_auto] gap-2 font-medium border-b border-zinc-400 pb-2 mb-2">
                <div>user</div>
                <div>host</div>
                <div>last connected</div>
                <div>status</div>
                <div></div>
            </div>
            
            <div className="space-y-2">
                {(
                    ssh_connections.map(lambda conn: any, i: any -> any:
                        <div key={i} className="grid grid-cols-[1fr_2fr_2fr_1fr_auto] gap-2 items-center text-sm">
                            <div className="truncate" title={conn["user"]}>{conn["user"]}</div>
                            <div className="truncate" title={conn["host"]}>{conn["host"]}</div>
                            <div className="text-zinc-600">{"N/A"}</div>
                            <div className="pl-2">
                                <div className={f"w-4 h-4 rounded-full {(
                                    'bg-green-500' if i in statusMap and statusMap[i] == 'success'
                                    else 'bg-red-500' if i in statusMap and statusMap[i] == 'failed'
                                    else 'bg-yellow-500 animate-pulse' if i in statusMap and statusMap[i] == 'pending'
                                    else 'bg-zinc-400'
                                )}"} />
                            </div>
                            <div>
                                <button
                                    onClick={lambda -> None { handleConnectToggle(i); }}
                                    className={f"px-3 py-1 rounded border text-xs font-bold w-24 transition-all {(
                                        'bg-zinc-100 border-zinc-400 hover:bg-zinc-200' if active_idx == i
                                        else 'bg-zinc-100/50 border-zinc-400/50 hover:bg-zinc-100'
                                    )}"}
                                >
                                    {("Disconnect" if active_idx == i else "Connect")}
                                </button>
                            </div>
                        </div>
                    )
                )}
                {(
                    <div className="text-zinc-500 text-center py-8 italic">No connections saved.</div>
                    if ssh_connections.length == 0
                    else null
                )}
            </div>
        </div>

        <div className="flex gap-3">
             <button
                onClick={lambda -> None { setView("add"); }}
                className="flex-1 h-12 bg-zinc-300 hover:bg-white text-zinc-900 font-bold rounded-lg transition-all shadow-lg hover:shadow-zinc-500/20"
            >
                Add New SSH
            </button>
            # Return button handled by parent Settings component logic (Back button)
        </div>
    </div>;
}
