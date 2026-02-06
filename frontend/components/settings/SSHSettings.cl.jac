"""SSH Settings Component"""

import from react { useState, useEffect }
import from ...assets.ChevronDownIcon { ChevronDownIcon }


def:pub SSHSettings(config: any, handleSaveSshConnection: any, handleRemoveSshConnection: any, saving: bool, checkSshConnectionWalker: any) -> any {
    ssh_connections = config["ssh_connections"] if "ssh_connections" in config else [];
    
    # Check if there's a last used connection to default to
    default_conn = {};
    if ssh_connections.length > 0 {
        default_conn = ssh_connections[0];
    }
    
    form_state = useState(default_conn);
    formData = form_state[0];
    setFormData = form_state[1];

    # Manage local selection state
    selected_idx_state = useState(0 if ssh_connections.length > 0 else -1);
    selectedIdx = selected_idx_state[0];
    setSelectedIdx = selected_idx_state[1];

    # Status: "unknown", "connecting", "connected", "failed"
    status_state = useState("unknown");
    status = status_state[0];
    setStatus = status_state[1];

    def handleChange(field: str, value: str) -> None {
        setFormData(lambda prev: any -> any {
            new_data = {**prev};
            new_data[field] = value;
            return new_data;
        });
    }

    # Auto-select the saved connection in the dropdown when config/list updates
    useEffect(
        lambda -> None {
            if formData and "host" in formData and "user" in formData {
                host = formData["host"];
                user = formData["user"];
                
                # Check if current formData matches any in the list
                found_idx = ssh_connections.findIndex(lambda c: any -> bool { 
                    return c["host"] == host and c["user"] == user; 
                });
                
                if found_idx != -1 and found_idx != selectedIdx {
                     setSelectedIdx(found_idx);
                }
            }
        }, [ssh_connections, formData]
    );

    # Simulate auto-connect when a saved connection is selected
    useEffect(
        lambda -> None {
            if selectedIdx != -1 {
                setStatus("connecting");
                
                # Check connection attempt
                async def check_connection() -> None {
                    conn = ssh_connections[selectedIdx];
                    if "host" in conn and conn["host"] and "user" in conn and conn["user"] {
                        # extracting password and key_path safely
                        pwd = conn["password"] if "password" in conn else "";
                        k_path = conn["key_path"] if "key_path" in conn else "";
                        
                        result = await checkSshConnectionWalker(conn["host"], conn["user"], pwd, k_path);
                        if result {
                            setStatus("connected");
                        } else {
                            setStatus("failed");
                        }
                    } else {
                         setStatus("failed");
                    }
                }
                
                check_connection();
            } else {
                setStatus("unknown");
            }
        }, [selectedIdx]
    );

    def handleSelect(idx: int) -> None {
        setSelectedIdx(idx);
        if idx >= 0 and idx < ssh_connections.length {
            setFormData(ssh_connections[idx]);
        } else {
            setFormData({"host": "", "user": "", "key_path": "", "password": ""});
        }
    }

    def onSave() -> None {
        handleSaveSshConnection(formData);
    }
    
    def onRemove() -> None {
        if selectedIdx >= 0 and selectedIdx < ssh_connections.length {
            handleRemoveSshConnection(selectedIdx);
            # Reset form logic handled by parent update or we can reset here
            setSelectedIdx(-1);
            setFormData({"host": "", "user": "", "key_path": "", "password": ""});
        }
    }

    # Removed dummy can_ssh function since frontend doesn't support python socket module directly

    return <div className="space-y-2">
        <div className="space-y-1">
            <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Saved Connections</label>
            <div className="relative">
                <select
                    value={selectedIdx}
                    onChange={lambda e: any -> None { handleSelect(parseInt(e.target.value)); }}
                    className="w-full h-10 appearance-none bg-zinc-900 border border-zinc-800 hover:border-zinc-700 rounded-lg px-4 text-zinc-200 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all cursor-pointer text-sm"
                >
                    <option value={-1}>New Connection</option>
                    {(
                        ssh_connections.map(lambda conn: any, i: any -> any:
                            <option value={i} key={i}>{f"{conn['user']}@{conn['host']}"}</option>
                        )
                    )}
                </select>
                <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-zinc-500">
                    <ChevronDownIcon />
                </div>
            </div>
        </div>

            <div className={f"flex items-center gap-3 px-3 py-2 bg-zinc-950 rounded-lg border border-zinc-900 {'invisible' if selectedIdx == -1 else ''}"}>
                <div className={f"w-2.5 h-2.5 rounded-full {(
                    'bg-yellow-500 animate-pulse' if status == 'connecting'
                    else 'bg-green-500' if status == 'connected'
                    else 'bg-red-500' if status == 'failed'
                    else 'bg-zinc-600'
                )}"} />
                <span className={f"text-sm font-medium font-mono {(
                    'text-yellow-500' if status == 'connecting'
                    else 'text-green-500' if status == 'connected'
                    else 'text-red-500' if status == 'failed'
                    else 'text-zinc-500'
                )}"}>
                    {(
                        "STATUS: CONNECTING..." if status == "connecting"
                        else "STATUS: CONNECTED" if status == "connected"
                        else "STATUS: FAILED CONNECTION" if status == "failed"
                        else "UNKNOWN STATUS"
                    )}
                </span>
            </div>

        <div className="space-y-2 p-3 bg-zinc-900/50 rounded-lg border border-zinc-800">
            <div className="space-y-1">
                <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Host IP</label>
                <input
                    type="text"
                    value={formData["host"] if "host" in formData else ""}
                    onChange={lambda e: any -> None { handleChange("host", e.target.value); }}
                    placeholder="192.168.1.100"
                    className="w-full h-9 bg-zinc-900 border border-zinc-800 rounded px-3 text-zinc-200 text-sm font-mono placeholder-zinc-600 focus:outline-none focus:border-zinc-600"
                />
            </div>
            <div className="space-y-1">
                <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">User</label>
                <input
                    type="text"
                    value={formData["user"] if "user" in formData else ""}
                    onChange={lambda e: any -> None { handleChange("user", e.target.value); }}
                    placeholder="dev_user"
                    className="w-full h-9 bg-zinc-900 border border-zinc-800 rounded px-3 text-zinc-200 text-sm font-mono placeholder-zinc-600 focus:outline-none focus:border-zinc-600"
                />
            </div>
            <div className="space-y-1">
                <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Password</label>
                <input
                    type="password"
                    value={formData["password"] if "password" in formData else ""}
                    onChange={lambda e: any -> None { handleChange("password", e.target.value); }}
                    placeholder="••••••••"
                    className="w-full h-9 bg-zinc-900 border border-zinc-800 rounded px-3 text-zinc-200 text-sm font-mono placeholder-zinc-600 focus:outline-none focus:border-zinc-600"
                />
            </div>
            <div className="space-y-1">
                <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Key Path</label>
                <input
                    type="text"
                    value={formData["key_path"] if "key_path" in formData else ""}
                    onChange={lambda e: any -> None { handleChange("key_path", e.target.value); }}
                    placeholder="/Users/me/.ssh/id_rsa"
                    className="w-full h-9 bg-zinc-900 border border-zinc-800 rounded px-3 text-zinc-200 text-sm font-mono placeholder-zinc-600 focus:outline-none focus:border-zinc-600"
                />
            </div>
        </div>

        <div className="flex gap-3">
             <button
                onClick={onSave}
                disabled={saving}
                className="flex-1 h-10 bg-zinc-100 hover:bg-white text-zinc-950 font-bold rounded-lg transition-all shadow-lg hover:shadow-zinc-500/20 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
            >
                {("Saving..." if saving else "Save Connection")}
            </button>
            {(
                <button
                    onClick={onRemove}
                    disabled={saving}
                    className="w-24 h-10 bg-red-900/20 hover:bg-red-900/40 text-red-400 hover:text-red-300 font-bold rounded-lg border border-red-900/50 transition-all text-sm"
                >
                    Remove
                </button>
                if selectedIdx >= 0
                else null
            )}
        </div>
    </div>;
}
