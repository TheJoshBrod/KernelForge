import from react { useState, useEffect }
import from "../../components/ui/Icons.cl.jac" { ChevronDownIcon, TrashIcon }

def:pub SSHSettings(
    config: dict,
    handleSaveSshConnection: func,
    handleRemoveSshConnection: func,
    handleSetActiveSshConnection: func,
    handleUpdateSshConnection: func,
    saving: bool,
    checkSshConnectionWalker: func
) -> Any {
    ssh_connections = config["ssh_connections"] if "ssh_connections" in config else [];
    active_idx = config["active_ssh_index"] if "active_ssh_index" in config else -1;

    view_state = useState("list");
    view = view_state[0];
    setView = view_state[1];

    form_state = useState({"host": "", "user": "", "key_path": ""});
    formData = form_state[0];
    setFormData = form_state[1];

    status_map_state = useState({});
    statusMap = status_map_state[0];
    setStatusMap = status_map_state[1];

    def updateStatus(idx: int, status: str) {
        setStatusMap(
            lambda prev: dict  -> dict { new_map = {** prev};new_map[idx] = status;return new_map; }
        );
    }

    def timeAgo(date_str: str) -> str {
        if not date_str {
            return "N/A";
        }
        try {
            now: float = Date.now();
            target: float = Date.parse(date_str);
            if isNaN(target) {
                return date_str;
            }
            diff: float = (now - target) / 1000;

            if diff < 60 {
                return "Just now";
            } elif diff < 3600 {
                mins: int = Math.floor(diff / 60);
                return f"{mins} minute{'s' if mins != 1 else ''} ago";
            } elif diff < 86400 {
                hours: int = Math.floor(diff / 3600);
                return f"{hours} hour{'s' if hours != 1 else ''} ago";
            } elif diff < 604800 {
                days: int = Math.floor(diff / 86400);
                return f"{days} day{'s' if days != 1 else ''} ago";
            } else {
                weeks: int = Math.floor(diff / 604800);
                return f"{weeks} week{'s' if weeks != 1 else ''} ago";
            }
        } except (e) {
            return date_str;
        }
    }

    async def performCheck(idx: int) {
        if idx == -1 or idx >= ssh_connections.length {
            return;
        }
        conn = ssh_connections[idx];
        updateStatus(idx, "pending");

        key_path = conn["key_path"] if "key_path" in conn else "";

        res = await checkSshConnectionWalker(conn["host"], conn["user"], "", key_path);
        success = res["success"] if "success" in res else False;

        if success {
            updateStatus(idx, "success");
            if "timestamp" in res {
                handleUpdateSshConnection(idx, {"last_connected": res["timestamp"]});
            }
        } else {
            updateStatus(idx, "failed");
        }
    }

    useEffect(
        lambda -> None { if active_idx != -1 and active_idx < ssh_connections.length {
            performCheck(active_idx);
        }},
        [active_idx]
    );

    def handleConnectToggle(idx: int) {
        if active_idx == idx {
            # Check if current status is failed
            is_failed = (idx in statusMap) and statusMap[idx] == "failed";
            if is_failed {
                # Retry logic
                performCheck(idx);
            } else {
                # Disconnect logic
                handleSetActiveSshConnection(-1);
                updateStatus(idx, "unknown");
            }
        } else {
            if active_idx != -1 {
                updateStatus(active_idx, "unknown");
            }
            updateStatus(idx, "pending");
            handleSetActiveSshConnection(idx);
        }
    }

    def handleChange(field: str, value: str) {
        setFormData(lambda prev: dict  -> dict { return {** prev, [field]: value}; });
    }

    def onSave {
        handleSaveSshConnection(formData);
        setView("list");
        setFormData({"host": "", "user": "", "key_path": ""});
    }

    if view == "add" {
        return
            <div className="space-y-4">
                <h2 className="text-xl font-bold">
                    Add New Connection
                </h2>
                <div
                    className="space-y-3 p-4 bg-zinc-900/50 rounded-lg border border-zinc-800"
                >
                    <div className="space-y-1">
                        <label
                            className="text-xs font-semibold text-zinc-400 uppercase"
                        >
                            Host IP
                        </label>
                        <input
                            type="text"
                            value={formData["host"]}
                            onChange={lambda e: any  -> any { handleChange(
                                "host", e.target.value
                            );}}
                            className="w-full h-10 bg-zinc-900 border border-zinc-800 rounded px-3 text-zinc-200"
                            placeholder="192.168.1.100"
                        />
                    </div>
                    <div className="space-y-1">
                        <label
                            className="text-xs font-semibold text-zinc-400 uppercase"
                        >
                            User
                        </label>
                        <input
                            type="text"
                            value={formData["user"]}
                            onChange={lambda e: any  -> any { handleChange(
                                "user", e.target.value
                            );}}
                            className="w-full h-10 bg-zinc-900 border border-zinc-800 rounded px-3 text-zinc-200"
                            placeholder="user"
                        />
                    </div>
                    <div className="space-y-1">
                        <label
                            className="text-xs font-semibold text-zinc-400 uppercase"
                        >
                            Key Path
                        </label>
                        <input
                            type="text"
                            value={formData["key_path"]}
                            onChange={lambda e: any  -> any { handleChange(
                                "key_path", e.target.value
                            );}}
                            className="w-full h-10 bg-zinc-900 border border-zinc-800 rounded px-3 text-zinc-200"
                            placeholder="~/.ssh/id_rsa"
                        />
                    </div>
                </div>
                <div className="flex gap-3">
                    <button
                        onClick={onSave}
                        disabled={saving}
                        className="flex-1 h-10 bg-zinc-100 hover:bg-white text-zinc-950 font-bold rounded-lg transition-all shadow-lg hover:shadow-zinc-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {"Saving..." if saving else "Save"}
                    </button>
                    <button
                        onClick={lambda -> None { setView("list");}}
                        className="flex-1 h-10 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 font-bold rounded-lg transition-all border border-zinc-700"
                    >
                        Cancel
                    </button>
                </div>
            </div>;
    }

    conn_list = [];
    i = 0;
    for conn in ssh_connections {
        # Capture current index in closure to avoid all buttons using final i value
        # Capture current index in closure to avoid all buttons using final i value
        # Capture current index in closure to avoid all buttons using final i value
        # Capture current index in closure to avoid all buttons using final i value
        current_idx = i;

        status_cls = "bg-zinc-700";
        if current_idx in statusMap {
            if statusMap[current_idx] == "success" {
                status_cls = "bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]";
            } elif statusMap[current_idx] == "failed" {
                status_cls = "bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]";
            } elif statusMap[current_idx] == "pending" {
                status_cls = "bg-yellow-500 animate-pulse shadow-[0_0_8px_rgba(234,179,8,0.5)]";
            }
        }

        is_pending = (current_idx in statusMap) and statusMap[current_idx] == "pending";

        is_failed = (current_idx in statusMap) and statusMap[current_idx] == "failed";

        btn_cls: str = "bg-zinc-800 text-zinc-300 border-zinc-700 hover:bg-zinc-700 hover:text-white";
        if active_idx == current_idx and not is_failed {
            # Disconnect style (Red) only if properly connected/pending logic flows
            # But wait, pending also uses this. Let's start with basic connect style.
            btn_cls = "bg-red-500/10 text-red-500 border-red-500/50 hover:bg-red-500/20";
        }

        if is_pending {
            btn_cls = btn_cls + " opacity-50 cursor-not-allowed";
        }

        btn_text = "Connect";
        if active_idx == current_idx {
            if is_failed {
                btn_text = "Retry";
                # Revert to connect style for Retry
                btn_cls = "bg-zinc-800 text-zinc-300 border-zinc-700 hover:bg-zinc-700 hover:text-white";
            } else {
                btn_text = "Disconnect";
            }
        }

        if is_pending {
            btn_text = "Connecting...";
        }

        conn_list = conn_list.concat(
            [
                <div
                    key={current_idx}
                    className="grid grid-cols-12 gap-2 items-center text-sm py-2 border-b border-zinc-800/50 last:border-0 hover:bg-zinc-900/40 rounded transition-colors px-2"
                >
                    <div
                        className="col-span-2 min-w-0 truncate text-zinc-300"
                        title={conn["user"]}
                    >
                        {conn["user"]}
                    </div>
                    <div
                        className="col-span-4 min-w-0 truncate text-zinc-300 font-mono"
                        title={conn["host"]}
                    >
                        {conn["host"]}
                    </div>
                    <div className="col-span-2 text-zinc-500 text-xs text-center">
                        {timeAgo(conn["last_connected"])
                        if "last_connected" in conn
                        else "N/A"}
                    </div>
                    <div className="col-span-1 flex justify-center">
                        <div className={f"w-2.5 h-2.5 rounded-full {status_cls}"} />
                    </div>
                    <div className="col-span-2 flex justify-end gap-2">
                        <button
                            onClick={lambda -> None { handleConnectToggle(current_idx);}}
                            disabled={is_pending}
                            className={f"flex-1 px-3 py-1.5 rounded text-xs font-bold transition-all border {btn_cls}"}
                        >
                            {btn_text}
                        </button>
                    </div>
                    <div className="col-span-1 flex justify-center">
                        <button
                            onClick={lambda -> None { handleRemoveSshConnection(
                                current_idx
                            );}}
                            className="p-1.5 rounded text-zinc-500 hover:text-red-400 hover:bg-red-400/10 transition-colors"
                            title="Delete Connection"
                        >
                            <TrashIcon />
                        </button>
                    </div>
                </div>
            ]
        );
        i += 1;
    }

    return
        <div className="space-y-4">
            <div className="flex items-center justify-between">
                <h2 className="text-xl font-bold">
                    SSH Connections
                </h2>
            </div>
            <div
                className="bg-zinc-900/50 rounded-lg p-4 border border-zinc-800 min-h-[300px]"
            >
                <div
                    className="grid grid-cols-12 gap-2 font-medium border-b border-zinc-800 pb-2 mb-2 text-zinc-400 uppercase text-xs tracking-wider px-2"
                >
                    <div className="col-span-2 min-w-0">
                        user
                    </div>
                    <div className="col-span-4 min-w-0">
                        host
                    </div>
                    <div className="col-span-2">
                        last connected
                    </div>
                    <div className="col-span-1 text-center">
                        status
                    </div>
                    <div className="col-span-2 text-center">
                        toggle
                    </div>
                    <div className="col-span-1 text-center">
                        Remove
                    </div>
                </div>
                <div className="space-y-2">
                    {conn_list}{<div
                        className="text-zinc-500 text-center py-12 flex flex-col items-center gap-2"
                    >
                        <span className="italic">
                            No connections saved
                        </span>
                    </div>
                    if ssh_connections.length == 0
                    else null}
                </div>
            </div>
            <div className="flex gap-3">
                <button
                    onClick={lambda -> None { setView("add");}}
                    className="flex-1 h-12 bg-zinc-100 hover:bg-white text-zinc-950 font-bold rounded-lg transition-all shadow-lg hover:shadow-zinc-500/20"
                >
                    Add New SSH
                </button>
            </div>
        </div>;
}
