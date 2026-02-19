import from react { useState, useEffect }
import from ...components.ui.Icons { ChevronDownIcon, TrashIcon }

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
        } except Exception as e {
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
                <h2 className="text-xl font-bold text-[var(--text-light)] industrial-font-display uppercase tracking-wider">
                    Add New Connection
                </h2>
                <div
                    className="space-y-3 p-4 bg-[var(--bg-darker)]/30 rounded-sm border border-[var(--border)]"
                >
                    <div className="space-y-1">
                        <label
                            className="text-xs font-mono text-[var(--text-muted)] uppercase tracking-widest"
                        >
                            Host IP
                        </label>
                        <input
                            type="text"
                            value={formData["host"]}
                            onChange={lambda e: any  -> any { handleChange(
                                "host", e.target.value
                            );}}
                            className="w-full h-10 bg-[var(--bg-darkest)] border border-[var(--border)] rounded-sm px-3 text-[var(--text-light)] focus:border-[var(--accent)] outline-none font-mono text-sm shadow-[inset_0_2px_10px_rgba(0,0,0,0.5)]"
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
                            className="w-full h-10 bg-[var(--bg-darkest)] border border-[var(--border)] rounded-sm px-3 text-[var(--text-light)] focus:border-[var(--accent)] outline-none font-mono text-sm shadow-[inset_0_2px_10px_rgba(0,0,0,0.5)]"
                            placeholder="user"
                        />
                    </div>
                    <div className="space-y-1">
                        <label
                            className="text-xs font-mono text-[var(--text-muted)] uppercase tracking-widest"
                        >
                            Key Path
                        </label>
                        <input
                            type="text"
                            value={formData["key_path"]}
                            onChange={lambda e: any  -> any { handleChange(
                                "key_path", e.target.value
                            );}}
                            className="w-full h-10 bg-[var(--bg-darkest)] border border-[var(--border)] rounded-sm px-3 text-[var(--text-light)] focus:border-[var(--accent)] outline-none font-mono text-sm shadow-[inset_0_2px_10px_rgba(0,0,0,0.5)]"
                            placeholder="~/.ssh/id_rsa"
                        />
                    </div>
                </div>
                <div className="flex gap-4">
                    <button
                        onClick={onSave}
                        disabled={saving}
                        className="app-button flex-1"
                    >
                        {"Saving..." if saving else "Save"}
                    </button>
                    <button
                        onClick={lambda -> None { setView("list");}}
                        className="app-button app-button--secondary flex-1"
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
                    className="grid grid-cols-12 gap-2 items-center text-sm py-2 border-b border-[var(--border)] last:border-0 hover:bg-[var(--surface-0)] transition-colors px-2"
                >
                    <div
                        className="col-span-2 min-w-0 truncate text-[var(--text-light)] font-mono text-xs"
                        title={conn["user"]}
                    >
                        {conn["user"]}
                    </div>
                    <div
                        className="col-span-4 min-w-0 truncate text-[var(--accent)] font-mono text-xs"
                        title={conn["host"]}
                    >
                        {conn["host"]}
                    </div>
                    <div className="col-span-2 text-[var(--text-muted)] font-mono text-xs text-center uppercase tracking-wider">
                        {timeAgo(conn["last_connected"])
                        if "last_connected" in conn
                        else "N/A"}
                    </div>
                    <div className="col-span-1 flex justify-center">
                        <div className={f"w-2.5 h-2.5 rounded-full {status_cls}"}/>
                    </div>
                    <div className="col-span-2 flex justify-end gap-2">
                        <button
                            onClick={lambda -> None { handleConnectToggle(current_idx);}}
                            disabled={is_pending}
                            className={f"flex-1 px-3 py-1.5 rounded-sm text-[10px] uppercase font-bold tracking-widest transition-all border {btn_cls} clip-anvil"}
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
                            <TrashIcon/>
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
                <h2 className="text-xl font-bold text-[var(--text-light)] industrial-font-display uppercase tracking-wider">
                    SSH Connections
                </h2>
            </div>
            <div
                className="bg-[var(--bg-darker)]/30 rounded-sm p-4 border border-[var(--border)] min-h-[300px]"
            >
                <div
                    className="grid grid-cols-12 gap-2 font-bold border-b border-[var(--border)] pb-2 mb-2 text-[var(--text-muted)] font-mono uppercase text-[10px] tracking-widest px-2"
                >
                    <div className="col-span-2 min-w-0">
                        User
                    </div>
                    <div className="col-span-4 min-w-0">
                        Host IP
                    </div>
                    <div className="col-span-2 text-center">
                        Last Active
                    </div>
                    <div className="col-span-1 text-center">
                        Status
                    </div>
                    <div className="col-span-2 text-center">
                        Action
                    </div>
                    <div className="col-span-1 text-center">
                        Drop
                    </div>
                </div>
                <div className="space-y-2">
                    {conn_list}
                    {(
                        <div
                            className="text-[var(--text-muted)] font-mono tracking-widest text-center py-12 flex flex-col items-center gap-2 uppercase text-xs"
                        >
                            <span className="italic opacity-50">
                                Empty Manifest
                            </span>
                        </div>
                    )
                    if ssh_connections.length == 0
                    else None}
                </div>
            </div>
            <div className="flex gap-4">
                <button
                    onClick={lambda -> None { setView("add");}}
                    className="app-button w-full justify-center"
                >
                    + NEW SSH LINK
                </button>
            </div>
        </div>;
}
