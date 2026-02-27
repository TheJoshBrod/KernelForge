"""Live job log tail modal component."""

import from ....components.ui.Modal { Modal }
import from ..shared.MetricsAdapter { to_list, list_len }

def:pub JobLogModal(props: any) -> Any {
    isOpen = props.isOpen;
    onClose = props.onClose;
    activeLogKey = props.activeLogKey;
    logTail = props.logTail;

    log_line_nodes = [];
    for log_line in to_list(logTail) {
        ls = str(log_line);
        lsl = ls.toLowerCase();
        lc = "truncate ";
        if lsl.includes("error") or lsl.includes("failed") or lsl.includes("exception") {
            lc = lc + "text-red-400";
        } elif lsl.includes("warning") {
            lc = lc + "text-yellow-400/80";
        } elif lsl.includes("success") or lsl.includes("passed") or lsl.includes("completed") {
            lc = lc + "text-emerald-400/80";
        } elif ls.startsWith("[workflow]") or ls.startsWith("[Remote]") or ls.startsWith("DEBUG:") {
            lc = lc + "text-zinc-300";
        }
        log_line_nodes.push(<div key={ls} className={lc}>{ls}</div>);
    }

    return
        <Modal
            isOpen={isOpen}
            onClose={onClose}
            title={f"{activeLogKey}.log — Live Tail"}
        >
            <div className="p-4 font-mono text-[11px] leading-[1.75] space-y-px">
                {log_line_nodes
                if list_len(log_line_nodes) > 0
                else <div className="text-zinc-500 text-xs">
                    No log output captured yet.
                </div>}
            </div>
        </Modal>;
}
