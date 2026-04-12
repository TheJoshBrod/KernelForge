"""OpNode — Custom ReactFlow node for a single profiled operator."""

import from reactflow { Handle, Position }

def:pub OpNode(props: any) -> Any {
    data = props.data if props and props.data else {};
    op = data["op"] if "op" in data else "";
    pytorch_ms = data["pytorch_ms"] if "pytorch_ms" in data else None;
    call_count = data["call_count"] if "call_count" in data else 0;
    winner = data["winner"] if "winner" in data else "";
    speedup = data["speedup"] if "speedup" in data else None;
    status = data["status"] if "status" in data else "";

    # Status stripe: left border color encodes kernel state
    stripe_color = "var(--accent-copper)";
    if winner == "optimized" {
        stripe_color = "#27ae60";
    } elif winner == "pytorch" {
        stripe_color = "#d63031";
    } elif status == "running" {
        stripe_color = "var(--molten-gold)";
    }

    # Timing label — guard with truthiness to avoid null.toFixed crash
    ms_label = "";
    if pytorch_ms {
        ms_label = pytorch_ms.toFixed(2) + "ms";
    }

    count_label = "";
    if call_count {
        count_label = "(" + call_count + "x)";
    }

    speedup_color = "#27ae60";
    speedup_label = "";
    if speedup {
        speedup_label = speedup.toFixed(2) + "x";
        if speedup < 1.0 {
            speedup_color = "#d63031";
        }
    }

    speedup_el = <span></span>;
    if speedup_label {
        speedup_el =
            <span
                style={{
                    "color": speedup_color,
                    "fontSize": "9px",
                    "fontFamily": "monospace",
                    "fontWeight": "600"
                }}
            >
                {speedup_label}
            </span>;
    }

    timing_row = <span></span>;
    if ms_label {
        timing_row =
            <div
                style={{
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "marginTop": "3px"
                }}
            >
                <span
                    style={{
                        "fontSize": "10px",
                        "color": "var(--text-muted)",
                        "fontFamily": "monospace"
                    }}
                >
                    {ms_label} {count_label}
                </span>
                {speedup_el}
            </div>;
    }

    return
        <div
            style={{
                "background": "var(--bg-panel)",
                "border": "1px solid var(--border)",
                "borderLeft": "3px solid " + stripe_color,
                "borderRadius": "4px",
                "padding": "6px 10px",
                "width": "180px",
                "boxSizing": "border-box",
                "cursor": "pointer"
            }}
        >
            <Handle
                type="target"
                position={Position.Top}
                style={{"background": "var(--border)", "width": "6px", "height": "6px", "border": "none"}}
            />
            <div
                style={{
                    "fontSize": "11px",
                    "fontFamily": "Oswald, sans-serif",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.08em",
                    "color": "var(--text-light)",
                    "fontWeight": "600",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                    "whiteSpace": "nowrap"
                }}
            >
                {op}
            </div>
            {timing_row}
            <Handle
                type="source"
                position={Position.Bottom}
                style={{"background": "var(--border)", "width": "6px", "height": "6px", "border": "none"}}
            />
        </div>;
}
