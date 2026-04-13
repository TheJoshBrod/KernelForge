"""FusionGroupNode — ReactFlow compound group node for a kernel fusion candidate."""

def:pub FusionGroupNode(props: any) -> Any {
    data = props.data if props and props.data else {};
    name = data["name"] if "name" in data else "Fusion Group";
    estimated_speedup = data["estimated_speedup"]
        if "estimated_speedup" in data
        else None;
    color = data["color"]
        if "color" in data
        else {"bg": "rgba(199,91,57,0.12)", "border": "#c75b39"};
    status = data["status"] if "status" in data else "proposed";

    speedup_label = "";
    if estimated_speedup {
        speedup_label = "~" + estimated_speedup.toFixed(1) + "x";
    }

    header_alpha = "rgba(0,0,0,0.25)";
    if status == "accepted" {
        header_alpha = "rgba(39,174,96,0.15)";
    }

    status_dot_color = "var(--text-muted)";
    if status == "accepted" {
        status_dot_color = "#27ae60";
    } elif status == "proposed" {
        status_dot_color = color["border"];
    }

    return
        <div
            style={{
                "background": color["bg"],
                "border": "2px dashed " + color["border"],
                "borderRadius": "8px",
                "width": "100%",
                "height": "100%",
                "position": "relative",
                "boxSizing": "border-box",
                "overflow": "hidden"
            }}
        >
            <div
                style={{
                    "position": "absolute",
                    "top": "0",
                    "left": "0",
                    "right": "0",
                    "padding": "4px 8px",
                    "background": header_alpha,
                    "borderBottom": "1px dashed " + color["border"],
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "boxSizing": "border-box",
                    "borderTopLeftRadius": "6px",
                    "borderTopRightRadius": "6px"
                }}
            >
                <span
                    title={name}
                    style={{
                        "fontFamily": "Oswald, sans-serif",
                        "fontSize": "9px",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.1em",
                        "color": color["border"],
                        "fontWeight": "600",
                        "overflow": "hidden",
                        "textOverflow": "ellipsis",
                        "whiteSpace": "nowrap",
                        "flex": "1",
                        "minWidth": "0"
                    }}
                >
                    {name}
                </span>
                <span
                    style={{
                        "fontSize": "9px",
                        "fontFamily": "monospace",
                        "color": status_dot_color,
                        "marginLeft": "6px",
                        "flexShrink": "0"
                    }}
                >
                    {speedup_label}
                </span>
            </div>
        </div>;
}
