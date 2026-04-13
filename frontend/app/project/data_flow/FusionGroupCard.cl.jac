"""FusionGroupCard — Sidebar card showing one fusion pattern and all its instances.

The card calls onSelect(patternName), onAccept(firstGroupId), onReject(firstGroupId)
using its own stable prop values — no closure over loop variables needed.
"""

import from ....components.ui.Icons { CheckIcon, XIcon }

def:pub FusionGroupCard(props: any) -> Any {
    patternName = props.patternName if props and props.patternName else "Fusion Group";
    instances = props.instances if props and props.instances else [];
    nodeMap = props.nodeMap if props and props.nodeMap else {};
    isSelected = props.isSelected if props and "isSelected" in props else False;
    colorBorder = props.colorBorder if props and props.colorBorder else "#c75b39";
    colorBg = props.colorBg if props and props.colorBg else "rgba(199,91,57,0.12)";
    score = props.score if props and "score" in props else None;
    estimatedSpeedup = props.estimatedSpeedup if props and "estimatedSpeedup" in props else None;
    rationale = props.rationale if props and props.rationale else "";
    status = props.status if props and props.status else "proposed";
    onSelect = props.onSelect if props and props.onSelect else None;
    onAccept = props.onAccept if props and props.onAccept else None;
    onReject = props.onReject if props and props.onReject else None;

    # Stable handlers — use the card's own props, not outer loop variables
    def handleSelect -> None {
        if onSelect { onSelect(patternName); }
    }

    def handleAction(cb: any) -> None {
        if cb and instances and instances.length > 0 {
            first = instances[0];
            gid = first["id"] if first and "id" in first else "";
            if gid { cb(gid); }
        }
    }

    instance_count = instances.length if instances else 0;
    first_instance = instances[0] if instance_count > 0 else {};
    members = first_instance["members"] if first_instance and "members" in first_instance else [];

    # Build op chain from first instance members
    ops_in_pattern = [];
    for member_id in members {
        n = nodeMap[member_id] if member_id in nodeMap else None;
        op_label = n["op"] if n and "op" in n else member_id;
        ops_in_pattern.push(op_label);
    }

    speedup_label = "";
    if estimatedSpeedup is not None {
        speedup_label = "~" + estimatedSpeedup.toFixed(1) + "x";
    }

    status_color = "var(--text-muted)";
    status_label = "proposed";
    if status == "accepted" {
        status_color = "#27ae60";
        status_label = "accepted";
    } elif status == "rejected" {
        status_color = "#d63031";
        status_label = "rejected";
    }

    card_border = "1px solid " + colorBorder if isSelected else "1px solid var(--border)";
    card_bg = colorBg if isSelected else "var(--bg-charcoal)";

    score_pct = "50%";
    score_color = "var(--text-muted)";
    if score is not None {
        score_pct = (score * 100).toFixed(0) + "%";
        if score >= 0.7 {
            score_color = "#27ae60";
        } elif score >= 0.4 {
            score_color = "var(--molten-gold)";
        } else {
            score_color = "#d63031";
        }
    }
    score_el = <span></span>;
    if score is not None {
        score_el =
            <div
                style={{
                    "height": "2px",
                    "background": "var(--surface-0)",
                    "borderRadius": "1px",
                    "marginBottom": "8px",
                    "overflow": "hidden"
                }}
            >
                <div
                    style={{
                        "height": "100%",
                        "width": score_pct,
                        "background": score_color,
                        "borderRadius": "1px"
                    }}
                />
            </div>;
    }

    op_chips = [];
    chip_idx = 0;
    for op_name in ops_in_pattern {
        is_last = chip_idx == ops_in_pattern.length - 1;
        arrow_display = "none" if is_last else "inline";
        op_chips.push(
            <span
                key={op_name + chip_idx}
                style={{"display": "inline-flex", "alignItems": "center", "gap": "2px"}}
            >
                <span
                    style={{
                        "fontFamily": "monospace",
                        "fontSize": "9px",
                        "background": "var(--surface-0)",
                        "color": "var(--text-light)",
                        "padding": "1px 5px",
                        "borderRadius": "3px",
                        "border": "1px solid var(--border)"
                    }}
                >
                    {op_name}
                </span>
                <span
                    style={{
                        "color": "var(--text-muted)",
                        "fontSize": "9px",
                        "display": arrow_display
                    }}
                >
                    →
                </span>
            </span>
        );
        chip_idx = chip_idx + 1;
    }

    accept_btn = <span></span>;
    reject_btn = <span></span>;
    if status == "proposed" {
        accept_btn =
            <button
                onClick={lambda e: any  -> None { e.stopPropagation(); handleAction(onAccept); }}
                title="Accept all instances"
                style={{
                    "background": "rgba(39,174,96,0.1)",
                    "border": "1px solid #27ae60",
                    "borderRadius": "4px",
                    "padding": "3px 6px",
                    "cursor": "pointer",
                    "display": "flex",
                    "alignItems": "center"
                }}
            >
                <CheckIcon className="w-3 h-3 text-emerald-400"/>
            </button>;
        reject_btn =
            <button
                onClick={lambda e: any  -> None { e.stopPropagation(); handleAction(onReject); }}
                title="Reject all instances"
                style={{
                    "background": "rgba(214,48,49,0.1)",
                    "border": "1px solid #d63031",
                    "borderRadius": "4px",
                    "padding": "3px 6px",
                    "cursor": "pointer",
                    "display": "flex",
                    "alignItems": "center"
                }}
            >
                <XIcon className="w-3 h-3 text-red-400"/>
            </button>;
    }

    instance_word = "instance" if instance_count == 1 else "instances";
    instance_summary = instance_count + " " + instance_word;
    if speedup_label {
        instance_summary = instance_summary + " · " + speedup_label;
    }

    return
        <div
            onClick={handleSelect}
            style={{
                "background": card_bg,
                "border": card_border,
                "borderLeft": "3px solid " + colorBorder,
                "borderRadius": "6px",
                "padding": "10px",
                "cursor": "pointer",
                "transition": "border-color 0.15s"
            }}
        >
            <div
                style={{
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "flex-start",
                    "marginBottom": "6px"
                }}
            >
                <div style={{"flex": "1", "minWidth": "0"}}>
                    <div
                        style={{
                            "fontFamily": "Oswald, sans-serif",
                            "fontSize": "11px",
                            "textTransform": "uppercase",
                            "letterSpacing": "0.08em",
                            "color": colorBorder,
                            "fontWeight": "600",
                            "overflow": "hidden",
                            "textOverflow": "ellipsis",
                            "whiteSpace": "nowrap"
                        }}
                    >
                        {patternName}
                    </div>
                    <div
                        style={{
                            "fontSize": "9px",
                            "color": "var(--text-muted)",
                            "fontFamily": "monospace",
                            "marginTop": "2px"
                        }}
                    >
                        {instance_summary}
                    </div>
                </div>
                <span
                    style={{
                        "fontSize": "9px",
                        "fontFamily": "monospace",
                        "color": status_color,
                        "textTransform": "uppercase",
                        "letterSpacing": "0.05em",
                        "marginLeft": "8px",
                        "flexShrink": "0"
                    }}
                >
                    {status_label}
                </span>
            </div>
            <div
                style={{
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "3px",
                    "alignItems": "center",
                    "marginBottom": "6px"
                }}
            >
                {op_chips}
            </div>
            {score_el}
            <div
                style={{
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "gap": "6px"
                }}
            >
                <p
                    style={{
                        "fontSize": "9px",
                        "color": "var(--text-muted)",
                        "fontFamily": "monospace",
                        "flex": "1",
                        "margin": "0",
                        "overflow": "hidden",
                        "textOverflow": "ellipsis",
                        "whiteSpace": "nowrap"
                    }}
                    title={rationale}
                >
                    {rationale}
                </p>
                <div style={{"display": "flex", "gap": "4px", "flexShrink": "0"}}>
                    {accept_btn}
                    {reject_btn}
                </div>
            </div>
        </div>;
}
