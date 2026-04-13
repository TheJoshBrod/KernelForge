"""FusionPanel — Right sidebar for managing kernel fusion groups.

Groups the flat fusionGroups list by pattern_name and shows one card per pattern.
Passes callbacks directly (no lambdas over loop vars) to avoid closure issues.
"""

import from ....components.ui.GlowButton { GlowButton }
import from .FusionGroupCard { FusionGroupCard }
import from "./fusionColors.js" { FUSION_COLORS }

def:pub FusionPanel(props: any) -> Any {
    fusionGroups = props.fusionGroups if props and props.fusionGroups else [];
    allNodes = props.allNodes if props and props.allNodes else [];
    selectedPatternName = props.selectedPatternName if props and props.selectedPatternName else None;
    onSelectPattern = props.onSelectPattern if props and props.onSelectPattern else None;
    onAccept = props.onAccept if props and props.onAccept else None;
    onReject = props.onReject if props and props.onReject else None;
    onPropose = props.onPropose if props and props.onPropose else None;
    onClose = props.onClose if props and props.onClose else None;

    # Build node map for member op lookup in FusionGroupCard
    node_map = {};
    for n in allNodes {
        nid = n["id"] if "id" in n else "";
        if nid { node_map[nid] = n; }
    }

    # Group instances by pattern_name, collecting metadata from the first entry
    patterns_map = {};
    patterns_order = [];
    for g in fusionGroups {
        g_status = g["status"] if "status" in g else "";
        if g_status == "rejected" { continue; }
        pname = g["pattern_name"] if "pattern_name" in g else g["name"] if "name" in g else "";
        if not pname { continue; }

        if not (pname in patterns_map) {
            cidx_raw = g["color_index"] if "color_index" in g else 0;
            cidx = cidx_raw if cidx_raw else 0;
            color = FUSION_COLORS[cidx % FUSION_COLORS.length];
            patterns_map[pname] = {
                "pattern_name": pname,
                "instances": [],
                "color": color,
                "score": g["score"] if "score" in g else 0.5,
                "estimated_speedup": g["estimated_speedup"] if "estimated_speedup" in g else None,
                "rationale": g["rationale"] if "rationale" in g else "",
                "status": g_status
            };
            patterns_order.push(pname);
        }

        patterns_map[pname]["instances"].push(g);
        if g_status == "accepted" {
            patterns_map[pname]["status"] = "accepted";
        }
    }

    # patterns_order is already score-sorted: the walker emits groups in descending
    # score order, and we push pattern names in first-seen order while iterating
    # fusionGroups, so insertion order == score order.

    total_count = patterns_order.length;
    has_groups = total_count > 0;

    accepted_count = 0;
    for pname in patterns_order {
        if patterns_map[pname]["status"] == "accepted" {
            accepted_count = accepted_count + 1;
        }
    }

    plural = "patterns" if total_count != 1 else "pattern";
    summary_label = str(total_count) + " " + plural;
    if accepted_count > 0 {
        summary_label = summary_label + " · " + str(accepted_count) + " accepted";
    }

    # Build cards — pass callbacks directly (no closures over loop vars)
    cards = [];
    for pname in patterns_order {
        pat = patterns_map[pname];
        color = pat["color"];
        instances = pat["instances"];

        cards.push(
            <FusionGroupCard
                key={pname}
                patternName={pname}
                instances={instances}
                nodeMap={node_map}
                isSelected={pname == selectedPatternName}
                colorBorder={color["border"]}
                colorBg={color["bg"]}
                score={pat["score"]}
                estimatedSpeedup={pat["estimated_speedup"]}
                rationale={pat["rationale"]}
                status={pat["status"]}
                onSelect={onSelectPattern}
                onAccept={onAccept}
                onReject={onReject}
            />
        );
    }

    propose_section = <span></span>;
    if not has_groups {
        propose_section =
            <div
                style={{
                    "padding": "16px",
                    "borderTop": "1px solid var(--border)",
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "10px"
                }}
            >
                <p
                    style={{
                        "fontSize": "11px",
                        "color": "var(--text-muted)",
                        "fontFamily": "monospace",
                        "margin": "0",
                        "textAlign": "center",
                        "lineHeight": "1.5"
                    }}
                >
                    No fusion groups yet. Run the LLM analyzer to detect candidates.
                </p>
                <GlowButton color="accent" onClick={onPropose}>
                    Propose Fusions
                </GlowButton>
            </div>;
    } else {
        propose_section =
            <div style={{"padding": "10px 14px", "borderTop": "1px solid var(--border)"}}>
                <button
                    onClick={onPropose}
                    style={{
                        "width": "100%",
                        "padding": "6px",
                        "background": "var(--surface-0)",
                        "border": "1px solid var(--border)",
                        "borderRadius": "4px",
                        "color": "var(--text-muted)",
                        "fontSize": "10px",
                        "fontFamily": "monospace",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.08em",
                        "cursor": "pointer"
                    }}
                >
                    Re-analyze
                </button>
            </div>;
    }

    return
        <div
            style={{
                "width": "300px",
                "height": "100%",
                "background": "var(--bg-panel)",
                "borderLeft": "1px solid var(--border)",
                "display": "flex",
                "flexDirection": "column",
                "flexShrink": "0"
            }}
        >
            <div
                style={{
                    "flexShrink": "0",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "padding": "14px 16px",
                    "borderBottom": "1px solid var(--border)"
                }}
            >
                <div>
                    <h3
                        className="industrial-font-display tracking-widest uppercase"
                        style={{
                            "fontSize": "12px",
                            "fontWeight": "700",
                            "color": "var(--text-light)",
                            "margin": "0"
                        }}
                    >
                        Fusion Groups
                    </h3>
                    <p
                        style={{
                            "fontSize": "10px",
                            "color": "var(--text-muted)",
                            "fontFamily": "monospace",
                            "margin": "2px 0 0 0"
                        }}
                    >
                        {summary_label}
                    </p>
                </div>
                <button
                    onClick={onClose}
                    style={{
                        "background": "none",
                        "border": "none",
                        "color": "var(--text-muted)",
                        "cursor": "pointer",
                        "fontSize": "20px",
                        "lineHeight": "1",
                        "padding": "2px 6px"
                    }}
                >
                    ×
                </button>
            </div>
            <div
                style={{"flex": "1", "overflowY": "auto", "padding": "10px"}}
                className="custom-scrollbar"
            >
                <div style={{"display": "flex", "flexDirection": "column", "gap": "8px"}}>
                    {cards}
                </div>
            </div>
            {propose_section}
        </div>;
}
