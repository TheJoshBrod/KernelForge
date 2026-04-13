"""FusionProposalModal — Modal for LLM-driven fusion proposal."""

import from react { useState, useEffect }
import from ....components.ui.Modal { Modal }
import from ....components.ui.GlowButton { GlowButton }
import from ....components.ui.Icons { BrainIcon }
import from ....components.ui.SearchableSelect { SearchableSelect }
import from ..shared.MetricsAdapter { to_list }

def:pub FusionProposalModal(props: any) -> Any {
    isOpen = props.isOpen if props and "isOpen" in props else False;
    onClose = props.onClose if props and props.onClose else None;
    onComplete = props.onComplete if props and props.onComplete else None;
    projectName = props.projectName if props and props.projectName else "";
    config = props.config if props and props.config else {};

    model_state = useState("");
    selectedModel = model_state[0];
    setSelectedModel = model_state[1];

    models_state = useState([]);
    models = models_state[0];
    setModels = models_state[1];

    loading_models_state = useState(False);
    loadingModels = loading_models_state[0];
    setLoadingModels = loading_models_state[1];

    running_state = useState(False);
    running = running_state[0];
    setRunning = running_state[1];

    error_state = useState("");
    errorMsg = error_state[0];
    setErrorMsg = error_state[1];

    useEffect(
        lambda -> None {
            if isOpen {
                async def fetchModels -> None {
                    setLoadingModels(True);
                    setErrorMsg("");
                    try {
                        resp = await __jacSpawn("GetOpenLLMModels", "", {"cfg": config});
                        data = resp.reports[0] if resp and resp.reports else {};
                        if data and "success" in data and data["success"] {
                            list_models = to_list(data["models"]);
                            setModels(list_models);
                            if list_models.length > 0 and not selectedModel {
                                setSelectedModel(list_models[0]["id"]);
                            }
                        }
                    } except Exception as e {
                        console.error("FusionProposalModal: model fetch error:", e);
                    }
                    setLoadingModels(False);
                }
                fetchModels();
            }
        },
        [isOpen]
    );

    async def runProposal -> None {
        if not selectedModel {
            setErrorMsg("Select a model first.");
            return;
        }
        setRunning(True);
        setErrorMsg("");
        try {
            resp = await __jacSpawn(
                "ProposeFusions", "",
                {"projectName": projectName, "model": selectedModel}
            );
            data = resp.reports[0] if resp and resp.reports else {};
            if data and "success" in data and data["success"] {
                groups = data["groups"] if "groups" in data else [];
                if onComplete { onComplete(groups); }
            } else {
                err = data["error"] if data and "error" in data else "Unknown error";
                setErrorMsg(err);
            }
        } except Exception as e {
            setErrorMsg("Request failed: " + str(e));
        }
        setRunning(False);
    }

    model_options = [];
    for m in models {
        model_options.push({
            "value": m["id"],
            "label": m["id"] + " (" + (m["provider"] if "provider" in m else "?") + ")"
        });
    }

    run_label = "Analyzing..." if running else "Propose Fusions";
    run_disabled = running or not selectedModel or loadingModels;

    error_el = <span></span>;
    if errorMsg {
        error_el =
            <div
                style={{
                    "background": "rgba(214,48,49,0.1)",
                    "border": "1px solid #d63031",
                    "borderRadius": "6px",
                    "padding": "10px 14px",
                    "fontSize": "11px",
                    "color": "#d63031",
                    "fontFamily": "monospace"
                }}
            >
                {errorMsg}
            </div>;
    }

    return
        <Modal isOpen={isOpen} onClose={onClose} title="Propose Kernel Fusions">
            <div style={{"padding": "20px", "display": "flex", "flexDirection": "column", "gap": "16px"}}>
                <div
                    style={{
                        "display": "flex",
                        "alignItems": "flex-start",
                        "gap": "12px",
                        "padding": "14px",
                        "background": "rgba(199,91,57,0.08)",
                        "border": "1px solid rgba(199,91,57,0.3)",
                        "borderRadius": "8px"
                    }}
                >
                    <BrainIcon className="w-5 h-5 text-[var(--accent-copper)] mt-0.5 flex-shrink-0"/>
                    <div>
                        <p
                            style={{
                                "fontSize": "12px",
                                "color": "var(--text-light)",
                                "margin": "0 0 4px 0",
                                "fontWeight": "600"
                            }}
                        >
                            LLM Fusion Analysis
                        </p>
                        <p
                            style={{
                                "fontSize": "11px",
                                "color": "var(--text-muted)",
                                "fontFamily": "monospace",
                                "margin": "0",
                                "lineHeight": "1.5"
                            }}
                        >
                            The compute graph is compressed into repeated subgraph patterns and sent to the selected model. The LLM identifies which adjacent op chains should be fused into single GPU kernels to reduce memory traffic.
                        </p>
                    </div>
                </div>
                <div>
                    <label
                        style={{
                            "display": "block",
                            "fontSize": "11px",
                            "fontFamily": "monospace",
                            "color": "var(--text-muted)",
                            "textTransform": "uppercase",
                            "letterSpacing": "0.08em",
                            "marginBottom": "6px"
                        }}
                    >
                        Model
                    </label>
                    <SearchableSelect
                        value={selectedModel}
                        onChange={setSelectedModel}
                        options={model_options}
                        placeholder="Select a model..."
                        loading={loadingModels}
                        disabled={running}
                    />
                </div>
                {error_el}
                <div style={{"display": "flex", "justifyContent": "flex-end", "gap": "10px"}}>
                    <button
                        onClick={onClose}
                        disabled={running}
                        style={{
                            "padding": "8px 16px",
                            "background": "var(--surface-0)",
                            "border": "1px solid var(--border)",
                            "borderRadius": "6px",
                            "color": "var(--text-muted)",
                            "fontSize": "12px",
                            "fontFamily": "monospace",
                            "cursor": "pointer"
                        }}
                    >
                        Cancel
                    </button>
                    <GlowButton label={run_label} onClick={runProposal} disabled={run_disabled}/>
                </div>
            </div>
        </Modal>;
}
