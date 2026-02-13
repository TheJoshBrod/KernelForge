"""Verification Page"""

import from "../../components/ui/Icons.cl.jac" {
    ShieldCheckIcon,
    XIcon
}
import from "../../components/ui/GlowButton.cl.jac" { GlowButton }

def:pub Verification {
    # Mock Test Suite
    tests = [
        {
            "name": "Standard Input",
            "desc": "Shape [32, 64, 224, 224]",
            "status": "pass",
            "error": None
        },
        {
            "name": "Odd Dimensions",
            "desc": "Shape [7, 63, 111, 113]",
            "status": "fail",
            "error": "atol > 1e-3"
        },
        {
            "name": "Zero Weights",
            "desc": "Sparsity Check",
            "status": "pass",
            "error": None
        }
    ];

    test_elements = [];
    for t in tests {
        border_class = "border-green-900/50";
        icon_color = "text-green-500";
        status_text_color = "text-green-400";
        status_text = "Pass";
        
        if t["status"] == "fail" {
            border_class = "border-red-900/50 ring-1 ring-red-500";
            icon_color = "text-red-500";
            status_text_color = "text-red-400";
            status_text = f"Fail ({t['error']})";
        }

        test_elements.push(
            <div key={t["name"]} className={"bg-zinc-950 border p-4 rounded-lg flex justify-between items-center cursor-pointer hover:bg-zinc-900 transition-colors " + border_class}>
                <div className="flex gap-3 items-center">
                    <ShieldCheckIcon className={icon_color + " w-5 h-5"} />
                    <div>
                        <div className="text-sm font-medium text-zinc-200">
                            {t["name"]}
                        </div>
                        <div className="text-xs text-zinc-500">
                            {t["desc"]}
                        </div>
                    </div>
                </div>
                <span className={"text-xs font-mono " + status_text_color}>
                    {status_text}
                </span>
            </div>
        );
    }
    
    # Heatmap Grid Generation
    heatmap_cells = [];
    i = 0;
    while i < 144 {
        is_error = (i % 7 == 0);
        bg_class = "bg-red-500" if is_error else "bg-green-900/20";
        opacity = 0.8 if is_error else 0.2;
        
        heatmap_cells.push(
            <div 
                key={i} 
                className={"w-3 h-3 rounded-sm " + bg_class}
                style={{"opacity": opacity}}
            ></div>
        );
        i += 1;
    }

    return
        <div className="h-full bg-zinc-900 text-zinc-100 flex p-6 gap-6 font-sans">
            
            <div className="w-1/3 flex flex-col gap-4">
                <div className="flex justify-between items-center">
                    <h2 className="text-xl font-bold">
                        Test Suite
                    </h2>
                    <button className="text-xs text-zinc-400 hover:text-white flex gap-1 items-center transition-colors">
                        Rerun All
                    </button>
                </div>
                <div className="space-y-2 overflow-y-auto">
                    {test_elements}
                </div>
            </div>

            
            <div className="flex-1 bg-zinc-950 border border-zinc-800 rounded-xl p-6 flex flex-col shadow-sm">
                <div className="flex justify-between mb-6">
                    <h3 className="font-semibold text-red-400 flex items-center gap-2">
                        <XIcon className="w-4 h-4" />
                        Failure Analysis: Odd Dimensions
                    </h3>
                    <div className="flex gap-4 text-xs font-mono">
                        <span className="text-zinc-500">
                            Max Diff:{" "}
                            <span className="text-red-400">
                                0.042
                            </span>
                        </span>
                        <span className="text-zinc-500">
                            Mean Diff:{" "}
                            <span className="text-yellow-400">
                                0.002
                            </span>
                        </span>
                    </div>
                </div>

                <div className="flex-1 flex flex-col items-center justify-center bg-zinc-900/50 rounded border border-dashed border-zinc-800 relative min-h-0 overflow-hidden">
                    <p className="absolute top-2 left-2 text-xs text-zinc-500 font-mono">
                        Error Heatmap (Channel 0)
                    </p>
                    
                    
                    <div className="grid grid-cols-12 gap-1 p-4">
                        {heatmap_cells}
                    </div>
                    <p className="mt-4 text-xs text-zinc-400">
                        Tip: Check boundary conditions in `tl.load` mask.
                    </p>
                </div>
            </div>
        </div>;
}
