"""Arena Page"""

import from ...components.ui.Icons { ActivityIcon }
import from ...components.ui.GlowButton { GlowButton }

def:pub Arena -> Any {
    # Mock data for now
    kernels = [
        {
            "name": "Baseline (PyTorch)",
            "latency": 450.2,
            "tflops": 12.5,
            "memory": 240,
            "accuracy": 0.00,
            "status": "baseline"
        },
        {
            "name": "Kernel_v4_Autotuned",
            "latency": 132.4,
            "tflops": 42.1,
            "memory": 120,
            "accuracy": 1e-5,
            "status": "winner"
        },
        {
            "name": "Kernel_v2_Tiled",
            "latency": 180.1,
            "tflops": 30.5,
            "memory": 125,
            "accuracy": 1e-6,
            "status": "candidate"
        }
    ];

    kernel_rows = [];
    for k in kernels {
        row_class = "hover:bg-zinc-800/50";
        name_class = "text-zinc-300";
        if k["status"] == "baseline" {
            row_class = "bg-zinc-900/30";
            name_class = "text-zinc-400";
        } elif k["status"] == "winner" {
            row_class = "bg-green-900/10 border-l-2 border-green-500";
            name_class = "text-white font-bold";
        }

        badge = <span></span>;
        if k["status"] == "winner" {
            badge = <span
                className="text-[10px] bg-green-900 text-green-300 px-1 rounded border border-green-700 ml-2"
            >
                BEST
            </span>;
        }

        kernel_rows.push(
            <tr key={k["name"]} className={row_class}>
                <td className={"px-6 py-4 font-mono flex items-center " + name_class}>
                    {k["name"]}{badge}
                </td>
                <td
                    className={"px-6 py-4 " + (
                        "text-green-400 font-bold" if k["status"] == "winner" else ""
                    )}
                >
                    {k["latency"]}
                </td>
                <td className="px-6 py-4">
                    {k["tflops"]}
                </td>
                <td className="px-6 py-4">
                    {k["memory"]}
                </td>
                <td
                    className={"px-6 py-4 " + (
                        "text-green-500" if k["accuracy"] < 1e-4 else "text-yellow-500"
                    )}
                >
                    {k["accuracy"]}
                </td>
                <td className="px-6 py-4">
                    <button
                        className="text-xs bg-zinc-800 hover:bg-zinc-700 px-2 py-1 rounded text-zinc-300 transition-colors"
                    >
                        View Code
                    </button>
                </td>
            </tr>
        );
    }

    return
        <div
            className="flex flex-col h-full bg-zinc-900 text-zinc-100 p-6 overflow-hidden font-sans"
        >
            <header className="mb-6 flex justify-between items-end">
                <div>
                    <h1 className="text-2xl font-bold mb-1">
                        Performance Arena
                    </h1>
                    <p className="text-zinc-400 text-sm">
                        Benchmarking
                        {" "}
                        <span className="font-mono text-indigo-400">
                            FusedConvBNReLU
                        </span>
                        {" "}
                        against baseline.
                    </p>
                </div>
                <div className="flex gap-4">
                    <div className="text-right">
                        <div
                            className="text-xs text-zinc-500 uppercase tracking-wider"
                        >
                            Best Speedup
                        </div>
                        <div className="text-xl font-mono text-green-400 font-bold">
                            3.4x
                        </div>
                    </div>
                    <div className="text-right">
                        <div
                            className="text-xs text-zinc-500 uppercase tracking-wider"
                        >
                            VRAM Savings
                        </div>
                        <div className="text-xl font-mono text-blue-400 font-bold">
                            -120 MB
                        </div>
                    </div>
                </div>
            </header>
            <div
                className="bg-zinc-950 border border-zinc-800 rounded-lg overflow-hidden mb-6 shadow-sm"
            >
                <table className="w-full text-left text-sm">
                    <thead
                        className="bg-zinc-900 text-zinc-500 border-b border-zinc-800 uppercase text-xs tracking-wider"
                    >
                        <tr>
                            <th className="px-6 py-3 font-medium">
                                Candidate
                            </th>
                            <th className="px-6 py-3 font-medium">
                                Latency (µs)
                            </th>
                            <th className="px-6 py-3 font-medium">
                                TFLOPS
                            </th>
                            <th className="px-6 py-3 font-medium">
                                Memory (MB)
                            </th>
                            <th className="px-6 py-3 font-medium">
                                Accuracy (L2)
                            </th>
                            <th className="px-6 py-3 font-medium">
                                Action
                            </th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-zinc-800">
                        {kernel_rows}
                    </tbody>
                </table>
            </div>
            <div className="flex-1 grid grid-cols-2 gap-6 min-h-0">
                <div
                    className="bg-zinc-950 border border-zinc-800 rounded-lg p-4 flex flex-col shadow-sm"
                >
                    <h3 className="text-sm font-semibold text-zinc-400 mb-4">
                        Roofline Analysis
                    </h3>
                    <div
                        className="flex-1 border border-dashed border-zinc-800 rounded flex items-center justify-center relative bg-zinc-900/20"
                    >
                        <span className="text-zinc-600 text-xs">
                            [Roofline Chart Component Placeholder]
                        </span>
                        <div
                            className="absolute top-[40%] left-[60%] w-3 h-3 bg-green-500 rounded-full shadow-[0_0_10px_rgba(34,197,94,0.5)]"
                        ></div>
                    </div>
                </div>
                <div
                    className="bg-zinc-950 border border-zinc-800 rounded-lg p-4 flex flex-col shadow-sm"
                >
                    <div className="flex justify-between items-center mb-4">
                        <h3
                            className="text-sm font-semibold text-zinc-400 flex gap-2 items-center"
                        >
                            Autotuning Grid
                        </h3>
                        <button
                            className="bg-indigo-600 hover:bg-indigo-500 text-white text-xs px-3 py-1.5 rounded flex gap-2 items-center transition-colors shadow-sm"
                        >
                            Run Sweep
                        </button>
                    </div>
                    <div className="space-y-4">
                        <div>
                            <label className="text-xs text-zinc-500 block mb-1">
                                BLOCK_SIZE_M
                            </label>
                            <div className="flex gap-2">
                                {(
                                    lambda() -> list { items = [];for v in [
                                        32,
                                        64,
                                        128,
                                        256
                                    ] {
                                        items.push(
                                            <div
                                                key={v}
                                                className="bg-zinc-900 border border-zinc-700 text-xs px-3 py-1 rounded text-zinc-300 hover:border-indigo-500 cursor-pointer transition-colors"
                                            >
                                                {v}
                                            </div>
                                        );
                                    }return items; }
                                )()}
                            </div>
                        </div>
                        <div>
                            <label className="text-xs text-zinc-500 block mb-1">
                                num_warps
                            </label>
                            <div className="flex gap-2">
                                {(
                                    lambda() -> list { items = [];for v in [4, 8] {
                                        items.push(
                                            <div
                                                key={v}
                                                className="bg-zinc-900 border border-zinc-700 text-xs px-3 py-1 rounded text-zinc-300 hover:border-indigo-500 cursor-pointer transition-colors"
                                            >
                                                {v}
                                            </div>
                                        );
                                    }return items; }
                                )()}
                            </div>
                        </div>
                        <div>
                            <label className="text-xs text-zinc-500 block mb-1">
                                num_stages
                            </label>
                            <div className="flex gap-2">
                                {(
                                    lambda() -> list { items = [];for v in [2, 3, 4] {
                                        items.push(
                                            <div
                                                key={v}
                                                className="bg-zinc-900 border border-zinc-700 text-xs px-3 py-1 rounded text-zinc-300 hover:border-indigo-500 cursor-pointer transition-colors"
                                            >
                                                {v}
                                            </div>
                                        );
                                    }return items; }
                                )()}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>;
}
