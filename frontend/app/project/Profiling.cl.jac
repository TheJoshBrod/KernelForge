"""Profiling Page"""

import from ...components.ui.Icons { ActivityIcon, UploadIcon }
import from ...components.ui.GlowButton { GlowButton }

def:pub Profiling {
    # Mock insights for now
    insights = [
        {
            "type": "fusion",
            "title": "Fusion Opportunity",
            "message": "Detected sequential Conv2d -> BatchNorm -> ReLU. Fusing reduces global memory reads by 35%.",
            "action": "Extract & Optimize"
        },
        {
            "type": "occupancy",
            "title": "Low Occupancy",
            "message": "Layer `AttentionBlock.0` has low SM occupancy due to small batch size.",
            "action": None
        }
    ];

    insight_elements = [];
    for insight in insights {
        action_button = <span></span>;
        if insight["action"] {
            action_button = <button
                className="w-full mt-3 py-2 bg-indigo-600 hover:bg-indigo-500 rounded text-xs font-semibold text-white transition-colors"
            >
                {insight["action"]}
            </button>;
        }

        insight_elements.push(
            <div
                key={insight["title"]}
                className="p-4 bg-zinc-900/50 border border-zinc-800 rounded-lg mb-4"
            >
                <div className="flex gap-3 mb-2 items-center">
                    <ActivityIcon className="text-indigo-400 w-4 h-4"/>
                    <span className="font-medium text-indigo-300 text-sm">
                        {insight["title"]}
                    </span>
                </div>
                <p className="text-sm text-zinc-400">
                    {insight["message"]}
                </p>
                {action_button}
            </div>
        );
    }

    return
        <div
            className="flex flex-col h-full bg-zinc-900 text-zinc-100 overflow-hidden"
        >
            <header
                className="h-16 border-b border-zinc-800 flex items-center justify-between px-8 bg-zinc-950 flex-shrink-0"
            >
                <h2 className="text-lg font-semibold">
                    Profiling & Discovery
                </h2>
                <div className="flex gap-3">
                    <button
                        className="flex items-center gap-2 bg-zinc-800 hover:bg-zinc-700 px-4 py-2 rounded text-sm transition-colors text-zinc-300"
                    >
                        <UploadIcon className="w-4 h-4"/>
                        Import Trace
                    </button>
                    <button
                        className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 px-4 py-2 rounded text-sm font-medium transition-colors text-white"
                    >
                        <ActivityIcon className="w-4 h-4"/>
                        Capture Run
                    </button>
                </div>
            </header>
            <div className="flex-1 grid grid-cols-12 overflow-hidden min-h-0">
                <div
                    className="col-span-8 p-6 overflow-y-auto border-r border-zinc-800 scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-zinc-900"
                >
                    <div className="mb-4 flex justify-between items-center">
                        <h3
                            className="text-zinc-400 text-sm uppercase tracking-wider font-semibold"
                        >
                            GPU Execution Timeline
                        </h3>
                        <span className="text-xs text-zinc-500 font-mono">
                            NVIDIA A100-80GB
                        </span>
                    </div>
                    <div
                        className="w-full h-64 bg-zinc-950 border border-zinc-800 rounded-lg relative overflow-hidden mb-8 shadow-inner"
                    >
                        <div
                            className="absolute top-4 left-4 w-3/4 h-8 bg-blue-900/40 border border-blue-700 rounded flex items-center px-2 text-xs text-blue-200"
                        >
                            Forward Pass (ResNet Block 3)
                        </div>
                        <div
                            className="absolute top-14 left-8 w-1/4 h-8 bg-red-900/40 border border-red-600 rounded flex items-center px-2 text-xs text-red-200 animate-pulse"
                        >
                            Conv2d (Bottleneck)
                        </div>
                        <div
                            className="absolute top-14 left-[35%] w-1/6 h-8 bg-green-900/40 border border-green-700 rounded flex items-center px-2 text-xs text-green-200"
                        >
                            BatchNorm
                        </div>
                    </div>
                    <h3
                        className="text-zinc-400 text-sm uppercase tracking-wider mb-4 font-semibold"
                    >
                        Operator Breakdown
                    </h3>
                    <table className="w-full text-left text-sm border-collapse">
                        <thead
                            className="text-zinc-500 border-b border-zinc-800 bg-zinc-900/50 sticky top-0"
                        >
                            <tr>
                                <th className="pb-2 font-medium pl-2">
                                    Name
                                </th>
                                <th className="pb-2 font-medium">
                                    Calls
                                </th>
                                <th className="pb-2 font-medium">
                                    Self CUDA Time
                                </th>
                                <th className="pb-2 font-medium">
                                    Status
                                </th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-zinc-800">
                            <tr
                                className="group hover:bg-zinc-800/50 cursor-pointer transition-colors"
                            >
                                <td className="py-3 pl-2 font-mono text-indigo-400">
                                    aten::conv2d
                                </td>
                                <td className="text-zinc-300">
                                    420
                                </td>
                                <td className="text-zinc-300">
                                    14.2s (45%)
                                </td>
                                <td>
                                    <span
                                        className="px-2 py-0.5 rounded bg-red-900/30 text-red-400 text-xs border border-red-900/50"
                                    >
                                        Compute Bound
                                    </span>
                                </td>
                            </tr>
                            <tr
                                className="group hover:bg-zinc-800/50 cursor-pointer transition-colors"
                            >
                                <td className="py-3 pl-2 font-mono text-zinc-300">
                                    aten::relu
                                </td>
                                <td className="text-zinc-300">
                                    420
                                </td>
                                <td className="text-zinc-300">
                                    4.1s (12%)
                                </td>
                                <td>
                                    <span
                                        className="px-2 py-0.5 rounded bg-yellow-900/30 text-yellow-400 text-xs border border-yellow-900/50"
                                    >
                                        Memory Bound
                                    </span>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                <div
                    className="col-span-4 bg-zinc-950 p-6 border-l border-zinc-800 flex flex-col h-full overflow-hidden"
                >
                    <div className="flex items-center gap-2 mb-6 flex-shrink-0">
                        <div
                            className="w-2 h-2 rounded-full bg-green-500 animate-pulse"
                        ></div>
                        <h3 className="font-semibold text-zinc-200">
                            Agent Insights
                        </h3>
                    </div>
                    <div
                        className="space-y-4 overflow-y-auto flex-1 pr-2 scrollbar-thin scrollbar-thumb-zinc-800"
                    >
                        {insight_elements}
                    </div>
                </div>
            </div>
        </div>;
}
