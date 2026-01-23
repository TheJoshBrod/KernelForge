"use client";

import Link from "next/link";
import React, { use, useState, useEffect } from 'react';
import ReactFlow, { Background, Controls } from 'reactflow';
import 'reactflow/dist/style.css';
import { useRouter } from 'next/navigation';

const initialNodes = [
    { id: '1', position: { x: 50, y: 50 }, data: { label: 'linear1' }, style: { background: '#f97316', color: '#fff' } },
    { id: '2', position: { x: 250, y: 150 }, data: { label: 'linear2 (20ms)' }, style: { background: '#22c55e', color: '#fff' } },
    { id: '3', position: { x: 450, y: 100 }, data: { label: 'linear3 (18ms)' }, style: { background: '#22c55e', color: '#fff' } },
    { id: '5', position: { x: 650, y: 200 }, data: { label: 'linear5 (10ms)' }, style: { background: '#3b82f6', color: '#fff' } },
];

const initialEdges = [
    { id: 'e1-2', source: '1', target: '2', animated: true },
    { id: 'e2-3', source: '2', target: '3', animated: true },
    { id: 'e3-5', source: '3', target: '5', animated: true },
];

// Mock data generator for different layers
const getLayerData = (layerName) => {
    // Deterministic pseudo-random based on string length/char codes to keep it consistent but different
    const seed = layerName.charCodeAt(layerName.length - 1);
    const isEven = seed % 2 === 0;

    const baseNodes = [
        { id: '1', position: { x: 50, y: 50 }, data: { label: `${layerName}_v1` }, style: { background: '#f97316', color: '#fff' } },
    ];

    // Add some random variations
    if (isEven) {
        baseNodes.push(
            { id: '2', position: { x: 250, y: isEven ? 100 : 200 }, data: { label: 'v2 (22ms)' }, style: { background: '#22c55e', color: '#fff' } },
            { id: '3', position: { x: 450, y: 150 }, data: { label: 'v3 (15ms)' }, style: { background: '#3b82f6', color: '#fff' } }
        );
    } else {
        baseNodes.push(
            { id: '2', position: { x: 250, y: 150 }, data: { label: 'v2 (19ms)' }, style: { background: '#22c55e', color: '#fff' } },
            { id: '3', position: { x: 450, y: 50 }, data: { label: 'v3 (17ms)' }, style: { background: '#22c55e', color: '#fff' } },
            { id: '4', position: { x: 650, y: 120 }, data: { label: 'v4 (12ms)' }, style: { background: '#3b82f6', color: '#fff' } }
        );
    }

    const edges = baseNodes.slice(0, -1).map((node, i) => ({
        id: `e${node.id}-${baseNodes[i + 1].id}`,
        source: node.id,
        target: baseNodes[i + 1].id,
        animated: true
    }));

    return { nodes: baseNodes, edges };
};

export default function OperatorPage({ params }) {
    const router = useRouter();
    const { name, op } = use(params);

    // Sidebar layers
    const layers = [`${op}1`, `${op}2`, `${op}3`, `${op}4`, `${op}5`];
    const [selectedLayer, setSelectedLayer] = useState(layers[0]);

    const [nodes, setNodes] = useState(initialNodes);
    const [edges, setEdges] = useState(initialEdges);

    // Update graph when layer changes
    useEffect(() => {
        if (selectedLayer) {
            const data = getLayerData(selectedLayer);
            setNodes(data.nodes);
            setEdges(data.edges);
        }
    }, [selectedLayer]);

    // Maintain Escape key nav
    useEffect(() => {
        const handleKeyDown = (event) => {
            if (event.key === "Escape") router.push(`/project/${name}`);
        };
        window.addEventListener("keydown", handleKeyDown);
        return () => window.removeEventListener("keydown", handleKeyDown);
    }, [name, router]);

    return (
        <div className="flex min-h-screen bg-zinc-950 text-zinc-100 font-sans">
            <div className="absolute top-2 right-6">
                <Link
                    href="/settings"
                    className="flex items-center justify-center p-2 rounded-full hover:bg-zinc-800 transition-colors text-zinc-400 hover:text-white"
                    aria-label="Settings"
                >
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    >
                        <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.1a2 2 0 0 1-1-1.74v-.51a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
                        <circle cx="12" cy="12" r="3" />
                    </svg>
                </Link>
            </div>
            <div className="absolute top-2 right-16 z-10">
                <Link
                    href={`/project/${name}`}
                    className="flex items-center justify-center p-2 rounded-full hover:bg-zinc-800 transition-colors text-zinc-400 hover:text-white bg-zinc-950/50 backdrop-blur-sm"
                    aria-label="Go Back"
                >
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    >
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="15" y1="9" x2="9" y2="15"></line>
                        <line x1="9" y1="9" x2="15" y2="15"></line>
                    </svg>
                </Link>
            </div>

            {/* Sidebar (25% width) */}
            <aside className="w-1/4 border-r border-zinc-900 flex flex-col p-4 space-y-6">
                <div>
                    <h2 className="text-xl font-bold mb-4 text-zinc-100">Layers:</h2>
                    <div className="flex gap-2">
                        <input
                            type="text"
                            placeholder="Search..."
                            className="flex-1 bg-zinc-900 border border-zinc-800 rounded px-3 py-2 text-sm focus:outline-none focus:border-zinc-700 transition-colors"
                        />
                        <button className="bg-zinc-800 hover:bg-zinc-700 px-4 py-2 rounded text-sm font-medium transition-colors">
                            Filter
                        </button>
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto space-y-2 pr-2 custom-scrollbar">
                    {layers.map((layer) => (
                        <button
                            key={layer}
                            onClick={() => setSelectedLayer(layer)}
                            className={`w-full text-left border rounded px-4 py-3 text-sm font-mono transition-all ${selectedLayer === layer
                                    ? 'bg-zinc-800 border-zinc-600 text-white'
                                    : 'bg-zinc-900/50 hover:bg-zinc-800 border-zinc-800/50 text-zinc-300'
                                }`}
                        >
                            {layer}
                        </button>
                    ))}
                </div>

                <div className="space-y-4 pt-4 border-t border-zinc-900">
                    <button className="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 rounded shadow-lg shadow-blue-900/20 transition-all">
                        New Layer
                    </button>
                </div>
            </aside>

            {/* Main Content Area (75% width) */}
            <main className="w-3/4 p-8 overflow-y-auto">
                <h1 className="text-3xl font-bold text-center mb-12 tracking-tight">
                    Optimizing Layer: <span className="text-emerald-400">{selectedLayer}</span>
                </h1>

                <div className="grid grid-cols-1 gap-8 mb-12">
                    {/* Optimization Info Graph */}
                    <div className="space-y-6">
                        <h3 className="text-xl font-semibold border-b border-zinc-800 pb-2">Kernel Refinement Graph:</h3>

                        <div className="h-[500px] bg-zinc-900/30 rounded-xl border border-zinc-800 p-4 relative">
                            <h4 className="absolute top-4 left-4 text-sm font-medium text-zinc-400 z-10">
                                Y-Axis: Execution Time (Lower is better) | X-Axis: Generation
                            </h4>
                            <ReactFlow
                                nodes={nodes}
                                edges={edges}
                                fitView
                                className="bg-zinc-900/30"
                                nodesConnectable={false}
                                nodesDraggable={true}
                                panOnDrag={true}
                                zoomOnScroll={true}
                                zoomOnPinch={true}
                                zoomOnDoubleClick={true}
                                proOptions={{ hideAttribution: true }}
                            >
                                <Background color="#27272a" gap={16} />
                                <Controls className="bg-zinc-800 border-zinc-700 fill-zinc-400" />
                            </ReactFlow>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}
