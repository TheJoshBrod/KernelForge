"use client";


import Link from "next/link";
import React, { use, useState, useEffect } from 'react';
import {
    PieChart, Pie,
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { useRouter } from 'next/navigation';

export default function ProjectPage({ params }) {
    const router = useRouter();
    const { name } = use(params);

    // Hardcoded Data
    const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];
    const pieData = [
        { name: 'linear', value: 1000, fill: COLORS[1] },
        { name: 'relu', value: 998, fill: COLORS[0] },
        { name: 'batch_norm', value: 500, fill: COLORS[3] },
        { name: 'embedding', value: 200, fill: COLORS[2] },
    ];

    const operatorFreqData = [
        { operator: 'linear', freq: 1000 },
        { operator: 'relu', freq: 998 },
        { operator: 'batch_norm', freq: 500 },
        { operator: 'embedding', freq: 200 },
    ];

    const barData = [
        { name: 'Baseline', time: 25, fill: '#22c55e' },  // Green
        { name: 'PyTorch', time: 20, fill: '#f97316' },   // Orange
        { name: 'Optimized', time: 10, fill: '#3b82f6' }, // Blue
    ];

    const timingData = [
        { ver: 'Default PyTorch', time: '20 ms' },
        { ver: 'Baseline', time: '25 ms' },
        { ver: 'Optimized', time: '10 ms' },
    ];

    // Logic to keep last_accessed updating
    useEffect(() => {
        fetch(`/api/projects/${name}`, { method: 'PATCH' }).catch(console.error);

        // Maintain Escape key nav
        const handleKeyDown = (event) => {
            if (event.key === "Escape") router.push("/");
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
                    href="/"
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
                    <h2 className="text-xl font-bold mb-4 text-zinc-100">Operators:</h2>
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
                    {['linear', 'conv2d', 'relu', 'batch_norm', 'dropout', 'embedding'].map((op) => (
                        <button
                            key={op}
                            className="w-full text-left bg-zinc-900/50 hover:bg-zinc-800 border border-zinc-800/50 rounded px-4 py-3 text-sm font-mono text-zinc-300 transition-all"
                        >
                            {op}
                        </button>
                    ))}
                </div>

                <div className="space-y-4 pt-4 border-t border-zinc-900">
                    <button className="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 rounded shadow-lg shadow-blue-900/20 transition-all">
                        Generate All Missing
                    </button>
                    <button className="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-3 rounded shadow-lg shadow-emerald-900/20 transition-all">
                        Optimize All
                    </button>

                    <div className="space-y-1">
                        <label className="text-xs font-semibold text-zinc-500 uppercase"># of Attempts (default = 10)</label>
                        <input
                            type="number"
                            defaultValue="10"
                            className="w-full bg-zinc-900 border border-zinc-800 rounded px-3 py-2 focus:outline-none focus:border-zinc-700 transition-colors"
                        />
                    </div>
                </div>
            </aside>

            {/* Main Content Area (75% width) */}
            <main className="w-3/4 p-8 overflow-y-auto">
                <h1 className="text-3xl font-bold text-center mb-12 tracking-tight">Project: {name}</h1>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
                    {/* Section 1: Model Info */}
                    <div className="space-y-6">
                        <h3 className="text-xl font-semibold border-b border-zinc-800 pb-2">Model Info:</h3>

                        {/* Pie Chart */}
                        <div className="h-64 bg-zinc-900/30 rounded-xl border border-zinc-800 p-4 flex flex-col items-center">
                            <h4 className="text-sm font-medium text-zinc-400 mb-2">PyTorch Operator Usage %</h4>
                            <ResponsiveContainer width="100%" height="100%">
                                <PieChart>
                                    <Pie
                                        data={pieData}
                                        innerRadius={60}
                                        outerRadius={80}
                                        paddingAngle={5}
                                        dataKey="value"
                                    />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', borderRadius: '8px' }}
                                        itemStyle={{ color: '#e4e4e7' }}
                                    />
                                    <Legend
                                        layout="vertical"
                                        verticalAlign="middle"
                                        align="right"
                                    />
                                </PieChart>
                            </ResponsiveContainer>
                        </div>

                        {/* Table */}
                        <div className="overflow-hidden rounded-xl border border-zinc-800">
                            <table className="w-full text-sm text-left">
                                <thead className="bg-zinc-900 text-zinc-400 font-medium uppercase text-xs">
                                    <tr>
                                        <th className="px-6 py-3">Operator</th>
                                        <th className="px-6 py-3">Freq</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-zinc-800 bg-zinc-900/50">
                                    {operatorFreqData.map((row, i) => (
                                        <tr key={i} className="hover:bg-zinc-800/50 transition-colors">
                                            <td className="px-6 py-3 font-mono text-zinc-300">{row.operator}</td>
                                            <td className="px-6 py-3 text-zinc-300">{row.freq}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {/* Section 2: Optimization Info */}
                    <div className="space-y-6">
                        <h3 className="text-xl font-semibold border-b border-zinc-800 pb-2">Optimization Info:</h3>

                        {/* Bar Chart */}
                        <div className="h-64 bg-zinc-900/30 rounded-xl border border-zinc-800 p-4 flex flex-col items-center">
                            <h4 className="text-sm font-medium text-zinc-400 mb-2">Time (ms) vs. Version</h4>
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={barData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                                    <XAxis dataKey="name" stroke="#71717a" tick={{ fill: '#71717a', fontSize: 12 }} axisLine={false} tickLine={false} />
                                    <YAxis stroke="#71717a" tick={{ fill: '#71717a', fontSize: 12 }} axisLine={false} tickLine={false} />
                                    <Tooltip
                                        cursor={{ fill: '#27272a' }}
                                        contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', borderRadius: '8px' }}
                                        itemStyle={{ color: '#e4e4e7' }}
                                    />
                                    <Bar
                                        dataKey="time"
                                        radius={[4, 4, 0, 0]}
                                        barSize={40}
                                        shape={(props) => {
                                            const { x, y, width, height, index } = props;
                                            return (
                                                <rect
                                                    x={x}
                                                    y={y}
                                                    width={width}
                                                    height={height}
                                                    fill={barData[index].fill}
                                                    rx={4}
                                                    ry={4}
                                                />
                                            );
                                        }}
                                    />
                                    <Tooltip
                                        cursor={{ fill: '#27272a' }}
                                        contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', borderRadius: '8px' }}
                                        itemStyle={{ color: '#e4e4e7' }}
                                        formatter={(value) => `${value} ms`}
                                    />
                                    <YAxis 
                                        stroke="#71717a" 
                                        tick={{ fill: '#71717a', fontSize: 12 }} 
                                        axisLine={false} 
                                        tickLine={false}
                                        domain={[0, Math.max(...barData.map(d => d.time))]}
                                    />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>

                        {/* Table */}
                        <div className="overflow-hidden rounded-xl border border-zinc-800">
                            <table className="w-full text-sm text-left">
                                <thead className="bg-zinc-900 text-zinc-400 font-medium uppercase text-xs">
                                    <tr>
                                        <th className="px-6 py-3">Version</th>
                                        <th className="px-6 py-3">Time</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-zinc-800 bg-zinc-900/50">
                                    {timingData.map((row, i) => (
                                        <tr key={i} className="hover:bg-zinc-800/50 transition-colors">
                                            <td className="px-6 py-3 text-zinc-300">{row.ver}</td>
                                            <td className="px-6 py-3 font-mono text-emerald-400">{row.time}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}
