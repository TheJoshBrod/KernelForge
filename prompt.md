Prerequisites
You would need to install Lucide icons:
npm install lucide-react

1. The Global Layout (app/layout.tsx & Sidebar)
This wraps all pages to create the "IDE" feel.

TypeScript
// components/Sidebar.tsx
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { 
  Activity, Code, Cpu, ShieldCheck, 
  Upload, Settings, Home 
} from 'lucide-react';

const navItems = [
  { name: 'Dashboard', href: '/', icon: Home },
  { name: 'Profiling', href: '/profiling', icon: Activity },
  { name: 'Workbench', href: '/workbench', icon: Code },
  { name: 'Arena', href: '/arena', icon: Cpu },
  { name: 'Verification', href: '/verification', icon: ShieldCheck },
  { name: 'Export', href: '/export', icon: Upload },
];

export default function Sidebar() {
  // In a real app, use usePathname() to highlight active link
  return (
    <div className="w-64 h-screen bg-zinc-950 border-r border-zinc-800 flex flex-col">
      <div className="p-6">
        <h1 className="text-xl font-bold text-white tracking-tight">
          Kernel<span className="text-indigo-500">Forge</span>
        </h1>
      </div>
      <nav className="flex-1 px-4 space-y-2">
        {navItems.map((item) => (
          <Link 
            key={item.name} 
            href={item.href}
            className="flex items-center gap-3 px-4 py-3 text-zinc-400 hover:text-white hover:bg-zinc-900 rounded-lg transition-all"
          >
            <item.icon size={18} />
            <span className="text-sm font-medium">{item.name}</span>
          </Link>
        ))}
      </nav>
      <div className="p-4 border-t border-zinc-800">
        <button className="flex items-center gap-3 px-4 py-3 text-zinc-400 hover:text-white w-full">
          <Settings size={18} />
          <span className="text-sm">Settings</span>
        </button>
      </div>
    </div>
  );
}
2. The Profiling Page (app/profiling/page.tsx)
Focus: Visualizing the bottleneck and Agent suggestions.

TypeScript
import { Play, FileUp, Zap, AlertTriangle } from 'lucide-react';

export default function ProfilingPage() {
  return (
    <div className="flex flex-col h-screen bg-zinc-900 text-zinc-100 overflow-hidden">
      {/* Top Bar */}
      <header className="h-16 border-b border-zinc-800 flex items-center justify-between px-8 bg-zinc-950">
        <h2 className="text-lg font-semibold">Profiling & Discovery</h2>
        <div className="flex gap-3">
          <button className="flex items-center gap-2 bg-zinc-800 hover:bg-zinc-700 px-4 py-2 rounded text-sm">
            <FileUp size={16} /> Import Trace
          </button>
          <button className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 px-4 py-2 rounded text-sm font-medium">
            <Play size={16} /> Capture Run
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 grid grid-cols-12 overflow-hidden">
        {/* Flame Graph Area */}
        <div className="col-span-8 p-6 overflow-y-auto border-r border-zinc-800">
          <div className="mb-4 flex justify-between items-center">
            <h3 className="text-zinc-400 text-sm uppercase tracking-wider">GPU Execution Timeline</h3>
            <span className="text-xs text-zinc-500">NVIDIA A100-80GB</span>
          </div>
          
          {/* Mock Flame Graph Visual */}
          <div className="w-full h-64 bg-zinc-950 border border-zinc-800 rounded-lg relative overflow-hidden mb-8">
             <div className="absolute top-4 left-4 w-3/4 h-8 bg-blue-900/40 border border-blue-700 rounded flex items-center px-2 text-xs text-blue-200">
               Forward Pass (ResNet Block 3)
             </div>
             <div className="absolute top-14 left-8 w-1/4 h-8 bg-red-900/40 border border-red-600 rounded flex items-center px-2 text-xs text-red-200 animate-pulse">
               Conv2d (Bottleneck)
             </div>
             <div className="absolute top-14 left-[35%] w-1/6 h-8 bg-green-900/40 border border-green-700 rounded flex items-center px-2 text-xs text-green-200">
               BatchNorm
             </div>
          </div>

          <h3 className="text-zinc-400 text-sm uppercase tracking-wider mb-4">Operator Breakdown</h3>
          <table className="w-full text-left text-sm">
            <thead className="text-zinc-500 border-b border-zinc-800">
              <tr>
                <th className="pb-2 font-medium">Name</th>
                <th className="pb-2 font-medium">Calls</th>
                <th className="pb-2 font-medium">Self CUDA Time</th>
                <th className="pb-2 font-medium">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800">
              <tr className="group hover:bg-zinc-800/50 cursor-pointer">
                <td className="py-3 font-mono text-indigo-400">aten::conv2d</td>
                <td>420</td>
                <td>14.2s (45%)</td>
                <td><span className="px-2 py-0.5 rounded bg-red-900/30 text-red-400 text-xs">Compute Bound</span></td>
              </tr>
              <tr className="group hover:bg-zinc-800/50 cursor-pointer">
                <td className="py-3 font-mono text-zinc-300">aten::relu</td>
                <td>420</td>
                <td>4.1s (12%)</td>
                <td><span className="px-2 py-0.5 rounded bg-yellow-900/30 text-yellow-400 text-xs">Memory Bound</span></td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Agent Insights Panel */}
        <div className="col-span-4 bg-zinc-950 p-6 border-l border-zinc-800">
          <div className="flex items-center gap-2 mb-6">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
            <h3 className="font-semibold text-zinc-200">Agent Insights</h3>
          </div>

          <div className="space-y-4">
            <div className="p-4 bg-zinc-900/50 border border-indigo-500/30 rounded-lg">
              <div className="flex gap-3 mb-2">
                <Zap className="text-indigo-400" size={18} />
                <span className="font-medium text-indigo-300">Fusion Opportunity</span>
              </div>
              <p className="text-sm text-zinc-400 mb-3">
                I detected a sequential pattern (Conv2d &rarr; BatchNorm &rarr; ReLU). Fusing these could reduce global memory reads by 35%.
              </p>
              <button className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 rounded text-xs font-semibold text-white transition-colors">
                Extract & Optimize Layer
              </button>
            </div>

            <div className="p-4 bg-zinc-900/50 border border-zinc-800 rounded-lg opacity-75">
               <div className="flex gap-3 mb-2">
                <AlertTriangle className="text-yellow-400" size={18} />
                <span className="font-medium text-yellow-200">Low Occupancy</span>
              </div>
              <p className="text-sm text-zinc-400">
                Layer `AttentionBlock.0` has low SM occupancy due to small batch size.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
3. The Workbench Page (app/workbench/page.tsx)
Focus: 3-Pane "IDE" layout for code generation.

TypeScript
import { Terminal, Bot, Sparkles, Save, Play } from 'lucide-react';

export default function WorkbenchPage() {
  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-200 overflow-hidden">
      {/* LEFT: Context Pane */}
      <div className="w-80 border-r border-zinc-800 flex flex-col">
        <div className="p-4 border-b border-zinc-800 bg-zinc-900">
          <h3 className="text-xs font-bold text-zinc-500 uppercase">Input Context</h3>
        </div>
        <div className="p-4 space-y-6 overflow-y-auto flex-1">
          <div>
            <label className="text-xs text-zinc-500 block mb-1">Target Operation</label>
            <div className="font-mono text-sm text-indigo-400 bg-zinc-900 p-2 rounded border border-zinc-800">
              FusedConvBNReLU
            </div>
          </div>
          <div>
            <label className="text-xs text-zinc-500 block mb-1">Tensor Shapes</label>
            <ul className="font-mono text-xs space-y-2">
              <li className="flex justify-between"><span>Input:</span> <span className="text-zinc-400">[32, 64, 224, 224]</span></li>
              <li className="flex justify-between"><span>Weights:</span> <span className="text-zinc-400">[64, 64, 3, 3]</span></li>
              <li className="flex justify-between"><span>Dtype:</span> <span className="text-orange-400">float16</span></li>
            </ul>
          </div>
          <div>
            <label className="text-xs text-zinc-500 block mb-1">Reference (Python)</label>
            <pre className="text-[10px] bg-zinc-900 p-2 rounded text-zinc-400 overflow-x-auto">
{`def forward(x, w):
  x = F.conv2d(x, w)
  x = F.relu(x)
  return x`}
            </pre>
          </div>
        </div>
      </div>

      {/* CENTER: Editor */}
      <div className="flex-1 flex flex-col bg-[#0d0d0d]">
        <div className="h-10 border-b border-zinc-800 flex items-center justify-between px-4">
          <span className="text-xs text-zinc-400 font-mono">generated_kernel.py (Triton)</span>
          <div className="flex gap-2">
             <button className="p-1 hover:text-white text-zinc-500"><Save size={14}/></button>
             <button className="p-1 hover:text-green-400 text-zinc-500"><Play size={14}/></button>
          </div>
        </div>
        <div className="flex-1 p-4 font-mono text-sm overflow-auto text-zinc-300">
          <div className="text-zinc-500"># Auto-generated by KernelForge Agent</div>
          <div className="mt-2"><span className="text-purple-400">import</span> triton</div>
          <div><span className="text-purple-400">import</span> triton.language <span className="text-purple-400">as</span> tl</div>
          <div className="mt-4"><span className="text-blue-400">@triton.jit</span></div>
          <div><span className="text-purple-400">def</span> <span className="text-yellow-200">fused_conv_kernel</span>(x_ptr, w_ptr...):</div>
          <div className="pl-4 text-zinc-500"># BLOCK_SIZE_M calculation...</div>
          <div className="pl-4">pid = tl.program_id(axis=0)</div>
          <div className="pl-4">...</div>
        </div>
        {/* Terminal / Output */}
        <div className="h-32 border-t border-zinc-800 bg-zinc-900 p-2 font-mono text-xs overflow-y-auto">
            <div className="flex items-center gap-2 text-zinc-400 mb-1"><Terminal size={12}/> Output</div>
            <div className="text-green-400">Successfully compiled in 0.42s.</div>
            <div className="text-zinc-400">Running verification... Passed.</div>
        </div>
      </div>

      {/* RIGHT: Agent Chat */}
      <div className="w-96 border-l border-zinc-800 bg-zinc-950 flex flex-col">
         <div className="p-3 border-b border-zinc-800 bg-zinc-900 flex justify-between items-center">
            <span className="text-sm font-semibold flex gap-2 items-center"><Bot size={16}/> Kernel Architect</span>
            <select className="bg-zinc-950 border border-zinc-700 text-xs rounded px-2 py-1">
                <option>Strategy: Latency</option>
                <option>Strategy: Memory</option>
            </select>
         </div>
         <div className="flex-1 p-4 space-y-4 overflow-y-auto">
            {/* Message 1 */}
            <div className="flex gap-3">
                <div className="w-6 h-6 rounded-full bg-indigo-600 flex items-center justify-center text-[10px]">AI</div>
                <div className="bg-zinc-800/50 p-3 rounded-lg rounded-tl-none text-sm text-zinc-300">
                    I've drafted a Triton kernel using block tiling of 128x128. This should optimize L2 cache hits.
                </div>
            </div>
             {/* Message 2 */}
            <div className="flex gap-3 flex-row-reverse">
                <div className="w-6 h-6 rounded-full bg-zinc-600 flex items-center justify-center text-[10px]">U</div>
                <div className="bg-indigo-900/20 border border-indigo-800 p-3 rounded-lg rounded-tr-none text-sm text-zinc-300">
                    Can we try using Warp Shuffle to reduce shared memory usage?
                </div>
            </div>
             {/* Thinking */}
            <div className="flex gap-3 animate-pulse opacity-70">
                <div className="w-6 h-6 rounded-full bg-indigo-600 flex items-center justify-center text-[10px]">AI</div>
                <div className="text-xs text-zinc-500 py-2">
                   Analyzing warp reduction primitives...
                </div>
            </div>
         </div>
         <div className="p-3 border-t border-zinc-800">
            <button className="w-full bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs py-2 rounded flex justify-center gap-2 items-center">
                <Sparkles size={14}/> Generate Variants
            </button>
         </div>
      </div>
    </div>
  );
}
4. The Arena Page (app/arena/page.tsx)
Focus: Dense data table and comparison charts.

TypeScript
import { ArrowUp, Sliders, PlayCircle } from 'lucide-react';

export default function ArenaPage() {
  return (
    <div className="flex flex-col h-screen bg-zinc-900 text-zinc-100 p-6 overflow-hidden">
        <header className="mb-6 flex justify-between items-end">
            <div>
                <h1 className="text-2xl font-bold mb-1">Performance Arena</h1>
                <p className="text-zinc-400 text-sm">Benchmarking <span className="font-mono text-indigo-400">FusedConvBNReLU</span> against baseline.</p>
            </div>
            <div className="flex gap-4">
                 <div className="text-right">
                    <div className="text-xs text-zinc-500 uppercase">Best Speedup</div>
                    <div className="text-xl font-mono text-green-400 font-bold">3.4x</div>
                 </div>
                 <div className="text-right">
                    <div className="text-xs text-zinc-500 uppercase">VRAM Savings</div>
                    <div className="text-xl font-mono text-blue-400 font-bold">-120 MB</div>
                 </div>
            </div>
        </header>

        {/* Leaderboard Table */}
        <div className="bg-zinc-950 border border-zinc-800 rounded-lg overflow-hidden mb-6">
            <table className="w-full text-left text-sm">
                <thead className="bg-zinc-900 text-zinc-500 border-b border-zinc-800">
                    <tr>
                        <th className="px-6 py-3 font-medium">Candidate</th>
                        <th className="px-6 py-3 font-medium">Latency (µs)</th>
                        <th className="px-6 py-3 font-medium">TFLOPS</th>
                        <th className="px-6 py-3 font-medium">Memory (MB)</th>
                        <th className="px-6 py-3 font-medium">Accuracy (L2)</th>
                        <th className="px-6 py-3 font-medium">Action</th>
                    </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800">
                    {/* Baseline */}
                    <tr className="bg-zinc-900/30">
                        <td className="px-6 py-4 font-mono text-zinc-400">Baseline (PyTorch)</td>
                        <td className="px-6 py-4">450.2</td>
                        <td className="px-6 py-4">12.5</td>
                        <td className="px-6 py-4">240</td>
                        <td className="px-6 py-4 text-green-500">0.00</td>
                        <td className="px-6 py-4">-</td>
                    </tr>
                     {/* Winner */}
                    <tr className="bg-green-900/10 border-l-2 border-green-500">
                        <td className="px-6 py-4 font-mono text-white flex gap-2 items-center">
                            Kernel_v4_Autotuned <span className="text-[10px] bg-green-900 text-green-300 px-1 rounded border border-green-700">BEST</span>
                        </td>
                        <td className="px-6 py-4 font-bold text-green-400">132.4</td>
                        <td className="px-6 py-4">42.1</td>
                        <td className="px-6 py-4">120</td>
                        <td className="px-6 py-4 text-yellow-500">1e-5</td>
                        <td className="px-6 py-4">
                             <button className="text-xs bg-zinc-800 hover:bg-zinc-700 px-2 py-1 rounded">View Code</button>
                        </td>
                    </tr>
                    {/* Others */}
                    <tr className="hover:bg-zinc-900/50">
                        <td className="px-6 py-4 font-mono text-zinc-300">Kernel_v2_Tiled</td>
                        <td className="px-6 py-4">180.1</td>
                        <td className="px-6 py-4">30.5</td>
                        <td className="px-6 py-4">125</td>
                        <td className="px-6 py-4 text-green-500">1e-6</td>
                        <td className="px-6 py-4">
                            <button className="text-xs bg-zinc-800 hover:bg-zinc-700 px-2 py-1 rounded">View Code</button>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>

        {/* Bottom Split: Roofline & Tuner */}
        <div className="flex-1 grid grid-cols-2 gap-6 min-h-0">
            {/* Visualizer */}
            <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-4 flex flex-col">
                <h3 className="text-sm font-semibold text-zinc-400 mb-4">Roofline Analysis</h3>
                <div className="flex-1 border border-dashed border-zinc-800 rounded flex items-center justify-center relative bg-zinc-900/20">
                    <span className="text-zinc-600 text-xs">[Roofline Chart Component Placeholder]</span>
                    {/* Simulated dot */}
                    <div className="absolute top-[40%] left-[60%] w-3 h-3 bg-green-500 rounded-full shadow-[0_0_10px_rgba(34,197,94,0.5)]"></div>
                </div>
            </div>

            {/* Tuner */}
            <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-4 flex flex-col">
                 <div className="flex justify-between items-center mb-4">
                    <h3 className="text-sm font-semibold text-zinc-400 flex gap-2 items-center"><Sliders size={14}/> Autotuning Grid</h3>
                    <button className="bg-indigo-600 hover:bg-indigo-500 text-white text-xs px-3 py-1.5 rounded flex gap-2 items-center">
                        <PlayCircle size={14}/> Run Sweep
                    </button>
                 </div>
                 <div className="space-y-4">
                    <div>
                        <label className="text-xs text-zinc-500 block mb-1">BLOCK_SIZE_M</label>
                        <div className="flex gap-2">
                             {[32, 64, 128, 256].map(v => (
                                 <div key={v} className="bg-zinc-900 border border-zinc-700 text-xs px-3 py-1 rounded text-zinc-300">{v}</div>
                             ))}
                        </div>
                    </div>
                    <div>
                        <label className="text-xs text-zinc-500 block mb-1">num_warps</label>
                        <div className="flex gap-2">
                             {[4, 8].map(v => (
                                 <div key={v} className="bg-zinc-900 border border-zinc-700 text-xs px-3 py-1 rounded text-zinc-300">{v}</div>
                             ))}
                        </div>
                    </div>
                    <div>
                        <label className="text-xs text-zinc-500 block mb-1">num_stages</label>
                        <div className="flex gap-2">
                             {[2, 3, 4].map(v => (
                                 <div key={v} className="bg-zinc-900 border border-zinc-700 text-xs px-3 py-1 rounded text-zinc-300">{v}</div>
                             ))}
                        </div>
                    </div>
                 </div>
            </div>
        </div>
    </div>
  );
}
5. The Verification Page (app/verification/page.tsx)
Focus: Correctness and stability checks.

TypeScript
import { CheckCircle, XCircle, RefreshCw } from 'lucide-react';

export default function VerificationPage() {
  return (
    <div className="h-screen bg-zinc-900 text-zinc-100 flex p-6 gap-6">
       {/* Test List */}
       <div className="w-1/3 flex flex-col gap-4">
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-bold">Test Suite</h2>
            <button className="text-xs text-zinc-400 hover:text-white flex gap-1 items-center"><RefreshCw size={12}/> Rerun All</button>
          </div>
          
          <div className="space-y-2">
             <div className="bg-zinc-950 border border-green-900/50 p-4 rounded-lg flex justify-between items-center cursor-pointer hover:bg-zinc-900">
                <div className="flex gap-3 items-center">
                    <CheckCircle className="text-green-500" size={20}/>
                    <div>
                        <div className="text-sm font-medium">Standard Input</div>
                        <div className="text-xs text-zinc-500">Shape [32, 64, 224, 224]</div>
                    </div>
                </div>
                <span className="text-xs font-mono text-green-400">Pass</span>
             </div>

             <div className="bg-zinc-950 border border-red-900/50 p-4 rounded-lg flex justify-between items-center cursor-pointer ring-1 ring-red-500">
                <div className="flex gap-3 items-center">
                    <XCircle className="text-red-500" size={20}/>
                    <div>
                        <div className="text-sm font-medium">Odd Dimensions</div>
                        <div className="text-xs text-zinc-500">Shape [7, 63, 111, 113]</div>
                    </div>
                </div>
                <span className="text-xs font-mono text-red-400">Fail (atol &gt; 1e-3)</span>
             </div>

             <div className="bg-zinc-950 border border-green-900/50 p-4 rounded-lg flex justify-between items-center cursor-pointer opacity-60">
                <div className="flex gap-3 items-center">
                    <CheckCircle className="text-green-500" size={20}/>
                    <div>
                        <div className="text-sm font-medium">Zero Weights</div>
                        <div className="text-xs text-zinc-500">Sparsity Check</div>
                    </div>
                </div>
                <span className="text-xs font-mono text-green-400">Pass</span>
             </div>
          </div>
       </div>

       {/* Visualization / Debugger */}
       <div className="flex-1 bg-zinc-950 border border-zinc-800 rounded-xl p-6 flex flex-col">
            <div className="flex justify-between mb-6">
                <h3 className="font-semibold text-red-400 flex items-center gap-2"><XCircle size={18}/> Failure Analysis: Odd Dimensions</h3>
                <div className="flex gap-4 text-xs font-mono">
                    <span className="text-zinc-500">Max Diff: <span className="text-red-400">0.042</span></span>
                    <span className="text-zinc-500">Mean Diff: <span className="text-yellow-400">0.002</span></span>
                </div>
            </div>

            <div className="flex-1 flex flex-col items-center justify-center bg-zinc-900/50 rounded border border-dashed border-zinc-800 relative">
                 <p className="absolute top-2 left-2 text-xs text-zinc-500">Error Heatmap (Channel 0)</p>
                 
                 {/* CSS Grid Heatmap Placeholder */}
                 <div className="grid grid-cols-12 gap-1 p-4">
                    {Array.from({length: 144}).map((_, i) => (
                        <div 
                            key={i} 
                            className={`w-3 h-3 rounded-sm ${i % 7 === 0 ? 'bg-red-500' : 'bg-green-900/20'}`}
                            style={{ opacity: i % 7 === 0 ? 0.8 : 0.2}}
                        ></div>
                    ))}
                 </div>
                 <p className="mt-4 text-xs text-zinc-400">Tip: Check boundary conditions in `tl.load` mask.</p>
            </div>
       </div>
    </div>
  );
}
6. The Export Page (app/export/page.tsx)
Focus: Configuration and final code delivery.

TypeScript
import { Package, Download, Copy, Check } from 'lucide-react';

export default function ExportPage() {
  return (
    <div className="max-w-4xl mx-auto py-12 px-6">
       <div className="mb-8">
         <h1 className="text-3xl font-bold text-white mb-2">Deploy Kernel</h1>
         <p className="text-zinc-400">Package your optimized kernel for production use.</p>
       </div>

       <div className="grid grid-cols-1 gap-8">
          {/* Configuration */}
          <div className="bg-zinc-900 p-6 rounded-lg border border-zinc-800">
             <h2 className="text-lg font-semibold text-zinc-200 mb-4 flex items-center gap-2">
                <Package size={20}/> Build Configuration
             </h2>
             <div className="grid grid-cols-2 gap-6">
                <div>
                    <label className="block text-sm text-zinc-500 mb-2">Project Name</label>
                    <input type="text" value="resnet_opt_layer3" className="w-full bg-zinc-950 border border-zinc-700 rounded p-2 text-zinc-200 text-sm" />
                </div>
                <div>
                    <label className="block text-sm text-zinc-500 mb-2">Integration Method</label>
                    <select className="w-full bg-zinc-950 border border-zinc-700 rounded p-2 text-zinc-200 text-sm">
                        <option>Inline (torch.compile backend)</option>
                        <option>Python Package (pip install)</option>
                        <option>C++ Extension (.so)</option>
                    </select>
                </div>
             </div>
          </div>

          {/* Preview */}
          <div className="bg-zinc-900 p-6 rounded-lg border border-zinc-800">
             <div className="flex justify-between items-center mb-4">
                 <h2 className="text-lg font-semibold text-zinc-200">Generated setup.py</h2>
                 <button className="text-xs flex items-center gap-1 text-indigo-400 hover:text-indigo-300">
                    <Copy size={14}/> Copy
                 </button>
             </div>
             <pre className="bg-zinc-950 p-4 rounded text-xs font-mono text-zinc-400 overflow-x-auto border border-zinc-800">
{`from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='resnet_opt_layer3',
    ext_modules=[
        CUDAExtension('resnet_opt_layer3', [
            'kernel_interface.cpp',
            'kernel_impl.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)`}
             </pre>
          </div>
          
          {/* Actions */}
          <div className="flex justify-end gap-4">
             <button className="px-6 py-3 rounded-lg border border-zinc-700 text-zinc-300 hover:bg-zinc-800 transition-colors">
                Download .zip
             </button>
             <button className="px-6 py-3 rounded-lg bg-indigo-600 text-white font-medium hover:bg-indigo-500 transition-colors flex items-center gap-2 shadow-lg shadow-indigo-900/20">
                <Download size={18}/> Build & Export
             </button>
          </div>
       </div>
    </div>
  );
}