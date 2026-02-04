"""Home Page Component"""

import from react { useEffect }
import from "@jac/runtime" { jacSpawn }
import from "react-router-dom" { useNavigate }
import from .GlowButton { GlowButton }
import from .Mascot { Mascot }
import from .IllustratedCard { IllustratedCard }

def:pub Home -> any {
    navigate = useNavigate();
    has projects: list = [],
        loading: bool = True;

    async def fetchProjects -> None {
        try {
            resp = await jacSpawn("GetProjects", "", {});
            data = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else None;
            if data and data["projects"] {
                projects = data["projects"];
            } else {
                projects = [];
            }
        } except Exception as e {
            console.error("Failed to fetch projects", e);
        } finally {
            loading = False;
        }
    }

    useEffect(lambda -> None { fetchProjects();}, []);

    def handleProjectSelect(e: any) -> None {
        projectId = e.target.value;
        if projectId {
            navigate("/project/" + projectId);
        }
    }

    recentProjects = [];
    recentLinks = [];
    projectOptions = [];
    hasRecent = False;
    idx = 0;
    for project in projects {
        projectOptions = projectOptions.concat(
            [
                <option key={project["name"]} value={project["name"]}>
                    {project["name"]}
                </option>
            ]
        );
        if idx < 3 {
            recentProjects = recentProjects.concat([project]);
            hasRecent = True;
            idx = idx + 1;
        }
    }
    for project in recentProjects {
        recentLinks = recentLinks.concat(
            [
                <a
                    key={project["name"]}
                    onClick={lambda -> None { navigate("/project/" + project["name"]);}}
                    className="nav-link"
                >
                    {project["name"]}
                </a>
            ]
        );
    }

    return
        <main className="page">
            <div className="container">
                <header className="navbar">
                    <div className="brand">
                        <Mascot variant="logo" />
                        <div className="brand-meta">
                            <span>
                                CGinS
                            </span>
                            <span className="brand-tag">
                                CUDA Ghost in the Shell
                            </span>
                        </div>
                    </div>
                    <div className="nav-links">
                        <a
                            className="nav-link"
                            onClick={lambda -> None { navigate("/settings");}}
                        >
                            Settings
                        </a>
                        <a
                            className="nav-link"
                            onClick={lambda -> None { navigate("/new-project");}}
                        >
                            New Project
                        </a>
                        <a className="nav-link" href="#existing">
                            Load Existing
                        </a>
                    </div>
                    <GlowButton
                        label="Start New Project"
                        onClick={lambda -> None { navigate("/new-project");}}
                        variant="primary"
                        className="glow-button--compact"
                    />
                </header>
            </div>
            <section className="hero-splash">
                <div className="container hero-grid">
                    <div className="hero-copy">
                        <span className="eyebrow reveal delay-1">
                            GPU Kernel Studio
                        </span>
                        <h1 className="hero-title reveal delay-2">
                            Make CUDA kernels feel effortless, precise, and cute.
                        </h1>
                        <p className="hero-sub reveal delay-3">
                            CGinS profiles your model, generates correct CUDA kernels, anditerates toward GPU-native performance with a workflow that feelshandcrafted and calm.
                        </p>
                        <div className="hero-actions reveal delay-4">
                            <GlowButton
                                label="Upload Full Model"
                                onClick={lambda -> None { navigate("/new-project");}}
                                variant="primary"
                            />
                            <button
                                className="ghost-button"
                                onClick={lambda -> None { navigate("/new-project");}}
                            >
                                Upload Weights
                            </button>
                        </div>
                        <div className="hero-badges">
                            <span className="badge">
                                GPU-aware generation
                            </span>
                            <span className="badge">
                                Correctness-first validation
                            </span>
                            <span className="badge">
                                Export-ready kernels
                            </span>
                        </div>
                    </div>
                    <div className="hero-visual">
                        <div className="mascot-halo"></div>
                        <div className="hero-orbits">
                            <span className="orb-one"></span>
                            <span className="orb-two"></span>
                            <span className="orb-three"></span>
                        </div>
                        <Mascot
                            variant="hero"
                            ambient={True}
                            className="reveal delay-3"
                        />
                    </div>
                </div>
            </section>
            <section className="section">
                <div className="container">
                    <h2 className="section-title">
                        CGinS feels like a studio
                    </h2>
                    <p className="section-sub">
                        A calm, structured flow that turns profiling, generation, and exportinto a repeatable, premium experience.
                    </p>
                    <div className="feature-grid">
                        <IllustratedCard
                            title="Profile in context"
                            text="Capture real operator inputs and focus on the kernels that move your model forward."
                            icon={<span>
                                ⚡
                            </span>}
                            tag="profile"
                        />
                        <IllustratedCard
                            title="Generate with guardrails"
                            text="Automated checks keep every kernel aligned with PyTorch outputs before moving on."
                            icon={<span>
                                ✓
                            </span>}
                            tag="validate"
                        />
                        <IllustratedCard
                            title="Optimize with intent"
                            text="Compare each iteration and keep only the best-performing kernel for your GPU."
                            icon={<span>
                                ⤴
                            </span>}
                            tag="optimize"
                        />
                        <IllustratedCard
                            title="Export-ready"
                            text="Ship the winning kernels, metadata, and logs in one exportable package."
                            icon={<span>
                                ⬚
                            </span>}
                            tag="export"
                        />
                    </div>
                </div>
            </section>
            <section className="section" id="existing">
                <div className="container">
                    <h2 className="section-title">
                        Start in minutes
                    </h2>
                    <p className="section-sub">
                        Upload a model or resume a project with your existing kernels.
                    </p>
                    <div className="quickstart-grid">
                        <div className="card">
                            <h3 className="card-title">
                                New project
                            </h3>
                            <p className="card-text">
                                Bring your model code and optional weights. CGinS handlesprofiling and kernel generation end-to-end.
                            </p>
                            <div className="hero-actions">
                                <GlowButton
                                    label="Create Project"
                                    onClick={lambda -> None { navigate("/new-project");}}
                                    variant="primary"
                                />
                                <button
                                    className="ghost-button"
                                    onClick={lambda -> None { navigate("/new-project");}}
                                >
                                    Upload Weights
                                </button>
                            </div>
                        </div>
                        <div className="card">
                            <h3 className="card-title">
                                Load existing
                            </h3>
                            <p className="card-text">
                                Pick up where you left off. Jump directly into your latestkernels and benchmarks.
                            </p>
                            <div>
                                <select
                                    defaultValue=""
                                    onChange={handleProjectSelect}
                                    className="select-field"
                                >
                                    <option value="" disabled>
                                        Select a project...
                                    </option>
                                    {projectOptions}
                                </select>
                            </div>
                            <div className="recent-links">
                                {recentLinks
                                if hasRecent
                                else <span className="card-text">
                                    {"Loading projects..."
                                    if loading
                                    else "No recent projects yet."}
                                </span>}
                            </div>
                        </div>
                    </div>
                </div>
            </section>
            <section className="section">
                <div className="container">
                    <div className="story-row">
                        <div className="story-panel">
                            <h2 className="section-title">
                                A sweet, serious workflow
                            </h2>
                            <p>
                                CGinS blends a cozy UI with production-grade tooling. Everykernel keeps a log of what improved, what failed, and whatshipped, so your team stays confident and aligned.
                            </p>
                            <p>
                                Think of it as a CUDA studio: a place where you tune, test, andexport without losing the human touch.
                            </p>
                        </div>
                        <div className="hero-visual">
                            <div className="mascot-halo"></div>
                            <Mascot variant="cta" ambient={True} />
                        </div>
                    </div>
                </div>
            </section>
            <section className="section">
                <div className="container">
                    <div className="quote">
                        <p className="quote-text">
                            “CGinS gives us a GPU-native workflow without waiting onhand-written kernels.”
                        </p>
                        <div className="quote-author">
                            — ML Infrastructure Team
                        </div>
                    </div>
                </div>
            </section>
            <section className="section">
                <div className="container">
                    <div className="cta">
                        <div>
                            <h2 className="cta-title">
                                Ready to ship faster kernels?
                            </h2>
                            <p className="cta-text">
                                Spin up a new project and start generating CUDA kernels inminutes.
                            </p>
                        </div>
                        <div className="hero-actions">
                            <GlowButton
                                label="Start New Project"
                                onClick={lambda -> None { navigate("/new-project");}}
                                variant="primary"
                            />
                            <Mascot variant="cta" ambient={True} />
                        </div>
                    </div>
                </div>
            </section>
        </main>;
}
