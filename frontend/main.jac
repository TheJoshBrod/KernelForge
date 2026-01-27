"""Main entry point for frontend."""

cl {
    import from react { useState, useEffect }
    import from sonner { Toaster }
    import from .context.ConfigContext { ConfigProvider }
    import from .components.Home { Home }
    import from .components.Project { Project }

    import "./styles.css";

    def:pub app() -> any {
        has page: str = "/";

        def navigate(path: str) -> None {
            console.log(f"Navigating to {path}");
            page = path;
            # window.history.pushState({}, "", path);
        }

        parts = page.split("/");
        # path is /project/name -> ["", "project", "name"]
        is_project = (parts[1] == "project") if parts.__len__() > 1 else False;
        project_name = parts[2] if is_project and parts.__len__() > 2 else "";

        return <ConfigProvider>
            <div className="bg-zinc-950 min-h-screen text-zinc-100 font-sans antialiased">
                {
                    <Home onNavigate={navigate} />
                    if page == "/" or page == ""
                    else 
                    <Project name={project_name} onNavigate={navigate} />
                    if is_project
                    else
                    <div className="flex min-h-screen flex-col items-center justify-center">
                        <h1>404 Not Found: {page}</h1>
                        <button onClick={lambda -> None { navigate("/"); }} className="mt-4 p-2 bg-zinc-800 rounded">
                            Back to Home
                        </button>
                    </div>
                }
                <Toaster richColors={True} theme="dark" />
            </div>
        </ConfigProvider>;
    }
}
