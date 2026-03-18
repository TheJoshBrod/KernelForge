// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::Write;
use std::io::{BufRead, BufReader};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{mpsc, Mutex};
use std::time::Duration;
use tauri::Manager;

static SIDECAR_PROCESS: Mutex<Option<Child>> = Mutex::new(None);
static API_BASE_URL: Mutex<Option<String>> = Mutex::new(None);
static UI_PROCESS: Mutex<Option<Child>> = Mutex::new(None);
static UI_BASE_URL: Mutex<Option<String>> = Mutex::new(None);

const PRODUCT_TITLE: &str = "Kernel Forge";
const GITHUB_REPO: &str = "TheJoshBrod/CGinS";
const GITHUB_RELEASES_URL: &str = "https://github.com/TheJoshBrod/CGinS/releases";
const APP_VERSION: &str = env!("CARGO_PKG_VERSION");
const CONFIGURED_BASE_URL: &str = "";
const SIDECAR_STARTUP_TIMEOUT: Duration = Duration::from_secs(20);
const ROUTE_READY_TIMEOUT: Duration = Duration::from_secs(20);

#[cfg(target_os = "linux")]
fn configure_linux_rendering_env() {
    // Keep the fastest Linux WebKit path as the default. On this machine, the
    // broad DMA-BUF/compositing fallbacks remove the blank-window risk but make
    // the exact origin/main UI unacceptably slow. Leave those fallback modes
    // opt-in for machines that still need them.
    let disable_dmabuf = std::env::var_os("KFORGE_DISABLE_DMABUF_RENDERER")
        .map(|value| value == "1")
        .unwrap_or(false);

    if disable_dmabuf {
        unsafe {
            if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
                std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
            }
        }
    }

    let disable_compositing = std::env::var_os("KFORGE_DISABLE_WEBKIT_COMPOSITING")
        .map(|value| value == "1")
        .unwrap_or(false);
    let force_software = std::env::var_os("KFORGE_FORCE_SOFTWARE_RENDERING")
        .map(|value| value == "1")
        .unwrap_or(false);

    if disable_compositing || force_software {
        unsafe {
            if std::env::var_os("WEBKIT_DISABLE_COMPOSITING_MODE").is_none() {
                std::env::set_var("WEBKIT_DISABLE_COMPOSITING_MODE", "1");
            }
        }
    }

    if force_software {
        unsafe {
            if std::env::var_os("GSK_RENDERER").is_none() {
                std::env::set_var("GSK_RENDERER", "cairo");
            }
            if std::env::var_os("LIBGL_ALWAYS_SOFTWARE").is_none() {
                std::env::set_var("LIBGL_ALWAYS_SOFTWARE", "1");
            }
        }
        eprintln!(
            "Linux software rendering fallback enabled via KFORGE_FORCE_SOFTWARE_RENDERING=1"
        );
    }

    if disable_compositing && !force_software {
        eprintln!("Linux WebKit compositing disabled via KFORGE_DISABLE_WEBKIT_COMPOSITING=1");
    }

    if disable_dmabuf {
        eprintln!("Linux DMA-BUF renderer disabled via KFORGE_DISABLE_DMABUF_RENDERER=1");
    }

    eprintln!(
        "Linux render env: WEBKIT_DISABLE_DMABUF_RENDERER={:?} WEBKIT_DISABLE_COMPOSITING_MODE={:?} GSK_RENDERER={:?} LIBGL_ALWAYS_SOFTWARE={:?}",
        std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER"),
        std::env::var_os("WEBKIT_DISABLE_COMPOSITING_MODE"),
        std::env::var_os("GSK_RENDERER"),
        std::env::var_os("LIBGL_ALWAYS_SOFTWARE"),
    );
}

#[cfg(not(target_os = "linux"))]
fn configure_linux_rendering_env() {}

fn build_sha() -> &'static str {
    option_env!("KFORGE_BUILD_SHA").unwrap_or("dev")
}

fn find_existing_path(candidates: &[PathBuf]) -> Option<PathBuf> {
    for candidate in candidates {
        if candidate.exists() {
            return Some(candidate.clone());
        }
    }
    None
}

fn find_sidecar_path(app: &tauri::AppHandle) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(resource_dir) = app.path().resource_dir() {
        if cfg!(windows) {
            candidates.push(resource_dir.join("binaries/jac-sidecar.exe"));
            candidates.push(resource_dir.join("binaries/jac-sidecar.bat"));
        } else {
            candidates.push(resource_dir.join("binaries/jac-sidecar"));
            candidates.push(resource_dir.join("binaries/jac-sidecar.sh"));
        }
    }

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let mut current = exe_dir.to_path_buf();
            loop {
                if cfg!(windows) {
                    candidates.push(current.join("src-tauri/binaries/jac-sidecar.exe"));
                    candidates.push(current.join("src-tauri/binaries/jac-sidecar.bat"));
                    candidates.push(current.join("binaries/jac-sidecar.exe"));
                    candidates.push(current.join("binaries/jac-sidecar.bat"));
                } else {
                    candidates.push(current.join("src-tauri/binaries/jac-sidecar"));
                    candidates.push(current.join("src-tauri/binaries/jac-sidecar.sh"));
                    candidates.push(current.join("binaries/jac-sidecar"));
                    candidates.push(current.join("binaries/jac-sidecar.sh"));
                }
                if !current.pop() {
                    break;
                }
            }
        }
    }

    find_existing_path(&candidates)
}

fn find_ui_server_path(app: &tauri::AppHandle, runtime_root: &Path) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Ok(resource_dir) = app.path().resource_dir() {
        candidates.push(resource_dir.join("binaries/kf-ui-server.py"));
        candidates.push(resource_dir.join("_up_/binaries/kf-ui-server.py"));
    }

    candidates.push(runtime_root.join("frontend/src-tauri/binaries/kf-ui-server.py"));
    candidates.push(runtime_root.join("src-tauri/binaries/kf-ui-server.py"));

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let mut current = exe_dir.to_path_buf();
            loop {
                candidates.push(current.join("src-tauri/binaries/kf-ui-server.py"));
                candidates.push(current.join("binaries/kf-ui-server.py"));
                if !current.pop() {
                    break;
                }
            }
        }
    }

    find_existing_path(&candidates)
}

fn find_module_path(app: &tauri::AppHandle) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(resource_dir) = app.path().resource_dir() {
        candidates.push(resource_dir.join("main.jac"));
        candidates.push(resource_dir.join("frontend/main.jac"));
        candidates.push(resource_dir.join("_up_/main.jac"));
        candidates.push(resource_dir.join("_up_/frontend/main.jac"));
    }

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let mut current = exe_dir.to_path_buf();
            loop {
                candidates.push(current.join("main.jac"));
                candidates.push(current.join("frontend/main.jac"));
                if !current.pop() {
                    break;
                }
            }
        }
    }

    find_existing_path(&candidates)
}

fn runtime_root_from_module(module_path: &Path) -> PathBuf {
    let module_dir = module_path
        .parent()
        .map(|path| path.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));

    if module_dir.join("src").exists() {
        return module_dir;
    }

    let bundled_runtime_root = module_dir.join("_up_");
    if bundled_runtime_root.join("src").exists() {
        return bundled_runtime_root;
    }

    if module_dir.file_name().and_then(|name| name.to_str()) == Some("frontend") {
        if let Some(parent) = module_dir.parent() {
            if parent.join("src").exists() {
                return parent.to_path_buf();
            }
        }
    }

    module_dir
}

fn python_bin_dir(runtime_root: &Path) -> Option<PathBuf> {
    let candidates = if cfg!(windows) {
        vec![
            runtime_root.join(".desktop-runtime/Scripts"),
            runtime_root.join(".venv/Scripts"),
            runtime_root.join("frontend/.desktop-runtime/Scripts"),
            runtime_root.join("frontend/.venv/Scripts"),
        ]
    } else {
        vec![
            runtime_root.join(".desktop-runtime/bin"),
            runtime_root.join(".venv/bin"),
            runtime_root.join("frontend/.desktop-runtime/bin"),
            runtime_root.join("frontend/.venv/bin"),
        ]
    };

    for candidate in candidates {
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn python_executable(runtime_root: &Path) -> PathBuf {
    let candidates = if cfg!(windows) {
        vec![
            runtime_root.join(".desktop-runtime/Scripts/python.exe"),
            runtime_root.join(".venv/Scripts/python.exe"),
            runtime_root.join("frontend/.desktop-runtime/Scripts/python.exe"),
            runtime_root.join("frontend/.venv/Scripts/python.exe"),
        ]
    } else {
        vec![
            runtime_root.join(".desktop-runtime/bin/python3"),
            runtime_root.join(".desktop-runtime/bin/python"),
            runtime_root.join(".venv/bin/python3"),
            runtime_root.join(".venv/bin/python"),
            runtime_root.join("frontend/.desktop-runtime/bin/python3"),
            runtime_root.join("frontend/.desktop-runtime/bin/python"),
            runtime_root.join("frontend/.venv/bin/python3"),
            runtime_root.join("frontend/.venv/bin/python"),
        ]
    };

    for candidate in candidates {
        if candidate.exists() {
            return candidate;
        }
    }

    if cfg!(windows) {
        PathBuf::from("python")
    } else {
        PathBuf::from("python3")
    }
}

fn reserve_local_port() -> Result<u16, Box<dyn std::error::Error>> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

fn find_dist_path(
    app: &tauri::AppHandle,
    module_path: &Path,
    runtime_root: &Path,
) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Ok(resource_dir) = app.path().resource_dir() {
        candidates.push(resource_dir.join("bundled-ui"));
        candidates.push(resource_dir.join("dist"));
        candidates.push(resource_dir.join("_up_/bundled-ui"));
        candidates.push(resource_dir.join(".jac/client/dist"));
        candidates.push(resource_dir.join("_up_/dist"));
        candidates.push(resource_dir.join("_up_/.jac/client/dist"));
    }

    if let Some(frontend_root) = module_path.parent() {
        candidates.push(frontend_root.join("src-tauri/bundled-ui"));
        candidates.push(frontend_root.join(".jac/client/dist"));
    }

    candidates.push(runtime_root.join("frontend/src-tauri/bundled-ui"));
    candidates.push(runtime_root.join("frontend/.jac/client/dist"));
    candidates.push(runtime_root.join(".jac/client/dist"));

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let mut current = exe_dir.to_path_buf();
            loop {
                candidates.push(current.join("bundled-ui"));
                candidates.push(current.join(".jac/client/dist"));
                candidates.push(current.join("src-tauri/bundled-ui"));
                candidates.push(current.join("frontend/.jac/client/dist"));
                if !current.pop() {
                    break;
                }
            }
        }
    }

    find_existing_path(&candidates)
}

fn prepend_env_path(current: Option<&OsStr>, prefix: &Path) -> OsString {
    let mut combined = OsString::from(prefix.as_os_str());
    if let Some(existing) = current {
        if !existing.is_empty() {
            let separator = if cfg!(windows) { ";" } else { ":" };
            combined.push(separator);
            combined.push(existing);
        }
    }
    combined
}

fn configure_child_env(
    cmd: &mut Command,
    app: &tauri::AppHandle,
    module_path: &Path,
    runtime_root: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let frontend_root = module_path
        .parent()
        .ok_or("Could not determine frontend root for Jac sidecar")?;
    let data_dir = app.path().app_data_dir()?;
    let config_dir = app.path().app_config_dir()?;

    fs::create_dir_all(&data_dir)?;
    fs::create_dir_all(&config_dir)?;

    cmd.current_dir(frontend_root);
    cmd.env("KFORGE_DESKTOP", "1");
    cmd.env("KFORGE_APP_VERSION", APP_VERSION);
    cmd.env("KFORGE_BUILD_SHA", build_sha());
    cmd.env("KFORGE_GITHUB_REPO", GITHUB_REPO);
    cmd.env("KFORGE_DATA_DIR", &data_dir);
    cmd.env("KFORGE_CONFIG_PATH", config_dir.join("config.json"));
    cmd.env("KFORGE_REPO_ROOT", runtime_root);
    cmd.env("PYTHONUNBUFFERED", "1");

    let python_path = prepend_env_path(std::env::var_os("PYTHONPATH").as_deref(), runtime_root);
    cmd.env("PYTHONPATH", python_path);

    if let Some(bin_dir) = python_bin_dir(runtime_root) {
        let path_value = prepend_env_path(std::env::var_os("PATH").as_deref(), &bin_dir);
        cmd.env("PATH", path_value);
    }

    Ok(())
}

fn find_and_start_sidecar(app: &tauri::AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    if !CONFIGURED_BASE_URL.is_empty() {
        let mut url = API_BASE_URL.lock().unwrap();
        *url = Some(CONFIGURED_BASE_URL.to_string());
        eprintln!("Using configured API base URL: {}", CONFIGURED_BASE_URL);
        return Ok(());
    }

    let sidecar_path = find_sidecar_path(app).ok_or("Jac sidecar launcher not found")?;
    let module_path =
        find_module_path(app).ok_or("Could not locate main.jac for the Jac sidecar")?;
    let frontend_root = module_path
        .parent()
        .map(|path| path.to_path_buf())
        .ok_or("Could not determine frontend root for main.jac")?;
    let runtime_root = runtime_root_from_module(&module_path);

    let mut cmd = if cfg!(windows) {
        if sidecar_path.extension().and_then(|s| s.to_str()) == Some("bat") {
            let mut command = Command::new("cmd");
            command.arg("/C");
            command.arg(&sidecar_path);
            command
        } else {
            Command::new(&sidecar_path)
        }
    } else if sidecar_path.extension().and_then(|s| s.to_str()) == Some("sh") {
        let mut command = Command::new("bash");
        command.arg(&sidecar_path);
        command
    } else {
        Command::new(&sidecar_path)
    };

    cmd.arg("--module-path").arg(&module_path);
    cmd.arg("--base-path").arg(&frontend_root);
    cmd.arg("--port").arg("0");
    cmd.arg("--host").arg("127.0.0.1");
    configure_child_env(&mut cmd, app, &module_path, &runtime_root)?;
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::inherit());

    match cmd.spawn() {
        Ok(mut child) => {
            let discovered_port = match wait_for_sidecar_port(&mut child) {
                Ok(port) => port,
                Err(error) => {
                    shutdown_child(&mut child);
                    return Err(error);
                }
            };

            let mut process = SIDECAR_PROCESS.lock().unwrap();
            *process = Some(child);

            let base_url = format!("http://127.0.0.1:{discovered_port}");
            wait_for_route_ready(discovered_port)?;
            let mut url = API_BASE_URL.lock().unwrap();
            *url = Some(base_url.clone());
            eprintln!("Sidecar started on {}", base_url);
            Ok(())
        }
        Err(error) => {
            eprintln!("Failed to start sidecar: {}", error);
            Err(Box::new(error))
        }
    }
}

fn wait_for_sidecar_port(child: &mut Child) -> Result<u16, Box<dyn std::error::Error>> {
    let stdout = child
        .stdout
        .take()
        .ok_or("Jac sidecar stdout was not available for port discovery")?;
    let (sender, receiver) = mpsc::channel::<Result<u16, String>>();

    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        let mut sent_port = false;
        for line in reader.lines() {
            match line {
                Ok(line) => {
                    eprintln!("[sidecar] {}", line);
                    if !sent_port {
                        if let Some(port_str) = line.strip_prefix("JAC_SIDECAR_PORT=") {
                            match port_str.trim().parse::<u16>() {
                                Ok(port) => {
                                    let _ = sender.send(Ok(port));
                                    sent_port = true;
                                }
                                Err(error) => {
                                    let _ = sender.send(Err(format!(
                                        "Jac sidecar reported an invalid port: {error}"
                                    )));
                                    return;
                                }
                            }
                        }
                    }
                }
                Err(error) => {
                    let _ = sender.send(Err(format!("Failed to read Jac sidecar stdout: {error}")));
                    return;
                }
            }
        }

        if !sent_port {
            let _ = sender.send(Err(
                "Jac sidecar exited before reporting its port".to_string()
            ));
        }
    });

    match receiver.recv_timeout(SIDECAR_STARTUP_TIMEOUT) {
        Ok(Ok(port)) => Ok(port),
        Ok(Err(message)) => Err(message.into()),
        Err(mpsc::RecvTimeoutError::Timeout) => Err(format!(
            "Jac sidecar did not report a port within {} seconds",
            SIDECAR_STARTUP_TIMEOUT.as_secs()
        )
        .into()),
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            Err("Jac sidecar stdout closed before reporting its port".into())
        }
    }
}

fn shutdown_child(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn wait_for_route_ready(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    wait_for_http_path_ready(port, "/")
}

fn wait_for_http_path_ready(port: u16, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let address = format!("127.0.0.1:{port}");
    let deadline = std::time::Instant::now() + ROUTE_READY_TIMEOUT;
    let request = format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");

    while std::time::Instant::now() < deadline {
        match TcpStream::connect(&address) {
            Ok(mut stream) => {
                stream.set_read_timeout(Some(Duration::from_secs(2)))?;
                stream.set_write_timeout(Some(Duration::from_secs(2)))?;
                stream.write_all(request.as_bytes())?;

                let mut reader = BufReader::new(stream);
                let mut status_line = String::new();
                if reader.read_line(&mut status_line).is_ok()
                    && (status_line.starts_with("HTTP/1.1 200")
                        || status_line.starts_with("HTTP/1.0 200"))
                {
                    return Ok(());
                }
            }
            Err(_) => {}
        }

        std::thread::sleep(Duration::from_millis(250));
    }

    Err(format!(
        "Jac sidecar did not serve the base route within {} seconds",
        ROUTE_READY_TIMEOUT.as_secs()
    )
    .into())
}

fn start_ui_server(app: &tauri::AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    let module_path = find_module_path(app).ok_or("Could not locate main.jac for the UI bundle")?;
    let runtime_root = runtime_root_from_module(&module_path);
    let dist_dir = find_dist_path(app, &module_path, &runtime_root)
        .ok_or("Could not locate built UI dist directory")?;
    let script_path =
        find_ui_server_path(app, &runtime_root).ok_or("Could not locate the UI server script")?;
    let api_base_url = {
        let url = API_BASE_URL.lock().unwrap();
        url.clone()
            .ok_or("Jac sidecar base URL was not available for the UI server")?
    };

    let port = reserve_local_port()?;
    let python = python_executable(&runtime_root);
    let mut cmd = Command::new(python);
    cmd.arg(&script_path)
        .arg("--bind")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--dist-dir")
        .arg(&dist_dir)
        .arg("--api-base-url")
        .arg(&api_base_url)
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());

    match cmd.spawn() {
        Ok(child) => {
            wait_for_http_path_ready(port, "/")?;
            let base_url = format!("http://127.0.0.1:{port}/");
            let mut process = UI_PROCESS.lock().unwrap();
            *process = Some(child);
            let mut url = UI_BASE_URL.lock().unwrap();
            *url = Some(base_url);
            Ok(())
        }
        Err(error) => Err(Box::new(error)),
    }
}

fn stop_children() {
    let mut ui_process = UI_PROCESS.lock().unwrap();
    if let Some(mut child) = ui_process.take() {
        shutdown_child(&mut child);
    }

    let mut process = SIDECAR_PROCESS.lock().unwrap();
    if let Some(mut child) = process.take() {
        shutdown_child(&mut child);
        eprintln!("Sidecar stopped");
    }
}

fn main() {
    configure_linux_rendering_env();

    tauri::Builder::default()
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            find_and_start_sidecar(app.handle())?;
            start_ui_server(app.handle())?;

            let desktop_payload = serde_json::json!({
                "desktop": true,
                "version": APP_VERSION,
                "commit": build_sha(),
                "github_repo": GITHUB_REPO,
                "releases_url": GITHUB_RELEASES_URL,
            });
            let init_js = format!("globalThis.__KFORGE_DESKTOP_BUILD__ = {};", desktop_payload);

            let webview_url = {
                let url = UI_BASE_URL.lock().unwrap();
                match &*url {
                    Some(base_url) => tauri::WebviewUrl::External(base_url.parse()?),
                    None => tauri::WebviewUrl::App("index.html".into()),
                }
            };

            let builder = tauri::WebviewWindowBuilder::new(app, "main", webview_url)
                .title(PRODUCT_TITLE)
                .inner_size(1200.0, 800.0)
                .min_inner_size(800.0, 600.0)
                .resizable(true)
                .initialization_script(&init_js);

            builder.build()?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if window.label() == "main"
                && matches!(event, tauri::WindowEvent::CloseRequested { .. })
            {
                stop_children();
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
