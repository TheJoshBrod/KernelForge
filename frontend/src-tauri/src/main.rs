// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{mpsc, Mutex};
use std::time::Duration;
use tauri::Manager;

static SIDECAR_PROCESS: Mutex<Option<Child>> = Mutex::new(None);
static API_BASE_URL: Mutex<Option<String>> = Mutex::new(None);

const PRODUCT_TITLE: &str = "Kernel Forge";
const GITHUB_REPO: &str = "TheJoshBrod/CGinS";
const GITHUB_RELEASES_URL: &str = "https://github.com/TheJoshBrod/CGinS/releases";
const APP_VERSION: &str = env!("CARGO_PKG_VERSION");
const CONFIGURED_BASE_URL: &str = "";
const SIDECAR_STARTUP_TIMEOUT: Duration = Duration::from_secs(20);

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

fn find_module_path(app: &tauri::AppHandle) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(resource_dir) = app.path().resource_dir() {
        candidates.push(resource_dir.join("main.jac"));
        candidates.push(resource_dir.join("frontend/main.jac"));
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
            runtime_root.join(".venv/Scripts"),
            runtime_root.join("frontend/.venv/Scripts"),
        ]
    } else {
        vec![
            runtime_root.join(".venv/bin"),
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

fn stop_sidecar() {
    let mut process = SIDECAR_PROCESS.lock().unwrap();
    if let Some(mut child) = process.take() {
        shutdown_child(&mut child);
        eprintln!("Sidecar stopped");
    }
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            find_and_start_sidecar(app.handle())?;

            let desktop_payload = serde_json::json!({
                "desktop": true,
                "version": APP_VERSION,
                "commit": build_sha(),
                "github_repo": GITHUB_REPO,
                "releases_url": GITHUB_RELEASES_URL,
            });
            let init_js = {
                let url = API_BASE_URL.lock().unwrap();
                match &*url {
                    Some(base_url) => format!(
                        "globalThis.__JAC_API_BASE_URL__ = '{}'; globalThis.__KFORGE_DESKTOP_BUILD__ = {};",
                        base_url,
                        desktop_payload
                    ),
                    None => format!("globalThis.__KFORGE_DESKTOP_BUILD__ = {};", desktop_payload),
                }
            };

            let builder = tauri::WebviewWindowBuilder::new(
                app,
                "main",
                tauri::WebviewUrl::App("index.html".into()),
            )
            .title(PRODUCT_TITLE)
            .inner_size(1200.0, 800.0)
            .min_inner_size(800.0, 600.0)
            .resizable(true)
            .initialization_script(&init_js);

            builder.build()?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if window.label() == "main" && matches!(event, tauri::WindowEvent::CloseRequested { .. }) {
                stop_sidecar();
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
