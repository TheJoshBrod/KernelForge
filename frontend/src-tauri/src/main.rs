// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{BufRead, BufReader};
use std::process::{Command, Child, Stdio};
use std::sync::Mutex;
use tauri::Manager;

// Global storage for sidecar process
static SIDECAR_PROCESS: Mutex<Option<Child>> = Mutex::new(None);
static API_BASE_URL: Mutex<Option<String>> = Mutex::new(None);

/// User-configured base URL from jac.toml (empty = dynamic discovery)
const CONFIGURED_BASE_URL: &str = "";

fn find_and_start_sidecar(app: &tauri::AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    // Skip sidecar launch â user manages their own backend
    if !CONFIGURED_BASE_URL.is_empty() {
        let mut url = API_BASE_URL.lock().unwrap();
        *url = Some(CONFIGURED_BASE_URL.to_string());
        eprintln!("Using configured API base URL: {}", CONFIGURED_BASE_URL);
        return Ok(());
    }

    // Try to find the sidecar in bundled resources
    let resource_dir = app.path().resource_dir()?;

    // Possible sidecar names
    let sidecar_names = if cfg!(windows) {
        vec!["binaries/jac-sidecar.exe", "binaries/jac-sidecar.bat"]
    } else {
        vec!["binaries/jac-sidecar", "binaries/jac-sidecar.sh"]
    };

    let mut sidecar_path = None;
    for name in &sidecar_names {
        let path = resource_dir.join(name);
        if path.exists() {
            sidecar_path = Some(path);
            break;
        }
    }

    // If not found in resources, try relative to executable
    if sidecar_path.is_none() {
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                let exe_dir = exe_dir.to_path_buf();
                for name in &sidecar_names {
                    let path = exe_dir.join(name);
                    if path.exists() {
                        sidecar_path = Some(path);
                        break;
                    }
                }
            }
        }
    }

    if let Some(sidecar_path) = sidecar_path {
        // Determine module path (try to find main.jac relative to app)
        let module_path = if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                // Look for main.jac in parent directories
                let mut current = exe_dir.to_path_buf();
                loop {
                    let main_jac = current.join("main.jac");
                    if main_jac.exists() {
                        break Some(main_jac);
                    }
                    if !current.pop() {
                        break None;
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        // Build command to start sidecar
        let mut cmd = if cfg!(windows) {
            if sidecar_path.extension().and_then(|s| s.to_str()) == Some("bat") {
                let mut c = Command::new("cmd");
                c.arg("/C");
                c.arg(&sidecar_path);
                c
            } else {
                Command::new(&sidecar_path)
            }
        } else {
            if sidecar_path.extension().and_then(|s| s.to_str()) == Some("sh") {
                let mut c = Command::new("sh");
                c.arg(&sidecar_path);
                c
            } else {
                Command::new(&sidecar_path)
            }
        };

        // Add arguments
        if let Some(ref mp) = module_path {
            cmd.arg("--module-path").arg(mp);
        } else {
            cmd.arg("--module-path").arg("main.jac");
        }
        cmd.arg("--port").arg("0"); // OS assigns free port
        cmd.arg("--host").arg("127.0.0.1");

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::inherit());

        match cmd.spawn() {
            Ok(mut child) => {
                let mut discovered_port: Option<u16> = None;
                if let Some(stdout) = child.stdout.take() {
                    let reader = BufReader::new(stdout);
                    for line in reader.lines() {
                        match line {
                            Ok(line) => {
                                eprintln!("[sidecar] {}", line);
                                if let Some(port_str) = line.strip_prefix("JAC_SIDECAR_PORT=") {
                                    if let Ok(port) = port_str.trim().parse::<u16>() {
                                        discovered_port = Some(port);
                                        break;
                                    }
                                }
                            }
                            Err(_) => break,
                        }
                    }
                }

                let mut process = SIDECAR_PROCESS.lock().unwrap();
                *process = Some(child);

                if let Some(port) = discovered_port {
                    let base_url = format!("http://127.0.0.1:{}", port);
                    eprintln!("Sidecar started on {}", base_url);
                    let mut url = API_BASE_URL.lock().unwrap();
                    *url = Some(base_url);
                } else {
                    eprintln!("Error: Sidecar started but did not report its port.");
                    eprintln!("       Expected JAC_SIDECAR_PORT=<port> on stdout.");
                    return Err("Sidecar port discovery failed".into());
                }
                Ok(())
            }
            Err(e) => {
                eprintln!("Failed to start sidecar: {}", e);
                Err(Box::new(e))
            }
        }
    } else {
        eprintln!("Sidecar not found in resources, skipping auto-start");
        Ok(())
    }
}

fn stop_sidecar() {
    let mut process = SIDECAR_PROCESS.lock().unwrap();
    if let Some(mut child) = process.take() {
        let _ = child.kill();
        let _ = child.wait();
        eprintln!("Sidecar stopped");
    }
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            // Start sidecar to discover dynamic port
            if let Err(e) = find_and_start_sidecar(app.handle()) {
                eprintln!("Warning: Could not start sidecar: {}", e);
            }

            // Build initialization script with API base URL
            let init_js = {
                let url = API_BASE_URL.lock().unwrap();
                match *url {
                    Some(ref base_url) => {
                        eprintln!("Injecting API base URL: {}", base_url);
                        format!(
                            "globalThis.__JAC_API_BASE_URL__ = '{}';",
                            base_url
                        )
                    }
                    None => String::new(),
                }
            };

            // Create window with initialization_script (runs BEFORE page JS)
            let mut builder = tauri::WebviewWindowBuilder::new(
                app,
                "main",
                tauri::WebviewUrl::App("index.html".into())
            )
            .title("frontend")
            .inner_size(1200.0, 800.0)
            .min_inner_size(800.0, 600.0)
            .resizable(true);

            if !init_js.is_empty() {
                builder = builder.initialization_script(&init_js);
            }

            builder.build()?;

            Ok(())
        })
        .on_window_event(|_window, event| {
            // Clean up sidecar when last window closes
            if matches!(event, tauri::WindowEvent::CloseRequested { .. }) {
                stop_sidecar();
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");

    // Ensure sidecar is stopped on exit
    stop_sidecar();
}
