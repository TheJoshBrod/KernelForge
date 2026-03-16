use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=KFORGE_BUILD_SHA");
    println!("cargo:rerun-if-changed=../../.git/HEAD");

    if let Ok(override_sha) = std::env::var("KFORGE_BUILD_SHA") {
        let sha = override_sha.trim();
        if !sha.is_empty() {
            println!("cargo:rustc-env=KFORGE_BUILD_SHA={sha}");
        }
    } else if let Ok(output) = Command::new("git").args(["rev-parse", "HEAD"]).output() {
        if output.status.success() {
            if let Ok(raw_sha) = String::from_utf8(output.stdout) {
                let sha = raw_sha.trim();
                if !sha.is_empty() {
                    println!("cargo:rustc-env=KFORGE_BUILD_SHA={sha}");
                }
            }
        }
    }

    tauri_build::build()
}
