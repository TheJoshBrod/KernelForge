# Kernel Forge Desktop

This repo now includes a Jac/Tauri desktop shell under `frontend/src-tauri`.

## What changed

- The desktop app is generated from the existing Jac frontend instead of a separate shell.
- Runtime data/config paths now honor desktop-safe env overrides:
  - `KFORGE_REPO_ROOT`
  - `KFORGE_DATA_DIR`
  - `KFORGE_CONFIG_PATH`
- The desktop launcher injects build metadata, starts the Jac sidecar in a native window, and shows an in-app GitHub `main` update notice when the packaged commit falls behind upstream.
- The Tauri updater path is configured for signed desktop releases via GitHub Releases, with `latest.json` expected at `https://github.com/TheJoshBrod/CGinS/releases/latest/download/latest.json`.
- Desktop sidecar launch now prefers the repo or bundled `.venv` and fails fast unless `KFORGE_ALLOW_SYSTEM_PYTHON=1` is set for local debugging.

## Commands

Development:

```bash
./scripts/desktop/dev.sh
```

Production build on Linux:

```bash
./scripts/desktop/build-linux.sh
```

## Updater release helpers

### 1) Generate signing keys

```bash
./scripts/desktop/generate-updater-keys.sh
```

This script creates:

- a local private key at `~/.config/kernel-forge-desktop/updater.key`
- a repo public-key snapshot at `frontend/src-tauri/updater.pub`

The private key stays outside the repo. The committed `frontend/src-tauri/updater.pub`
is what the desktop app uses to verify signed updates.

### 2) Build release artifacts

After signing key generation, run the normal build flow:

```bash
./scripts/desktop/build-linux.sh
```

The build script exports `TAURI_SIGNING_PRIVATE_KEY_PATH` automatically from:

- `KFORGE_TAURI_UPDATER_KEY_PATH`, or
- `~/.config/kernel-forge-desktop/updater.key`

This helper flow expects the Tauri updater bundle output to be present under:

- `frontend/src-tauri/target/release/bundle/appimage`
- `frontend/src-tauri/target/release/bundle/macos`
- `frontend/src-tauri/target/release/bundle/nsis`
- `frontend/src-tauri/target/release/bundle/msi`

### 3) Prepare update manifest + staged assets

```bash
./scripts/desktop/prepare-update-release.sh \
  --version 1.2.3 \
  --base-url https://github.com/<org>/<repo>/releases/download
```

Output staging directory defaults to:

`docs/desktop/releases/main/<version>/`

The command copies discovered bundle artifacts and generated signatures into that directory and creates:

`docs/desktop/releases/main/<version>/latest.json`

If no artifact/signature is found for a platform, placeholders are left in the manifest for manual follow-up.

The manifest template used by the script is:

`docs/desktop/update-manifest.template.json`

## Expected updater config touch points (reference)

`frontend/src-tauri/tauri.conf.json` (no edits made by these scripts)

- `bundle.createUpdaterArtifacts` should be `true` (when ready to publish signed artifacts).
- `plugins.updater.endpoints` should point to the published `latest.json` URL.
- `plugins.updater.pubkey` currently points at `frontend/src-tauri/updater.pub`.

### 4) Publish

Upload:
- all staged platform installers/signatures
- the matching `latest.json`

to a GitHub release so Tauri clients can reach `latest.json` from the configured endpoint.

## Linux prerequisites

The current desktop build on this machine is blocked by missing native GLib/GTK/WebKit development packages. The immediate error is:

```text
The system library `glib-2.0` required by crate `glib-sys` was not found.
```

Install the Linux prerequisites from the official Tauri docs for your distro, then rerun the build:

- https://v2.tauri.app/start/prerequisites/

On Ubuntu/Debian systems that usually means the GTK/WebKit/GLib development packages, not just Rust and Cargo.

## Expected runtime layout

For desktop builds, the Tauri bundle includes:

- the Jac frontend project files from `frontend/`
- Python backend sources from `src/`
- baseline kernel templates from `kernels/generated/`
- the repo `.venv` for sidecar startup

Mutable application state is expected to live outside the install directory via app-data paths injected by the Tauri launcher.
