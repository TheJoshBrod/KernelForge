# Kernel Forge Desktop

This repo now includes a Jac/Tauri desktop shell under `frontend/src-tauri`.

## What changed

- The desktop app is generated from the existing Jac frontend instead of a separate shell.
- Runtime data/config paths now honor desktop-safe env overrides:
  - `KFORGE_REPO_ROOT`
  - `KFORGE_DATA_DIR`
  - `KFORGE_CONFIG_PATH`
- The desktop launcher starts the Jac sidecar in a native window and keeps desktop-specific behavior in `frontend/src-tauri` rather than forking the shared frontend UI.
- The Tauri updater path is configured for signed desktop releases via GitHub Releases, with `latest.json` expected at `https://github.com/TheJoshBrod/CGinS/releases/latest/download/latest.json`.
- Desktop sidecar launch now prefers a dedicated repo or bundled `.desktop-runtime` and only falls back to repo `.venv` for local development compatibility.
- `./scripts/desktop/prepare-runtime.sh` bootstraps the dedicated desktop runtime with a CUDA-capable Torch wheel channel instead of inheriting the repo CPU-only Torch install.
- On Linux, the desktop shell now keeps the same visual frontend as `origin/main` and leaves slower WebKit fallback modes opt-in instead of default.

## Commands

Development:

```bash
./scripts/desktop/dev.sh
```

Production build on Linux:

```bash
./scripts/desktop/build-linux.sh
```

Bootstrap or refresh the dedicated desktop runtime explicitly:

```bash
./scripts/desktop/prepare-runtime.sh
```

Useful runtime environment overrides:

- `KFORGE_TORCH_CHANNEL=cu130` to pick a different official Torch accelerator channel.
- `KFORGE_TORCH_INDEX_URL=...` to override the Torch wheel index directly.
- `KFORGE_DESKTOP_RUNTIME_REBUILD=1` to recreate `.desktop-runtime` from scratch.
- `KFORGE_DESKTOP_RUNTIME_SYNC=1` to force a dependency refresh into an existing runtime.
- `KFORGE_ENABLE_DMABUF_RENDERER=1` to opt back into the faster Linux WebKit DMA-BUF path on machines where it paints correctly.
- `KFORGE_DISABLE_WEBKIT_COMPOSITING=1` to force the slower non-composited Linux fallback.
- `KFORGE_FORCE_SOFTWARE_RENDERING=1` to force the slowest software-rendered Linux fallback when debugging graphics issues.

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
`generate-updater-keys.sh` now fails fast if the output key path resolves inside
the repository.

### 2) Build release artifacts

After signing key generation, run the normal build flow:

```bash
./scripts/desktop/build-linux.sh
```

The build script exports both `TAURI_SIGNING_PRIVATE_KEY_PATH` and
`TAURI_SIGNING_PRIVATE_KEY` automatically from:

- `KFORGE_TAURI_UPDATER_KEY_PATH`, or
- `~/.config/kernel-forge-desktop/updater.key`

The build script fails if the resolved signing key path points inside the repo.
On Linux, `./scripts/desktop/build-linux.sh` now builds a `.deb` installer by default.
Override bundle types with `KFORGE_TAURI_BUNDLES` if you explicitly want others.

This helper flow expects the Tauri updater bundle output to be present under:

- `frontend/src-tauri/target/release/bundle/appimage`
- `frontend/src-tauri/target/release/bundle/macos`
- `frontend/src-tauri/target/release/bundle/nsis`
- `frontend/src-tauri/target/release/bundle/msi`

### 3) Prepare update manifest + staged assets

```bash
./scripts/desktop/prepare-update-release.sh \
  --version 1.2.3 \
  --base-url https://github.com/TheJoshBrod/CGinS/releases/download
```

Output staging directory defaults to:

`docs/desktop/releases/main/<version>/`

The command copies discovered bundle artifacts and required signatures into that directory and creates:

`docs/desktop/releases/main/<version>/latest.json`

If no artifact is found for a platform, placeholders are left in the manifest for manual follow-up.

If an artifact is found but its signature file is missing, the script now fails fast and stops.

`--base-url` must be a concrete URL; templated placeholders (for example `<org>` or `{{...}}`) fail immediately.

The updater config currently points to:

`https://github.com/TheJoshBrod/CGinS/releases/latest/download/latest.json`

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

Desktop builds need the normal Tauri Linux prerequisites and, for CUDA-capable
desktop runtimes on NVIDIA systems, a compatible NVIDIA driver/runtime.

- https://v2.tauri.app/start/prerequisites/

The runtime bootstrap script validates CUDA separately. On hosts with
`nvidia-smi` available, `prepare-runtime.sh` will fail if the selected Torch
channel still produces `torch.cuda.is_available() == false`.

## Expected runtime layout

For desktop builds, the Tauri bundle includes:

- the Jac frontend project files from `frontend/`
- Python backend sources from `src/`
- baseline kernel templates from `kernels/generated/`
- the dedicated `.desktop-runtime` for sidecar startup and job execution

Mutable application state is expected to live outside the install directory via app-data paths injected by the Tauri launcher.
