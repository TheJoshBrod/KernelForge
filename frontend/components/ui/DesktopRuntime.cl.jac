"""Helpers for desktop-specific runtime behavior."""

def:pub getDesktopBuildInfo() -> Any {
    return (
        window["__KFORGE_DESKTOP_BUILD__"]
        if "__KFORGE_DESKTOP_BUILD__" in window and window["__KFORGE_DESKTOP_BUILD__"]
        else None
    );
}

def:pub isDesktopBuild() -> bool {
    build_info = getDesktopBuildInfo();
    return bool(
        build_info
        and "desktop" in build_info
        and build_info["desktop"]
    );
}

def:pub isDocumentVisible() -> bool {
    try {
        if "hidden" in document {
            return not bool(document.hidden);
        }
        if "visibilityState" in document {
            return str(document.visibilityState) != "hidden";
        }
    } except Exception {}
    return True;
}
