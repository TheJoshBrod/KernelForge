"""Mascot component with static KernelForge SVG."""

import from react { useRef }
import from "@jac-client/assets/KernelForge.svg" { default as mascotImage }

def:pub Mascot(
    variant: str = "hero",
    className: str = "",
    alt: str = "Kernal Forge mascot",
    ambient: bool = False,
    enableVideo: bool = False
) -> Any {
    has _unused_enable_video = enableVideo;

    containerRef = useRef(None);
    classes = f"mascot mascot--{variant}";
    if className {
        classes += f" {className}";
    }
    ambientMarkup = (
        <div className="mascot-ambient">
            <span className="ambient-one"></span>
            <span className="ambient-two"></span>
            <span className="ambient-three"></span>
        </div>
    )
    if ambient
    else None;

    return
        <div
            className={classes}
            ref={containerRef}
        >
            {ambientMarkup}
            <div className="mascot-media">
                <img
                    className="mascot-img"
                    src={mascotImage}
                    alt={alt}
                    draggable={False}
                />
            </div>
        </div>;
}
