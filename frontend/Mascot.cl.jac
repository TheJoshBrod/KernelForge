"""Mascot component with PNG default and MP4 hover (desktop only)."""

import from react { useEffect, useRef }

def:pub Mascot(
    variant: str = "hero",
    className: str = "",
    alt: str = "CGinS mascot",
    ambient: bool = False,
    enableVideo: bool = False
) -> any {
    has isHovered: bool = False,
        shouldLoadVideo: bool = False,
        videoReady: bool = False,
        reduceMotion: bool = False,
        isTouch: bool = False;

    containerRef = useRef(None);
    videoRef = useRef(None);

    useEffect(
        lambda -> None { try {
            mq = window.matchMedia("(prefers-reduced-motion: reduce)");
            reduceMotion = mq.matches;
            def onChange(e: any) -> None {
                reduceMotion = e.matches;
            }
            if mq and mq.addEventListener {
                mq.addEventListener("change", onChange);
            }
            if not isTouch {
                isTouch = ("ontouchstart" in window)
                or (navigator.maxTouchPoints and navigator.maxTouchPoints > 0);
            }
            return lambda -> None { if mq and mq.removeEventListener {
                mq.removeEventListener("change", onChange);
            }};
        } except Exception as e {
            return lambda -> None { };
        }},
        []
    );

    def handleEnter -> None {
        if reduceMotion or isTouch or not enableVideo {
            return;
        }
        shouldLoadVideo = True;
        isHovered = True;
        if videoRef and videoRef.current {
            try {
                videoRef.current.currentTime = 0;
                videoRef.current.play();
            } except Exception as e {
                return;
            }
        }
    }

    def handleLeave -> None {
        if reduceMotion or isTouch or not enableVideo {
            return;
        }
        isHovered = False;
        if videoRef and videoRef.current {
            try {
                videoRef.current.pause();
                videoRef.current.currentTime = 0;
            } except Exception as e {
                return;
            }
        }
    }

    def handleCanPlay -> None {
        videoReady = True;
        if isHovered and videoRef and videoRef.current {
            try {
                videoRef.current.play();
            } except Exception as e {
                return;
            }
        }
    }

    playing = enableVideo and isHovered and videoReady;
    classes = f"mascot mascot--{variant}";
    if playing {
        classes += " is-playing";
    }
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

    videoMarkup = (
        <video
            className="mascot-video"
            ref={videoRef}
            src={"/mascot.mp4" if shouldLoadVideo else ""}
            poster="/new-mascot.png"
            muted={True}
            loop={True}
            playsInline={True}
            preload="none"
            controls={False}
            onCanPlay={handleCanPlay}
        />
    )
    if enableVideo
    else None;

    return
        <div
            className={classes}
            ref={containerRef}
            onMouseEnter={handleEnter}
            onMouseLeave={handleLeave}
        >
            {ambientMarkup}
            <div className="mascot-media">
                <img
                    className="mascot-img"
                    src="/new-mascot.png"
                    alt={alt}
                    draggable={False}
                />
                {videoMarkup}
            </div>
        </div>;
}
