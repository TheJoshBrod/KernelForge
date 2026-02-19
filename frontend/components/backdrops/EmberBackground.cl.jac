"""Ember / Spark Background Component"""

import from react { useEffect, useRef }

def:pub EmberBackground(className: str = "") -> any {
    canvasRef = useRef(None);

    useEffect(
        lambda -> None { canvas = canvasRef.current;if not canvas {
            return;
        }ctx = canvas.getContext("2d");
        # Config
        particleCount = 60;particles = [];def createParticle(
            reset: bool = False
        ) -> dict {
            width = window.innerWidth;
            height = window.innerHeight;
            return {
                "x": Math.random() * width,
                "y": height + (
                    Math.random() * 100 if reset else Math.random() * height
                ),
                "size": Math.random() * 2 + 1,
                "speed": Math.random() * 1.5 + 0.5,
                "opacity": Math.random() * 0.5 + 0.2,
                "drift": Math.random() * 0.5 - 0.25
            };
        } def resize() -> None {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            # Re-init particles on resize to fill screen
            nonlocal particles ;
            particles = [];
            i = 0;
            while i < particleCount {
                particles.push(createParticle(False));
                i += 1;
            }
        } resize();window.addEventListener("resize", resize);def draw() -> None {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            # Update and Draw
            for p in particles {
                p["y"] -= p["speed"];
                p["x"] += Math.sin(p["y"] * 0.05) * 0.2 + p["drift"];

                if p["y"] < -10 {
                    # Reset to bottom
                    newP = createParticle(True);
                    p["x"] = newP["x"];
                    p["y"] = newP["y"];
                    p["opacity"] = newP["opacity"];
                }

                ctx.beginPath();
                ctx.arc(p["x"], p["y"], p["size"], 0, Math.PI * 2);
                ctx.fillStyle = f"rgba(225, 112, 85, {p['opacity']})";  # Copper/Orange
                ctx.fill();

                # Optional glow
                ctx.shadowBlur = 10;
                ctx.shadowColor = "rgba(214, 48, 49, 0.5)";  # Ember Red
            }
            ctx.shadowBlur = 0;

            requestAnimationFrame(draw);
        } animationId = requestAnimationFrame(draw);return lambda -> None { cancelAnimationFrame(
            animationId
        );window.removeEventListener("resize", resize);}; },
        []
    );

    baseClass = "app-life-canvas";  # Keeping base class for absolute positioning
    if className {
        baseClass = baseClass + f" {className}";
    }

    return
        <canvas ref={canvasRef} className={baseClass}/>;
}
