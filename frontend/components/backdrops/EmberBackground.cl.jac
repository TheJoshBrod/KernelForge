"""Ember / Spark Background Component"""

import from react { useEffect, useRef }

def:pub EmberBackground(className: str = "") -> Any {
    canvasRef = useRef(None);
    desktop_build = window["__KFORGE_DESKTOP_BUILD__"]
    if "__KFORGE_DESKTOP_BUILD__" in window and window["__KFORGE_DESKTOP_BUILD__"]
    else None;
    isDesktop = bool(
        desktop_build and "desktop" in desktop_build and desktop_build["desktop"]
    );

    baseClass = "app-life-canvas";  # Keeping base class for absolute positioning
    if className {
        baseClass = baseClass + f" {className}";
    }

    if isDesktop {
        return
            <div className={baseClass + " app-backdrop-static app-backdrop-static--ember"}/>;
    }

    useEffect(
        lambda -> Any { canvas = canvasRef.current;if not canvas {
            return;
        }ctx = canvas.getContext("2d");
        # Config
        particleCount = 60;particles = [];animationId = 0;active = True;def createParticle(
            reset: bool = False
        ) -> dict {
            width = window.innerWidth;
            height = window.innerHeight;
            return {
                "x": Math.random() * width,
                "y": height + (
                    Math.random() * 100 if reset else Math.random() * height
                ),
                "size": Math.random() * 3 + 2,
                "speed": Math.random() * 2 + 1,
                "opacity": Math.random() * 0.5 + 0.7,
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
            nonlocal animationId ;
            nonlocal active ;
            if not active {
                return;
            }
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            time = Date.now() / 1000;
            breathe = (Math.sin(time * Math.PI / 2) + 1) / 2;

            # 1. Upward Glow
            glowGrad = ctx.createRadialGradient(canvas.width/2, canvas.height, 0, canvas.width/2, canvas.height, canvas.height * 0.8);
            glowGrad.addColorStop(0, f"rgba(214, 48, 49, {0.6 + 0.2 * breathe})");
            glowGrad.addColorStop(0.5, f"rgba(199, 91, 57, {0.2 + 0.1 * breathe})");
            glowGrad.addColorStop(1, "rgba(0, 0, 0, 0)");
            
            ctx.fillStyle = glowGrad;
            ctx.globalCompositeOperation = "source-over";
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            # 2. Layered Flames (now slow-moving lava)
            gradient = ctx.createLinearGradient(0, canvas.height, 0, canvas.height - 200 - 40 * breathe);
            gradient.addColorStop(0, "rgba(214, 48, 49, 0.9)");      # Ember Red
            gradient.addColorStop(0.5, "rgba(225, 112, 85, 0.6)"); # Copper/Orange
            gradient.addColorStop(0.8, "rgba(214, 48, 49, 0.2)");  # Ember Red
            gradient.addColorStop(1, "rgba(225, 112, 85, 0)");

            ctx.globalCompositeOperation = "screen";
            for i in [0, 1, 2, 3] {
                ctx.beginPath();
                ctx.moveTo(0, canvas.height);
                
                amp = 50 + 20 * i + 10 * breathe;
                freq = 0.002 + 0.0005 * i;
                speed = 0.3 + 0.1 * i;
                
                x = 0;
                while x <= canvas.width + 20 {
                    v = Math.sin((x * freq) + time * speed) + Math.sin((x * freq * 1.7) - time * speed * 0.8) * 0.5;
                    y = canvas.height - ((v + 1.5) * amp) - (30 * i);
                    ctx.lineTo(x, y);
                    x += 20;
                }
                ctx.lineTo(canvas.width, canvas.height);
                ctx.closePath();
                
                ctx.fillStyle = gradient;
                ctx.fill();
            }
            
            # Reset composite operation for sparks
            ctx.globalCompositeOperation = "source-over";

            # Update and Draw Sparks
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
                ctx.shadowBlur = 25;
                ctx.shadowColor = "rgba(253, 203, 110, 1.0)";  # Molten Gold
            }
            ctx.shadowBlur = 0;

            animationId = requestAnimationFrame(draw);
        } animationId = requestAnimationFrame(draw);return lambda -> None { nonlocal active ;active = False;cancelAnimationFrame(
            animationId
        );window.removeEventListener("resize", resize);}; },
        []
    );

    return
        <canvas ref={canvasRef} className={baseClass}/>;
}
