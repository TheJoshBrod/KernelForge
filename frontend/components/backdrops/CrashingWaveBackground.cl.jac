"""Crashing Wave Background Component"""

import from react { useEffect, useRef }

def:pub CrashingWaveBackground() -> any {
    canvasRef = useRef(None);

    useEffect(lambda -> None {
        canvas = canvasRef.current;
        if not canvas { return; }

        ctx = canvas.getContext("2d");
        
        # Configuration
        fontSize = 12;
        # Blue-ish palette for water/foam
        colors = ["#0f172a", "#1e293b", "#334155", "#475569", "#64748b", "#94a3b8", "#cbd5e1", "#f1f5f9"];
        chars = [" ", ".", ",", "-", "~", "+", "*", "%", "#", "@"];
        
        cols = 0;
        rows = 0;
        time = 0;

        def resize() -> None {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            cols = Math.floor(canvas.width / fontSize);
            rows = Math.floor(canvas.height / fontSize);
        }

        resize();
        window.addEventListener("resize", resize);

        def draw() -> None {
            # Clear with very low opacity black to create trails/motion blur feel
            # or just opaque clear for crisp movement. Let's do partial clear for a "wet" look?
            # Actually, standard clear is better for this wave sim.
            ctx.fillStyle = "#020617"; # Very dark blue/black bg
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.font = f"{fontSize}px monospace";

            r = 0;
            while r < rows {
                c = 0;
                while c < cols {
                    # Simulation Logic:
                    # Create a wave effect using sin waves
                    # y-coordinate affects the 'phase' to make it look like it's moving down or up
                    # We want waves crashing on shore (bottom). So closer to bottom = 'breaking'
                    
                    # Normalize coordinates
                    nx = c / cols;
                    ny = r / rows; # 0 at top, 1 at bottom

                    # Wave calculation
                    # Main wave movement
                    wave1 = Math.sin((nx * 10) + (time * 0.05) + (ny * 5));
                    # Secondary detail wave
                    wave2 = Math.sin((nx * 25) - (time * 0.1) + (ny * 10));
                    # Breaking randomness
                    noise = Math.random() * 0.5;

                    # Combine: Focus on 'bands' of waves moving down
                    # The (ny * WHATEVER) - time controls movement direction.
                    # ny * 20 - time * 2 means as time increases, we track a phase moving UP???
                    # To move DOWN, we want phase = k * (y - ct). NO.
                    # sin(y - t). As t increases, matching y must increase to keep phase constant -> moves DOWN.
                    
                    movingPhase = (ny * 10) - (time * 2); 
                    
                    # Intensity 'h' based on the wave function
                    h = Math.sin(movingPhase) + (wave2 * 0.3) + noise;
                    
                    # Shore crash effect: Near bottom (ny > 0.8), waves break harder
                    if ny > 0.8 {
                        h += Math.random() * ((ny - 0.8) * 5); 
                    }

                    # Map 'h' to character and color
                    # h is roughly -2 to +2. Normalize to 0-1ish for array index
                    val = (h + 1.5) / 3.0;
                    if val < 0 { val = 0; }
                    if val > 1 { val = 1; }

                    charIdx = Math.floor(val * chars.length);
                    if charIdx >= chars.length { charIdx = chars.length - 1; }

                    colorIdx = Math.floor(val * colors.length);
                    if colorIdx >= colors.length { colorIdx = colors.length - 1; }

                    # Only draw if not empty to save perf and visuals
                    if charIdx > 0 {
                        ctx.fillStyle = colors[colorIdx];
                        char = chars[charIdx];
                        ctx.fillText(char, c * fontSize, r * fontSize);
                    }
                    
                    c += 1;
                }
                r += 1;
            }

            time += 0.05;
            requestAnimationFrame(draw);
        }

        animationId = requestAnimationFrame(draw);

        return lambda -> None {
            cancelAnimationFrame(animationId);
            window.removeEventListener("resize", resize);
        };
    }, []);

    return <canvas
        ref={canvasRef}
        className="fixed top-0 left-0 w-full h-full pointer-events-none z-0 opacity-40"
    />;
}
