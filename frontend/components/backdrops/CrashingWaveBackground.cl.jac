"""Crashing Wave Background Component"""

import from react { useEffect, useRef }

def:pub CrashingWaveBackground() -> Any {
    canvasRef = useRef(None);

    useEffect(
        lambda -> Any { canvas = canvasRef.current;if not canvas {
            return;
        }ctx = canvas.getContext("2d");
        # Configuration
        fontSize = 14;
        # Color Palettes
        # Deep Ocean -> Shallow -> Foam -> Wet Sand
        waterColors = ["#0c4a6e", "#0369a1", "#0ea5e9", "#38bdf8", "#7dd3fc"];foamColors = [
            "#e0f2fe",
            "#f0f9ff",
            "#ffffff"
        ];wetSandColors = ["#262626", "#171717"];  # Very dark to blend with BG
        waterChars = ["~", "≈", "-", "=", "«"];foamChars = [
            "*",
            "%",
            "#",
            "@",
            "o",
            "O"
        ];cols = 0;rows = 0;time = 0;def resize() -> None {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            cols = Math.floor(canvas.width / fontSize);
            rows = Math.floor(canvas.height / fontSize);
        } resize();window.addEventListener("resize", resize);def draw() -> None {
            # Clear screen
            ctx.fillStyle = "#020617";  # Zinc-950 equivalent
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.font = f"bold {fontSize}px monospace";

            # Global Tide Movement
            # oscillate between 0.3 (high tide/recede) and 0.9 (low tide/crash)
            # Use a specialized wave shape for fast-in, slow-out
            # t cycles every ~200 frames
            cycle = (time * 0.015) % (Math.PI * 2);

            # Base water line position (0 to 1)
            # sin(cycle) gives -1 to 1.
            # We want a sharp "crash" forward and slow "recede" back.
            # Using simple sin for now, maybe skew it later.
            tideLevel = 0.5 + 0.4 * Math.sin(cycle);

            # Add some perlin-ish noise to the tide line so it's not a straight line
            # varying intensity of the "swash" across columns
            c = 0;
            while c < cols {
                # Column-specific noise for irregular shoreline
                colNoise = Math.sin(c * 0.1) * 0.1 + Math.sin(c * 0.3 + time * 0.05) * 0.05;

                # Actual water line for this column
                currentWaterLine = tideLevel + colNoise;

                r = 0;
                while r < rows {
                    normalizedY = r / rows;

                    # Distance from water line (positive = below/on beach, negative = inside water)
                    dist = normalizedY - currentWaterLine;

                    charToDraw = "";
                    colorToUse = "";

                    if dist < 0 {
                        # IN WATER
                        # Logic: Deep water (far from line) vs Shallow/Foam (close to line)
                        depth = -dist;  # 0 at line, 1 at top
                        if depth < 0.15 {
                            # FOAM / BREAKING WAVE
                            # Chaotic, white/light blue
                            idx = Math.floor(Math.random() * foamColors.length);
                            colorToUse = foamColors[idx];
                            cIdx = Math.floor(Math.random() * foamChars.length);
                            charToDraw = foamChars[cIdx];
                        } else {
                            # BODY OF WATER
                            # Darker, calmer
                            # Map depth to color index (0 is deep/dark, last is shallow/light)
                            # depth 0.15 -> index 4, depth 1.0 -> index 0
                            wVal = 1.0 - ((depth - 0.15) / 0.85);  # 1 near shore, 0 far
                            if wVal < 0 {
                                wVal = 0;
                            }
                            wIdx = Math.floor(wVal * waterColors.length);
                            if wIdx >= waterColors.length {
                                wIdx = waterColors.length - 1;
                            }
                            colorToUse = waterColors[wIdx];
                            # Add some movement "glint"
                            if Math.random() > 0.95 {
                                charToDraw = "≈";
                            } elif Math.random() > 0.9 {
                                charToDraw = "~";
                            } else {
                                charToDraw = "-";
                            }
                        }
                    } elif dist < 0.15 {
                        # WET SAND (Just receded)
                        # Only show this if tide is receding? Or just make it always look wet near edge.
                        # Let's make it sparse "wet bubbles"
                        if Math.random() > 0.8 {
                            colorToUse = "#334155";
                            charToDraw = ".";
                        }
                    }

                    if charToDraw != "" {
                        ctx.fillStyle = colorToUse;
                        ctx.fillText(charToDraw, c * fontSize, r * fontSize);
                    }

                    r += 1;
                }
                c += 1;
            }

            time += 1;
            requestAnimationFrame(draw);
        } animationId = requestAnimationFrame(draw);return lambda -> None { cancelAnimationFrame(
            animationId
        );window.removeEventListener("resize", resize);}; },
        []
    );

    return
        <canvas
            ref={canvasRef}
            className="fixed top-0 left-0 w-full h-full pointer-events-none z-0 opacity-50"
        />;
}
