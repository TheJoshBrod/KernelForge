"""Matrix Rain Background Component"""

import from react { useEffect, useRef }

def:pub MatrixBackground() -> Any {
    canvasRef = useRef(None);
    desktop_build = window["__KFORGE_DESKTOP_BUILD__"]
    if "__KFORGE_DESKTOP_BUILD__" in window and window["__KFORGE_DESKTOP_BUILD__"]
    else None;
    isDesktop = bool(
        desktop_build and "desktop" in desktop_build and desktop_build["desktop"]
    );

    if isDesktop {
        return
            <div className="app-life-canvas app-backdrop-static app-backdrop-static--matrix"/>;
    }

    useEffect(
        lambda -> Any { canvas = canvasRef.current;if not canvas {
            return;
        }ctx = canvas.getContext("2d");
        # Set canvas size to window size
        def resize() -> None {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        } resize();window.addEventListener("resize", resize);
        # Characters to use (Hiragana + Latin)
        chars = "アァカサタナハマヤャラワガザダバパイィキシチニヒミリヰギジヂビピウゥクスツヌフムユュルグズブヅプエェケセテネヘメレヱゲゼデベペオォコソトノホモヨョロヲゴゾドボポヴッン0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";charArray = chars.split(
            ""
        );fontSize = 14;columns = Math.floor(canvas.width / fontSize);
        # Array of drops - one per column
        drops = [];col_idx = 0;while col_idx < columns {
            drops.push(1);
            col_idx += 1;
        }def draw() -> None {
            # Black BG for the trail effect (opacity controls trail length)
            ctx.fillStyle = "rgba(0, 0, 0, 0.05)";
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.fillStyle = "#0F0";  # Green text
            ctx.font = f"{fontSize}px monospace";

            draw_idx = 0;
            while draw_idx < drops.length {
                text = charArray[Math.floor(Math.random() * charArray.length)];

                # Soft green with varying opacity for depth
                opacity = Math.random() * 0.5 + 0.1;
                ctx.fillStyle = f"rgba(20, 255, 50, {opacity})";

                ctx.fillText(text, draw_idx * fontSize, drops[draw_idx] * fontSize);

                # Sending the drop back to the top randomly after it has crossed the screen
                # Adding a randomness to the reset to make the drops scattered on both Y and Time axes
                if drops[draw_idx] * fontSize > canvas.height and Math.random() > 0.975 {
                    drops[draw_idx] = 0;
                }

                # Incrementing Y coordinate
                drops[draw_idx] += 1;
                draw_idx += 1;
            }
        } interval = setInterval(draw, 33);return lambda -> None { clearInterval(
            interval
        );window.removeEventListener("resize", resize);}; },
        []
    );

    return
        <canvas
            ref={canvasRef}
            className="fixed top-0 left-0 w-full h-full pointer-events-none z-0 opacity-70"
        />;
}
