"""Game of Life Background Component"""

import from react { useEffect, useRef }

def:pub GameOfLifeBackground(className: str = "") -> Any {
    canvasRef = useRef(None);
    desktop_build = window["__KFORGE_DESKTOP_BUILD__"]
    if "__KFORGE_DESKTOP_BUILD__" in window and window["__KFORGE_DESKTOP_BUILD__"]
    else None;
    isDesktop = bool(
        desktop_build and "desktop" in desktop_build and desktop_build["desktop"]
    );

    baseClass = "app-life-canvas";
    if className {
        baseClass = baseClass + f" {className}";
    }

    if isDesktop {
        return
            <div className={baseClass + " app-backdrop-static app-backdrop-static--grid"}/>;
    }

    useEffect(
        lambda -> Any { canvas = canvasRef.current;if not canvas {
            return;
        }ctx = canvas.getContext("2d");
        # Configuration
        cellSize = 10;deadColor = "rgba(0, 0, 0, 0)";aliveColor = "rgba(134, 230, 169, 0.8)";frameCount = 0;tickInterval = 3;cols = 0;rows = 0;grid = [];animationId = 0;active = True;
        # Track reset state to prevent multiple timers
        resetPending = False;def createGrid() -> list {
            newGrid = [];
            r = 0;
            while r < rows {
                c = 0;
                row = [];
                while c < cols {
                    # Random noise: 12% chance of starting alive
                    row.push(1 if Math.random() > 0.88 else 0);
                    c += 1;
                }
                newGrid.push(row);
                r += 1;
            }
            return newGrid;
        } def resize() -> None {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            cols = Math.floor(canvas.width / cellSize);
            rows = Math.floor(canvas.height / cellSize);
            grid = createGrid();
            # Reset pending state on resize/restart
            nonlocal resetPending ;
            resetPending = False;
        } resize();window.addEventListener("resize", resize);def draw(
            currentGrid: list
        ) -> None {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = aliveColor;

            r = 0;
            while r < rows {
                c = 0;
                while c < cols {
                    if currentGrid[r][c] == 1 {
                        ctx.fillRect(
                            c * cellSize, r * cellSize, cellSize - 1, cellSize - 1
                        );
                    }
                    c += 1;
                }
                r += 1;
            }
        } def countNeighbors(myGrid: list, x: int, y: int) -> int {
            sum = 0;
            i = -1;
            while i < 2 {
                j = -1;
                while j < 2 {
                    # Wrap around edges (toroidal)
                    col = (x + i + cols) % cols;
                    row = (y + j + rows) % rows;
                    sum += myGrid[row][col];
                    j += 1;
                }
                i += 1;
            }
            sum -= myGrid[y][x];
            return sum;
        } def update() -> None {
            nonlocal animationId ;
            nonlocal active ;
            nonlocal resetPending ;
            nonlocal frameCount ;
            if not active {
                return;
            }

            # If we are already waiting for a reset, just keep drawing current state (or do nothing)
            if resetPending {
                draw(grid);
                animationId = requestAnimationFrame(update);
                return;
            }

            frameCount += 1;
            if frameCount % tickInterval != 0 {
                draw(grid);
                animationId = requestAnimationFrame(update);
                return;
            }

            nextGrid = [];
            hasChanged = False;

            r = 0;
            while r < rows {
                c = 0;
                newRow = [];
                while c < cols {
                    state = grid[r][c];
                    neighbors = countNeighbors(grid, c, r);

                    newState = state;
                    if state == 0 and neighbors == 3 {
                        newState = 1;
                    } elif state == 1 and (neighbors < 2 or neighbors > 3) {
                        newState = 0;
                    }

                    if newState != state {
                        hasChanged = True;
                    }
                    newRow.push(newState);
                    c += 1;
                }
                nextGrid.push(newRow);
                r += 1;
            }

            if not hasChanged {
                # Stasis detected (stable or empty)
                resetPending = True;
                setTimeout(
                    lambda -> None { grid = createGrid();nonlocal resetPending ;resetPending = False;},
                    3000
                );
            } else {
                grid = nextGrid;
            }

            draw(grid);
            animationId = requestAnimationFrame(update);
        } animationId = requestAnimationFrame(update);return lambda -> None { nonlocal active ;active = False;cancelAnimationFrame(
            animationId
        );window.removeEventListener("resize", resize);}; },
        []
    );

    return
        <canvas ref={canvasRef} className={baseClass}/>;
}
