"""Game of Life Background Component"""

import from react { useEffect, useRef }

def:pub GameOfLifeBackground() -> any {
    canvasRef = useRef(None);

    useEffect(lambda -> None {
        canvas = canvasRef.current;
        if not canvas { return; }

        ctx = canvas.getContext("2d");
        
        # Configuration
        cellSize = 8;
        deadColor = "rgba(0, 0, 0, 0)"; 
        aliveColor = "rgba(39, 39, 42, 1.0)";
        
        cols = 0;
        rows = 0;
        grid = [];

        # Track reset state to prevent multiple timers
        resetPending = False;

        def createGrid() -> list {
            newGrid = [];
            r = 0;
            while r < rows {
                c = 0;
                row = [];
                while c < cols {
                    # Random noise: 15% chance of starting alive
                    row.push(1 if Math.random() > 0.85 else 0);
                    c += 1;
                }
                newGrid.push(row);
                r += 1;
            }
            return newGrid;
        }

        def resize() -> None {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            cols = Math.floor(canvas.width / cellSize);
            rows = Math.floor(canvas.height / cellSize);
            grid = createGrid();
            # Reset pending state on resize/restart
            nonlocal resetPending;
            resetPending = False;
        }

        resize();
        window.addEventListener("resize", resize);

        def draw(currentGrid: list) -> None {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = aliveColor;

            r = 0;
            while r < rows {
                c = 0;
                while c < cols {
                    if currentGrid[r][c] == 1 {
                        ctx.fillRect(c * cellSize, r * cellSize, cellSize - 1, cellSize - 1);
                    }
                    c += 1;
                }
                r += 1;
            }
        }

        def countNeighbors(myGrid: list, x: int, y: int) -> int {
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
        }

        def update() -> None {
            nonlocal resetPending;
            
            # If we are already waiting for a reset, just keep drawing current state (or do nothing)
            if resetPending {
                draw(grid);
                requestAnimationFrame(update);
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
                setTimeout(lambda -> None {
                    grid = createGrid();
                    nonlocal resetPending;
                    resetPending = False;
                }, 3000);
            } else {
                grid = nextGrid;
            }
            
            draw(grid);
            requestAnimationFrame(update);
        }

        animationId = requestAnimationFrame(update);

        return lambda -> None {
            cancelAnimationFrame(animationId);
            window.removeEventListener("resize", resize);
        };
    }, []);

    return <canvas
        ref={canvasRef}
        className="fixed top-0 left-0 w-full h-full pointer-events-none z-0"
    />;
}
