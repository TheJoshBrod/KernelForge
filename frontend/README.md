# CGinS Frontend

A modern, agentic user interface built with **Jac** and **React** for interacting with the PyTorch Model Optimizer.
This GUI allows users to manage projects, upload models/weights, view optimization graphs, and monitor system resources.

## Features

- **Project Management**: Create, list, and search optimization projects.
- **Visualizations**: Interactive graphs showing kernel optimization nodes and their performance lineage.
- **System Monitoring**: Real-time display of GPU specifications and availability.
- **Workflow**: Upload full models or individual weight files for optimization.

## Project Structure

```
frontend/
├── jac.toml              # Project configuration
├── main.jac              # Main application entry & walkers
├── ConfigContext.cl.jac  # Config context provider
├── Home.cl.jac           # Home page
├── NewProject.cl.jac     # Project upload page
├── OpDetails.cl.jac      # Op details page
├── Project.cl.jac        # Project page
├── Settings.cl.jac       # Settings page
├── SplashScreen.cl.jac   # Splash screen
├── Button.cl.jac         # Shared UI component
├── backdrops/            # Animated background components
├── assets/               # Static assets (Icons, Logos)
└── styles.css            # Global styles (Tailwind utilities)
```

## Getting Started

1.  **Install Dependencies**:
    Ensure you have `jaclang` and the necessary plugins installed.

    ```bash
    pip install jaseci
    ```

    Install frontend npm packages:

    ```bash
    cd frontend
    jac install
    ```

2.  **Run the Application**:
    Start the development server from the `frontend` directory:

    ```bash
    jac start main.jac
    ```

    The application will launch in your default browser (usually at `http://localhost:8000`).

## Development

-   **Components**: UI components are written in `.cl.jac` files, mixing Python-like Jac syntax with React JSX.
-   **Walkers**: Backend logic is implemented as Jac walkers (e.g., `GetProjects`, `GetSystemInfo`) in `main.jac`.
-   **Styling**: Uses Tailwind CSS classes directly in JSX `className` attributes.
