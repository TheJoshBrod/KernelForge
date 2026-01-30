# frontend

A Jac client-side application with React + router support.

## Project Structure

```
frontend/
|-- jac.toml              # Project configuration
|-- main.jac              # Main application entry
|-- ConfigContext.cl.jac  # Config context provider
|-- Home.cl.jac           # Home page
|-- NewProject.cl.jac     # Project upload page
|-- OpDetails.cl.jac      # Op details page
|-- Project.cl.jac        # Project page
|-- Settings.cl.jac       # Settings page
|-- Button.cl.jac         # Example component
|-- styles.css            # Global styles
|-- config.json           # Local config storage
```

## Getting Started

Start the development server:

```bash
jac start main.jac
```

## Components

Create Jac components as `.cl.jac` files and import them:

```jac
cl import from .Button { Button }
```

## Adding Dependencies

Add npm packages with the --cl flag:

```bash
jac add --cl react-router-dom
```
