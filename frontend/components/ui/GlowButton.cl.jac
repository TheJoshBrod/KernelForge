"""GlowButton component with animated ring hover."""

def:pub GlowButton(
    label: str,
    onClick: any,
    variant: str = "primary",
    disabled: bool = False,
    className: str = ""
) -> Any {
    classes = f"glow-button {variant}";
    if disabled {
        classes += " is-disabled";
    }
    if className {
        classes += f" {className}";
    }

    return
        <button
            className={classes}
            onClick={onClick}
            disabled={disabled}
            type="button"
        >
            <span className="glow-button__label">
                {label}
            </span>
        </button>;
}
