"""GlowButton component with animated ring hover."""

def:pub GlowButton(
    label: str,
    onClick: any,
    variant: str = "primary",
    disabled: bool = False,
    className: str = ""
) -> any {
    classes = "glow-button " + variant + (" is-disabled" if disabled else "") + (
        " " + className if className else ""
    );

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
