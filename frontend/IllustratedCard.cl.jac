"""Illustrated card component for landing page."""

def:pub IllustratedCard(
    title: str,
    text: str,
    icon: any,
    tag: str = ""
) -> any {
    return
        <div className="card illustrated-card">
            <div className="card-icon">
                {icon}
            </div>
            <h3 className="card-title">{title}</h3>
            <p className="card-text">{text}</p>
            {<span className="card-tag">{tag}</span> if tag else None}
        </div>;
}
