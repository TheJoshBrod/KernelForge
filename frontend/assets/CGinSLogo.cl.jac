"""CGinS Logo Component"""

import from "./CGinS-logo.svg" { default as cginsLogo }

def:pub CGinSLogo(className: str = "") -> any {
    return <img src={cginsLogo} alt="CGinS Logo" className={className} />;
}
