"""Workbench Page"""

import from react { useEffect }
import from "react-router-dom" { useParams, useNavigate }
import from .OpSelector { OpSelector }
import from .KernelWorkbench { KernelWorkbench }

def:pub Workbench {
    params = useParams();
    projectName = params.projectName;
    opName = params.opName;
    navigate = useNavigate();

    if not projectName {
        return
            <div
                className="flex items-center justify-center h-screen bg-zinc-950 text-zinc-500"
            >
                No project selected.
            </div>;
    }

    if not opName {
        return
            <OpSelector projectName={projectName} />;
    }

    return
        <KernelWorkbench projectName={projectName} opName={opName} />;
}
