"""Workbench Page"""

import from react { useEffect }
import from "react-router-dom" { useParams, useNavigate }
import from .OpSelector { OpSelector }
import from .OperatorWorkbenchView { OperatorWorkbenchView }

def:pub OperatorWorkbench -> Any {
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
            <OpSelector projectName={projectName}/>;
    }

    return
        <OperatorWorkbenchView projectName={projectName} opName={opName}/>;
}
