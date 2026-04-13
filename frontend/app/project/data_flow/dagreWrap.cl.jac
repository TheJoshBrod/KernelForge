"""Thin wrapper around the dagre graph layout library."""

import from "dagre" { default as dagre }

def:pub createGraph -> Any {
    return Reflect.construct(dagre.graphlib.Graph, [{"compound": True}]);
}

def:pub layoutGraph(g: any) -> None {
    dagre.layout(g);
}
