import dagre from 'dagre';

export function createGraph() {
    return new dagre.graphlib.Graph();
}

export function layoutGraph(g) {
    dagre.layout(g);
}
