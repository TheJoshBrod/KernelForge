import dagre from 'dagre';

export function createGraph() {
    return new dagre.graphlib.Graph({ compound: true });
}

export function layoutGraph(g) {
    dagre.layout(g);
}
