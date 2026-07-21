import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import defaultdict


def visualize_subgraph(graph, start_measure, end_measure, path=None):
    sub_nodes = [
        n for n, data in graph.nodes(data=True)
        if 'measure_num' in data
        and start_measure <= int(data['measure_num']) <= end_measure
    ]

    subgraph = graph.subgraph(sub_nodes).copy()

    if len(subgraph.nodes) == 0:
        print(f"No nodes found between measures {start_measure} and {end_measure}.")
        return

    # Use layer_index for horizontal placement through time
    for node in subgraph.nodes:
        subgraph.nodes[node]['subset'] = int(subgraph.nodes[node]['layer_index'])

    pos = nx.multipartite_layout(subgraph, subset_key='subset')

    plt.figure(figsize=(16, 6))

    # Determine which nodes and edges are on the path
    path_set = set(path) if path else set()
    path_edges = set()
    if path:
        for i in range(len(path) - 1):
            if path[i] in sub_nodes and path[i + 1] in sub_nodes:
                path_edges.add((path[i], path[i + 1]))

    # Node colors
    node_colors = ['#FF6B6B' if n in path_set else 'lightblue' for n in subgraph.nodes]

    nx.draw_networkx_nodes(subgraph, pos, node_size=300, node_color=node_colors)

    # Draw non-path edges
    other_edges = [e for e in subgraph.edges if e not in path_edges]
    nx.draw_networkx_edges(
        subgraph, pos,
        edgelist=other_edges,
        width=0.5, alpha=0.3, edge_color='gray'
    )

    # Draw path edges
    if path_edges:
        nx.draw_networkx_edges(
            subgraph, pos,
            edgelist=list(path_edges),
            width=3.0, alpha=1.0, edge_color='#FF6B6B'
        )

    # Node labels
    labels = {}
    for n in subgraph.nodes:
        s = graph.nodes[n]['staff'].split('_')[1]
        v = graph.nodes[n]['voice'].split('_')[1]
        u = f"{graph.nodes[n]['features']['mean_utility']:.2f}"
        labels[n] = f"s{s}v{v}\n{u}"

    nx.draw_networkx_labels(subgraph, pos, labels, font_size=7)

    # Edge labels with offset
    layer_pairs = defaultdict(list)
    for src, dst in subgraph.edges:
        src_layer = int(graph.nodes[src]['layer_index'])
        dst_layer = int(graph.nodes[dst]['layer_index'])
        layer_pairs[(src_layer, dst_layer)].append((src, dst))

    for pair, edges in layer_pairs.items():
        n_edges = len(edges)
        for idx, (src, dst) in enumerate(edges):
            spread = 0.03
            offset = (idx - (n_edges - 1) / 2) * spread

            src_pos = np.array(pos[src])
            dst_pos = np.array(pos[dst])
            mid = (src_pos + dst_pos) / 2

            direction = dst_pos - src_pos
            perp = np.array([-direction[1], direction[0]])
            norm = np.linalg.norm(perp)
            if norm > 0:
                perp = perp / norm

            label_pos = mid + offset * perp

            is_path_edge = (src, dst) in path_edges
            plt.annotate(
                f"{subgraph.edges[src, dst]['weight']:.2f}",
                xy=label_pos,
                fontsize=5,
                ha='center',
                va='center',
                color='#CC0000' if is_path_edge else 'black',
                fontweight='bold' if is_path_edge else 'normal'
            )

    plt.title(f"Measures {start_measure}–{end_measure}")
    plt.tight_layout()
    plt.show()