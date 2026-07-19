from collections import OrderedDict
import partitura as pt
import argparse
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from music21 import stream, note as m21_note, meter, tempo, key, metadata, instrument, pitch as m21, converter

from utils.renderGraphPath import extract_score_context, attach_render_payload, render_best_path

INFERENCE_DIR = 'outputs/inference/'
SCORE_DIR = 'reduction_scores/'
OUTPUT_DIR = 'outputs/reductions/'
RAW_DIR = 'outputs/raw_graph_reductions/'

# Split scores by voices

def split_voices(scorePath):

    score = pt.load_score(scorePath, force_note_ids='keep')
    result = OrderedDict()
    result['rests'] = OrderedDict()

    part = next(pt.score.iter_parts(score.parts))

    for obj in part.iter_all():
        if isinstance(obj, pt.score.Note):

            staff = obj.staff if obj.staff is not None else 1
            voice = obj.voice if obj.voice is not None else 1

            result.setdefault(f'staff_{staff}', OrderedDict()) \
                  .setdefault(f'voice_{voice}', []) \
                  .append(obj)

        elif isinstance(obj, pt.score.Rest):

            staff = obj.staff if obj.staff is not None else 1
            voice = obj.voice if obj.voice is not None else 1

            result["rests"].setdefault(f'staff_{staff}', OrderedDict()) \
                .setdefault(f'voice_{voice}', []) \
                .append(obj)

    return result, part

# Split voices into controlled horizontal slices

def split_into_candidates(score_dictionary, part, segments):
    strands = OrderedDict()

    def note_onset_q(n):
        if hasattr(part, 'quarter_map'):
            return float(np.asarray(part.quarter_map(n.start.t)).reshape(-1)[0])
        return float(n.start.t)

    for staff_key, voices in score_dictionary.items():
        if staff_key == 'rests':
            continue
    
        strands[staff_key] = OrderedDict()

        for voice_key, notes in voices.items():
            strands[staff_key][voice_key] = OrderedDict()

            for seg in segments:
                seg_notes = [
                    n for n in notes if seg['start_q'] <= note_onset_q(n) < seg['end_q']
                ]

                if seg_notes:
                    strands[staff_key][voice_key][seg['id']] = {
                        'segment': seg,
                        'notes': seg_notes
                    }

    return strands


def get_measure_ts(part, measure):
    try:
        ts = np.asarray(part.time_signature_map(measure.start.t)).reshape(-1)

        beats = int(ts[0])
        beat_type = int(ts[1])

    except Exception:
        beats = 4
        beat_type = 4
    
    return beats, beat_type


def build_slices(part, mode = 'measure', subdivisions = 2, window_ql = 1.0):
    measures = sorted(part.iter_all(pt.score.Measure), key=lambda m: m.start.t)
    segments = []
    seg_idx = 1

    def qmap(t):
        if hasattr(part, "quarter_map"):
            return float(np.asarray(part.quarter_map(t)).reshape(-1)[0])
        return float(t)

    for i, m in enumerate(measures):
        m_start_t = float(m.start.t)
        if i + 1 < len(measures):
            m_end_t = float(measures[i + 1].start.t)
        elif hasattr(m, "end") and m.end is not None:
            m_end_t = float(m.end.t)
        else:
            m_end_t = m_start_t

        m_start_q = qmap(m_start_t)
        m_end_q = qmap(m_end_t)
        m_len_q = m_end_q - m_start_q

        measure_num = i + 1

        if mode == 'measure':
            segments.append({
                'id': f'seg_{seg_idx}',
                'label': f'measure_{measure_num}',
                'measure_num': measure_num,
                'start_t': m_start_t,
                'end_t': m_end_t,
                'start_q': m_start_q,
                'end_q': m_end_q,
                'layer_index': seg_idx
            })
            seg_idx += 1

        elif mode == "beat":
            beats, beat_type = get_measure_ts(part, m)
            beat_len_q = 4.0 / beat_type

            for b in range(beats):
                start_q = m_start_q + b * beat_len_q
                end_q = min(start_q + beat_len_q, m_end_q)

                segments.append({
                    "id": f"seg_{seg_idx}",
                    "label": f"measure_{measure_num}_beat_{b + 1}",
                    "measure_num": measure_num,
                    "beat_index": b + 1,
                    "start_t": m_start_t,
                    "end_t": m_end_t,
                    "start_q": start_q,
                    "end_q": end_q,
                    "layer_index": seg_idx,
                })

                seg_idx += 1

        elif mode == "subbeat":

            beats, beat_type = get_measure_ts(part, m)
            beat_len_q = 4.0 / beat_type

            sub_len_q = beat_len_q / subdivisions
            n_sub = int(round(m_len_q / sub_len_q))

            for s in range(n_sub):
                start_q = m_start_q + s * sub_len_q
                end_q = min(start_q + sub_len_q, m_end_q)

                segments.append({
                    "id": f"seg_{seg_idx}",
                    "label": f"measure_{measure_num}_sub_{s + 1}",
                    "measure_num": measure_num,
                    "sub_index": s + 1,
                    "start_t": m_start_t,
                    "end_t": m_end_t,
                    "start_q": start_q,
                    "end_q": end_q,
                    "layer_index": seg_idx,
                })

                seg_idx += 1

        elif mode == 'fixed':
            start_q = m_start_q
            local_idx = 1
            while start_q < m_end_q - 1e-9:
                end_q = min(start_q + window_ql, m_end_q)

                segments.append({
                    'id': f'seg_{seg_idx}',
                    'label': f'measure_{measure_num}_win_{local_idx}',
                    'measure_num': measure_num,
                    'window_index': local_idx,
                    'start_t': m_start_t,
                    'end_t': m_end_t,
                    'start_q': start_q,
                    'end_q': end_q,
                    'layer_index': seg_idx
                })

                seg_idx += 1
                local_idx += 1
                start_q = end_q

    return segments

# Build DAG and node feature vector

def build_dag(strands):
    G = nx.DiGraph()
    layers = OrderedDict()

    for staff_key, voices in strands.items():
        for voice_key, segs in voices.items():
            for seg_id, payload in segs.items():
                seg = payload['segment']
                notes = payload['notes']

                node = {
                    'id': f'{staff_key}_{voice_key}_{seg_id}',
                    'staff': staff_key,
                    'voice': voice_key,
                    'segment_id': seg_id,
                    'segment_label': seg['label'],
                    'measure_num': seg['measure_num'],
                    'layer_index': seg['layer_index'],
                    'start_q': seg['start_q'],
                    'end_q': seg['end_q'],
                    'notes': notes
                }

                layers.setdefault(seg['layer_index'], []).append(node)

    layer_keys = sorted(layers.keys())

    for layer_key in layer_keys:
        for node in layers[layer_key]:
            G.add_node(node['id'], **node)

    for i in range(len(layer_keys) - 1):
        current_layer = layers[layer_keys[i]]
        next_layer = layers[layer_keys[i + 1]]

        for src in current_layer:
            for dst in next_layer:
                G.add_edge(src['id'], dst['id'], weight=1)

    return G


def build_feature_vector(node, inference_json):
    notes = node['notes']

    pitches = [n.midi_pitch for n in notes]
    durations = [n.duration_from_symbolic for n in notes]
    onsets = [n.start.t for n in notes]

    # Features included in the vector
    features = {
        'pitches': pitches,
        'mean_pitch': np.mean(pitches),
        'median_pitch': np.median(pitches),
        'pitch_range': max(pitches) - min(pitches),
        'highest_pitch': max(pitches),
        'lowest_pitch': min(pitches),
        'first_pitch': pitches[0],
        'last_pitch': pitches[-1],

        'note_count': len(notes),
        'durations': durations,
        'mean_duration': np.mean(durations),
        'total_duration': sum(durations),
        'duration_density': 1 - (np.mean(durations) / sum(durations)),

        'intervals': [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)],
        'mean_interval': np.mean([abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]) if len(pitches) > 1 else 0,

        'staff': int(node['staff'].split('_')[1]),
        'voice': int(node['voice'].split('_')[1])
    }

    # Merging in the structural inference results
    for n in notes:
        note_id = n.id
        if note_id in inference_json:
            entry = inference_json[note_id]
            if 'layer_scores' in entry:
                features.setdefault('utility', []).append(entry['layer_scores'][2]) # Hardcoded layer 2

    features['mean_utility'] = np.mean(features.get('utility', [0.0]))
    
    # utility_density = np.dot(features['utility'], features['durations']) #COME BACK
    

    return features

def attach_features(graph, inference_lookup):
    for node_id in graph.nodes:
        node = graph.nodes[node_id]
        features = build_feature_vector(node, inference_lookup)
        node['features'] = features

# Scoring

def normalize_interval(semitones, max_interval = 24):
    return min(semitones / max_interval, 1.0)

def edge(source_node, destination_node, args):
    utility = destination_node['features'].get('mean_utility', 0.0)
    utility_cost = 1.0 - utility    

    final_pitch = source_node['notes'][-1].midi_pitch
    first_pitch = destination_node['notes'][0].midi_pitch
    continuity_cost = normalize_interval(abs(final_pitch - first_pitch), 12)

    duration_density = destination_node['features'].get('duration_density', 0.0)

    intervals = destination_node['features'].get('intervals', 0.0)
    
    if intervals:
        contour_cost = normalize_interval(
            np.mean(np.abs(intervals)),
            12
        )
    else:
        contour_cost = 0.0

    return(
        args.utility * utility_cost + 
        args.contour * contour_cost + 
        args.continuity * continuity_cost
    )

def assign_edge_weights(graph, args):
    for source_id, destination_id in graph.edges:
        graph.edges[source_id, destination_id]['weight'] = edge(graph.nodes[source_id], graph.nodes[destination_id], args)

# Visualizing the graph
def visualize_subgraph(graph, start_measure, end_measure, path=None):
    sub_nodes = [
        n for n in graph.nodes
        if 'measure' in graph.nodes[n]
        and start_measure <= int(graph.nodes[n]['measure'].split('_')[1]) <= end_measure
    ]

    subgraph = graph.subgraph(sub_nodes)

    for node in subgraph.nodes:
        subgraph.nodes[node]['subset'] = int(subgraph.nodes[node]['measure'].split('_')[1])

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
    nx.draw_networkx_edges(subgraph, pos, edgelist=other_edges, width=0.5, alpha=0.3, edge_color='gray')

    # Draw path edges
    if path_edges:
        nx.draw_networkx_edges(subgraph, pos, edgelist=list(path_edges), width=3.0, alpha=1.0, edge_color='#FF6B6B')

    # Node labels
    labels = {}
    for n in subgraph.nodes:
        s = graph.nodes[n]['staff'].split('_')[1]
        v = graph.nodes[n]['voice'].split('_')[1]
        u = f"{graph.nodes[n]['features']['mean_utility']:.2f}"
        labels[n] = f"s{s}v{v}\n{u}"

    nx.draw_networkx_labels(subgraph, pos, labels, font_size=7)

    # Edge labels with offset
    from collections import defaultdict
    layer_pairs = defaultdict(list)
    for src, dst in subgraph.edges:
        src_layer = int(graph.nodes[src]['measure'].split('_')[1])
        dst_layer = int(graph.nodes[dst]['measure'].split('_')[1])
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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--piece', required = True)
    parser.add_argument('--utility', type = float, default = 0.7)
    parser.add_argument('--continuity', type = float, default = 0.2)
    parser.add_argument('--duration', type = float, default = 0.4)
    parser.add_argument('--contour', type = float, default = 0.5)
    parser.add_argument('--method', default = 'measure')
    args = parser.parse_args()

    input_score = SCORE_DIR + args.piece
    render_context = extract_score_context(input_score)
    mono_raw = RAW_DIR + args.piece
    mono_score = OUTPUT_DIR + args.piece
    results, part = split_voices(input_score)

    segments = build_slices(part, mode = args.method)
    candidates = split_into_candidates(results, part, segments)
    graph = build_dag(candidates)

    for node in graph.nodes:
        graph.nodes[node]['subset'] = graph.nodes[node]['layer_index']

    piece_name = args.piece.replace('.musicxml', '')

    with open(f'outputs/inference/{piece_name}.json') as f:
        inference_json = json.load(f)

    inference_lookup = {}
    for note in inference_json['notes']:
        inference_lookup[note['note_id']] = note

    attach_features(graph, inference_lookup)

    # print(graph.nodes['staff_1_voice_2_measure_89']['features']) # Checking one strand

    assign_edge_weights(graph, args)

    # starting_nodes = [n for n in graph.nodes if graph.nodes[n]['layer_index'] == 1]

    first_layer = min(
        graph.nodes[n]["layer_index"]
        for n in graph.nodes
    )

    starting_nodes = [
        n for n in graph.nodes
        if graph.nodes[n]["layer_index"] == first_layer
    ]
    starting_node = max(starting_nodes, key = lambda n: graph.nodes[n]['features']['mean_utility'])
    end_nodes = [n for n in graph.nodes if graph.out_degree(n) == 0]

    graph.add_node('sink')
    for n in end_nodes:
        graph.add_edge(n, 'sink', weight = 0)
    
    # Dijkstra
    path = nx.shortest_path(graph, starting_node, 'sink', weight = 'weight')
    cost = nx.path_weight(graph, path, weight = 'weight')
    path = path[:-1]

    # print(f"Cost: {cost:.15f}")
    # print("Path:")
    # for p in path:
    #     print("   ", p)

    attach_render_payload(graph, part, simultaneous = "highest")
    render_best_path(
        path,
        graph,
        mono_raw,
        render_context,
        input_score
    )

    # Subgraph visual
    # visualize_subgraph(graph, 1, 30, path)