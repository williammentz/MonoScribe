from utils.renderGraphPath import render_interwoven_primary_secondary
from collections import OrderedDict
import partitura as pt
import argparse
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from music21 import stream, note as m21_note, meter, tempo, key, metadata, instrument, pitch as m21, converter

from utils.renderGraphPath import extract_score_context, attach_render_payload, render_best_path, render_interwoven_primary_secondary, restrict_path_to_core, restore_path_notes
from utils.visualizeGraph import visualize_subgraph
from pprint import pprint

INFERENCE_DIR = 'outputs/inference/'
SCORE_DIR = 'outputs/clean_scores/'
OUTPUT_DIR = 'outputs/reductions/'
INFERENCE_DIR = 'outputs/graph_selections/'
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


def build_slices(part, mode = 'measure', subdivisions = 2, offset = 0.25):
    measures = sorted(part.iter_all(pt.score.Measure), key=lambda m: m.start.t)
    segments = []
    seg_idx = 1

    if offset < 0:
        raise ValueError(f"Entered offset must be nonnegative; received {offset}")

    def qmap(t):
        if hasattr(part, "quarter_map"):
            return float(np.asarray(part.quarter_map(t)).reshape(-1)[0])
        return float(t)
    
    if not measures:
        return segments

    last_measure = measures[-1]

    if hasattr(last_measure, 'end') and last_measure.end is not None:
        score_end_q = qmap(float(last_measure.end.t))
    else:
        score_end_q = qmap(float(last_measure.start.t))

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

        measure_num = getattr(m, "number", None)
        if measure_num is None:
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

        elif mode == 'measure_offset':
            segment_end_q = min(m_end_q + offset, score_end_q)

            segments.append({
                'id': f'seg_{seg_idx}',
                'label': f'measure_{measure_num}_offset{offset:g}',
                'measure_num': measure_num,
                'start_t': m_start_t,
                'end_t': m_end_t,
                'start_q': m_start_q,
                'measure_end_q': m_end_q,
                'end_q': segment_end_q,
                'offset': segment_end_q - m_end_q,
                'requested_offset': offset,
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

        else:
            raise ValueError(f'Unsupported horizontal slicing method: {mode}')

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
                    'measure_end_q': seg.get('measure_end_q', seg['end_q']),
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
                features.setdefault('utility', []).append(entry['layer_scores'][2]) # Hardcoded layer 3

    features['mean_utility'] = np.mean(features.get('utility', [0.0]))
    features['onset_density'] = features['note_count'] / (node['end_q'] - node['start_q'])
    
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

def normalize_density(graph):
    densities = [graph.nodes[n]['features']['onset_density'] for n in graph.nodes]

    density_min = min(densities)
    density_max = max(densities)

    for n in graph.nodes:
        density = graph.nodes[n]['features']['onset_density']

        if density_max == density_min:
            normalized_density = 0.0
        
        else:
            normalized_density = (density - density_min) / (density_max - density_min)

        graph.nodes[n]['features']['normalized_onset_density'] = normalized_density


def edge(source_node, destination_node, args):
    utility = destination_node['features'].get('mean_utility', 0.0)
    utility_cost = 1.0 - utility    

    final_pitch = source_node['notes'][-1].midi_pitch
    first_pitch = destination_node['notes'][0].midi_pitch
    continuity_cost = normalize_interval(abs(final_pitch - first_pitch), 12)

    # Min-max normalized duration density
    onset_density = destination_node['features']['normalized_onset_density']
    density_cost = 1.0 - onset_density

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
        args.continuity * continuity_cost +
        args.density * density_cost
    )

def assign_edge_weights(graph, args):
    for source_id, destination_id in graph.edges:
        graph.edges[source_id, destination_id]['weight'] = edge(graph.nodes[source_id], graph.nodes[destination_id], args)

# Select primary/secondary nodes for each time layer in the optimized graph

def build_layer_index(graph):
    layers = OrderedDict()

    for node_id, data in graph.nodes(data = True):
        if node_id == 'sink':
            continue

        layer = data.get('layer_index')
        if layer is None:
            continue

        layers.setdefault(layer, []).append(node_id)

    for layer in layers:
        layers[layer] = sorted(layers[layer])

    return layers

def extract_pairs(graph, path, sink = 'sink'):
    if not path:
        return []

    real_path = [n for n in path if n != sink]

    if not real_path:
        return []

    layers = build_layer_index(graph)
    rows = []

    for i, primary_node in enumerate(real_path):
        primary_layer = graph.nodes[primary_node]['layer_index']
        layer_nodes = layers.get(primary_layer, [])

        alternatives = [n for n in layer_nodes if n != primary_node]


        # i == 0: Starting node logic (matches graph optimizer)
        if i == 0:
            ranked = sorted(
                layer_nodes,
                key=lambda n: (-graph.nodes[n]['features'].get('mean_utility', 0.0), n)
            )

            primary_rank_entry = None
            secondary_rank_entry = None

            for n in ranked:
                if n == primary_node:
                    primary_rank_entry = {
                        'node': n,
                        'score': graph.nodes[n]['features'].get('mean_utility', 0.0)
                    }
                    break

            for n in ranked:
                if n != primary_node:
                    secondary_rank_entry = {
                        'node': n,
                        'score': graph.nodes[n]['features'].get('mean_utility', 0.0)
                    }
                    break

            rows.append({
                'layer_index': primary_layer,
                'measure_num': graph.nodes[primary_node].get('measure_num'),
                'segment_label': graph.nodes[primary_node].get('segment_label'),

                'primary_node': primary_node,
                'primary_score_type': 'start_utility',
                'primary_score': primary_rank_entry['score'] if primary_rank_entry else None,

                'secondary_node': secondary_rank_entry['node'] if secondary_rank_entry else None,
                'secondary_score_type': 'start_utility',
                'secondary_score': secondary_rank_entry['score'] if secondary_rank_entry else None,

                'delta_to_secondary': (
                    (primary_rank_entry['score'] - secondary_rank_entry['score'])
                    if primary_rank_entry and secondary_rank_entry else None
                )
            })
            continue

        # Rest of graph
        prev_primary = real_path[i - 1]
        ranked = []

        for alt in layer_nodes:
            if not graph.has_edge(prev_primary, alt):
                continue

            incoming_cost = graph.edges[prev_primary, alt]['weight']
            ranked.append({
                'node': alt,
                'incoming_cost': incoming_cost
            })

        ranked.sort(key=lambda x: (x['incoming_cost'], x['node']))

        primary_rank_entry = None
        secondary_rank_entry = None

        for entry in ranked:
            if entry['node'] == primary_node:
                primary_rank_entry = entry
                break

        for entry in ranked:
            if entry['node'] != primary_node:
                secondary_rank_entry = entry
                break

        rows.append({
            'layer_index': primary_layer,
            'measure_num': graph.nodes[primary_node].get('measure_num'),
            'segment_label': graph.nodes[primary_node].get('segment_label'),

            'previous_primary': prev_primary,

            'primary_node': primary_node,
            'primary_score_type': 'incoming_edge',
            'primary_score': primary_rank_entry['incoming_cost'] if primary_rank_entry else None,

            'secondary_node': secondary_rank_entry['node'] if secondary_rank_entry else None,
            'secondary_score_type': 'incoming_edge',
            'secondary_score': secondary_rank_entry['incoming_cost'] if secondary_rank_entry else None,

            'delta_to_secondary': (
                (secondary_rank_entry['incoming_cost'] - primary_rank_entry['incoming_cost'])
                if primary_rank_entry and secondary_rank_entry else None
            )
        })

    return rows

def save_primary_secondary_by_layer(rows, out_path):
    with open(out_path, 'w') as f:
        json.dump(rows, f, indent=2)

def save_second_choice_transitions(rows, out_path):
    with open(out_path, 'w') as f:
        json.dump(rows, f, indent=2)


# Main Reducer
def graph_reducer(args):
    input_score = SCORE_DIR + args.piece
    mono_raw = RAW_DIR + args.piece
    mono_score = OUTPUT_DIR + args.piece
    
    render_context = extract_score_context(input_score)
    results, part = split_voices(input_score)

    segments = build_slices(part, mode = args.method, offset = args.offset)
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
    normalize_density(graph)
    print_segment_features(graph, 'seg_3')
    assign_edge_weights(graph, args)

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
    
    # For debugging:

    # print(f"Cost: {cost:.15f}")
    # print("Path:")
    # for p in path:
    #     print("   ", p)


    # Range Check for features['onset_density']
    # for node_id, data in graph.nodes(data=True):
    #     if 'features' not in data:
    #         continue

    #     f = data['features']

    #     print(
    #         node_id,
    #         "notes:", f['note_count'],
    #         "duration:", f['total_duration'],
    #         "density:", f['onset_density'],
    #         'normalized', f['normalized_onset_density'],
    #         "segment:", data['end_q'] - data['start_q']
    #     )

    # # Extract secondary choices
    inference_json = INFERENCE_DIR + args.piece.replace('.musicxml', '_transitions.json')

    primary_secondary_pairs = extract_pairs(graph, path, 'sink')
    inference_json = INFERENCE_DIR + args.piece.replace('.musicxml', '_primary_secondary_nodes.json')
    save_primary_secondary_by_layer(primary_secondary_pairs, inference_json)

    # # Remove 'sink' node
    path = path[:-1]

    # visualize_subgraph(graph, 1, 20, path)

    # # Raw graph monophonic rendering
    # attach_render_payload(graph, part, simultaneous = "highest")
    # render_best_path(
    #     path,
    #     graph,
    #     mono_raw,
    #     render_context,
    #     input_score
    # )

    original_path_notes = restrict_path_to_core(graph, path, part)

    try:
        attach_render_payload(graph, part, simultaneous='highest')
        render_best_path(path, graph, mono_raw, render_context, input_score)

    finally:
        restore_path_notes(graph, original_path_notes)


    # # Interweaving primary and secondary nodes
    # interwoven_out = OUTPUT_DIR + args.piece.replace('.musicxml', '_interwoven.musicxml')
    # render_interwoven_primary_secondary(primary_secondary_pairs, graph, input_score, interwoven_out, render_context)

# Debugging
def print_segment_features(graph, segment_identifier):
    matches = []

    for node_id, node_data in graph.nodes(data=True):
        if (
            node_data.get('segment_id') == segment_identifier
            or node_data.get('segment_label') == segment_identifier
        ):
            matches.append((node_id, node_data))

    if not matches:
        print(f'No nodes found for segment: {segment_identifier}')
        return

    print(f'\nFeature vectors for segment: {segment_identifier}')

    for node_id, node_data in matches:
        print('\n' + '=' * 70)
        print(f'Node:    {node_id}')
        print(f'Staff:   {node_data["staff"]}')
        print(f'Voice:   {node_data["voice"]}')
        print(f'Segment: {node_data["segment_label"]}')
        pprint(node_data['features'], sort_dicts=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--piece', required = True)
    parser.add_argument('--utility', type = float, default = 0.7)
    parser.add_argument('--continuity', type = float, default = 0.2) # Higher <=> More pitch continuity between nodes
    parser.add_argument('--density', type = float, default = 0.4) # Higher <=> Denser
    parser.add_argument('--contour', type = float, default = 0.5) # Higher <=> More contour within each node
    parser.add_argument('--method', default = 'measure') # Horizontal slicing method, default: method
    parser.add_argument('--offset', type = float, default = 0.25)

    args = parser.parse_args()

    graph_reducer(args)