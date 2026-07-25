from collections import OrderedDict
import numpy as np
from copy import deepcopy
from fractions import Fraction
from music21 import stream, note as m21_note, meter, tempo, key, metadata, instrument, pitch as m21, converter, duration as m21_duration
import partitura as pt

EPSILON = 1e-7

def rational_value(x, max_denominator=96, allow_zero=True):
    if x is None:
        return None
    q = Fraction(float(x)).limit_denominator(max_denominator)
    if allow_zero:
        if q < 0:
            return None
    else:
        if q <= 0:
            return None
    return q

def snap_value_float(x, max_denominator=96, allow_zero=True):
    q = rational_value(x, max_denominator=max_denominator, allow_zero=allow_zero)
    return float(q) if q is not None else None

def first_measure_q_origin(part):
    measures = sorted(part.iter_all(pt.score.Measure), key=lambda m: m.start.t)
    if not measures:
        return 0.0

    first_measure = measures[0]

    try:
        if hasattr(part, "quarter_map"):
            return float(scalar(part.quarter_map(first_measure.start.t)))
    except Exception:
        pass

    return float(first_measure.start.t)

# Build Selected (Raw) Reduction

def extract_score_context(score_path):
    src = converter.parse(score_path)

    context = {
        "title": None,
        "composer": None,
        "time_signatures": [],
        "tempos": [],
        "key_signatures": [],
    }

    if src.metadata is not None:
        context["title"] = src.metadata.title
        context["composer"] = src.metadata.composer

    if len(src.parts) == 0:
        return context

    src_part = src.parts[0]

    # Deduplicate by (offset, value)
    seen_ts = set()
    seen_tempo = set()
    seen_ks = set()

    for ts in src_part.recurse().getElementsByClass(meter.TimeSignature):
        off = float(ts.getOffsetInHierarchy(src_part))
        sig = ts.ratioString
        key_id = (off, sig)
        if key_id not in seen_ts:
            seen_ts.add(key_id)
            context["time_signatures"].append((off, deepcopy(ts)))

    for mm in src_part.recurse().getElementsByClass(tempo.MetronomeMark):
        off = float(mm.getOffsetInHierarchy(src_part))
        num = float(mm.number) if mm.number is not None else None
        key_id = (off, num, str(mm))
        if key_id not in seen_tempo:
            seen_tempo.add(key_id)
            context["tempos"].append((off, deepcopy(mm)))

    for ks in src_part.recurse().getElementsByClass(key.KeySignature):
        off = float(ks.getOffsetInHierarchy(src_part))
        fifths = int(ks.sharps) if ks.sharps is not None else 0
        key_id = (off, fifths)
        if key_id not in seen_ks:
            seen_ks.add(key_id)
            context["key_signatures"].append((off, deepcopy(ks)))

    context["time_signatures"].sort(key=lambda x: x[0])
    context["tempos"].sort(key=lambda x: x[0])
    context["key_signatures"].sort(key=lambda x: x[0])

    return context

def apply_score_context_to_part(part_stream, context):
    if not context:
        return

    for off, ts in context.get("time_signatures", []):
        part_stream.insert(float(off), deepcopy(ts))

    for off, ks in context.get("key_signatures", []):
        part_stream.insert(float(off), deepcopy(ks))

    for off, mm in context.get("tempos", []):
        part_stream.insert(float(off), deepcopy(mm))


def scalar(x):
    arr = np.asarray(x)
    if arr.size == 0:
        return 0.0
    return float(arr.reshape(-1)[0])


def symbolic_to_quarter_length(sym):
    if sym is None:
        return None

    if callable(sym):
        sym = sym()

    if isinstance(sym, (int, float)):
        return float(sym)

    if isinstance(sym, tuple) and len(sym) == 2:
        num, den = sym
        return float(num) / float(den) * 4.0

    if isinstance(sym, dict):
        base_map = {
            "whole": 4.0,
            "half": 2.0,
            "quarter": 1.0,
            "eighth": 0.5,
            "16th": 0.25,
            "32nd": 0.125,
            "64th": 0.0625,
            "128th": 0.03125,
        }
        base = base_map.get(sym.get("type"), None)
        if base is None:
            return None

        dots = int(sym.get("dots", 0) or 0)
        ql = base
        add = base / 2.0
        for _ in range(dots):
            ql += add
            add /= 2.0
        return ql

    return None


def note_onset_quarter(n, part):
    try:
        if hasattr(part, "quarter_map"):
            return float(scalar(part.quarter_map(n.start.t)))
    except Exception:
        pass

    return None


def note_duration_quarter(n, part):
    try:
        if hasattr(part, "quarter_map") and hasattr(n, "end") and n.end is not None:
            start_q = float(scalar(part.quarter_map(n.start.t)))
            end_q = float(scalar(part.quarter_map(n.end.t)))
            dur = end_q - start_q
            if dur > 0:
                return dur
    except Exception:
        pass

    if hasattr(n, "duration_from_symbolic"):
        ql = symbolic_to_quarter_length(n.duration_from_symbolic)
        if ql is not None and ql > 0:
            return float(ql)

    if hasattr(n, "symbolic_duration"):
        ql = symbolic_to_quarter_length(n.symbolic_duration)
        if ql is not None and ql > 0:
            return float(ql)

    return None


def build_render_payload(notes, part, simultaneous="highest"):
    notes_sorted = sorted(
        notes,
        key=lambda n: (n.start.t, getattr(n, "midi_pitch", -999))
    )

    q_origin = first_measure_q_origin(part)

    render_events = []
    onset_buckets = OrderedDict()
    note_ids = []

    for n in notes_sorted:
        onset_q = note_onset_quarter(n, part)
        dur_q = note_duration_quarter(n, part)

        if onset_q is None or dur_q is None or dur_q <= 0:
            continue

        onset_q = float(onset_q) - q_origin
        dur_q = float(dur_q)

        if abs(onset_q) <= EPSILON:
            onset_q = 0.0

        ev = {
            "note_id": getattr(n, "id", None),
            "pitch": int(n.midi_pitch),
            "onset": onset_q,
            "duration": dur_q,
            "onset_t": float(n.start.t),
            "end_t": float(n.end.t) if hasattr(n, "end") and n.end is not None else None,
        }

        render_events.append(ev)
        note_ids.append(ev["note_id"])
        onset_buckets.setdefault(round(ev["onset"], 8), []).append(ev)

    primary_events = []
    max_simultaneity = 0

    for _, bucket in onset_buckets.items():
        max_simultaneity = max(max_simultaneity, len(bucket))

        if simultaneous == "lowest":
            chosen = min(bucket, key=lambda e: e["pitch"])
        elif simultaneous == "first":
            chosen = bucket[0]
        else:
            chosen = max(bucket, key=lambda e: e["pitch"])

        primary_events.append({
            "note_id": chosen["note_id"],
            "pitch": chosen["pitch"],
            "onset": chosen["onset"],
            "duration": chosen["duration"],
        })

    if render_events:
        start_q = min(ev["onset"] for ev in render_events)
        end_q = max(ev["onset"] + ev["duration"] for ev in render_events)
    else:
        start_q = None
        end_q = None

    representative = primary_events[0] if primary_events else None

    return {
        "note_ids": note_ids,
        "start_q": start_q,
        "end_q": end_q,
        "duration_q": (end_q - start_q) if start_q is not None and end_q is not None else None,
        "render_events": render_events,
        "primary_events": primary_events,
        "representative_pitch": representative["pitch"] if representative else None,
        "representative_onset": representative["onset"] if representative else None,
        "representative_duration": representative["duration"] if representative else None,
        "max_simultaneity": max_simultaneity,
        "is_monophonic": max_simultaneity <= 1,
    }


def attach_render_payload(graph, part, simultaneous="highest"):
    for node_id in graph.nodes:
        node = graph.nodes[node_id]
        if "notes" not in node:
            continue
        node["render"] = build_render_payload(
            node["notes"],
            part,
            simultaneous=simultaneous
        )


def path_to_render_events(path, graph, truncate_overlaps = False):
    path_events = []

    for node_id in path:
        node = graph.nodes[node_id]
        render_payload = node.get("render", None)
        if render_payload is None:
            raise ValueError(
                f"Node {node_id} has no render payload. "
                "Call attach_render_payload(graph, part) first."
            )

        for ev in render_payload.get("primary_events", []):
            path_events.append({
                "pitch": int(ev["pitch"]),
                "onset": float(ev["onset"]),
                "duration": float(ev["duration"]),
                'node_id': node_id
            })

    if not path_events:
        return []

    path_events.sort(key=lambda x: (x["onset"], x["pitch"]))

    if not path_events:
        return []

    path_events.sort(key=lambda x: (x["onset"], x["pitch"]))

    return normalize_path_events(path_events, truncate_overlaps = truncate_overlaps)


def render_best_path(path, graph, out_path, render_context, source_score_path, truncate_overlaps = False):
    path_events = path_to_render_events(path, graph, truncate_overlaps = truncate_overlaps)

    if not path_events:
        raise ValueError("No renderable events found in the selected path.")

    return render_raw_path(
        path_events,
        source_score_path = source_score_path,
        out_path = out_path,
        title = render_context.get('title'),
        composer = render_context.get('composer'),
    )

def normalize_path_events(path_events, merge_adjacent_same_pitch = False, truncate_overlaps = False):
    events = sorted(path_events, key=lambda x: (x["onset"], x["pitch"]))
    cleaned = []

    for ev in events:
        onset = float(ev["onset"])
        dur = float(ev["duration"])
        p = ev["pitch"]

        if dur <= 0:
            continue

        if not cleaned:
            cleaned.append({"pitch": p, "onset": onset, "duration": dur})
            continue

        prev = cleaned[-1]
        prev_end = prev["onset"] + prev["duration"]

        # Current event begins before previous ends
        if onset < prev_end - EPSILON:
            if not truncate_overlaps:
                raise ValueError(
                    f"Overlapping notes detected: "
                    f"{prev} overlaps {ev}"
                )

            truncated_duration = onset - prev["onset"]

            if truncated_duration <= EPSILON:
                raise ValueError(f"Cannot truncate overlapping notes with identical onsets: {prev} overlaps {ev}")
            
            prev["duration"] = truncated_duration
            prev_end = onset

        if abs(onset - prev_end) <= EPSILON:
            onset = prev_end

        # Merge adjacent same-pitch notes if desired
        if merge_adjacent_same_pitch:
            prev_end = prev["onset"] + prev["duration"]
            if abs(onset - prev_end) < EPSILON and p == prev["pitch"]:
                prev["duration"] += dur
                continue

        cleaned.append({"pitch": p, "onset": onset, "duration": dur})

    return cleaned

def get_measure_by_number(part_stream, number):
    for m in part_stream.getElementsByClass(stream.Measure):
        if m.number == number:
            return m
    return None


def clone_measure_shell(measure):
    new_measure = stream.Measure(number=measure.number)

    if measure.leftBarline is not None:
        new_measure.leftBarline = deepcopy(measure.leftBarline)
    if measure.rightBarline is not None:
        new_measure.rightBarline = deepcopy(measure.rightBarline)

    for element in measure:
        if isinstance(element, stream.Voice):
            continue
        if isinstance(element, m21_note.GeneralNote):
            continue

        new_measure.insert(element.offset, deepcopy(element))

    return new_measure


def get_measure_length(measure):
    if measure.barDuration is not None:
        return float(measure.barDuration.quarterLength)

    ts = measure.getContextByClass(meter.TimeSignature)
    if ts is not None:
        return float(ts.barDuration.quarterLength)

    return 4.0


def tempo_marks(source_score, reduced_part):
    for mark in source_score.recurse().getElementsByClass(tempo.MetronomeMark):
        src_measure = mark.getContextByClass(stream.Measure)
        if src_measure is None:
            continue

        target_measure = get_measure_by_number(reduced_part, src_measure.number)
        if target_measure is None:
            continue

        local_offset = float(mark.getOffsetInHierarchy(src_measure))
        target_measure.insert(local_offset, deepcopy(mark))


def build_source_measure_table(src_part):
    measures = list(src_part.getElementsByClass(stream.Measure))
    table = []

    for m in measures:
        start = float(m.getOffsetInHierarchy(src_part))
        length = get_measure_length(m)
        end = start + length

        table.append({
            "number": m.number,
            "source_measure": m,
            "start": start,
            "end": end,
            "length": length,
        })

    return table


def render_raw_path(
    path_events,
    source_score_path,
    out_path="reduction.musicxml",
    title=None,
    composer=None,
    part_name="Reduction",
    inst=None,
):
    path_events = normalize_path_events(path_events, merge_adjacent_same_pitch = False)

    # Optional printout for debugging
    # for ev in path_events[:10]:
    #     print(ev)

    src_score = converter.parse(source_score_path)
    if len(src_score.parts) == 0:
        raise ValueError("Source score contains no parts.")

    src_part = src_score.parts[0]
    measures = list(src_part.getElementsByClass(stream.Measure))
    measure0 = next((m for m in measures if m.number == 0), None)
    measure1 = next((m for m in measures if m.number == 1), None)

    if measure0 is not None and measure1 is not None:
        source_measure0_has_notes = len(list(measure0.notes)) > 0
        if not source_measure0_has_notes:
            measure1_start = float(measure1.getOffsetInHierarchy(src_part))
            path_events = [
                ev for ev in path_events
                if float(ev["onset"]) >= measure1_start - EPSILON
            ]

    # m1 = list(src_part.getElementsByClass(stream.Measure))[0]

    # print("SOURCE MEASURE 1")
    # print("number:", m1.number)
    # print("offset in part:", float(m1.getOffsetInHierarchy(src_part)))
    # print("paddingLeft:", getattr(m1, "paddingLeft", None))
    # print("paddingRight:", getattr(m1, "paddingRight", None))
    # print("barDuration:", float(m1.barDuration.quarterLength) if m1.barDuration else None)

    # print("SOURCE MEASURE 1 NOTES/RESTS")
    # for x in m1.notesAndRests:
    #     print(type(x).__name__, "offset=", float(x.offset), "ql=", float(x.duration.quarterLength))

    # print("FIRST SOURCE NOTES (music21)")
    # for n in list(src_part.recurse().notes)[:10]:
    #     meas = n.getContextByClass(stream.Measure)
    #     print(
    #         n,
    #         "part_offset=", float(n.getOffsetInHierarchy(src_part)),
    #         "measure=", meas.number if meas else None,
    #         "measure_offset=", float(n.offset)
    #     )

    measure_table = build_source_measure_table(src_part)

    sc = stream.Score()
    sc.metadata = metadata.Metadata()
    sc.metadata.title = title
    if composer:
        sc.metadata.composer = composer

    reduced_part = stream.Part()
    reduced_part.id = part_name

    if inst is not None:
        reduced_part.insert(0, inst)
    else:
        reduced_part.insert(0, instrument.Instrument(partName=part_name))

    # Clone all source measure shells first
    target_measures = OrderedDict()
    for row in measure_table:
        cloned = clone_measure_shell(row["source_measure"])
        target_measures[row["number"]] = cloned
        reduced_part.append(cloned)

    # Insert each event into the appropriate source measure(s)
    for ev in path_events:
        start = snap_value_float(ev["onset"], allow_zero = True)
        dur = snap_value_float(ev["duration"], allow_zero = False)
        pitch_value = ev["pitch"]

        if start is None or dur is None or dur <= 0:
            continue

        end = snap_value_float(start + dur, allow_zero = True)
        cursor = start


        while cursor < end - EPSILON:
            containing = None

            for row in measure_table:
                if row["start"] <= cursor < row["end"] - EPSILON:
                    containing = row
                    break

            # Edge case: event lands exactly on final boundary
            if containing is None:
                for row in reversed(measure_table):
                    if abs(cursor - row["end"]) < EPSILON:
                        continue
                    if row["start"] <= cursor <= row["end"]:
                        containing = row
                        break

            if containing is None:
                break

            seg_end = snap_value_float(min(end, containing["end"]), allow_zero = True)
            seg_dur = snap_value_float(seg_end - cursor, allow_zero = False)

            if seg_end is None or seg_dur is None or seg_dur <= EPSILON:
                break

            n = m21_note.Note()
            if isinstance(pitch_value, int):
                n.pitch.midi = pitch_value
            else:
                n.pitch = m21.Pitch(pitch_value)

            n.duration = m21_duration.Duration(rational_value(seg_dur, allow_zero = False))

            local_offset = snap_value_float(max(0.0, cursor - containing["start"]), allow_zero = True)
            target_measures[containing["number"]].insert(local_offset, n)
            cursor = seg_end

    # Fill gaps with rests inside the already-correct measure shells
    # for m in reduced_part.getElementsByClass(stream.Measure):
    #     try:
    #         m.makeRests(fillGaps=True, inPlace=True)
    #     except Exception:
    #         pass

    tempo_marks(src_score, reduced_part)

    sc.insert(0, reduced_part)

    try:
        sc.makeTies(inPlace=True)
    except Exception:
        pass

    sc.write("musicxml", fp=out_path)
    print("Path optimization reduction written.")
    return sc

# Primary/Secondary node pairs from graph path
def build_measure_context_table(source_score_path):
    """
    Map measure number -> beat_q_len / measure_len using the source score.
    """
    src_score = converter.parse(source_score_path)
    if len(src_score.parts) == 0:
        raise ValueError("Source score contains no parts.")

    src_part = src_score.parts[0]
    table = {}

    for m in src_part.getElementsByClass(stream.Measure):
        ts = m.getContextByClass(meter.TimeSignature)

        if ts is not None:
            beat_q_len = float(ts.beatDuration.quarterLength)
            measure_len = float(ts.barDuration.quarterLength)
        else:
            beat_q_len = 1.0
            measure_len = float(m.barDuration.quarterLength) if m.barDuration is not None else 4.0

        table[m.number] = {
            'beat_q_len': beat_q_len,
            'measure_len': measure_len
        }

    return table


def clip_node_events_to_layer(node_id, graph, layer_start_q, layer_end_q, event_key):
    """
    Clip one node's render events to the current layer bounds.
    Returns absolute-time events.
    """
    if node_id is None:
        return []

    node = graph.nodes[node_id]
    render_payload = node.get('render', {})
    raw_events = render_payload.get(event_key, [])

    clipped = []

    for ev in raw_events:
        onset = float(ev['onset'])
        end = onset + float(ev['duration'])

        start = max(onset, layer_start_q)
        stop = min(end, layer_end_q)
        dur = stop - start

        if dur <= EPSILON:
            continue

        clipped.append({
            'pitch': int(ev['pitch']),
            'onset': float(start),
            'duration': float(dur),
        })

    clipped.sort(key=lambda x: (x['onset'], x['pitch']))
    return clipped


def choose_primary_protected_duration(primary_event, secondary_candidates, beat_q_len, span_len):
    """
    First-pass graph analog of the homophonic heuristic:

    - if there is secondary material and primary span is longer than the first
      secondary duration, protect only that shorter chunk
    - otherwise protect one beat
    """
    if span_len <= EPSILON:
        return 0.0

    if secondary_candidates:
        first_secondary_dur = float(secondary_candidates[0]['duration'])
        if span_len > first_secondary_dur:
            return min(span_len, first_secondary_dur)

    return min(span_len, beat_q_len)


def collapse_monophonic_with_priority(events, end_limit=None):
    """
    Resolve a mixed primary/secondary event list into one monophonic stream.

    Priority:
      primary before secondary at same onset
    """
    if not events:
        return []

    def priority(ev):
        return 0 if ev.get('source') == 'primary' else 1

    events = sorted(events, key=lambda e: (e['onset'], priority(e), e['pitch']))
    cleaned = []
    current_time = events[0]['onset']

    for i, ev in enumerate(events):
        start = max(float(ev['onset']), current_time)

        next_onset = float('inf')
        if i + 1 < len(events):
            next_onset = float(events[i + 1]['onset'])

        raw_end = float(ev['onset']) + float(ev['duration'])
        end = min(raw_end, next_onset)

        if end_limit is not None:
            end = min(end, end_limit)

        dur = end - start
        if dur <= EPSILON:
            continue

        cleaned.append({
            'pitch': int(ev['pitch']),
            'onset': float(start),
            'duration': float(dur),
            'source': ev.get('source')
        })

        current_time = start + dur

    return cleaned


def interweave_layer_events(primary_events, secondary_events, layer_start_q, layer_end_q, beat_q_len):
    """
    Layer-level primary/secondary interweaving.

    Logic analog to the reference homophonic reducer:
    - primary owns the onset
    - primary gets a protected duration
    - secondary fills after that protected duration within the primary span
    """
    if not primary_events:
        return []

    selected = []

    for i, p in enumerate(primary_events):
        p_onset = float(p['onset'])
        p_end = p_onset + float(p['duration'])

        next_primary_onset = layer_end_q
        if i + 1 < len(primary_events):
            next_primary_onset = float(primary_events[i + 1]['onset'])

        span_end = min(p_end, next_primary_onset, layer_end_q)
        span_len = span_end - p_onset

        if span_len <= EPSILON:
            continue

        # Secondary events that belong to this primary span
        candidates = [
            {
                'pitch': int(s['pitch']),
                'onset': float(s['onset']),
                'duration': min(float(s['duration']), span_end - float(s['onset']))
            }
            for s in secondary_events
            if p_onset <= float(s['onset']) < span_end
        ]

        candidates = [c for c in candidates if c['duration'] > EPSILON]
        candidates.sort(key=lambda x: (x['onset'], x['pitch']))

        protected = choose_primary_protected_duration(
            p,
            candidates,
            beat_q_len,
            span_len
        )

        if protected <= EPSILON:
            continue

        # Primary gets first claim
        selected.append({
            'pitch': int(p['pitch']),
            'onset': p_onset,
            'duration': protected,
            'source': 'primary'
        })

        fill_start = p_onset + protected
        fill_end = span_end

        for c in candidates:
            if float(c['onset']) < fill_start:
                continue

            clipped_end = min(float(c['onset']) + float(c['duration']), fill_end)
            clipped_dur = clipped_end - float(c['onset'])

            if clipped_dur <= EPSILON:
                continue

            selected.append({
                'pitch': int(c['pitch']),
                'onset': float(c['onset']),
                'duration': float(clipped_dur),
                'source': 'secondary'
            })

    return collapse_monophonic_with_priority(selected, end_limit=layer_end_q)


def build_interwoven_path_events(primary_secondary_rows, graph, source_score_path):
    """
    Build one monophonic event stream by interweaving primary/secondary nodes
    layer by layer.
    """
    measure_context = build_measure_context_table(source_score_path)
    path_events = []

    for row in primary_secondary_rows:
        primary_node = row['primary_node']
        secondary_node = row.get('secondary_node')

        node_data = graph.nodes[primary_node]
        layer_start_q = float(node_data['start_q'])
        layer_end_q = float(node_data['end_q'])
        measure_num = int(node_data['measure_num'])

        beat_q_len = measure_context.get(measure_num, {}).get('beat_q_len', 1.0)

        primary_events = clip_node_events_to_layer(
            primary_node,
            graph,
            layer_start_q,
            layer_end_q,
            event_key='primary_events'
        )

        secondary_events = clip_node_events_to_layer(
            secondary_node,
            graph,
            layer_start_q,
            layer_end_q,
            event_key='render_events'
        ) if secondary_node is not None else []

        layer_events = interweave_layer_events(
            primary_events,
            secondary_events,
            layer_start_q,
            layer_end_q,
            beat_q_len
        )

        path_events.extend(layer_events)

    # Strip source before final normalization if desired
    flat = [
        {
            'pitch': int(ev['pitch']),
            'onset': float(ev['onset']),
            'duration': float(ev['duration'])
        }
        for ev in path_events
    ]

    flat.sort(key=lambda x: (x['onset'], x['pitch']))
    return normalize_path_events(flat, merge_adjacent_same_pitch=False)

def render_interwoven_primary_secondary(
    primary_secondary_rows,
    graph,
    source_score_path,
    out_path,
    render_context
):
    path_events = build_interwoven_path_events(
        primary_secondary_rows,
        graph,
        source_score_path
    )

    if not path_events:
        raise ValueError("No interwoven events were produced.")

    return render_raw_path(
        path_events,
        source_score_path=source_score_path,
        out_path=out_path,
        title=render_context.get('title'),
        composer=render_context.get('composer'),
    )

def restrict_path_to_core(graph, path, part):
    original_notes = {}

    def note_onset_q(n):
        if hasattr(part, 'quarter_map'):
            return float(np.asarray(part.quarter_map(n.start.t)).reshape(-1)[0])
        return float(n.start.t)

    for node_id in path:
        if node_id == 'sink':
            continue
        
        node = graph.nodes[node_id]
        original_notes[node_id] = node['notes']

        core_start_q = node['start_q']
        core_end_q = node.get('measure_end_q', node['end_q'])

        node['notes'] = [n for n in node['notes'] if core_start_q <= note_onset_q(n) < core_end_q]

    return original_notes

def restore_path_notes(graph, original_notes):
    for node_id, notes in original_notes.items():
        graph.nodes[node_id]['notes'] = notes