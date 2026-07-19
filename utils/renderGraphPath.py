from collections import OrderedDict
import partitura as pt
import argparse
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from music21 import stream, note as m21_note, meter, tempo, key, metadata, instrument, pitch as m21, converter

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

    render_events = []
    onset_buckets = OrderedDict()
    note_ids = []

    for n in notes_sorted:
        onset_q = note_onset_quarter(n, part)
        dur_q = note_duration_quarter(n, part)

        if onset_q is None or dur_q is None or dur_q <= 0:
            continue

        onset_q = round(float(onset_q), 8)
        dur_q = round(float(dur_q), 8)

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
            chosen = max(bucket, key=lambda e: e["pitch"])  # default highest

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


def path_to_render_events(path, graph):
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
            })

    if not path_events:
        return []

    path_events.sort(key=lambda x: (x["onset"], x["pitch"]))

    if not path_events:
        return []

    path_events.sort(key=lambda x: (x["onset"], x["pitch"]))

    return normalize_path_events(
        path_events,
        merge_adjacent_same_pitch=True
    )


def render_best_path(path, graph, out_path, render_context, source_score_path):
    path_events = path_to_render_events(path, graph)

    if not path_events:
        raise ValueError("No renderable events found in the selected path.")

    return render_raw_path(
        path_events,
        source_score_path = source_score_path,
        out_path = out_path,
        title = render_context.get('title'),
        composer = render_context.get('composer'),
    )

def normalize_path_events(path_events, merge_adjacent_same_pitch=True):
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

        # If this note starts before previous ends, truncate previous
        if onset < prev_end:
            raise ValueError(
                f"Overlapping notes detected: "
                f"{prev} overlaps {ev}"
            )

        # Merge adjacent same-pitch notes if desired
        if merge_adjacent_same_pitch:
            prev_end = prev["onset"] + prev["duration"]
            if abs(onset - prev_end) < 1e-9 and p == prev["pitch"]:
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
    path_events = normalize_path_events(path_events)

    src_score = converter.parse(source_score_path)
    if len(src_score.parts) == 0:
        raise ValueError("Source score contains no parts.")

    src_part = src_score.parts[0]
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
        start = float(ev["onset"])
        dur = float(ev["duration"])
        pitch_value = ev["pitch"]

        if dur <= 0:
            continue

        end = start + dur
        cursor = start

        while cursor < end - 1e-9:
            containing = None

            for row in measure_table:
                if row["start"] <= cursor < row["end"] - 1e-9:
                    containing = row
                    break

            # Edge case: event lands exactly on final boundary
            if containing is None:
                for row in reversed(measure_table):
                    if abs(cursor - row["end"]) < 1e-9:
                        continue
                    if row["start"] <= cursor <= row["end"]:
                        containing = row
                        break

            if containing is None:
                break

            seg_end = min(end, containing["end"])
            seg_dur = seg_end - cursor

            if seg_dur <= 1e-9:
                break

            n = m21_note.Note()
            if isinstance(pitch_value, int):
                n.pitch.midi = pitch_value
            else:
                n.pitch = m21.Pitch(pitch_value)

            n.duration.quarterLength = seg_dur

            local_offset = cursor - containing["start"]
            target_measures[containing["number"]].insert(local_offset, n)

            cursor = seg_end

    # Fill gaps with rests inside the already-correct measure shells
    for m in reduced_part.getElementsByClass(stream.Measure):
        try:
            m.makeRests(fillGaps=True, inPlace=True)
        except Exception:
            pass

    tempo_marks(src_score, reduced_part)

    sc.insert(0, reduced_part)

    try:
        sc.makeTies(inPlace=True)
    except Exception:
        pass

    sc.write("musicxml", fp=out_path)
    print("Raw path reduction written.")
    return sc