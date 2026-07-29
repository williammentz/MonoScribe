"""
utils/highlightReduction.py

Two responsibilities:
  1) Export the surviving monophonic line (graph-selected primary events) as a
     JSON manifest keyed by partitura note ids.
  2) Write a highlighted copy of the ORIGINAL score, recoloring the surviving
     notes and greying the rest.

Highlighting bridge (Option A -- nothing upstream required):
  The graph gives us the surviving partitura note ids. We re-load the SAME
  score with partitura (force_note_ids='keep' -> identical ids) to recover each
  selected note's musical coordinates, then match those coordinates to the
  <note> elements in the MusicXML to apply color.

  Join key:  (part_index, onset_in_quarters, midi_pitch, staff)
    - quarters make it independent of the file's <divisions> resolution
    - midi makes it independent of enharmonic spelling
    - staff disambiguates cross-hand unisons; voice is intentionally omitted
      (voice numbering round-trips less reliably than staff)

Requires: lxml, partitura, numpy
"""

import json
from pathlib import Path

import numpy as np
import partitura as pt
from lxml import etree


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def build_selected_notes(graph, path):
    """
    Collect one entry per surviving primary event across all path nodes.
    Matching downstream is by note id, so playability octave shifts do not
    affect this. 'pitch'/'onset' are kept as human-readable diagnostics.
    """
    selected_notes = []

    for node_id in path:
        if node_id == "sink":
            continue

        node = graph.nodes[node_id]
        render_payload = node.get("render", {})

        for event in render_payload.get("primary_events", []):
            note_id = event.get("note_id")
            if note_id is None:
                continue

            selected_notes.append({
                "note_id": str(note_id),
                "pitch": int(event["pitch"]),        # diagnostic only
                "onset": float(event["onset"]),      # diagnostic only
                "duration": float(event["duration"]),
                "node_id": node_id,
                "segment_label": node.get("segment_label"),
                "measure_num": node.get("measure_num"),
            })

    return selected_notes


def save_selected_path_manifest(graph, path, source_score_path, method, out_path):
    """
    Write the selected-line manifest to disk and return the output Path.
    Call AFTER restrict_path_to_core() and attach_render_payload().
    """
    selected_notes = build_selected_notes(graph, path)

    manifest = {
        "source_score": str(source_score_path),
        "method": method,
        "note_count": len(selected_notes),
        "selected_notes": selected_notes,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return out_path


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

_STEP_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
COLORABLE_CHILDREN = ("notehead", "stem", "beam", "accidental", "dot", "flag")


def _ln(tag):
    """Local tag name, namespace-stripped. None for comments/PIs."""
    if not isinstance(tag, str):
        return None
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _child(el, name):
    for c in el:
        if _ln(c.tag) == name:
            return c
    return None


def _child_int(el, name, default=None):
    c = _child(el, name)
    if c is not None and c.text is not None:
        try:
            return int(c.text)
        except ValueError:
            return default
    return default


def _note_midi(note_el):
    """MIDI number from a <note>'s <pitch>, or None for rests/unpitched."""
    pitch_el = _child(note_el, "pitch")
    if pitch_el is None:
        return None

    step = None
    octave = None
    alter = 0
    for c in pitch_el:
        n = _ln(c.tag)
        if n == "step":
            step = c.text
        elif n == "octave":
            octave = int(c.text)
        elif n == "alter" and c.text is not None:
            alter = int(round(float(c.text)))

    if step is None or octave is None or step not in _STEP_SEMITONE:
        return None
    return (octave + 1) * 12 + _STEP_SEMITONE[step] + alter


def _apply_color(note_el, color):
    note_el.set("color", color)
    for child in note_el:
        if _ln(child.tag) in COLORABLE_CHILDREN:
            child.set("color", color)


def _iter_xml_note_keys(root, precision=6):
    """
    Yield (note_element, key, is_rest) for every <note> in document order.

    key = (part_index, round(onset_q, precision), midi, staff), or None for rests.
    onset_q is absolute quarter-length from the start of each part.
    """
    parts = [el for el in root if _ln(el.tag) == "part"]
    if not parts:
        parts = [el for el in root.iter() if _ln(el.tag) == "part"]

    for part_index, part in enumerate(parts):
        divisions = 1
        cumulative_q = 0.0

        for measure in (m for m in part if _ln(m.tag) == "measure"):
            position = 0     # divisions, within this measure
            last_onset = 0
            measure_max = 0

            for el in measure:
                name = _ln(el.tag)

                if name == "attributes":
                    d = _child_int(el, "divisions")
                    if d:
                        divisions = d

                elif name == "note":
                    is_chord = _child(el, "chord") is not None
                    is_grace = _child(el, "grace") is not None
                    dur = _child_int(el, "duration", default=0) or 0

                    if is_chord:
                        onset = last_onset
                    else:
                        onset = position
                        last_onset = onset
                        if not is_grace:
                            position += dur
                    measure_max = max(measure_max, position)

                    is_rest = _child(el, "rest") is not None
                    midi = None if is_rest else _note_midi(el)
                    staff = _child_int(el, "staff", default=1)
                    onset_q = cumulative_q + onset / divisions

                    key = None if midi is None else (
                        part_index, round(onset_q, precision), midi, staff
                    )
                    yield el, key, is_rest

                elif name == "backup":
                    position -= (_child_int(el, "duration", default=0) or 0)

                elif name == "forward":
                    position += (_child_int(el, "duration", default=0) or 0)
                    measure_max = max(measure_max, position)

            cumulative_q += measure_max / divisions


def _partitura_key_by_id(source_score_path, precision=6):
    """Map each partitura note id -> coordinate key, using the same load call as the pipeline."""
    score = pt.load_score(str(source_score_path), force_note_ids="keep")
    key_by_id = {}

    for part_index, part in enumerate(pt.score.iter_parts(score.parts)):
        qmap = part.quarter_map
        for n in part.iter_all(pt.score.Note):
            onset_q = float(np.asarray(qmap(n.start.t)).reshape(-1)[0])
            midi = int(n.midi_pitch)
            staff = int(n.staff) if n.staff is not None else 1
            key_by_id[str(n.id)] = (part_index, round(onset_q, precision), midi, staff)

    return key_by_id


# ---------------------------------------------------------------------------
# Highlighting
# ---------------------------------------------------------------------------

def annotate_by_partitura_ids(
    source_score_path,
    selected_ids,
    selected_color="#d1008f",
    background_color="#b0b0b0",
    color_rests=False,
    precision=6,
):
    """
    Color <note> elements whose partitura coordinate matches a selected id.
    Returns (lxml_tree, report).
    """
    # 1) selected ids -> coordinate multiset (via partitura)
    key_by_id = _partitura_key_by_id(source_score_path, precision=precision)

    remaining = {}
    missing_ids = []
    for sid in selected_ids:
        key = key_by_id.get(str(sid))
        if key is None:
            missing_ids.append(str(sid))
            continue
        remaining[key] = remaining.get(key, 0) + 1

    # 2) parse the SAME file with lxml and color by coordinate match
    parser = etree.XMLParser(
        remove_blank_text=False,
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
    )
    tree = etree.parse(str(source_score_path), parser)
    root = tree.getroot()

    total = 0
    selected_count = 0
    background_count = 0

    for note_el, key, is_rest in _iter_xml_note_keys(root, precision=precision):
        if is_rest and not color_rests:
            continue

        total += 1
        if key is not None and remaining.get(key, 0) > 0:
            _apply_color(note_el, selected_color)
            remaining[key] -= 1
            selected_count += 1
        else:
            if background_color is not None:
                _apply_color(note_el, background_color)
            background_count += 1

    unmatched_selected = sum(v for v in remaining.values() if v > 0)

    report = {
        "total_pitched_notes": total,
        "selected_notes": selected_count,
        "background_notes": background_count,
        "expected_selected": len([s for s in selected_ids]) - len(missing_ids),
        "unmatched_selected": unmatched_selected,
        "missing_ids": sorted(missing_ids),
    }
    return tree, report


def write_musicxml(tree, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    docinfo = tree.docinfo
    doctype = docinfo.doctype if docinfo and docinfo.doctype else None

    tree.write(
        str(out_path),
        xml_declaration=True,
        encoding="UTF-8",
        doctype=doctype,
        pretty_print=False,
    )
    return out_path


def write_highlighted_score(
    graph,
    path,
    source_score_path,
    out_path,
    selected_color="#d1008f",
    background_color="#b0b0b0",
    color_rests=False,
):
    """
    Build the selected-id set from the graph and write a highlighted copy of the
    original score, matched via partitura coordinates. Returns (out_path, report).

    Call AFTER attach_render_payload().
    """
    selected_ids = {n["note_id"] for n in build_selected_notes(graph, path)}

    tree, report = annotate_by_partitura_ids(
        source_score_path=source_score_path,
        selected_ids=selected_ids,
        selected_color=selected_color,
        background_color=background_color,
        color_rests=color_rests,
    )
    write_musicxml(tree, out_path)

    if report["selected_notes"] == 0 and selected_ids:
        print("WARNING: 0 notes highlighted -- likely a partitura/XML onset-origin "
              "mismatch. Check that the score has a normal measure start.")
    if report["unmatched_selected"]:
        print(f"WARNING: {report['unmatched_selected']} selected note(s) had no XML "
              f"coordinate match (possible unison edge case).")
    if report["missing_ids"]:
        print(f"WARNING: {len(report['missing_ids'])} selected id(s) not found by partitura.")

    return Path(out_path), report
