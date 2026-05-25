import argparse
import copy
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import music21
from music21 import chord, note, stream

"""
uv run reduceAutoSchA.py \
  --json outputs/test.json \
  --output-xml outputs/test_reduction.musicxml \
  --layer 2 \
  --threshold 0.5
"""

EPSILON = 1e-6


@dataclass
class ScoreEvent:
    element: note.Note | chord.Chord
    pitch_index: int | None
    staff: int
    measure: int
    measure_offset: float
    quarter_length: float
    pitch_midi: int


def q(value):
    return round(float(value), 6)


def event_key(staff, measure, measure_offset, pitch_midi, quarter_length):
    return (
        int(staff),
        int(measure),
        q(measure_offset),
        int(pitch_midi),
        q(quarter_length),
    )


def loose_event_key(staff, measure, measure_offset, pitch_midi):
    return (
        int(staff),
        int(measure),
        q(measure_offset),
        int(pitch_midi),
    )


def row_key(row):
    return event_key(
        row.get("staff") or 1,
        row["measure"],
        row["measure_offset"],
        row["pitch_midi"],
        row["quarter_length"],
    )


def row_loose_key(row):
    return loose_event_key(
        row.get("staff") or 1,
        row["measure"],
        row["measure_offset"],
        row["pitch_midi"],
    )


def staff_measure_pitch_key(staff, measure, pitch_midi):
    return (
        int(staff),
        int(measure),
        int(pitch_midi),
    )


def event_identity(event):
    return (
        id(event.element),
        event.pitch_index,
        event.pitch_midi,
    )


def score_for_layer(row, layer):
    layer_index = layer - 1
    scores = row["layer_scores"]

    if not 0 <= layer_index < len(scores):
        raise ValueError(
            f"--layer must be between 1 and {len(scores)}, got {layer}"
        )

    return float(scores[layer_index])


def source_xml_path(data, json_path, source_xml):
    if source_xml is not None:
        return Path(source_xml)

    if data.get("source_xml"):
        path = Path(data["source_xml"])
        if path.exists():
            return path

        candidate = Path(json_path).parent / path
        if candidate.exists():
            return candidate

    raise ValueError(
        "No source MusicXML found. Re-run scoreStructure.py with the updated "
        "exporter, or pass --source-xml explicitly."
    )


def measure_offset_for(element, measure):
    return float(element.getOffsetInHierarchy(measure))


def measure_number_for(measure):
    if measure.number is None:
        return 0
    return int(measure.number)


def iter_score_events(score):
    for part_index, part in enumerate(score.parts, start=1):
        staff = part_index

        for measure in part.getElementsByClass(stream.Measure):
            measure_number = measure_number_for(measure)

            for element in measure.recurse().notes:
                measure_offset = measure_offset_for(element, measure)
                quarter_length = float(element.duration.quarterLength)

                if isinstance(element, note.Note):
                    yield ScoreEvent(
                        element=element,
                        pitch_index=None,
                        staff=staff,
                        measure=measure_number,
                        measure_offset=measure_offset,
                        quarter_length=quarter_length,
                        pitch_midi=element.pitch.midi,
                    )

                elif isinstance(element, chord.Chord):
                    for pitch_index, pitch in enumerate(element.pitches):
                        yield ScoreEvent(
                            element=element,
                            pitch_index=pitch_index,
                            staff=staff,
                            measure=measure_number,
                            measure_offset=measure_offset,
                            quarter_length=quarter_length,
                            pitch_midi=pitch.midi,
                        )


def build_measure_lengths(data):
    return {
        int(m["number"]): float(m["quarter_length"])
        for m in data.get("measures", [])
    }


def row_positions(row, measure_lengths):
    measure = int(row["measure"])
    offset = float(row["measure_offset"])
    positions = [(measure, offset)]

    while measure in measure_lengths and offset >= measure_lengths[measure] - EPSILON:
        offset -= measure_lengths[measure]
        measure += 1
        positions.append((measure, offset))

    return positions


def build_event_indexes(score):
    exact_index = defaultdict(list)
    loose_index = defaultdict(list)
    pitch_index = defaultdict(list)

    for event in iter_score_events(score):
        exact_index[
            event_key(
                event.staff,
                event.measure,
                event.measure_offset,
                event.pitch_midi,
                event.quarter_length,
            )
        ].append(event)
        loose_index[
            loose_event_key(
                event.staff,
                event.measure,
                event.measure_offset,
                event.pitch_midi,
            )
        ].append(event)
        pitch_index[
            staff_measure_pitch_key(
                event.staff,
                event.measure,
                event.pitch_midi,
            )
        ].append(event)

    return exact_index, loose_index, pitch_index


def choose_unmatched_event(candidates, used_events):
    for event in candidates:
        identity = event_identity(event)
        if identity not in used_events:
            used_events.add(identity)
            return event

    return None


def choose_closest_unmatched_event(candidates, row, used_events):
    available = [
        event
        for event in candidates
        if event_identity(event) not in used_events
    ]

    if not available:
        return None

    row_offset = float(row["measure_offset"])
    row_duration = float(row["quarter_length"])
    event = min(
        available,
        key=lambda candidate: (
            abs(candidate.measure_offset - row_offset),
            abs(candidate.quarter_length - row_duration),
        ),
    )
    used_events.add(event_identity(event))
    return event


def replace_with_rest(element):
    parent = element.activeSite
    if parent is None:
        return False

    rest = note.Rest()
    rest.duration = copy.deepcopy(element.duration)

    parent_offset = element.getOffsetBySite(parent)
    parent.remove(element, recurse=False)
    parent.insert(parent_offset, rest)
    return True


def apply_removals(note_removals, chord_pitch_removals):
    removed_notes = 0
    removed_chord_pitches = 0
    replaced_chords = 0

    for element in note_removals:
        if replace_with_rest(element):
            removed_notes += 1

    for element, pitch_indices in chord_pitch_removals.items():
        keep = [
            pitch
            for index, pitch in enumerate(element.pitches)
            if index not in pitch_indices
        ]

        removed_chord_pitches += len(pitch_indices)

        if keep:
            element.pitches = keep
        elif replace_with_rest(element):
            replaced_chords += 1

    return removed_notes, removed_chord_pitches, replaced_chords


def build_reduction(
    json_path,
    output_xml,
    layer,
    threshold,
    source_xml=None,
):
    json_path = Path(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    source_path = source_xml_path(data, json_path, source_xml)
    score = music21.converter.parse(str(source_path))
    for part in score.parts:
        part.insert(0, music21.instrument.Piano())
    notes = data["notes"]
    measure_lengths = build_measure_lengths(data)
    exact_index, loose_index, pitch_index = build_event_indexes(score)

    matched = 0
    exact_matches = 0
    loose_matches = 0
    fuzzy_matches = 0
    unmatched_rows = []
    kept = 0
    removed = 0
    used_events = set()
    note_removals = set()
    chord_pitch_removals = defaultdict(set)

    for row in notes:
        event = None
        positions = row_positions(row, measure_lengths)
        staff = row.get("staff") or 1

        for measure, offset in positions:
            event = choose_unmatched_event(
                exact_index.get(
                    event_key(
                        staff,
                        measure,
                        offset,
                        row["pitch_midi"],
                        row["quarter_length"],
                    ),
                    [],
                ),
                used_events,
            )
            if event is not None:
                break

        if event is not None:
            exact_matches += 1
        else:
            for measure, offset in positions:
                event = choose_unmatched_event(
                    loose_index.get(
                        loose_event_key(
                            staff,
                            measure,
                            offset,
                            row["pitch_midi"],
                        ),
                        [],
                    ),
                    used_events,
                )
                if event is not None:
                    loose_matches += 1
                    break

        if event is None:
            for measure, _ in positions:
                event = choose_closest_unmatched_event(
                    pitch_index.get(
                        staff_measure_pitch_key(
                            staff,
                            measure,
                            row["pitch_midi"],
                        ),
                        [],
                    ),
                    row,
                    used_events,
                )
                if event is not None:
                    fuzzy_matches += 1
                    break

        if event is None:
            unmatched_rows.append(row)
            continue

        matched += 1

        if score_for_layer(row, layer) >= threshold:
            kept += 1
            continue

        removed += 1
        if isinstance(event.element, note.Note):
            note_removals.add(event.element)
        else:
            chord_pitch_removals[event.element].add(event.pitch_index)

    removed_notes, removed_chord_pitches, replaced_chords = apply_removals(
        note_removals,
        chord_pitch_removals,
    )

    score.write("musicxml", fp=output_xml)

    print(f"Source: {source_path}")
    print(f"Matched {matched} / {len(notes)} scored notes")
    print(
        f"  exact matches: {exact_matches}; "
        f"onset/pitch fallback: {loose_matches}; "
        f"measure/pitch fuzzy: {fuzzy_matches}"
    )
    print(f"Keeping {kept}; removing {removed}")
    print(
        "Applied removals: "
        f"{removed_notes} notes, "
        f"{removed_chord_pitches} chord pitches, "
        f"{replaced_chords} emptied chords"
    )

    if unmatched_rows:
        print(f"WARNING: {len(unmatched_rows)} scored notes could not be matched")
        for row in unmatched_rows[:10]:
            print(
                "  unmatched "
                f"index={row['note_index']} "
                f"staff={row.get('staff')} "
                f"measure={row['measure']} "
                f"offset={row['measure_offset']} "
                f"pitch={row['pitch_name']} "
                f"dur={row['quarter_length']}"
            )

    print(f"Wrote reduction to {output_xml}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json",
        required=True,
    )

    parser.add_argument(
        "--output-xml",
        required=True,
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--source-xml",
        help="Optional override for the source MusicXML path stored in the JSON.",
    )

    args = parser.parse_args()

    build_reduction(
        json_path=args.json,
        output_xml=args.output_xml,
        layer=args.layer,
        threshold=args.threshold,
        source_xml=args.source_xml,
    )


if __name__ == "__main__":
    main()
