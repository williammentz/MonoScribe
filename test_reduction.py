import argparse
import copy
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import music21
from music21 import chord, note, stream

"""
uv run test_reduction.py \
  --json outputs/rach.json \
  --output-xml outputs/rach-test.musicxml \
  --layer 2
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


@dataclass
class MatchedTone:
    row: dict
    event: ScoreEvent
    utility: float
    start_abs: float
    end_abs: float


@dataclass
class MeasureInfo:
    index: int
    number: int
    start_abs: float
    end_abs: float


@dataclass
class MeasureCell:
    start_local: float
    end_local: float
    pitches: tuple[int, ...]


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


def match_row_to_event(
    row,
    measure_lengths,
    exact_index,
    loose_index,
    pitch_index,
    used_events,
):
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
            return event, "exact"

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
            return event, "loose"

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
            return event, "fuzzy"

    return None, None


def build_measure_infos(score):
    first_part = score.parts[0]
    infos = []

    for index, measure in enumerate(first_part.getElementsByClass(stream.Measure)):
        start_abs = float(measure.getOffsetInHierarchy(score))
        ql = float(measure.duration.quarterLength)
        infos.append(
            MeasureInfo(
                index=index,
                number=measure_number_for(measure),
                start_abs=start_abs,
                end_abs=start_abs + ql,
            )
        )

    return infos


def tone_sort_key(tone, slice_start):
    # Utility dominates.
    # In ties, prefer continuing notes to reduce needless chopping.
    # If you want to prefer new attacks in ties, swap `continuing`
    # for `attacked_here`.
    continuing = 1 if tone.start_abs < slice_start - EPSILON else 0
    return (tone.utility, continuing, tone.event.pitch_midi)


def build_cells_by_staff(matched_tones, measure_infos, staff_count):
    boundaries = set()

    for measure in measure_infos:
        boundaries.add(q(measure.start_abs))
        boundaries.add(q(measure.end_abs))

    for tone in matched_tones:
        boundaries.add(q(tone.start_abs))
        boundaries.add(q(tone.end_abs))

    times = sorted(boundaries)
    cells_by_staff = {
        staff: defaultdict(list)
        for staff in range(1, staff_count + 1)
    }

    if len(times) < 2:
        return cells_by_staff, 0

    measure_ptr = 0
    slice_count = 0

    for t0, t1 in zip(times, times[1:]):
        if t1 <= t0 + EPSILON:
            continue

        while (
            measure_ptr + 1 < len(measure_infos)
            and t0 >= measure_infos[measure_ptr].end_abs - EPSILON
        ):
            measure_ptr += 1

        measure = measure_infos[measure_ptr]

        active = [
            tone
            for tone in matched_tones
            if tone.start_abs <= t0 + EPSILON and tone.end_abs >= t1 - EPSILON
        ]

        chosen = sorted(
            active,
            key=lambda tone: tone_sort_key(tone, t0),
            reverse=True,
        )[:2]

        by_staff = defaultdict(list)
        for tone in chosen:
            by_staff[tone.event.staff].append(tone.event.pitch_midi)

        local_start = t0 - measure.start_abs
        local_end = t1 - measure.start_abs

        for staff in range(1, staff_count + 1):
            pitches = tuple(sorted(set(by_staff.get(staff, []))))
            cells_by_staff[staff][measure.index].append(
                MeasureCell(
                    start_local=local_start,
                    end_local=local_end,
                    pitches=pitches,
                )
            )

        slice_count += 1

    return cells_by_staff, slice_count


def merge_measure_cells(cells):
    merged = []

    for cell in cells:
        if not merged:
            merged.append(
                MeasureCell(cell.start_local, cell.end_local, cell.pitches)
            )
            continue

        prev = merged[-1]
        if (
            prev.pitches == cell.pitches
            and abs(prev.end_local - cell.start_local) <= EPSILON
        ):
            prev.end_local = cell.end_local
        else:
            merged.append(
                MeasureCell(cell.start_local, cell.end_local, cell.pitches)
            )

    return merged


def clone_measure_shell(measure):
    new_measure = stream.Measure(number=measure.number)

    if measure.leftBarline is not None:
        new_measure.leftBarline = copy.deepcopy(measure.leftBarline)
    if measure.rightBarline is not None:
        new_measure.rightBarline = copy.deepcopy(measure.rightBarline)

    for element in measure:
        if isinstance(element, stream.Voice):
            continue
        if isinstance(element, note.GeneralNote):
            continue
        if isinstance(element, chord.Chord):
            continue

        new_measure.insert(element.offset, copy.deepcopy(element))

    return new_measure


def make_general_note(pitches, quarter_length):
    if not pitches:
        element = note.Rest()
    elif len(pitches) == 1:
        element = note.Note()
        element.pitch.midi = int(pitches[0])
    else:
        element = chord.Chord([int(p) for p in pitches])

    element.duration.quarterLength = float(quarter_length)
    return element


def build_reduced_score(score, cells_by_staff):
    reduced = stream.Score()

    if score.metadata is not None:
        reduced.metadata = copy.deepcopy(score.metadata)

    for staff_index, part in enumerate(score.parts, start=1):
        new_part = stream.Part(id=part.id)
        new_part.partName = part.partName
        new_part.partAbbreviation = part.partAbbreviation
        new_part.insert(0, music21.instrument.Piano())

        measures = list(part.getElementsByClass(stream.Measure))

        for measure_index, measure in enumerate(measures):
            new_measure = clone_measure_shell(measure)
            raw_cells = cells_by_staff[staff_index].get(measure_index, [])

            merged_cells = merge_measure_cells(raw_cells)

            if not merged_cells:
                rest = note.Rest()
                rest.duration.quarterLength = float(measure.duration.quarterLength)
                new_measure.insert(0, rest)
            else:
                for cell in merged_cells:
                    dur = cell.end_local - cell.start_local
                    if dur <= EPSILON:
                        continue

                    new_measure.insert(
                        cell.start_local,
                        make_general_note(cell.pitches, dur),
                    )

            new_part.append(new_measure)

        reduced.append(new_part)

    return reduced


def build_reduction(
    json_path,
    output_xml,
    layer,
    source_xml=None,
):
    json_path = Path(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    source_path = source_xml_path(data, json_path, source_xml)
    score = music21.converter.parse(str(source_path))

    notes = data["notes"]
    measure_lengths = build_measure_lengths(data)
    exact_index, loose_index, pitch_index = build_event_indexes(score)

    matched = 0
    exact_matches = 0
    loose_matches = 0
    fuzzy_matches = 0
    unmatched_rows = []

    used_events = set()
    matched_tones = []

    for row in notes:
        event, match_kind = match_row_to_event(
            row=row,
            measure_lengths=measure_lengths,
            exact_index=exact_index,
            loose_index=loose_index,
            pitch_index=pitch_index,
            used_events=used_events,
        )

        if event is None:
            unmatched_rows.append(row)
            continue

        matched += 1

        if match_kind == "exact":
            exact_matches += 1
        elif match_kind == "loose":
            loose_matches += 1
        elif match_kind == "fuzzy":
            fuzzy_matches += 1

        utility = score_for_layer(row, layer)
        start_abs = float(event.element.getOffsetInHierarchy(score))
        end_abs = start_abs + float(event.quarter_length)

        matched_tones.append(
            MatchedTone(
                row=row,
                event=event,
                utility=utility,
                start_abs=start_abs,
                end_abs=end_abs,
            )
        )

    measure_infos = build_measure_infos(score)
    cells_by_staff, slice_count = build_cells_by_staff(
        matched_tones=matched_tones,
        measure_infos=measure_infos,
        staff_count=len(score.parts),
    )

    reduced_score = build_reduced_score(score, cells_by_staff)
    reduced_score.write("musicxml", fp=output_xml)

    print(f"Source: {source_path}")
    print(f"Matched {matched} / {len(notes)} scored notes")
    print(
        f"  exact matches: {exact_matches}; "
        f"onset/pitch fallback: {loose_matches}; "
        f"measure/pitch fuzzy: {fuzzy_matches}"
    )
    print(f"Built {slice_count} time slices")
    print("Guaranteed maximum simultaneous sounding tones: 2")

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
        "--source-xml",
        help="Optional override for the source MusicXML path stored in the JSON.",
    )

    args = parser.parse_args()

    build_reduction(
        json_path=args.json,
        output_xml=args.output_xml,
        layer=args.layer,
        source_xml=args.source_xml,
    )


if __name__ == "__main__":
    main()