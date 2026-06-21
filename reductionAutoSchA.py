import argparse
import copy
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import music21
from music21 import chord, note, stream

"""
uv run reductionAutoSchA.py \
  --json outputs/inference/Mozart_12.json \
  --output-xml outputs/reductions/mozart_new-test.musicxml \ (unnecessary)
  --layer 2
"""

EPSILON = 1e-6

projRoot = Path(__file__).resolve().parent
outputDir = projRoot / 'outputs/reductions/'
outputDir.mkdir(parents = True, exist_ok = True)

@dataclass
class MeasureInfo:
    index: int
    number: int
    start_abs: float
    end_abs: float
    quarter_length: float


@dataclass
class ReducedTone:
    row: dict
    utility: float
    start_abs: float
    end_abs: float
    pitch_midi: int


@dataclass
class MeasureCell:
    start_local: float
    end_local: float
    pitches: tuple[int, ...]

@dataclass
class SliceSelection:
    measure_index: int
    measure_number: int
    start_local: float
    end_local: float
    start_abs: float
    end_abs: float
    tones: tuple[ReducedTone, ...]



def q(value):
    return round(float(value), 6)


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


def build_measure_lengths(data):
    return {
        int(m["number"]): float(m["quarter_length"])
        for m in data.get("measures", [])
    }


def build_measure_infos_from_json(data):
    measure_entries = data.get("measures", [])
    if not measure_entries:
        raise ValueError("JSON is missing measure metadata")

    infos = []
    cursor = 0.0

    for index, m in enumerate(sorted(measure_entries, key=lambda x: int(x["number"]))):
        ql = float(m["quarter_length"])
        infos.append(
            MeasureInfo(
                index=index,
                number=int(m["number"]),
                start_abs=cursor,
                end_abs=cursor + ql,
                quarter_length=ql,
            )
        )
        cursor += ql

    return infos


def build_measure_start_map(measure_infos):
    return {m.number: m.start_abs for m in measure_infos}


def row_to_reduced_tone(row, measure_start_map, layer):
    measure = int(row["measure"])
    if measure not in measure_start_map:
        raise ValueError(f"Measure {measure} not found in measure metadata")

    start_abs = measure_start_map[measure] + float(row["measure_offset"])
    end_abs = start_abs + float(row["quarter_length"])

    return ReducedTone(
        row=row,
        utility=score_for_layer(row, layer),
        start_abs=start_abs,
        end_abs=end_abs,
        pitch_midi=int(row["pitch_midi"]),
    )


def tone_sort_key(tone, slice_start):
    attacked_here = 1 if abs(tone.start_abs - slice_start) <= EPSILON else 0
    return (tone.utility, attacked_here, tone.pitch_midi)


def format_tone_debug(tone, slice_start):
    attacked_here = abs(tone.start_abs - slice_start) <= EPSILON
    return (
        f"m={tone.row['measure']} "
        f"off={q(tone.row['measure_offset'])} "
        f"pitch={tone.pitch_midi} "
        f"util={tone.utility:.6f} "
        f"start={q(tone.start_abs)} "
        f"end={q(tone.end_abs)} "
        f"{'ATTACK' if attacked_here else 'HELD'}"
    )


def build_selected_slices(matched_tones, measure_infos, debug_slices=False):
    boundaries = set()

    for measure in measure_infos:
        boundaries.add(q(measure.start_abs))
        boundaries.add(q(measure.end_abs))

    for tone in matched_tones:
        boundaries.add(q(tone.start_abs))
        boundaries.add(q(tone.end_abs))

    times = sorted(boundaries)
    slices = []

    if len(times) < 2:
        return slices, 0

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
            if tone.start_abs < t1 - EPSILON and tone.end_abs > t0 + EPSILON
        ]

        chosen = sorted(
            active,
            key=lambda tone: tone_sort_key(tone, t0),
            reverse=True,
        )[:2]

        chosen = tuple(
            sorted(chosen, key=lambda tone: tone.pitch_midi, reverse=True)
        )

        if debug_slices:
            print()
            print(
                f"[slice {slice_count + 1}] "
                f"measure={measure.number} "
                f"abs=({q(t0)} -> {q(t1)}) "
                f"local=({q(t0 - measure.start_abs)} -> {q(t1 - measure.start_abs)})"
            )
            print("  active:")
            for tone in sorted(
                active,
                key=lambda tone: tone_sort_key(tone, t0),
                reverse=True,
            ):
                print("   ", format_tone_debug(tone, t0))

            print("  chosen:")
            for tone in chosen:
                print("   ", format_tone_debug(tone, t0))

        slices.append(
            SliceSelection(
                measure_index=measure.index,
                measure_number=measure.number,
                start_local=t0 - measure.start_abs,
                end_local=t1 - measure.start_abs,
                start_abs=t0,
                end_abs=t1,
                tones=chosen,
            )
        )

        slice_count += 1

    return slices, slice_count

def assign_slices_to_lanes(slices):
    """
    Convert per-slice top-2 selections into two persistent monophonic lanes.
    Each lane is merged independently, so a sustained note can continue
    while the other lane changes.
    """
    lane_cells = {
        1: defaultdict(list),  # upper lane
        2: defaultdict(list),  # lower lane
    }

    prev_pitch = {
        1: None,
        2: None,
    }

    for sl in slices:
        tones = list(sl.tones)
        assignment = {1: None, 2: None}
        remaining = tones[:]

        # First preserve exact pitch continuity in each lane when possible.
        for lane in (1, 2):
            if prev_pitch[lane] is None:
                continue

            for tone in list(remaining):
                if tone.pitch_midi == prev_pitch[lane]:
                    assignment[lane] = tone
                    remaining.remove(tone)
                    break

        unassigned_lanes = [lane for lane in (1, 2) if assignment[lane] is None]

        if len(remaining) == 2:
            # Assign higher pitch to upper lane, lower pitch to lower lane.
            remaining = sorted(remaining, key=lambda t: t.pitch_midi, reverse=True)

            if len(unassigned_lanes) == 2:
                assignment[1] = remaining[0]
                assignment[2] = remaining[1]
            else:
                # Defensive fallback; shouldn't normally happen.
                for lane, tone in zip(unassigned_lanes, remaining):
                    assignment[lane] = tone

        elif len(remaining) == 1:
            tone = remaining[0]

            if len(unassigned_lanes) == 2:
                # Choose the lane whose previous pitch is closer.
                candidates = []
                for lane in unassigned_lanes:
                    if prev_pitch[lane] is None:
                        dist = 0
                    else:
                        dist = abs(tone.pitch_midi - prev_pitch[lane])
                    candidates.append((dist, lane))

                chosen_lane = min(candidates)[1]
                assignment[chosen_lane] = tone
            elif len(unassigned_lanes) == 1:
                assignment[unassigned_lanes[0]] = tone

        # Emit cells for both lanes.
        for lane in (1, 2):
            pitches = ()
            if assignment[lane] is not None:
                pitches = (assignment[lane].pitch_midi,)

            lane_cells[lane][sl.measure_index].append(
                MeasureCell(
                    start_local=sl.start_local,
                    end_local=sl.end_local,
                    pitches=pitches,
                )
            )

            prev_pitch[lane] = (
                assignment[lane].pitch_midi
                if assignment[lane] is not None
                else None
            )

    return lane_cells


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


def build_reduced_score_single_staff(source_score, lane_cells):
    reduced = stream.Score()

    if source_score.metadata is not None:
        reduced.metadata = copy.deepcopy(source_score.metadata)

    source_part = source_score.parts[0]
    new_part = stream.Part(id=source_part.id or "Reduction")
    new_part.partName = "Reduction"
    new_part.partAbbreviation = "Red."
    new_part.insert(0, music21.instrument.Piano())

    measures = list(source_part.getElementsByClass(stream.Measure))

    for measure_index, measure in enumerate(measures):
        new_measure = clone_measure_shell(measure)

        for lane in (1, 2):
            voice_stream = stream.Voice(id=str(lane))
            raw_cells = lane_cells[lane].get(measure_index, [])
            merged_cells = merge_measure_cells(raw_cells)

            if not merged_cells:
                rest = note.Rest()
                rest.duration.quarterLength = float(measure.duration.quarterLength)
                voice_stream.insert(0, rest)
            else:
                for cell in merged_cells:
                    dur = cell.end_local - cell.start_local
                    if dur <= EPSILON:
                        continue

                    voice_stream.insert(
                        cell.start_local,
                        make_general_note(cell.pitches, dur),
                    )

            new_measure.insert(0, voice_stream)

        new_part.append(new_measure)

    reduced.append(new_part)
    return reduced



def build_reduction(
    json_path,
    output_xml,
    layer,
    source_xml=None,
    debug_slices=False,
):
    json_path = Path(json_path)
    if not json_path.is_absolute():
        json_path = projRoot / json_path
    
    if output_xml is None:
        output_xml = outputDir / f"{json_path.stem}-reduction.musicxml"
    else:
        # output_xml = outputDir / Path(output_xml).name
        output_xml = output_xml

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    source_path = source_xml_path(data, json_path, source_xml)
    source_score = music21.converter.parse(str(source_path))

    notes = data["notes"]
    measure_infos = build_measure_infos_from_json(data)
    measure_start_map = build_measure_start_map(measure_infos)

    reduced_tones = [
        row_to_reduced_tone(row, measure_start_map, layer)
        for row in notes
    ]

    selected_slices, slice_count = build_selected_slices(
    matched_tones=reduced_tones,
    measure_infos=measure_infos,
    debug_slices=debug_slices,
)

    lane_cells = assign_slices_to_lanes(selected_slices)

    reduced_score = build_reduced_score_single_staff(
        source_score=source_score,
        lane_cells=lane_cells,
    )
    reduced_score.write("musicxml", fp=output_xml)

    # print(f"Source: {source_path}")
    # print(f"Loaded {len(notes)} notes from JSON")
    # print(f"Built {slice_count} time slices")
    # print("Guaranteed maximum simultaneous sounding tones: 2")
    # print("Selection rule: top-2 active tones globally per slice")
    # print("Output format: single staff")

    print(f"Wrote reduction to {output_xml}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json",
        required=True,
    )

    parser.add_argument(
        "--output-xml",
        required=False,
        default=None,
        help="Optional output filename. File is always written to the fixed output folder."
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

    parser.add_argument(
        "--debug-slices",
        action="store_true",
        help="Print active/chosen tones for every time slice.",
    )

    args = parser.parse_args()

    build_reduction(
        json_path=args.json,
        output_xml=args.output_xml,
        layer=args.layer,
        source_xml=args.source_xml,
        debug_slices=args.debug_slices,
    )


if __name__ == "__main__":
    main()