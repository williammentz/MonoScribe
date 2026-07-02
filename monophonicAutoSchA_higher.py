import argparse
import copy
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import music21
from music21 import duration, meter, note, stream, tempo

from reductionAutoSchA import (
    EPSILON,
    q,
    projRoot,
    outputDir,
    MeasureCell,
    SliceSelection,
    source_xml_path,
    build_measure_infos_from_json,
    build_measure_start_map,
    row_to_reduced_tone,
    merge_measure_cells
)

"""
Where highest score is considered the primary line to keep
Example terminal call:

uv run monophonicAutoSchA_highest.py \
  --json outputs/inference/Bach_880.json \
  --output-xml outputs/reductions/Bach_880-monophonic-1.musicxml \
  --layer 2

"""


@dataclass
class TimedEvent:
    offset: float
    duration: float
    pitch: music21.pitch.Pitch
    source: str

    @property
    def end(self):
        return self.offset + self.duration


def tone_sort_key(tone, slice_start):
    attacked_here = 1 if abs(tone.start_abs - slice_start) <= EPSILON else 0
    return (tone.utility, attacked_here)


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
    """
    Build vertical slices and keep the top two tones by structural utility.

    Important difference from the polyphonic AutoSchA reducer:
    - chosen tones remain in UTILITY order
    - no re-sorting by pitch
    """
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

        ranked = sorted(
            active,
            key=lambda tone: tone_sort_key(tone, t0),
            reverse=True,
        )

        chosen = []
        if ranked:
            chosen.append(ranked[0])

            for tone in ranked[1:]:
                if tone.pitch_midi != chosen[0].pitch_midi:
                    chosen.append(tone)
                    break

            # Keep fallback behavior from the original reducer:
            # if all active tones share the same pitch, still keep a second tone
            # so we don't drop the second stream artificially.
            if len(chosen) < 2 and len(ranked) > 1:
                chosen.append(ranked[1])

        # IMPORTANT: keep utility order, do NOT sort by pitch.
        chosen = tuple(chosen)

        if debug_slices:
            print()
            print(
                f"[slice {slice_count + 1}] "
                f"measure={measure.number} "
                f"abs=({q(t0)} -> {q(t1)}) "
                f"local=({q(t0 - measure.start_abs)} -> {q(t1 - measure.start_abs)})"
            )

            print("  active:")
            for tone in ranked:
                print("   ", format_tone_debug(tone, t0))

            print("  chosen (utility-ranked):")
            labels = ["PRIMARY", "SECONDARY"]
            for i, tone in enumerate(chosen):
                label = labels[i] if i < len(labels) else f"TONE{i + 1}"
                print("   ", label, format_tone_debug(tone, t0))

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


def assign_slices_to_utility_streams(slices):
    """
    Stream 1 = highest utility selected tone in the slice
    Stream 2 = second-highest utility selected tone in the slice

    No pitch ordering, no lane continuity heuristics, no crossing logic.
    """
    stream_cells = {
        1: defaultdict(list),  # primary utility stream
        2: defaultdict(list),  # secondary utility stream
    }

    for sl in slices:
        for stream_index in (1, 2):
            pitches = ()
            if len(sl.tones) >= stream_index:
                pitches = (sl.tones[stream_index - 1].pitch_midi,)

            stream_cells[stream_index][sl.measure_index].append(
                MeasureCell(
                    start_local=sl.start_local,
                    end_local=sl.end_local,
                    pitches=pitches,
                )
            )

    return stream_cells


def pitch_from_midi(midi_value):
    p = music21.pitch.Pitch()
    p.midi = int(midi_value)
    return p


def cells_to_timed_events(raw_cells, source_name):
    merged_cells = merge_measure_cells(raw_cells)
    events = []

    for cell in merged_cells:
        dur = float(cell.end_local - cell.start_local)
        if dur <= EPSILON:
            continue

        if not cell.pitches:
            continue

        events.append(
            TimedEvent(
                offset=float(cell.start_local),
                duration=dur,
                pitch=pitch_from_midi(cell.pitches[0]),
                source=source_name,
            )
        )

    return events


def get_measure_by_number(part, number):
    for m in part.getElementsByClass(stream.Measure):
        if m.number == number:
            return m
    return None


def tempo_marks(source, part):
    for mark in source.recurse().getElementsByClass(tempo.MetronomeMark):
        src_measure = mark.getContextByClass(stream.Measure)
        if src_measure is None:
            continue

        target_measure = get_measure_by_number(part, src_measure.number)
        if target_measure is None:
            continue

        local_offset = float(mark.getOffsetInHierarchy(src_measure))
        target_measure.insert(local_offset, copy.deepcopy(mark))


def get_beat_length(measure):
    ts = measure.getContextByClass(meter.TimeSignature)
    if ts is not None:
        return float(ts.beatDuration.quarterLength)
    return 1.0


def get_measure_length(measure):
    if measure.barDuration is not None:
        return float(measure.barDuration.quarterLength)

    ts = measure.getContextByClass(meter.TimeSignature)
    if ts is not None:
        return float(ts.barDuration.quarterLength)

    return 4.0


def render_monophonic_measure(measure_number, ts, measure_length, events, show_ts=False):
    new_measure = stream.Measure(number=measure_number)

    if show_ts and ts is not None:
        new_measure.timeSignature = copy.deepcopy(ts)

    def priority(e):
        return 0 if e.source == "primary" else 1

    events = sorted(events, key=lambda e: (e.offset, priority(e), e.pitch.midi))

    current_time = 0.0

    for i, ev in enumerate(events):
        start = max(ev.offset, current_time)
        if start >= measure_length:
            break

        next_offset = measure_length
        if i + 1 < len(events):
            next_offset = min(next_offset, events[i + 1].offset)

        end = min(ev.end, next_offset, measure_length)
        dur = end - start

        if dur <= EPSILON:
            continue

        gap = start - current_time
        if gap > EPSILON:
            r = note.Rest()
            r.duration = duration.Duration(gap)
            new_measure.append(r)

        n = note.Note(ev.pitch)
        n.duration = duration.Duration(dur)
        new_measure.append(n)

        current_time = start + dur

    tail = measure_length - current_time
    if tail > EPSILON:
        r = note.Rest()
        r.duration = duration.Duration(tail)
        new_measure.append(r)

    return new_measure


def reduce_measure(primary_events, secondary_events, source_measure, show_ts=False):
    """
    Homophonic-style behavior:
    - primary utility stream acts like melody
    - secondary utility stream fills around it
    """
    beat_q_len = get_beat_length(source_measure)
    measure_len = get_measure_length(source_measure)

    selected = []

    for prim in primary_events:
        prim_end = min(prim.end, measure_len)

        if len(secondary_events) > 0:
            if prim.duration > float(secondary_events[0].duration):
                selected.append(
                    TimedEvent(
                        offset=prim.offset,
                        duration=secondary_events[0].duration,
                        pitch=prim.pitch,
                        source="primary",
                    )
                )

                fill_start = prim.offset + float(secondary_events[0].duration)
                fill_end = prim_end

                candidates = [
                    e for e in secondary_events
                    if fill_start <= e.offset < fill_end
                ]
                for c in candidates:
                    selected.append(c)

                continue

        selected.append(
            TimedEvent(
                offset=prim.offset,
                duration=beat_q_len,
                pitch=prim.pitch,
                source="primary",
            )
        )

        fill_start = prim.offset + beat_q_len
        fill_end = prim_end

        candidates = [
            e for e in secondary_events
            if fill_start <= e.offset < fill_end
        ]

        for c in candidates:
            selected.append(c)

    return render_monophonic_measure(
        measure_number=source_measure.number,
        ts=source_measure.getContextByClass(meter.TimeSignature),
        measure_length=measure_len,
        events=selected,
        show_ts=show_ts,
    )


def build_monophonic_part(source_score, stream_cells, measure_infos):
    reduced_part = stream.Part(id="ReducedMonophonicUtility")
    reduced_part.partName = "Reduced Monophonic"
    reduced_part.partAbbreviation = "Red. Mono."

    source_part = source_score.parts[0]
    source_measures_by_num = {
        m.number: m for m in source_part.getElementsByClass(stream.Measure)
    }

    last_ts_ratio = None

    for mi in measure_infos:
        source_measure = source_measures_by_num.get(mi.number)
        if source_measure is None:
            continue

        primary_events = cells_to_timed_events(
            stream_cells[1].get(mi.index, []),
            source_name="primary",
        )
        secondary_events = cells_to_timed_events(
            stream_cells[2].get(mi.index, []),
            source_name="secondary",
        )

        ts = source_measure.getContextByClass(meter.TimeSignature)
        ts_ratio = ts.ratioString if ts is not None else None
        show_ts = ts_ratio != last_ts_ratio

        reduced_measure = reduce_measure(
            primary_events=primary_events,
            secondary_events=secondary_events,
            source_measure=source_measure,
            show_ts=show_ts,
        )

        if source_measure.leftBarline is not None:
            reduced_measure.leftBarline = copy.deepcopy(source_measure.leftBarline)
        if source_measure.rightBarline is not None:
            reduced_measure.rightBarline = copy.deepcopy(source_measure.rightBarline)

        reduced_part.append(reduced_measure)
        last_ts_ratio = ts_ratio

    tempo_marks(source_score, reduced_part)
    return reduced_part


def build_reduction_higher(
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
        output_xml = outputDir / f"{json_path.stem}-monophonic_higher.musicxml"
    else:
        output_xml = Path(output_xml)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    source_path = source_xml_path(data, json_path, source_xml)
    source_score = music21.converter.parse(str(source_path))

    measure_infos = build_measure_infos_from_json(data)
    measure_start_map = build_measure_start_map(measure_infos)

    reduced_tones = [
        row_to_reduced_tone(row, measure_start_map, layer)
        for row in data["notes"]
    ]

    selected_slices, slice_count = build_selected_slices(
        matched_tones=reduced_tones,
        measure_infos=measure_infos,
        debug_slices=debug_slices,
    )

    stream_cells = assign_slices_to_utility_streams(selected_slices)

    reduced_part = build_monophonic_part(
        source_score=source_score,
        stream_cells=stream_cells,
        measure_infos=measure_infos,
    )

    out_score = stream.Score()
    if source_score.metadata is not None:
        out_score.metadata = copy.deepcopy(source_score.metadata)

    out_score.insert(0, reduced_part)
    out_score.write("musicxml", fp=output_xml)

    print(f"Built {slice_count} slices")
    print(f"Wrote monophonic reduction to {output_xml}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json",
        required=True,
        help="Inference JSON with structural layer scores.",
    )

    parser.add_argument(
        "--output-xml",
        default=None,
        help="Optional output path for the monophonic reduction MusicXML.",
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=2,
        help="Structural layer to use from layer_scores (1-indexed).",
    )

    parser.add_argument(
        "--source-xml",
        default=None,
        help="Optional override for the source MusicXML path stored in the JSON.",
    )

    parser.add_argument(
        "--debug-slices",
        action="store_true",
        help="Print active/chosen tones for every time slice.",
    )

    args = parser.parse_args()

    build_reduction_higher(
        json_path=args.json,
        output_xml=args.output_xml,
        layer=args.layer,
        source_xml=args.source_xml,
        debug_slices=args.debug_slices,
    )


if __name__ == "__main__":
    main()
