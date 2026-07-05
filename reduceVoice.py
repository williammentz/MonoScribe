import argparse
import copy
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import music21
from music21 import chord, duration, meter, note, stream, tempo

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
    merge_measure_cells,
)

"""
Voice-aware monophonic reducer.

Instead of choosing the primary note purely by highest utility at each slice.

Example:
uv run reduceVoice.py \
  --json outputs/inference/Bach_883.json \
  --output-xml outputs/reductions/Bach_883-monophonic-voice-aware.musicxml \
  --layer 2
"""


# Conservative first-pass defaults
DEFAULT_SAME_VOICE_BONUS = 0.08
DEFAULT_SWITCH_THRESHOLD = 0.05
DEFAULT_ATTACK_BONUS = 0.01
DEFAULT_CONFIRM_SLICES = 1


@dataclass
class TimedEvent:
    offset: float
    duration: float
    pitch: music21.pitch.Pitch
    source: str

    @property
    def end(self):
        return self.offset + self.duration


@dataclass
class ActiveSlice:
    measure_index: int
    measure_number: int
    start_local: float
    end_local: float
    start_abs: float
    end_abs: float
    active_tones: tuple


@dataclass
class VoiceCandidate:
    voice_id: str
    tone: object
    base_score: float
    attacked_here: bool
    melodic_bonus: float


def tone_sort_key(tone, slice_start):
    attacked_here = 1 if abs(tone.start_abs - slice_start) <= EPSILON else 0
    util = getattr(tone, "selection_utility", tone.utility)
    return (util, attacked_here)


def format_tone_debug(tone, slice_start):
    attacked_here = abs(tone.start_abs - slice_start) <= EPSILON
    voice_id = getattr(tone, "voice_id", "unknown")
    util = getattr(tone, "selection_utility", tone.utility)
    return (
        f"m={tone.row['measure']} "
        f"off={q(tone.row['measure_offset'])} "
        f"pitch={tone.pitch_midi} "
        f"util={util:.6f} "
        f"start={q(tone.start_abs)} "
        f"end={q(tone.end_abs)} "
        f"voice={voice_id} "
        f"{'ATTACK' if attacked_here else 'HELD'}"
    )


def tone_selection_utility(tone):
    return float(getattr(tone, "selection_utility", tone.utility))


def tone_voice_id(tone):
    return str(getattr(tone, "voice_id", "unknown"))


def normalize_tone_utilities(tones):
    if not tones:
        return

    values = [float(t.utility) for t in tones]
    lo = min(values)
    hi = max(values)
    span = hi - lo

    for tone in tones:
        if span <= EPSILON:
            tone.selection_utility = 0.5
        else:
            tone.selection_utility = (float(tone.utility) - lo) / span


def canonical_voice_id(element):
    voice_ctx = element.getContextByClass(stream.Voice)
    if voice_ctx is not None and voice_ctx.id is not None:
        voice_label = str(voice_ctx.id)
    else:
        voice_label = "1"

    staff_num = getattr(element, "staffNumber", None)
    if staff_num is None:
        staff_num = 1

    return f"staff={staff_num}|voice={voice_label}"


def build_source_voice_lookups(source_score):
    source_part = source_score.parts[0]
    exact_lookup = defaultdict(list)
    fuzzy_lookup = defaultdict(list)

    for measure in source_part.getElementsByClass(stream.Measure):
        for element in measure.recurse().notes:
            local_offset = q(element.getOffsetInHierarchy(measure))
            dur = q(element.duration.quarterLength)
            voice_id = canonical_voice_id(element)

            if isinstance(element, note.Note):
                midi_values = [int(element.pitch.midi)]
            elif isinstance(element, chord.Chord):
                midi_values = [int(p.midi) for p in element.pitches]
            else:
                continue

            for midi_value in midi_values:
                exact_key = (int(measure.number), local_offset, dur, midi_value)
                fuzzy_key = (int(measure.number), local_offset, midi_value)
                exact_lookup[exact_key].append(voice_id)
                fuzzy_lookup[fuzzy_key].append(voice_id)

    return exact_lookup, fuzzy_lookup


def annotate_tones_with_musicxml_voices(source_score, reduced_tones):
    exact_lookup, fuzzy_lookup = build_source_voice_lookups(source_score)

    exact_available = {k: list(v) for k, v in exact_lookup.items()}
    fuzzy_available = {k: list(v) for k, v in fuzzy_lookup.items()}

    unmatched = 0

    for tone in reduced_tones:
        measure_number = int(tone.row["measure"])
        measure_offset = q(float(tone.row["measure_offset"]))
        quarter_length = q(float(tone.row["quarter_length"]))
        pitch_midi = int(tone.pitch_midi)

        exact_key = (measure_number, measure_offset, quarter_length, pitch_midi)
        fuzzy_key = (measure_number, measure_offset, pitch_midi)

        voice_id = None
        if exact_key in exact_available and exact_available[exact_key]:
            voice_id = exact_available[exact_key].pop(0)
        elif fuzzy_key in fuzzy_available and fuzzy_available[fuzzy_key]:
            voice_id = fuzzy_available[fuzzy_key].pop(0)
        elif "voice" in tone.row:
            voice_id = f"staff=1|voice={tone.row['voice']}"
        else:
            unmatched += 1
            voice_id = (
                f"unknown|m={measure_number}|off={measure_offset}|"
                f"pitch={pitch_midi}"
            )

        tone.voice_id = voice_id

    return unmatched


def build_activity_slices(matched_tones, measure_infos, debug_slices=False):
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

        active = tuple(
            sorted(
                active,
                key=lambda tone: tone_sort_key(tone, t0),
                reverse=True,
            )
        )

        if debug_slices:
            print()
            print(
                f"[activity slice {slice_count + 1}] "
                f"measure={measure.number} "
                f"abs=({q(t0)} -> {q(t1)}) "
                f"local=({q(t0 - measure.start_abs)} -> {q(t1 - measure.start_abs)})"
            )
            print("  active:")
            for tone in active:
                print("   ", format_tone_debug(tone, t0))

        slices.append(
            ActiveSlice(
                measure_index=measure.index,
                measure_number=measure.number,
                start_local=t0 - measure.start_abs,
                end_local=t1 - measure.start_abs,
                start_abs=t0,
                end_abs=t1,
                active_tones=active,
            )
        )

        slice_count += 1

    return slices, slice_count


def melodic_continuity_bonus(prev_pitch_midi, cand_pitch_midi):
    if prev_pitch_midi is None:
        return 0.0

    interval = abs(int(cand_pitch_midi) - int(prev_pitch_midi))

    if interval == 0:
        return 0.04
    if interval <= 2:
        return 0.03
    if interval <= 5:
        return 0.015
    if interval <= 12:
        return 0.0
    if interval <= 19:
        return -0.02
    return -0.04


def build_voice_candidates_for_slice(active_slice, prev_primary_pitch, attack_bonus):
    grouped = defaultdict(list)
    for tone in active_slice.active_tones:
        grouped[tone_voice_id(tone)].append(tone)

    candidates = {}

    for voice_id, tones in grouped.items():
        def tone_rank_key(tone):
            attacked_here = 1 if abs(tone.start_abs - active_slice.start_abs) <= EPSILON else 0
            melodic = melodic_continuity_bonus(prev_primary_pitch, tone.pitch_midi)
            closeness = 0
            if prev_primary_pitch is not None:
                closeness = -abs(int(tone.pitch_midi) - int(prev_primary_pitch))
            return (
                tone_selection_utility(tone),
                attacked_here,
                melodic,
                closeness,
                tone.pitch_midi,
            )

        best_tone = max(tones, key=tone_rank_key)
        attacked_here = abs(best_tone.start_abs - active_slice.start_abs) <= EPSILON
        melodic = melodic_continuity_bonus(prev_primary_pitch, best_tone.pitch_midi)
        base_score = tone_selection_utility(best_tone) + melodic
        if attacked_here:
            base_score += attack_bonus

        candidates[voice_id] = VoiceCandidate(
            voice_id=voice_id,
            tone=best_tone,
            base_score=base_score,
            attacked_here=attacked_here,
            melodic_bonus=melodic,
        )

    return candidates


def candidate_choice_key(candidate):
    return (
        candidate.base_score,
        1 if candidate.attacked_here else 0,
        tone_selection_utility(candidate.tone),
        candidate.tone.pitch_midi,
    )


def choose_best_candidate(candidates):
    if not candidates:
        return None
    return max(candidates.values(), key=candidate_choice_key)


def choose_secondary_candidate(candidates, primary_voice_id, primary_pitch):
    others = [
        candidate
        for voice_id, candidate in candidates.items()
        if voice_id != primary_voice_id
    ]
    if not others:
        return None

    distinct_pitch = [c for c in others if c.tone.pitch_midi != primary_pitch]
    pool = distinct_pitch if distinct_pitch else others
    return max(pool, key=candidate_choice_key)


def switch_is_confirmed(
    activity_slices,
    start_index,
    current_voice_id,
    challenger,
    confirm_slices,
    attack_bonus,
):
    if confirm_slices <= 0:
        return True

    future_slices = [
        sl for sl in activity_slices[start_index + 1 :]
        if len(sl.active_tones) > 0
    ]
    if not future_slices:
        return True

    needed = min(confirm_slices, len(future_slices))
    checks = 0
    prev_pitch = challenger.tone.pitch_midi

    for future_slice in future_slices:
        future_candidates = build_voice_candidates_for_slice(
            future_slice,
            prev_primary_pitch=prev_pitch,
            attack_bonus=attack_bonus,
        )

        future_challenger = future_candidates.get(challenger.voice_id)
        if future_challenger is None:
            return False

        future_current = future_candidates.get(current_voice_id)
        if future_current is not None:
            if future_challenger.base_score <= future_current.base_score:
                return False

        prev_pitch = future_challenger.tone.pitch_midi
        checks += 1
        if checks >= needed:
            return True

    return False


def select_primary_secondary_slices(
    activity_slices,
    same_voice_bonus=DEFAULT_SAME_VOICE_BONUS,
    switch_threshold=DEFAULT_SWITCH_THRESHOLD,
    attack_bonus=DEFAULT_ATTACK_BONUS,
    confirm_slices=DEFAULT_CONFIRM_SLICES,
    debug_slices=False,
):
    selected = []
    current_voice_id = None
    prev_primary_pitch = None

    for index, active_slice in enumerate(activity_slices):
        candidates = build_voice_candidates_for_slice(
            active_slice,
            prev_primary_pitch=prev_primary_pitch,
            attack_bonus=attack_bonus,
        )

        primary_candidate = None
        decision = "REST"
        stay_candidate = None
        best_competitor = None

        if candidates:
            if current_voice_id is None or current_voice_id not in candidates:
                primary_candidate = choose_best_candidate(candidates)
                if current_voice_id is None:
                    decision = "START"
                else:
                    decision = "RESELECT"
            else:
                stay_candidate = candidates[current_voice_id]
                competitors = [
                    candidate
                    for voice_id, candidate in candidates.items()
                    if voice_id != current_voice_id
                ]
                best_competitor = (
                    max(competitors, key=candidate_choice_key)
                    if competitors else None
                )

                stay_score = stay_candidate.base_score + same_voice_bonus

                if (
                    best_competitor is not None
                    and best_competitor.base_score > stay_score + switch_threshold
                    and switch_is_confirmed(
                        activity_slices=activity_slices,
                        start_index=index,
                        current_voice_id=current_voice_id,
                        challenger=best_competitor,
                        confirm_slices=confirm_slices,
                        attack_bonus=attack_bonus,
                    )
                ):
                    primary_candidate = best_competitor
                    decision = "SWITCH"
                else:
                    primary_candidate = stay_candidate
                    decision = "STAY"

        tones = []
        if primary_candidate is not None:
            current_voice_id = primary_candidate.voice_id
            prev_primary_pitch = primary_candidate.tone.pitch_midi
            tones.append(primary_candidate.tone)

            secondary_candidate = choose_secondary_candidate(
                candidates=candidates,
                primary_voice_id=primary_candidate.voice_id,
                primary_pitch=primary_candidate.tone.pitch_midi,
            )
            if secondary_candidate is not None:
                tones.append(secondary_candidate.tone)
        else:
            current_voice_id = None

        if debug_slices:
            print()
            print(
                f"[primary slice {index + 1}] "
                f"measure={active_slice.measure_number} "
                f"abs=({q(active_slice.start_abs)} -> {q(active_slice.end_abs)}) "
                f"local=({q(active_slice.start_local)} -> {q(active_slice.end_local)})"
            )
            print(f"  decision={decision}")

            if candidates:
                print("  candidates:")
                for candidate in sorted(
                    candidates.values(),
                    key=candidate_choice_key,
                    reverse=True,
                ):
                    stay_mark = " <= current" if candidate.voice_id == current_voice_id else ""
                    print(
                        "   "
                        f"voice={candidate.voice_id} "
                        f"pitch={candidate.tone.pitch_midi} "
                        f"base={candidate.base_score:.6f} "
                        f"util={tone_selection_utility(candidate.tone):.6f} "
                        f"mel={candidate.melodic_bonus:.6f} "
                        f"{'ATTACK' if candidate.attacked_here else 'HELD'}"
                        f"{stay_mark}"
                    )

                if stay_candidate is not None:
                    print(
                        f"  stay_score={stay_candidate.base_score + same_voice_bonus:.6f} "
                        f"(base {stay_candidate.base_score:.6f} + same_voice_bonus {same_voice_bonus:.6f})"
                    )
                if best_competitor is not None:
                    print(
                        f"  best_competitor={best_competitor.voice_id} "
                        f"score={best_competitor.base_score:.6f}"
                    )

            if tones:
                print("  chosen:")
                labels = ["PRIMARY", "SECONDARY"]
                for i, tone_obj in enumerate(tones):
                    label = labels[i] if i < len(labels) else f"TONE{i + 1}"
                    print("   ", label, format_tone_debug(tone_obj, active_slice.start_abs))
            else:
                print("  chosen: rest")

        selected.append(
            SliceSelection(
                measure_index=active_slice.measure_index,
                measure_number=active_slice.measure_number,
                start_local=active_slice.start_local,
                end_local=active_slice.end_local,
                start_abs=active_slice.start_abs,
                end_abs=active_slice.end_abs,
                tones=tuple(tones),
            )
        )

    return selected, len(activity_slices)


def assign_slices_to_utility_streams(slices):
    """
    Stream 1 = primary line selected by the voice-aware tracker
    Stream 2 = highest remaining active candidate for local fill
    """
    stream_cells = {
        1: defaultdict(list),
        2: defaultdict(list),
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
    
    beat_q_len = get_beat_length(source_measure)
    measure_len = get_measure_length(source_measure)

    primary_events = sorted(primary_events, key=lambda e: e.offset)
    secondary_events = sorted(secondary_events, key=lambda e: e.offset)

    selected = []

    def clipped_events_in_span(events, span_start, span_end, source_name):
        clipped = []

        if span_end <= span_start + EPSILON:
            return clipped

        for e in events:
            if e.end <= span_start + EPSILON:
                continue
            if e.offset >= span_end - EPSILON:
                break

            start = max(float(e.offset), float(span_start))
            end = min(float(e.end), float(span_end))
            dur = end - start

            if dur <= EPSILON:
                continue

            clipped.append(
                TimedEvent(
                    offset=start,
                    duration=dur,
                    pitch=e.pitch,
                    source=source_name,
                )
            )

        return clipped

    # No primary voice check
    if not primary_events:
        selected.extend(
            clipped_events_in_span(
                secondary_events,
                0.0,
                measure_len,
                "secondary",
            )
        )

        return render_monophonic_measure(
            measure_number=source_measure.number,
            ts=source_measure.getContextByClass(meter.TimeSignature),
            measure_length=measure_len,
            events=selected,
            show_ts=show_ts,
        )

    current_time = 0.0

    for prim in primary_events:
        prim_start = max(0.0, float(prim.offset))
        prim_end = min(float(prim.end), float(measure_len))

        if prim_end <= prim_start + EPSILON:
            continue

        # 1) Before the primary enters, let the secondary stream lead.
        if prim_start > current_time + EPSILON:
            selected.extend(
                clipped_events_in_span(
                    secondary_events,
                    current_time,
                    prim_start,
                    "secondary",
                )
            )

        # 2) Keep the current primary-note reduction behavior.
        if len(secondary_events) > 0 and prim.duration > float(secondary_events[0].duration):
            primary_dur = min(float(secondary_events[0].duration), prim_end - prim_start)
        else:
            primary_dur = min(float(beat_q_len), prim_end - prim_start)

        if primary_dur > EPSILON:
            selected.append(
                TimedEvent(
                    offset=prim_start,
                    duration=primary_dur,
                    pitch=prim.pitch,
                    source="primary",
                )
            )

        # 3) After the reduced primary note ends, if there is uncovered span
        # inside the original primary span, let the secondary stream carry it
        # verbatim rather than treating it as ad hoc filler.
        uncovered_start = prim_start + primary_dur
        uncovered_end = prim_end

        if uncovered_end > uncovered_start + EPSILON:
            selected.extend(
                clipped_events_in_span(
                    secondary_events,
                    uncovered_start,
                    uncovered_end,
                    "secondary",
                )
            )

        current_time = max(current_time, prim_end)

    # 4) After the last primary event, let the secondary stream lead the tail.
    if current_time < measure_len - EPSILON:
        selected.extend(
            clipped_events_in_span(
                secondary_events,
                current_time,
                measure_len,
                "secondary",
            )
        )

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


def build_reduction_voice(
    json_path,
    output_xml,
    layer,
    source_xml=None,
    debug_slices=False,
    same_voice_bonus=DEFAULT_SAME_VOICE_BONUS,
    switch_threshold=DEFAULT_SWITCH_THRESHOLD,
    attack_bonus=DEFAULT_ATTACK_BONUS,
    confirm_slices=DEFAULT_CONFIRM_SLICES,
):
    json_path = Path(json_path)
    if not json_path.is_absolute():
        json_path = projRoot / json_path

    if output_xml is None:
        output_xml = outputDir / f"{json_path.stem}-monophonic_voice.musicxml"
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

    normalize_tone_utilities(reduced_tones)
    unmatched = annotate_tones_with_musicxml_voices(source_score, reduced_tones)

    activity_slices, raw_slice_count = build_activity_slices(
        matched_tones=reduced_tones,
        measure_infos=measure_infos,
        debug_slices=False,
    )

    selected_slices, slice_count = select_primary_secondary_slices(
        activity_slices=activity_slices,
        same_voice_bonus=same_voice_bonus,
        switch_threshold=switch_threshold,
        attack_bonus=attack_bonus,
        confirm_slices=confirm_slices,
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

    print(f"Built {raw_slice_count} activity slices")
    print(f"Selected {slice_count} primary slices")
    print(f"Unmatched MusicXML voice assignments: {unmatched}")
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
        "--same-voice-bonus",
        type=float,
        default=DEFAULT_SAME_VOICE_BONUS,
        help="Bonus for staying on the current MusicXML voice.",
    )

    parser.add_argument(
        "--switch-threshold",
        type=float,
        default=DEFAULT_SWITCH_THRESHOLD,
        help="Extra margin required before switching to another voice.",
    )

    parser.add_argument(
        "--attack-bonus",
        type=float,
        default=DEFAULT_ATTACK_BONUS,
        help="Small bonus for a newly attacked note over a held note.",
    )

    parser.add_argument(
        "--confirm-slices",
        type=int,
        default=DEFAULT_CONFIRM_SLICES,
        help="How many future slices must support a switch before it occurs.",
    )

    parser.add_argument(
        "--debug-slices",
        action="store_true",
        help="Print candidate voices and stay/switch decisions for every slice.",
    )

    args = parser.parse_args()

    build_reduction_voice(
        json_path=args.json,
        output_xml=args.output_xml,
        layer=args.layer,
        source_xml=args.source_xml,
        debug_slices=args.debug_slices,
        same_voice_bonus=args.same_voice_bonus,
        switch_threshold=args.switch_threshold,
        attack_bonus=args.attack_bonus,
        confirm_slices=args.confirm_slices,
    )


if __name__ == "__main__":
    main()
