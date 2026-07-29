import copy
from collections import defaultdict
from dataclasses import dataclass
import music21
from music21 import chord, converter, duration, environment, meter, note, stream, tempo
import json

EPSILON = 1e-6

us = environment.UserSettings()
us['musicxmlPath'] = '/Applications/MuseScore 4.app/Contents/MacOS/mscore'


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
class TimedSonority:
    offset: float
    duration: float
    pitches: tuple
    source: str = "support"

    @property
    def end(self):
        return self.offset + self.duration


def is_real_note(el):
    if isinstance(el, note.Rest):
        return False
    if not hasattr(el, "duration"):
        return False
    if el.duration.isGrace:
        return False
    return isinstance(el, (note.Note, chord.Chord))


def get_measure_by_number(part, number):
    for m in part.getElementsByClass(stream.Measure):
        if m.number == number:
            return m
    return None


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


def voice_sort_key(v):
    if v.id is None:
        return (1, "")
    try:
        return (0, int(v.id))
    except Exception:
        return (1, str(v.id))


def copy_general_notes(source, target):
    for el in source:
        if isinstance(el, note.GeneralNote):
            target.insert(el.offset, copy.deepcopy(el))


def split_single_part_into_two_lanes(part):
    upper_lane = stream.Part(id="UpperLane")
    lower_lane = stream.Part(id="LowerLane")

    upper_lane.partName = "Upper Lane"
    lower_lane.partName = "Lower Lane"

    for measure in part.getElementsByClass(stream.Measure):
        upper_measure = clone_measure_shell(measure)
        lower_measure = clone_measure_shell(measure)

        voices = sorted(
            measure.getElementsByClass(stream.Voice),
            key=voice_sort_key
        )

        if len(voices) > 2:
            raise ValueError(
                f"Measure {measure.number} has more than 2 voices; "
                "this adapter expects at most 2."
            )

        if len(voices) == 0:
            copy_general_notes(measure, upper_measure)
        elif len(voices) == 1:
            copy_general_notes(voices[0], upper_measure)
        else:
            copy_general_notes(voices[0], upper_measure)
            copy_general_notes(voices[1], lower_measure)

        upper_lane.append(upper_measure)
        lower_lane.append(lower_measure)

    return upper_lane, lower_lane


def load_score_as_lanes(file_path):
    score = converter.parse(file_path)
    parts = score.parts

    if len(parts) >= 2:
        upper_lane = parts[0]
        lower_lane = parts[1]
        return score, upper_lane, lower_lane

    if len(parts) == 1:
        upper_lane, lower_lane = split_single_part_into_two_lanes(parts[0])
        return score, upper_lane, lower_lane

    raise ValueError("Score contains no parts.")


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


def element_pitch_events(el, source):
    off = float(el.offset)
    ql = float(el.duration.quarterLength)

    if isinstance(el, note.Note):
        return [
            TimedEvent(
                offset=off,
                duration=ql,
                pitch=copy.deepcopy(el.pitch),
                source=source
            )
        ]

    return [
        TimedEvent(
            offset=off,
            duration=ql,
            pitch=copy.deepcopy(p),
            source=source
        )
        for p in el.pitches
    ]


def extract_upper_material(measure):
    offset_map = defaultdict(list)

    for el in measure.flatten().notesAndRests:
        if not is_real_note(el):
            continue

        if isinstance(el, note.Note) and el.tie and el.tie.type in ("stop", "continue"):
            continue

        for event in element_pitch_events(el, "upper"):
            offset_map[event.offset].append(event)

    melody = []
    support = []

    for off in sorted(offset_map):
        candidates = offset_map[off]
        top = max(candidates, key=lambda e: e.pitch.midi)

        melody.append(
            TimedEvent(
                offset=top.offset,
                duration=top.duration,
                pitch=copy.deepcopy(top.pitch),
                source="melody"
            )
        )

        seen = {top.pitch.midi}
        for event in sorted(candidates, key=lambda e: e.pitch.midi):
            midi = event.pitch.midi
            if midi in seen:
                continue

            seen.add(midi)
            support.append(
                TimedEvent(
                    offset=event.offset,
                    duration=event.duration,
                    pitch=copy.deepcopy(event.pitch),
                    source="support"
                )
            )

    return melody, support


def extract_lower_material(measure):
    events = []

    for el in measure.flatten().notesAndRests:
        if not is_real_note(el):
            continue
        events.extend(element_pitch_events(el, "support"))

    return events


def group_support_events(events):
    offset_map = defaultdict(list)

    for event in events:
        offset_map[event.offset].append(event)

    sonorities = []

    for off in sorted(offset_map):
        grouped = offset_map[off]
        pitches = {}

        for event in grouped:
            pitches.setdefault(event.pitch.midi, copy.deepcopy(event.pitch))

        sonorities.append(
            TimedSonority(
                offset=off,
                duration=max(event.end for event in grouped) - off,
                pitches=tuple(
                    pitches[midi] for midi in sorted(pitches)
                )
            )
        )

    return sonorities


def unique_ordered_pitches(pitches, excluded_midi=None):
    excluded_midi = set() if excluded_midi is None else set(excluded_midi)
    unique = {}

    for p in pitches:
        if p.midi in excluded_midi:
            continue
        unique.setdefault(p.midi, copy.deepcopy(p))

    return [unique[midi] for midi in sorted(unique)]


def get_local_rhythmic_unit(sonorities, index, default_unit):
    sonority = sonorities[index]
    candidates = [default_unit]

    if sonority.duration > EPSILON:
        candidates.append(sonority.duration)

    if index + 1 < len(sonorities):
        onset_distance = sonorities[index + 1].offset - sonority.offset
        if onset_distance > EPSILON:
            candidates.append(onset_distance)

    return min(value for value in candidates if value > EPSILON)


def available_slots(start, end, rhythmic_unit):
    if end - start <= EPSILON:
        return 0

    return max(1, int(((end - start) + EPSILON) / rhythmic_unit))


def select_melody_block_pitches(melody_pitch, support_pitches):
    support = unique_ordered_pitches(support_pitches, excluded_midi={melody_pitch.midi})

    selected = [(copy.deepcopy(melody_pitch), "melody")]

    if not support:
        return selected

    bass = support[0]
    selected.append((copy.deepcopy(bass), "support"))

    remaining = support[1:]
    if remaining:
        selected.append((copy.deepcopy(remaining[-1]), "support"))

    return selected


def select_support_block_pitches(pitches, slots):
    ordered = unique_ordered_pitches(pitches)

    if not ordered or slots <= 0:
        return []

    if len(ordered) == 1 or slots == 1:
        return [(ordered[0], "support")]

    if len(ordered) == 2 or slots == 2:
        return [(ordered[0], "support"), (ordered[-1], "support")]

    inner = ordered[1:-1]
    if not inner:
        return [(ordered[0], "support"), (ordered[-1], "support")]

    middle_pitch = inner[len(inner) // 2]
    return [(ordered[0], "support"), (middle_pitch, "support"), (ordered[-1], "support")]


def schedule_on_rhythmic_grid(start, end, pitch_sources, rhythmic_unit):
    slots = available_slots(start, end, rhythmic_unit)
    events = []

    for index, (pitch_value, source) in enumerate(pitch_sources[:slots]):
        offset = start + index * rhythmic_unit
        event_end = min(offset + rhythmic_unit, end)

        if event_end - offset <= EPSILON:
            break

        events.append(
            TimedEvent(
                offset=offset,
                duration=event_end - offset,
                pitch=copy.deepcopy(pitch_value),
                source=source
            )
        )

    return events


def render_support_region(
    sonorities,
    region_start,
    region_end,
    default_unit
):
    groups = [
        sonority for sonority in sonorities
        if region_start - EPSILON <= sonority.offset < region_end - EPSILON
    ]

    events = []

    for index, sonority in enumerate(groups):
        next_offset = region_end
        if index + 1 < len(groups):
            next_offset = min(
                next_offset,
                groups[index + 1].offset
            )

        start = max(region_start, sonority.offset)
        end = min(region_end, sonority.end, next_offset)

        if end - start <= EPSILON:
            continue

        pitches = unique_ordered_pitches(sonority.pitches)

        if len(pitches) == 1:
            events.append(
                TimedEvent(
                    offset=start,
                    duration=end - start,
                    pitch=copy.deepcopy(pitches[0]),
                    source="support"
                )
            )
            continue

        rhythmic_unit = get_local_rhythmic_unit(
            groups,
            index,
            default_unit
        )
        slots = available_slots(start, end, rhythmic_unit)
        selected = select_support_block_pitches(pitches, slots)

        events.extend(
            schedule_on_rhythmic_grid(
                start,
                end,
                selected,
                rhythmic_unit
            )
        )

    return events


def reduce_melody_window(
    melody,
    window_end,
    sonorities,
    default_unit
):
    events = []
    groups = [
        sonority for sonority in sonorities
        if melody.offset - EPSILON <= sonority.offset < window_end - EPSILON
    ]

    start_group = None
    later_groups = []

    for sonority in groups:
        if (start_group is None and abs(sonority.offset - melody.offset) <= EPSILON):
            start_group = sonority
        else:
            later_groups.append(sonority)

    melody_end = min(melody.end, window_end)

    if start_group is None:
        first_support_offset = (
            later_groups[0].offset
            if later_groups
            else window_end
        )
        end = min(melody_end, first_support_offset)

        if end - melody.offset > EPSILON:
            events.append(
                TimedEvent(
                    offset=melody.offset,
                    duration=end - melody.offset,
                    pitch=copy.deepcopy(melody.pitch),
                    source="melody"
                )
            )

        events.extend(
            render_support_region(later_groups, melody.offset, window_end, default_unit))
        return events

    support_pitches = unique_ordered_pitches(start_group.pitches, excluded_midi={melody.pitch.midi})

    if not support_pitches:
        first_support_offset = (
            later_groups[0].offset
            if later_groups
            else window_end
        )
        end = min(melody_end, first_support_offset)

        if end - melody.offset > EPSILON:
            events.append(
                TimedEvent(
                    offset=melody.offset,
                    duration=end - melody.offset,
                    pitch=copy.deepcopy(melody.pitch),
                    source="melody"
                )
            )

        events.extend(
            render_support_region(later_groups, melody.offset, window_end, default_unit)
        )
        return events

    local_groups = [start_group] + later_groups
    rhythmic_unit = get_local_rhythmic_unit(
        local_groups,
        0,
        default_unit
    )

    later_during_melody = [
        sonority for sonority in later_groups
        if sonority.offset < melody_end - EPSILON
    ]

    is_broken_chord_start = (
        len(support_pitches) == 1
        and len(later_during_melody) > 0
    )

    if is_broken_chord_start:
        first_later_offset = later_during_melody[0].offset
        protected_end = min(
            melody_end,
            melody.offset + rhythmic_unit,
            first_later_offset
        )

        if protected_end - melody.offset > EPSILON:
            events.append(
                TimedEvent(
                    offset=melody.offset,
                    duration=protected_end - melody.offset,
                    pitch=copy.deepcopy(melody.pitch),
                    source="melody"
                )
            )
    else:
        next_support_offset = (
            later_groups[0].offset
            if later_groups
            else window_end
        )
        block_end = min(
            window_end,
            next_support_offset,
            max(melody.end, start_group.end)
        )

        selected = select_melody_block_pitches(
            melody.pitch,
            support_pitches
        )

        events.extend(
            schedule_on_rhythmic_grid(
                melody.offset,
                block_end,
                selected,
                rhythmic_unit
            )
        )

    events.extend(
        render_support_region(
            later_groups,
            melody.offset,
            window_end,
            default_unit
        )
    )

    return events


def reduce_measure(upper_measure, lower_measure, show_ts=False):
    measure_len = get_measure_length(upper_measure)
    beat_q_len = get_beat_length(upper_measure)

    melody_events, upper_support = extract_upper_material(upper_measure)
    lower_support = extract_lower_material(lower_measure)
    sonorities = group_support_events(upper_support + lower_support)

    selected = []

    if not melody_events:
        selected.extend(
            render_support_region(
                sonorities,
                0.0,
                measure_len,
                beat_q_len
            )
        )
    else:
        support_index = 0
        first_melody_offset = melody_events[0].offset

        prelude = []
        while (
            support_index < len(sonorities)
            and sonorities[support_index].offset < first_melody_offset - EPSILON
        ):
            prelude.append(sonorities[support_index])
            support_index += 1

        selected.extend(
            render_support_region(
                prelude,
                0.0,
                first_melody_offset,
                beat_q_len
            )
        )

        for index, melody in enumerate(melody_events):
            window_end = measure_len

            if index + 1 < len(melody_events):
                window_end = min(
                    window_end,
                    melody_events[index + 1].offset
                )

            window_support = []
            while (
                support_index < len(sonorities)
                and sonorities[support_index].offset < window_end - EPSILON
            ):
                if sonorities[support_index].offset >= melody.offset - EPSILON:
                    window_support.append(sonorities[support_index])
                support_index += 1

            selected.extend(
                reduce_melody_window(
                    melody,
                    window_end,
                    window_support,
                    beat_q_len
                )
            )

    return render_monophonic_measure(
        measure_number=upper_measure.number,
        ts=upper_measure.getContextByClass(meter.TimeSignature),
        measure_length=measure_len,
        events=selected,
        show_ts=show_ts
    )


def range_check(part, instrument='piano'):
    with open("utils/instruments.json") as f:
        ranges = {k: tuple(v) for k, v in json.load(f).items()}

    instrument = instrument.lower()

    for n in part.flatten().notes:
        while (
            n.pitch.midi < ranges[instrument][0]
            or n.pitch.midi > ranges[instrument][1]
        ):
            if n.pitch.midi < ranges[instrument][0]:
                n.pitch.midi += 12
            elif n.pitch.midi > ranges[instrument][1]:
                n.pitch.midi -= 12


def render_monophonic_measure(
    measure_number,
    ts,
    measure_length,
    events,
    show_ts=False
):
    new_measure = stream.Measure(number=measure_number)

    if show_ts and ts is not None:
        new_measure.timeSignature = copy.deepcopy(ts)

    def priority(event):
        return 0 if event.source == "melody" else 1

    events = sorted(
        events,
        key=lambda e: (e.offset, priority(e), e.pitch.midi)
    )

    current_time = 0.0

    for index, event in enumerate(events):
        start = max(event.offset, current_time)
        if start >= measure_length - EPSILON:
            break

        next_offset = measure_length
        if index + 1 < len(events):
            next_offset = min(next_offset, events[index + 1].offset)

        end = min(event.end, next_offset, measure_length)
        event_duration = end - start

        if event_duration <= EPSILON:
            continue

        gap = start - current_time
        if gap > EPSILON:
            r = note.Rest()
            r.duration = duration.Duration(gap)
            new_measure.append(r)

        n = note.Note(copy.deepcopy(event.pitch))
        n.duration = duration.Duration(event_duration)
        new_measure.append(n)

        current_time = end

    tail = measure_length - current_time
    if tail > EPSILON:
        r = note.Rest()
        r.duration = duration.Duration(tail)
        new_measure.append(r)

    return new_measure


def reduce_score(score, upper_lane, lower_lane, instrument='piano'):
    reduced_part = stream.Part()
    reduced_part.id = "ReducedMonophonic"

    upper_measures = list(upper_lane.getElementsByClass(stream.Measure))
    last_ts_ratio = None

    for upper_measure in upper_measures:
        lower_measure = get_measure_by_number(
            lower_lane,
            upper_measure.number
        )
        if lower_measure is None:
            lower_measure = stream.Measure(number=upper_measure.number)

        ts = upper_measure.getContextByClass(meter.TimeSignature)
        ts_ratio = ts.ratioString if ts is not None else None
        show_ts = ts_ratio != last_ts_ratio

        reduced_measure = reduce_measure(
            upper_measure,
            lower_measure,
            show_ts=show_ts
        )
        reduced_part.append(reduced_measure)

        last_ts_ratio = ts_ratio

    tempo_marks(score, reduced_part)
    range_check(reduced_part, instrument)

    return reduced_part


def print_measure_debug(upper_lane, lower_lane, measure_number=1):
    upper_measure = get_measure_by_number(upper_lane, measure_number)
    lower_measure = get_measure_by_number(lower_lane, measure_number)

    if upper_measure is None:
        print(f"No upper-lane measure {measure_number}")
        return

    if lower_measure is None:
        lower_measure = stream.Measure(number=measure_number)

    melody, upper_support = extract_upper_material(upper_measure)
    lower_support = extract_lower_material(lower_measure)
    sonorities = group_support_events(upper_support + lower_support)

    print(f"\nMeasure {measure_number} — melody:")
    for event in melody:
        print(
            f"  pitch={event.pitch.midi}, "
            f"offset={event.offset}, dur={event.duration}"
        )

    print(f"\nMeasure {measure_number} — support sonorities:")
    for sonority in sonorities:
        pitches = [p.midi for p in sonority.pitches]
        print(
            f"  pitches={pitches}, "
            f"offset={sonority.offset}, dur={sonority.duration}"
        )


if __name__ == "__main__":
    score, upper_lane, lower_lane = load_score_as_lanes(
        'reduction_scores/Mozart_12.musicxml'
    )

    print_measure_debug(
        upper_lane,
        lower_lane,
        measure_number=1
    )

    reduced = reduce_score(
        score,
        upper_lane,
        lower_lane,
        instrument='clarinet'
    )

    out_score = stream.Score()
    out_score.insert(0, reduced)
    out_score.show("musicxml")