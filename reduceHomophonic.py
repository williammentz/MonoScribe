import copy
from collections import defaultdict
from dataclasses import dataclass
import music21
from music21 import converter, stream, note, chord, duration, meter, environment, tempo
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
    """
    Prefer stable ordering by numeric id if possible, otherwise by string id.
    This lets Voice 1 map to upper lane and Voice 2 to lower lane when present.
    """
    if v.id is None:
        return (1, "")
    try:
        return (0, int(v.id))
    except Exception:
        return (1, str(v.id))


def split_single_part_into_two_lanes(part):
    """
    Convert a single part with up to two voices per measure into two synthetic parts:
    upper_lane and lower_lane.

    This is only an input adapter. It does not change the reduction logic.
    """
    upper_lane = stream.Part(id="UpperLane")
    lower_lane = stream.Part(id="LowerLane")

    upper_lane.partName = "Upper Lane"
    lower_lane.partName = "Lower Lane"

    measures = list(part.getElementsByClass(stream.Measure))

    for measure in measures:
        upper_measure = clone_measure_shell(measure)
        lower_measure = clone_measure_shell(measure)

        voices = list(measure.getElementsByClass(stream.Voice))
        voices = sorted(voices, key=voice_sort_key)

        if len(voices) > 2:
            raise ValueError(
                f"Measure {measure.number} has more than 2 voices; "
                "this adapter currently expects at most 2."
            )

        if len(voices) == 0:
            # No explicit voices: leave measure shells empty.
            # Could be extended later if needed.
            pass

        elif len(voices) == 1:
            # Put the only voice in upper lane by default.
            for el in voices[0]:
                if isinstance(el, note.GeneralNote):
                    upper_measure.insert(el.offset, copy.deepcopy(el))

        else:
            # Voice 1 -> upper lane, Voice 2 -> lower lane
            for el in voices[0]:
                if isinstance(el, note.GeneralNote):
                    upper_measure.insert(el.offset, copy.deepcopy(el))

            for el in voices[1]:
                if isinstance(el, note.GeneralNote):
                    lower_measure.insert(el.offset, copy.deepcopy(el))

        upper_lane.append(upper_measure)
        lower_lane.append(lower_measure)

    return upper_lane, lower_lane


def load_score_as_lanes(file_path):
    score = converter.parse(file_path)
    parts = score.parts

    if len(parts) >= 2:
        # Preserve current behavior exactly:
        # part 0 = upper lane, part 1 = lower lane
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


def extract_melody(measure):
    offset_map = defaultdict(list)

    for el in measure.flatten().notesAndRests:
        if not is_real_note(el):
            continue

        if isinstance(el, note.Note) and el.tie and el.tie.type in ("stop", "continue"):
            continue

        offset_map[float(el.offset)].append(el)

    melody = []

    for off in sorted(offset_map.keys()):
        elems = offset_map[off]
        candidates = []

        for e in elems:
            ql = float(e.duration.quarterLength)

            if isinstance(e, chord.Chord):
                for p in e.pitches:
                    candidates.append((p, ql))
            elif isinstance(e, note.Note):
                candidates.append((e.pitch, ql))

        if not candidates:
            continue

        top_pitch, top_dur = max(candidates, key=lambda x: x[0].midi)

        melody.append(
            TimedEvent(
                offset=off,
                duration=top_dur,
                pitch=top_pitch,
                source="melody"
            )
        )

    return melody


def get_lower_lane_events(lower_measure):
    lower_events = []

    for el in lower_measure.flatten().notesAndRests:
        if not is_real_note(el):
            continue

        off = float(el.offset)
        ql = float(el.duration.quarterLength)

        if isinstance(el, chord.Chord):
            for p in el.pitches:
                lower_events.append(
                    TimedEvent(
                        offset=off,
                        duration=ql,
                        pitch=p,
                        source="lh"
                    )
                )
        elif isinstance(el, note.Note):
            lower_events.append(
                TimedEvent(
                    offset=off,
                    duration=ql,
                    pitch=el.pitch,
                    source="lh"
                )
            )

    lower_events.sort(key=lambda e: (e.offset, e.pitch.midi))
    return lower_events


def reduce_measure(upper_measure, lower_measure, show_ts=False):
    beat_q_len = get_beat_length(upper_measure)
    measure_len = get_measure_length(upper_measure)

    melody_events = extract_melody(upper_measure)
    lower_events = get_lower_lane_events(lower_measure)

    selected = []

    for mel in melody_events:
        mel_end = min(mel.end, measure_len)

        if len(lower_events) > 0:
            if mel.duration > float(lower_events[0].duration):
                selected.append(
                    TimedEvent(
                        offset=mel.offset,
                        duration=lower_events[0].duration,
                        pitch=mel.pitch,
                        source="melody"
                    )
                )
                fill_start = mel.offset + float(lower_events[0].duration)
                fill_end = mel_end

                candidates = [
                    e for e in lower_events
                    if fill_start <= e.offset < fill_end
                ]
                for c in candidates:
                    selected.append(c)
                continue

        selected.append(
            TimedEvent(
                offset=mel.offset,
                duration=beat_q_len,
                pitch=mel.pitch,
                source="melody"
            )
        )

        fill_start = mel.offset + beat_q_len
        fill_end = mel_end

        candidates = [
            e for e in lower_events
            if fill_start <= e.offset < fill_end
        ]

        for c in candidates:
            selected.append(c)

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
        while n.pitch.midi < ranges[instrument][0] or n.pitch.midi > ranges[instrument][1]:
            if n.pitch.midi < ranges[instrument][0]:
                n.pitch.midi = n.pitch.midi + 12
            elif n.pitch.midi > ranges[instrument][1]:
                n.pitch.midi = n.pitch.midi - 12


def render_monophonic_measure(measure_number, ts, measure_length, events, show_ts=False):
    new_measure = stream.Measure(number=measure_number)

    if show_ts and ts is not None:
        new_measure.timeSignature = copy.deepcopy(ts)

    def priority(e):
        return 0 if e.source == "melody" else 1

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


def reduce_score(score, upper_lane, lower_lane, instrument='piano'):
    reduced_part = stream.Part()
    reduced_part.id = "ReducedMonophonic"

    upper_measures = list(upper_lane.getElementsByClass(stream.Measure))
    last_ts_ratio = None

    for upper_measure in upper_measures:
        lower_measure = get_measure_by_number(lower_lane, upper_measure.number)
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


def print_melody_debug(upper_lane, num_measures=4):
    for m in upper_lane.getElementsByClass(stream.Measure)[:num_measures]:
        print(f"\nMeasure {m.number} — melody extraction:")
        for ev in extract_melody(m):
            print(
                f"  melody: pitch={ev.pitch.midi}, offset={ev.offset}, dur={ev.duration}"
            )


def print_lower_lane_debug(lower_lane, measure_number=1):
    m = get_measure_by_number(lower_lane, measure_number)
    if m is None:
        print(f"No lower-lane measure {measure_number}")
        return

    print(f"\nLower-lane events for measure {measure_number}:")
    for ev in get_lower_lane_events(m):
        print(
            f"  lower: pitch={ev.pitch.midi}, offset={ev.offset}, dur={ev.duration}"
        )

if __name__ == "__main__":
    score, upper_lane, lower_lane = load_score_as_lanes('reduction_scores/Mozart_12.musicxml')

    print_melody_debug(upper_lane, num_measures=4)
    print_lower_lane_debug(lower_lane, measure_number=1)

    reduced = reduce_score(score, upper_lane, lower_lane, 'clarinet')

    out_score = stream.Score()
    out_score.insert(0, reduced)

    out_score.show("musicxml")