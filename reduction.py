import copy
from collections import defaultdict
from dataclasses import dataclass
import music21
from music21 import converter, stream, note, chord, duration, meter, environment, tempo
import json

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

def load_score(file_path):
    score = converter.parse(file_path)
    parts = score.parts

    if len(parts) < 2:
        raise ValueError("Expected at least two parts: treble and bass.")

    treble = parts[0]
    bass = parts[1]
    return score, treble, bass

def is_real_note(el):
    if isinstance(el, note.Rest):
        return False
    if not hasattr(el, "duration"):
        return False
    if el.duration.isGrace:
        return False
    return isinstance(el, (note.Note, chord.Chord))

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


def get_measure_by_number(part, number):
    for m in part.getElementsByClass(stream.Measure):
        if m.number == number:
            return m
    return None


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

        # skip tie continuations as fresh melodic attacks
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


def get_lh_arpeggiation_events(bass_measure):
    lh_events = []

    for el in bass_measure.flatten().notesAndRests:
        if not is_real_note(el):
            continue

        off = float(el.offset)
        ql = float(el.duration.quarterLength)

        if isinstance(el, chord.Chord):
            for p in el.pitches:
                lh_events.append(
                    TimedEvent(
                        offset=off,
                        duration=ql,
                        pitch=p,
                        source="lh"
                    )
                )
        elif isinstance(el, note.Note):
            lh_events.append(
                TimedEvent(
                    offset=off,
                    duration=ql,
                    pitch=el.pitch,
                    source="lh"
                )
            )

    lh_events.sort(key=lambda e: (e.offset, e.pitch.midi))
    return lh_events

def reduce_measure(measure, bass_measure, show_ts = False):
    beat_q_len = get_beat_length(measure)
    measure_len = get_measure_length(measure)

    melody_events = extract_melody(measure)
    lh_events = get_lh_arpeggiation_events(bass_measure)

    selected = []

    for mel in melody_events:
        mel_end = min(mel.end, measure_len)

        if len(lh_events) > 0:
            if mel.duration > float(lh_events[0].duration):
                selected.append(
                    TimedEvent(
                        offset=mel.offset,
                        duration=lh_events[0].duration,
                        pitch=mel.pitch,
                        source="melody"
                    )
                )
                fill_start = mel.offset + float(lh_events[0].duration)
                fill_end = mel_end

                candidates = [
                    e for e in lh_events
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
            e for e in lh_events
            if fill_start <= e.offset < fill_end
        ]

        for c in candidates:
            selected.append(c)

    return render_monophonic_measure(
        measure_number=measure.number,
        ts=measure.getContextByClass(meter.TimeSignature),
        measure_length=measure_len,
        events=selected,
        show_ts=show_ts
    )

def range_check(part, instrument = 'piano'):
    with open("instruments.json") as f:
        ranges = {k: tuple(v) for k, v in json.load(f).items()}

    instrument = instrument.lower()

    for note in part.flatten().notes:
        while note.pitch.midi < ranges[instrument][0] or note.pitch.midi > ranges[instrument][1]:
            if note.pitch.midi < ranges[instrument][0]:
                note.pitch.midi = note.pitch.midi + 12
            elif note.pitch.midi > ranges[instrument][1]:
                note.pitch.midi = note.pitch.midi - 12

def render_monophonic_measure(measure_number, ts, measure_length, events, show_ts = False):
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

        if dur <= 0:
            continue

        if start > current_time:
            r = note.Rest()
            r.duration = duration.Duration(start - current_time)
            new_measure.append(r)

        n = note.Note(ev.pitch)
        n.duration = duration.Duration(dur)
        new_measure.append(n)

        current_time = start + dur

    if current_time < measure_length:
        r = note.Rest()
        r.duration = duration.Duration(measure_length - current_time)
        new_measure.append(r)

    return new_measure


def reduce_score(score, treble, bass, instrument = 'piano'):
    reduced_part = stream.Part()
    reduced_part.id = "ReducedMonophonic"

    treble_measures = list(treble.getElementsByClass(stream.Measure))
    last_ts_ratio = None

    for treble_measure in treble_measures:
        bass_measure = get_measure_by_number(bass, treble_measure.number)
        if bass_measure is None:
            bass_measure = stream.Measure(number=treble_measure.number)

        ts = treble_measure.getContextByClass(meter.TimeSignature)
        ts_ratio = ts.ratioString if ts is not None else None
        show_ts = ts_ratio != last_ts_ratio

        reduced_measure = reduce_measure(
            treble_measure,
            bass_measure,
            show_ts=show_ts
        )
        reduced_part.append(reduced_measure)

        last_ts_ratio = ts_ratio

    tempo_marks(score, reduced_part)
    range_check(reduced_part, instrument)

    return reduced_part


def print_melody_debug(treble, num_measures=4):
    for m in treble.getElementsByClass(stream.Measure)[:num_measures]:
        print(f"\nMeasure {m.number} — melody extraction:")
        for ev in extract_melody(m):
            print(
                f"  melody: pitch={ev.pitch.midi}, offset={ev.offset}, dur={ev.duration}"
            )


def print_lh_debug(bass, measure_number=1):
    m = get_measure_by_number(bass, measure_number)
    if m is None:
        print(f"No bass measure {measure_number}")
        return

    print(f"\nLH arpeggiation events for measure {measure_number}:")
    for ev in get_lh_arpeggiation_events(m):
        print(
            f"  lh: pitch={ev.pitch.midi}, offset={ev.offset}, dur={ev.duration}"
        )

if __name__ == "__main__":
    score, treble, bass = load_score('reduction_scores/Mozart_12.musicxml')

    print_melody_debug(treble, num_measures=4)
    print_lh_debug(bass, measure_number=1)

    reduced = reduce_score(score, treble, bass, 'clarinet')

    out_score = stream.Score()
    out_score.insert(0, reduced)

    out_score.show("musicxml")