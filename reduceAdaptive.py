from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
Adaptive-span monophonic reducer.

Example:
uv run reduceAdaptive.py \
  --json outputs/inference/Mozart_12.json \
  --output-xml outputs/reductions/Mozart_12-monophonic-adaptive.musicxml \
  --layer 2 \
  --density-mode balanced \
  --debug-spans
"""


DEFAULT_ATTACK_BONUS = 0.015
DEFAULT_MIN_SPAN_BEATS = 1
DEFAULT_MAX_SPAN_BEATS = 6
DEFAULT_PRIMARY_CONTINUITY_WEIGHT = 0.75
DEFAULT_SECONDARY_CONTINUITY_WEIGHT = 0.20
DEFAULT_SECONDARY_SUPPORT_WEIGHT = 0.45
DEFAULT_BOUNDARY_FIT_WEIGHT = 0.30
DEFAULT_SWITCH_PENALTY = 0.55
DEFAULT_COMPLEXITY_WEIGHT = 0.40
DEFAULT_MAX_SECONDARY_INSERTIONS_PER_SPAN = 3
DEFAULT_TOP_SPANS_PER_START = 24

@dataclass
class TimedEvent:
    offset: float
    duration: float
    pitch: music21.pitch.Pitch
    source: str
    source_note_id: Optional[str] = None
    span_id: Optional[int] = None
    strand_id: Optional[str] = None

    @property
    def end(self) -> float:
        return self.offset + self.duration


@dataclass
class ActiveSlice:
    measure_index: int
    measure_number: int
    start_local: float
    end_local: float
    start_abs: float
    end_abs: float
    active_tones: Tuple[object, ...]

    @property
    def duration(self) -> float:
        return self.end_abs - self.start_abs


@dataclass
class CandidateStrand:
    strand_id: str
    source_voice_ids: Tuple[str, ...]
    staff_num: int
    mean_pitch: float
    attack_density: float
    role_hint: str
    role_priors: Dict[str, float]


@dataclass
class BeatStrandFeatures:
    utility_mass: float = 0.0
    attack_mass: float = 0.0
    coverage: float = 0.0
    metric_salience: float = 0.0
    avg_pitch: Optional[float] = None
    continuity: float = 0.0
    melodic_score: float = 0.0
    support_score: float = 0.0
    sounding_duration: float = 0.0
    attack_count: int = 0
    ursatz_like_count: int = 0
    high_utility_count: int = 0


@dataclass
class BeatNode:
    beat_index: int
    measure_index: int
    measure_number: int
    beat_in_measure: int
    beats_in_measure: int
    start_abs: float
    end_abs: float
    metric_strength: float
    boundary_strength: float = 0.0
    strand_features: Dict[str, BeatStrandFeatures] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_abs - self.start_abs


@dataclass
class SpanHypothesis:
    span_id: int
    start_beat: int
    end_beat: int
    primary_strand_id: str
    secondary_strand_id: Optional[str]
    primary_score: float
    secondary_score: float
    boundary_fit: float
    complexity_penalty: float
    total_score: float


@dataclass
class ReductionChoice:
    primary_tone: Optional[object]
    secondary_tone: Optional[object]
    span_id: Optional[int]
    primary_strand_id: Optional[str]
    secondary_strand_id: Optional[str]
    decision: str

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def tone_selection_utility(tone) -> float:
    return float(getattr(tone, "selection_utility", tone.utility))


def tone_voice_id(tone) -> str:
    return str(getattr(tone, "voice_id", "unknown"))


def tone_note_id(tone) -> str:
    measure = tone.row.get("measure", "?")
    off = q(float(tone.row.get("measure_offset", 0.0)))
    dur = q(float(tone.row.get("quarter_length", 0.0)))
    pitch = int(tone.pitch_midi)
    return f"m={measure}|off={off}|dur={dur}|pitch={pitch}"


def attacked_at(tone, abs_time: float) -> bool:
    return abs(float(tone.start_abs) - float(abs_time)) <= EPSILON


def tone_sort_key(tone, slice_start: float):
    return (
        tone_selection_utility(tone),
        1 if attacked_at(tone, slice_start) else 0,
        int(tone.pitch_midi),
    )


def format_tone_debug(tone, slice_start: float) -> str:
    return (
        f"m={tone.row['measure']} "
        f"off={q(tone.row['measure_offset'])} "
        f"pitch={tone.pitch_midi} "
        f"util={tone_selection_utility(tone):.6f} "
        f"start={q(tone.start_abs)} "
        f"end={q(tone.end_abs)} "
        f"voice={tone_voice_id(tone)} "
        f"{'ATTACK' if attacked_at(tone, slice_start) else 'HELD'}"
    )


def normalize_tone_utilities(tones: Iterable[object]) -> None:
    tones = list(tones)
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


def canonical_voice_id(element) -> str:
    voice_ctx = element.getContextByClass(stream.Voice)
    if voice_ctx is not None and voice_ctx.id is not None:
        voice_label = str(voice_ctx.id)
    else:
        voice_label = "1"

    staff_num = getattr(element, "staffNumber", None)
    if staff_num is None:
        staff_num = 1

    return f"staff={staff_num}|voice={voice_label}"


def parse_staff_num(voice_id: str) -> int:
    for part in str(voice_id).split("|"):
        if part.startswith("staff="):
            try:
                return int(part.split("=", 1)[1])
            except ValueError:
                return 1
    return 1


def pitch_from_midi(midi_value: int):
    p = music21.pitch.Pitch()
    p.midi = int(midi_value)
    return p


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
                f"unknown|m={measure_number}|off={measure_offset}|pitch={pitch_midi}"
            )

        tone.voice_id = voice_id

    return unmatched


def get_measure_by_number(part, number):
    for m in part.getElementsByClass(stream.Measure):
        if m.number == number:
            return m
    return None


def get_beat_length(measure) -> float:
    ts = measure.getContextByClass(meter.TimeSignature)
    if ts is not None:
        return float(ts.beatDuration.quarterLength)
    return 1.0


def get_measure_length(measure) -> float:
    if measure.barDuration is not None:
        return float(measure.barDuration.quarterLength)

    ts = measure.getContextByClass(meter.TimeSignature)
    if ts is not None:
        return float(ts.barDuration.quarterLength)

    return 4.0


def metric_strength_for_beat(beat_in_measure: int, beats_in_measure: int) -> float:
    if beat_in_measure == 0:
        return 1.0
    if beats_in_measure >= 4 and beat_in_measure == beats_in_measure // 2:
        return 0.75
    if beat_in_measure == beats_in_measure - 1:
        return 0.50
    return 0.45


# Build out micro-slices

def build_activity_slices(matched_tones, measure_infos, source_score, include_beat_boundaries=True, debug_slices=False):
    source_part = source_score.parts[0]
    source_measures_by_num = {
        m.number: m for m in source_part.getElementsByClass(stream.Measure)
    }

    boundaries = set()

    for measure in measure_infos:
        boundaries.add(q(measure.start_abs))
        boundaries.add(q(measure.end_abs))

        if include_beat_boundaries:
            src_measure = source_measures_by_num.get(measure.number)
            if src_measure is not None:
                beat_len = get_beat_length(src_measure)
                pos = float(measure.start_abs)
                while pos < float(measure.end_abs) - EPSILON:
                    boundaries.add(q(pos))
                    pos += beat_len
                boundaries.add(q(measure.end_abs))

    for tone in matched_tones:
        boundaries.add(q(tone.start_abs))
        boundaries.add(q(tone.end_abs))

    times = sorted(boundaries)
    slices: List[ActiveSlice] = []
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

        active = tuple(sorted(active, key=lambda tone: tone_sort_key(tone, t0), reverse=True))

        if debug_slices:
            print()
            print(
                f"[activity slice {slice_count + 1}] "
                f"measure={measure.number} "
                f"abs=({q(t0)} -> {q(t1)}) "
                f"local=({q(t0 - measure.start_abs)} -> {q(t1 - measure.start_abs)})"
            )
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


# Candidate Strands

def infer_role_for_voice(staff_num: int, mean_pitch: float) -> Tuple[str, Dict[str, float]]:
    if staff_num <= 1 and mean_pitch >= 60:
        return "upper_melodic", {"melodic": 1.0, "support": 0.30, "bass": 0.05}
    if staff_num <= 1:
        return "upper_inner", {"melodic": 0.70, "support": 0.45, "bass": 0.10}
    if mean_pitch <= 52:
        return "bass_support", {"melodic": 0.45, "support": 0.95, "bass": 1.0}
    return "lower_inner", {"melodic": 0.35, "support": 0.80, "bass": 0.70}


def build_candidate_strands(reduced_tones) -> Dict[str, CandidateStrand]:
    grouped = defaultdict(list)
    for tone in reduced_tones:
        grouped[tone_voice_id(tone)].append(tone)

    strands: Dict[str, CandidateStrand] = {}
    for voice_id, tones in grouped.items():
        tones_sorted = sorted(tones, key=lambda t: (t.start_abs, t.end_abs, t.pitch_midi))
        attack_count = 0
        prev_end = None
        for tone in tones_sorted:
            if prev_end is None or tone.start_abs > prev_end + EPSILON:
                attack_count += 1
            else:
                attack_count += 1
            prev_end = tone.end_abs

        mean_pitch = sum(int(t.pitch_midi) for t in tones_sorted) / max(1, len(tones_sorted))
        staff_num = parse_staff_num(voice_id)
        role_hint, role_priors = infer_role_for_voice(staff_num, mean_pitch)
        span = max(EPSILON, tones_sorted[-1].end_abs - tones_sorted[0].start_abs)
        attack_density = attack_count / span

        strands[voice_id] = CandidateStrand(
            strand_id=voice_id,
            source_voice_ids=(voice_id,),
            staff_num=staff_num,
            mean_pitch=mean_pitch,
            attack_density=attack_density,
            role_hint=role_hint,
            role_priors=role_priors,
        )

    return strands


# Aggregation

def build_beat_nodes(source_score, measure_infos, activity_slices, strands):
    source_part = source_score.parts[0]
    source_measures_by_num = {
        m.number: m for m in source_part.getElementsByClass(stream.Measure)
    }

    nodes: List[BeatNode] = []
    global_beat_index = 0

    for measure in measure_infos:
        src_measure = source_measures_by_num.get(measure.number)
        if src_measure is None:
            continue

        beat_len = get_beat_length(src_measure)
        measure_len = get_measure_length(src_measure)
        beats_in_measure = max(1, int(round(measure_len / max(EPSILON, beat_len))))

        beat_start = float(measure.start_abs)
        for beat_in_measure in range(beats_in_measure):
            beat_end = min(float(measure.end_abs), beat_start + beat_len)
            if beat_end <= beat_start + EPSILON:
                continue

            node = BeatNode(
                beat_index=global_beat_index,
                measure_index=measure.index,
                measure_number=measure.number,
                beat_in_measure=beat_in_measure,
                beats_in_measure=beats_in_measure,
                start_abs=beat_start,
                end_abs=beat_end,
                metric_strength=metric_strength_for_beat(beat_in_measure, beats_in_measure),
                strand_features={sid: BeatStrandFeatures() for sid in strands},
            )

            prev_avg_pitch_by_strand = {
                sid: None if not nodes else nodes[-1].strand_features[sid].avg_pitch
                for sid in strands
            }

            for sid, strand in strands.items():
                feat = node.strand_features[sid]
                pitch_weight_sum = 0.0
                weighted_pitch_sum = 0.0
                span_duration = max(EPSILON, node.duration)

                for sl in activity_slices:
                    if sl.end_abs <= node.start_abs + EPSILON:
                        continue
                    if sl.start_abs >= node.end_abs - EPSILON:
                        break

                    overlap_start = max(sl.start_abs, node.start_abs)
                    overlap_end = min(sl.end_abs, node.end_abs)
                    overlap = overlap_end - overlap_start
                    if overlap <= EPSILON:
                        continue

                    strand_tones = [t for t in sl.active_tones if tone_voice_id(t) == sid]
                    if not strand_tones:
                        continue

                    best_tone = max(strand_tones, key=lambda t: tone_selection_utility(t))
                    util = tone_selection_utility(best_tone)
                    feat.utility_mass += util * overlap
                    feat.sounding_duration += overlap
                    feat.coverage += overlap / span_duration
                    weighted_pitch_sum += int(best_tone.pitch_midi) * overlap * max(util, 0.1)
                    pitch_weight_sum += overlap * max(util, 0.1)

                    if best_tone.start_abs >= node.start_abs - EPSILON and best_tone.start_abs < node.end_abs - EPSILON:
                        feat.attack_count += 1
                        feat.attack_mass += util
                        feat.metric_salience += util * node.metric_strength
                    if util >= 0.70:
                        feat.high_utility_count += 1
                    if util >= 0.90:
                        feat.ursatz_like_count += 1

                if pitch_weight_sum > EPSILON:
                    feat.avg_pitch = weighted_pitch_sum / pitch_weight_sum
                else:
                    feat.avg_pitch = None

                prev_pitch = prev_avg_pitch_by_strand[sid]
                if feat.avg_pitch is not None and prev_pitch is not None:
                    interval = abs(feat.avg_pitch - prev_pitch)
                    feat.continuity = clamp(1.0 - interval / 12.0, 0.0, 1.0)
                elif feat.avg_pitch is not None:
                    feat.continuity = 0.5
                else:
                    feat.continuity = 0.0

                upperness = clamp((strand.mean_pitch - 50.0) / 22.0, 0.0, 1.0)
                lowerness = clamp((68.0 - strand.mean_pitch) / 22.0, 0.0, 1.0)

                feat.melodic_score = (
                    0.30 * feat.utility_mass
                    + 0.20 * feat.attack_mass
                    + 0.18 * feat.continuity
                    + 0.12 * feat.metric_salience
                    + 0.10 * feat.coverage
                    + 0.10 * upperness
                ) * strand.role_priors["melodic"]

                feat.support_score = (
                    0.30 * feat.utility_mass
                    + 0.18 * feat.attack_mass
                    + 0.18 * feat.coverage
                    + 0.14 * feat.metric_salience
                    + 0.10 * lowerness
                    + 0.10 * strand.role_priors["bass"]
                ) * max(strand.role_priors["support"], strand.role_priors["bass"])

            nodes.append(node)
            global_beat_index += 1
            beat_start = beat_end

    return nodes


# Boundaries

def active_strands_for_node(node: BeatNode) -> set:
    return {
        sid for sid, feat in node.strand_features.items()
        if feat.sounding_duration > EPSILON
    }


def estimate_boundary_strength(nodes: List[BeatNode]) -> None:
    for i, node in enumerate(nodes):
        strength = 0.0

        if node.beat_in_measure == 0:
            strength += 0.35

        utility_sum = sum(f.utility_mass for f in node.strand_features.values())
        attack_sum = sum(f.attack_mass for f in node.strand_features.values())
        active_now = active_strands_for_node(node)

        if not active_now:
            strength += 0.25

        if i > 0:
            prev = nodes[i - 1]
            prev_utility = sum(f.utility_mass for f in prev.strand_features.values())
            prev_active = active_strands_for_node(prev)

            union = len(active_now | prev_active)
            if union > 0:
                change = 1.0 - (len(active_now & prev_active) / union)
                strength += 0.18 * change

            strength += 0.12 * clamp(abs(utility_sum - prev_utility), 0.0, 1.0)

            prev_pitches = [f.avg_pitch for f in prev.strand_features.values() if f.avg_pitch is not None]
            cur_pitches = [f.avg_pitch for f in node.strand_features.values() if f.avg_pitch is not None]
            if prev_pitches and cur_pitches:
                prev_center = sum(prev_pitches) / len(prev_pitches)
                cur_center = sum(cur_pitches) / len(cur_pitches)
                strength += 0.10 * clamp(abs(cur_center - prev_center) / 10.0, 0.0, 1.0)

        strength += 0.08 * clamp(attack_sum, 0.0, 1.0)
        node.boundary_strength = clamp(strength, 0.0, 1.0)


# Scoring

def aggregate_span_features(nodes: List[BeatNode], start: int, end: int, strand_id: str) -> BeatStrandFeatures:
    out = BeatStrandFeatures()
    pitches = []
    count_pitch = 0
    span_len = max(1, end - start)

    for i in range(start, end):
        feat = nodes[i].strand_features[strand_id]
        out.utility_mass += feat.utility_mass
        out.attack_mass += feat.attack_mass
        out.coverage += feat.coverage
        out.metric_salience += feat.metric_salience
        out.continuity += feat.continuity
        out.melodic_score += feat.melodic_score
        out.support_score += feat.support_score
        out.sounding_duration += feat.sounding_duration
        out.attack_count += feat.attack_count
        out.ursatz_like_count += feat.ursatz_like_count
        out.high_utility_count += feat.high_utility_count
        if feat.avg_pitch is not None:
            pitches.append(feat.avg_pitch)
            count_pitch += 1

    out.coverage /= span_len
    out.continuity /= span_len
    out.avg_pitch = sum(pitches) / count_pitch if count_pitch else None
    return out


def primary_score_for_span(nodes: List[BeatNode], start: int, end: int, strand: CandidateStrand) -> Tuple[float, float]:
    feat = aggregate_span_features(nodes, start, end, strand.strand_id)
    span_len = max(1, end - start)

    fragmentation = 0
    previously_active = False
    for i in range(start, end):
        active_now = nodes[i].strand_features[strand.strand_id].sounding_duration > EPSILON
        if active_now and not previously_active and i > start:
            fragmentation += 1
        previously_active = active_now

    score = (
        0.30 * feat.utility_mass
        + 0.16 * feat.attack_mass
        + 0.14 * feat.metric_salience
        + 0.16 * feat.continuity
        + 0.12 * feat.coverage
        + 0.12 * feat.melodic_score
        + 0.06 * feat.ursatz_like_count
    )

    complexity = (
        0.50 * fragmentation / span_len
        + 0.25 * (1.0 - feat.coverage)
        + 0.25 * (1.0 - clamp(feat.continuity, 0.0, 1.0))
    )
    return score, complexity


def secondary_score_for_span(nodes: List[BeatNode], start: int, end: int, primary_id: str, secondary_strand: Optional[CandidateStrand]) -> float:
    if secondary_strand is None:
        return 0.0

    secondary_id = secondary_strand.strand_id
    sec_feat = aggregate_span_features(nodes, start, end, secondary_id)
    span_len = max(1, end - start)

    complementarity = 0.0
    competition = 0.0
    for i in range(start, end):
        p = nodes[i].strand_features[primary_id]
        s = nodes[i].strand_features[secondary_id]
        p_active = p.sounding_duration > EPSILON
        s_active = s.sounding_duration > EPSILON
        if (not p_active) and s_active:
            complementarity += 1.0
        elif p_active and s_active:
            competition += 1.0

    complementarity /= span_len
    competition /= span_len

    return (
        0.28 * sec_feat.support_score
        + 0.22 * sec_feat.utility_mass
        + 0.15 * sec_feat.attack_mass
        + 0.20 * complementarity
        + 0.10 * sec_feat.coverage
        - 0.10 * competition
    )


def boundary_fit_for_span(nodes: List[BeatNode], start: int, end: int) -> float:
    start_fit = nodes[start].boundary_strength
    end_fit = nodes[end - 1].boundary_strength
    return 0.45 * start_fit + 0.85 * end_fit


def generate_span_hypotheses(nodes: List[BeatNode], strands: Dict[str, CandidateStrand], min_span_beats: int, max_span_beats: int, secondary_support_weight: float, boundary_fit_weight: float, complexity_weight: float, top_spans_per_start: int):
    hypotheses_by_start: Dict[int, List[SpanHypothesis]] = {i: [] for i in range(len(nodes))}
    span_id = 0
    strand_ids = list(strands.keys())

    for start in range(len(nodes)):
        candidate_ends = set()
        for length in range(min_span_beats, max_span_beats + 1):
            end = start + length
            if end <= len(nodes):
                candidate_ends.add(end)

        current_measure = nodes[start].measure_number
        for end in range(start + 1, min(len(nodes), start + max_span_beats) + 1):
            if nodes[end - 1].measure_number != current_measure or end == len(nodes):
                candidate_ends.add(end)
                break

        for end in sorted(candidate_ends):
            if end <= start:
                continue

            active_primaries = []
            for sid in strand_ids:
                feat = aggregate_span_features(nodes, start, end, sid)
                if feat.sounding_duration > EPSILON:
                    active_primaries.append(sid)

            if not active_primaries:
                continue

            for primary_id in active_primaries:
                primary_strand = strands[primary_id]
                primary_score, complexity = primary_score_for_span(nodes, start, end, primary_strand)

                secondary_candidates = [None] + [sid for sid in active_primaries if sid != primary_id]
                for secondary_id in secondary_candidates:
                    secondary_strand = strands[secondary_id] if secondary_id is not None else None
                    secondary_score = secondary_score_for_span(nodes, start, end, primary_id, secondary_strand)
                    boundary_fit = boundary_fit_for_span(nodes, start, end)
                    total = (
                        primary_score
                        + secondary_support_weight * secondary_score
                        + boundary_fit_weight * boundary_fit
                        - complexity_weight * complexity
                    )
                    hypotheses_by_start[start].append(
                        SpanHypothesis(
                            span_id=span_id,
                            start_beat=start,
                            end_beat=end,
                            primary_strand_id=primary_id,
                            secondary_strand_id=secondary_id,
                            primary_score=primary_score,
                            secondary_score=secondary_score,
                            boundary_fit=boundary_fit,
                            complexity_penalty=complexity,
                            total_score=total,
                        )
                    )
                    span_id += 1

        hypotheses_by_start[start].sort(key=lambda h: h.total_score, reverse=True)
        hypotheses_by_start[start] = hypotheses_by_start[start][:top_spans_per_start]

    return hypotheses_by_start


# Optimization

def optimize_span_chain(nodes: List[BeatNode], hypotheses_by_start: Dict[int, List[SpanHypothesis]], primary_continuity_weight: float, secondary_continuity_weight: float, switch_penalty: float) -> List[SpanHypothesis]:
    @lru_cache(maxsize=None)
    def solve(i: int, prev_primary: Optional[str], prev_secondary: Optional[str]):
        if i >= len(nodes):
            return 0.0, ()

        candidates = hypotheses_by_start.get(i, [])
        if not candidates:
            return -1e12, ()

        best_score = -1e18
        best_path = ()

        for hyp in candidates:
            transition = 0.0
            if prev_primary is not None:
                if hyp.primary_strand_id == prev_primary:
                    transition += primary_continuity_weight
                else:
                    transition -= switch_penalty

            if prev_secondary is not None and hyp.secondary_strand_id is not None:
                if hyp.secondary_strand_id == prev_secondary:
                    transition += secondary_continuity_weight
                else:
                    transition -= 0.20 * switch_penalty

            future_score, future_path = solve(hyp.end_beat, hyp.primary_strand_id, hyp.secondary_strand_id)
            total = hyp.total_score + transition + future_score
            if total > best_score:
                best_score = total
                best_path = (hyp.span_id,) + future_path

        return best_score, best_path

    _, path_ids = solve(0, None, None)
    by_id = {
        hyp.span_id: hyp
        for hyps in hypotheses_by_start.values()
        for hyp in hyps
    }
    return [by_id[hid] for hid in path_ids if hid in by_id]


# Monophonic generation

def build_time_to_beat_map(nodes: List[BeatNode]):
    def beat_at_time(abs_time: float) -> Optional[int]:
        for node in nodes:
            if abs_time >= node.start_abs - EPSILON and abs_time < node.end_abs - EPSILON:
                return node.beat_index
        if nodes and abs(abs_time - nodes[-1].end_abs) <= EPSILON:
            return nodes[-1].beat_index
        return None

    return beat_at_time


def best_tone_for_strand(active_slice: ActiveSlice, strand_id: Optional[str]):
    if strand_id is None:
        return None
    tones = [t for t in active_slice.active_tones if tone_voice_id(t) == strand_id]
    if not tones:
        return None
    return max(tones, key=lambda t: tone_sort_key(t, active_slice.start_abs))


def best_overall_tone(active_slice: ActiveSlice, exclude_voice_ids: Tuple[str, ...] = ()):
    tones = [t for t in active_slice.active_tones if tone_voice_id(t) not in exclude_voice_ids]
    if not tones:
        return None
    return max(tones, key=lambda t: tone_sort_key(t, active_slice.start_abs))


def realize_slices_from_spans(activity_slices: List[ActiveSlice], nodes: List[BeatNode], chosen_spans: List[SpanHypothesis], density_mode: str = "balanced", max_secondary_insertions_per_span: int = DEFAULT_MAX_SECONDARY_INSERTIONS_PER_SPAN, debug_slices: bool = False):
    beat_at_time = build_time_to_beat_map(nodes)
    span_by_beat = {}
    for hyp in chosen_spans:
        for beat_idx in range(hyp.start_beat, hyp.end_beat):
            span_by_beat[beat_idx] = hyp

    selected: List[SliceSelection] = []
    debug_rows = []
    secondary_counts = defaultdict(int)

    for index, sl in enumerate(activity_slices):
        midpoint = (sl.start_abs + sl.end_abs) / 2.0
        beat_idx = beat_at_time(midpoint)
        hyp = span_by_beat.get(beat_idx)

        primary_tone = None
        secondary_tone = None
        decision = "REST"
        primary_id = None
        secondary_id = None
        span_id = None

        if hyp is not None:
            primary_id = hyp.primary_strand_id
            secondary_id = hyp.secondary_strand_id
            span_id = hyp.span_id

            primary_tone = best_tone_for_strand(sl, primary_id)
            secondary_tone = best_tone_for_strand(sl, secondary_id)

            if primary_tone is None and secondary_tone is not None:
                primary_tone = secondary_tone
                secondary_tone = None
                decision = "PROMOTE_SECONDARY"
            elif primary_tone is None:
                fallback = best_overall_tone(sl)
                if fallback is not None:
                    primary_tone = fallback
                    decision = "GLOBAL_FALLBACK"
                else:
                    decision = "REST"
            else:
                decision = "PRIMARY"

            if secondary_tone is not None and primary_tone is not None:
                if int(secondary_tone.pitch_midi) == int(primary_tone.pitch_midi):
                    secondary_tone = None
                else:
                    attacked_secondary = attacked_at(secondary_tone, sl.start_abs)
                    attacked_primary = attacked_at(primary_tone, sl.start_abs)
                    sec_ok = False
                    if density_mode == "sparse":
                        sec_ok = primary_tone is None
                    elif density_mode == "balanced":
                        sec_ok = (
                            attacked_secondary
                            and (not attacked_primary or tone_selection_utility(secondary_tone) > tone_selection_utility(primary_tone) + 0.05)
                            and secondary_counts[span_id] < max_secondary_insertions_per_span
                        )
                    else:
                        sec_ok = secondary_counts[span_id] < max_secondary_insertions_per_span + 2

                    if sec_ok:
                        secondary_counts[span_id] += 1
                        decision += "+SECONDARY"
                    else:
                        secondary_tone = None

        tones = tuple(t for t in (primary_tone, secondary_tone) if t is not None)

        if debug_slices:
            print()
            print(
                f"[adaptive slice {index + 1}] "
                f"measure={sl.measure_number} abs=({q(sl.start_abs)} -> {q(sl.end_abs)}) "
                f"beat={beat_idx} span={span_id} decision={decision}"
            )
            if hyp is not None:
                print(f"  primary_strand={primary_id} secondary_strand={secondary_id}")
            for t in tones:
                label = "PRIMARY" if t is primary_tone else "SECONDARY"
                print("   ", label, format_tone_debug(t, sl.start_abs))

        debug_rows.append(
            {
                "slice_index": index,
                "measure_number": sl.measure_number,
                "start_abs": sl.start_abs,
                "end_abs": sl.end_abs,
                "beat_index": beat_idx,
                "span_id": span_id,
                "primary_strand_id": primary_id,
                "secondary_strand_id": secondary_id,
                "decision": decision,
                "tones": [tone_note_id(t) for t in tones],
            }
        )

        selected.append(
            SliceSelection(
                measure_index=sl.measure_index,
                measure_number=sl.measure_number,
                start_local=sl.start_local,
                end_local=sl.end_local,
                start_abs=sl.start_abs,
                end_abs=sl.end_abs,
                tones=tones,
            )
        )

    return selected, debug_rows


def assign_slices_to_utility_streams(slices):
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

    if not primary_events:
        selected.extend(clipped_events_in_span(secondary_events, 0.0, measure_len, "secondary"))
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

        if prim_start > current_time + EPSILON:
            selected.extend(clipped_events_in_span(secondary_events, current_time, prim_start, "secondary"))

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

        uncovered_start = prim_start + primary_dur
        uncovered_end = prim_end
        if uncovered_end > uncovered_start + EPSILON:
            selected.extend(clipped_events_in_span(secondary_events, uncovered_start, uncovered_end, "secondary"))

        current_time = max(current_time, prim_end)

    if current_time < measure_len - EPSILON:
        selected.extend(clipped_events_in_span(secondary_events, current_time, measure_len, "secondary"))

    return render_monophonic_measure(
        measure_number=source_measure.number,
        ts=source_measure.getContextByClass(meter.TimeSignature),
        measure_length=measure_len,
        events=selected,
        show_ts=show_ts,
    )


def build_monophonic_part(source_score, stream_cells, measure_infos):
    reduced_part = stream.Part(id="ReducedMonophonicAdaptive")
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

        primary_events = cells_to_timed_events(stream_cells[1].get(mi.index, []), source_name="primary")
        secondary_events = cells_to_timed_events(stream_cells[2].get(mi.index, []), source_name="secondary")

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


# Debugging

def write_debug_json(path: Path, nodes: List[BeatNode], spans: List[SpanHypothesis], slice_rows: List[dict], strands: Dict[str, CandidateStrand]) -> None:
    payload = {
        "strands": {
            sid: {
                "strand_id": strand.strand_id,
                "source_voice_ids": list(strand.source_voice_ids),
                "staff_num": strand.staff_num,
                "mean_pitch": strand.mean_pitch,
                "attack_density": strand.attack_density,
                "role_hint": strand.role_hint,
                "role_priors": strand.role_priors,
            }
            for sid, strand in strands.items()
        },
        "beat_nodes": [
            {
                "beat_index": node.beat_index,
                "measure_index": node.measure_index,
                "measure_number": node.measure_number,
                "beat_in_measure": node.beat_in_measure,
                "beats_in_measure": node.beats_in_measure,
                "start_abs": node.start_abs,
                "end_abs": node.end_abs,
                "metric_strength": node.metric_strength,
                "boundary_strength": node.boundary_strength,
                "strand_features": {
                    sid: asdict(feat) for sid, feat in node.strand_features.items()
                },
            }
            for node in nodes
        ],
        "chosen_spans": [asdict(span) for span in spans],
        "slice_choices": slice_rows,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# Main rendering

def build_reduction_adaptive(
    json_path,
    output_xml,
    layer,
    source_xml=None,
    density_mode="balanced",
    min_span_beats=DEFAULT_MIN_SPAN_BEATS,
    max_span_beats=DEFAULT_MAX_SPAN_BEATS,
    primary_continuity_weight=DEFAULT_PRIMARY_CONTINUITY_WEIGHT,
    secondary_continuity_weight=DEFAULT_SECONDARY_CONTINUITY_WEIGHT,
    secondary_support_weight=DEFAULT_SECONDARY_SUPPORT_WEIGHT,
    boundary_fit_weight=DEFAULT_BOUNDARY_FIT_WEIGHT,
    switch_penalty=DEFAULT_SWITCH_PENALTY,
    complexity_weight=DEFAULT_COMPLEXITY_WEIGHT,
    max_secondary_insertions_per_span=DEFAULT_MAX_SECONDARY_INSERTIONS_PER_SPAN,
    top_spans_per_start=DEFAULT_TOP_SPANS_PER_START,
    debug_slices=False,
    debug_spans=False,
    debug_json=None,
):
    json_path = Path(json_path)
    if not json_path.is_absolute():
        json_path = projRoot / json_path

    if output_xml is None:
        output_xml = outputDir / f"{json_path.stem}-monophonic_adaptive.musicxml"
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
        source_score=source_score,
        include_beat_boundaries=True,
        debug_slices=False,
    )

    strands = build_candidate_strands(reduced_tones)
    beat_nodes = build_beat_nodes(
        source_score=source_score,
        measure_infos=measure_infos,
        activity_slices=activity_slices,
        strands=strands,
    )
    estimate_boundary_strength(beat_nodes)

    hypotheses_by_start = generate_span_hypotheses(
        nodes=beat_nodes,
        strands=strands,
        min_span_beats=min_span_beats,
        max_span_beats=max_span_beats,
        secondary_support_weight=secondary_support_weight,
        boundary_fit_weight=boundary_fit_weight,
        complexity_weight=complexity_weight,
        top_spans_per_start=top_spans_per_start,
    )

    chosen_spans = optimize_span_chain(
        nodes=beat_nodes,
        hypotheses_by_start=hypotheses_by_start,
        primary_continuity_weight=primary_continuity_weight,
        secondary_continuity_weight=secondary_continuity_weight,
        switch_penalty=switch_penalty,
    )

    selected_slices, slice_rows = realize_slices_from_spans(
        activity_slices=activity_slices,
        nodes=beat_nodes,
        chosen_spans=chosen_spans,
        density_mode=density_mode,
        max_secondary_insertions_per_span=max_secondary_insertions_per_span,
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

    if debug_json is not None:
        debug_json = Path(debug_json)
        write_debug_json(
            path=debug_json,
            nodes=beat_nodes,
            spans=chosen_spans,
            slice_rows=slice_rows,
            strands=strands,
        )

    print(f"Built {raw_slice_count} activity slices")
    print(f"Built {len(beat_nodes)} beat nodes")
    print(f"Chose {len(chosen_spans)} adaptive spans")
    print(f"Unmatched MusicXML voice assignments: {unmatched}")
    print(f"Wrote monophonic reduction to {output_xml}")
    if debug_json is not None:
        print(f"Wrote debug JSON to {debug_json}")

    if debug_spans:
        print()
        print("Chosen spans:")
        for hyp in chosen_spans:
            print(
                f"  span={hyp.span_id:>3} "
                f"beats=[{hyp.start_beat},{hyp.end_beat}) "
                f"P={hyp.primary_strand_id:<20} "
                f"Q={str(hyp.secondary_strand_id):<20} "
                f"total={hyp.total_score:.4f} "
                f"primary={hyp.primary_score:.4f} "
                f"secondary={hyp.secondary_score:.4f} "
                f"boundary={hyp.boundary_fit:.4f} "
                f"complexity={hyp.complexity_penalty:.4f}"
            )


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
        "--density-mode",
        choices=["sparse", "balanced", "rich"],
        default="balanced",
        help="Rendering density for secondary support insertion.",
    )
    parser.add_argument(
        "--min-span-beats",
        type=int,
        default=DEFAULT_MIN_SPAN_BEATS,
        help="Minimum adaptive span length in beats.",
    )
    parser.add_argument(
        "--max-span-beats",
        type=int,
        default=DEFAULT_MAX_SPAN_BEATS,
        help="Maximum adaptive span length in beats.",
    )
    parser.add_argument(
        "--primary-continuity-weight",
        type=float,
        default=DEFAULT_PRIMARY_CONTINUITY_WEIGHT,
        help="Reward for retaining the same primary strand across adjacent spans.",
    )
    parser.add_argument(
        "--secondary-continuity-weight",
        type=float,
        default=DEFAULT_SECONDARY_CONTINUITY_WEIGHT,
        help="Small reward for secondary continuity across spans.",
    )
    parser.add_argument(
        "--secondary-support-weight",
        type=float,
        default=DEFAULT_SECONDARY_SUPPORT_WEIGHT,
        help="Weight applied to the secondary-support span score.",
    )
    parser.add_argument(
        "--boundary-fit-weight",
        type=float,
        default=DEFAULT_BOUNDARY_FIT_WEIGHT,
        help="Weight applied to the boundary-fit span score.",
    )
    parser.add_argument(
        "--switch-penalty",
        type=float,
        default=DEFAULT_SWITCH_PENALTY,
        help="Penalty for switching primary strands across spans.",
    )
    parser.add_argument(
        "--complexity-weight",
        type=float,
        default=DEFAULT_COMPLEXITY_WEIGHT,
        help="Penalty weight for internally unstable span hypotheses.",
    )
    parser.add_argument(
        "--max-secondary-insertions-per-span",
        type=int,
        default=DEFAULT_MAX_SECONDARY_INSERTIONS_PER_SPAN,
        help="Maximum number of explicit secondary insertions per chosen span.",
    )
    parser.add_argument(
        "--top-spans-per-start",
        type=int,
        default=DEFAULT_TOP_SPANS_PER_START,
        help="Beam width: keep only the top N span hypotheses per beat start.",
    )
    parser.add_argument(
        "--debug-slices",
        action="store_true",
        help="Print realized slice-level primary/secondary decisions.",
    )
    parser.add_argument(
        "--debug-spans",
        action="store_true",
        help="Print the chosen adaptive spans and their scores.",
    )
    parser.add_argument(
        "--debug-json",
        default=None,
        help="Optional path for a JSON dump of beat nodes, spans, and slice decisions.",
    )

    args = parser.parse_args()

    build_reduction_adaptive(
        json_path=args.json,
        output_xml=args.output_xml,
        layer=args.layer,
        source_xml=args.source_xml,
        density_mode=args.density_mode,
        min_span_beats=args.min_span_beats,
        max_span_beats=args.max_span_beats,
        primary_continuity_weight=args.primary_continuity_weight,
        secondary_continuity_weight=args.secondary_continuity_weight,
        secondary_support_weight=args.secondary_support_weight,
        boundary_fit_weight=args.boundary_fit_weight,
        switch_penalty=args.switch_penalty,
        complexity_weight=args.complexity_weight,
        max_secondary_insertions_per_span=args.max_secondary_insertions_per_span,
        top_spans_per_start=args.top_spans_per_start,
        debug_slices=args.debug_slices,
        debug_spans=args.debug_spans,
        debug_json=args.debug_json,
    )


if __name__ == "__main__":
    main()