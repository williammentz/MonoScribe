import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import music21
from music21 import note, chord, stream

def measure_number_for(measure):
    if measure.number is None:
        return 0
    return int(measure.number)


def score_to_dataframe(xml_path):
    score = music21.converter.parse(xml_path)
    rows = []

    for staff_index, part in enumerate(score.parts, start=1):
        for measure in part.getElementsByClass(stream.Measure):
            measure_num = measure_number_for(measure)

            for element in measure.recurse().notes:
                start_abs = float(element.getOffsetInHierarchy(score))
                measure_offset = float(element.getOffsetInHierarchy(measure))
                dur = float(element.duration.quarterLength)
                end_abs = start_abs + dur

                if isinstance(element, note.Note):
                    rows.append(
                        {
                            "staff": staff_index,
                            "measure": measure_num,
                            "measure_offset": measure_offset,
                            "time_position": start_abs,
                            "duration_quarter": dur,
                            "pitch_midi": int(element.pitch.midi),
                            "pitch_name": element.pitch.nameWithOctave,
                            "end_position": end_abs,
                        }
                    )

                elif isinstance(element, chord.Chord):
                    for p in element.pitches:
                        rows.append(
                            {
                                "staff": staff_index,
                                "measure": measure_num,
                                "measure_offset": measure_offset,
                                "time_position": start_abs,
                                "duration_quarter": dur,
                                "pitch_midi": int(p.midi),
                                "pitch_name": p.nameWithOctave,
                                "end_position": end_abs,
                            }
                        )

    return pd.DataFrame(rows)

EPSILON = 1e-6


def merge_intervals(intervals):
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [list(intervals[0])]

    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1] + EPSILON:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])

    return [(a, b) for a, b in merged]


def build_interval_index(df):
    index = {}

    grouped = df.groupby(["staff", "pitch_midi"])
    for key, group in grouped:
        intervals = list(zip(group["time_position"], group["end_position"]))
        index[key] = merge_intervals(intervals)

    return index


def intersect_interval_with_set(start, end, intervals):
    overlaps = []

    for a, b in intervals:
        if b <= start + EPSILON:
            continue
        if a >= end - EPSILON:
            break

        overlap_start = max(start, a)
        overlap_end = min(end, b)

        if overlap_end > overlap_start + EPSILON:
            overlaps.append((overlap_start, overlap_end))

    return overlaps


def subtract_intervals(start, end, kept_intervals):
    removed = []
    cursor = start

    for a, b in kept_intervals:
        if a > cursor + EPSILON:
            removed.append((cursor, a))
        cursor = max(cursor, b)

    if end > cursor + EPSILON:
        removed.append((cursor, end))

    return removed


def classify_original_against_reduced(original_df, reduced_df):
    reduced_index = build_interval_index(reduced_df)
    rows = []

    for _, row in original_df.iterrows():
        start = float(row["time_position"])
        end = float(row["end_position"])
        key = (int(row["staff"]), int(row["pitch_midi"]))

        kept_segments = intersect_interval_with_set(
            start,
            end,
            reduced_index.get(key, []),
        )
        removed_segments = subtract_intervals(start, end, kept_segments)

        for seg_start, seg_end in kept_segments:
            rows.append(
                {
                    "staff": row["staff"],
                    "measure": row["measure"],
                    "measure_offset": row["measure_offset"],
                    "time_position": seg_start,
                    "end_position": seg_end,
                    "duration_quarter": seg_end - seg_start,
                    "pitch_midi": row["pitch_midi"],
                    "pitch_name": row["pitch_name"],
                    "status": "kept",
                }
            )

        for seg_start, seg_end in removed_segments:
            rows.append(
                {
                    "staff": row["staff"],
                    "measure": row["measure"],
                    "measure_offset": row["measure_offset"],
                    "time_position": seg_start,
                    "end_position": seg_end,
                    "duration_quarter": seg_end - seg_start,
                    "pitch_midi": row["pitch_midi"],
                    "pitch_name": row["pitch_name"],
                    "status": "removed",
                }
            )

    return pd.DataFrame(rows)


def plot_piano_roll(df, title, color="darkorange", cmap=None, score_col=None, status_col = None):
    fig, ax = plt.subplots(figsize=(14, 6))

    use_scores = score_col is not None and score_col in df.columns and df[score_col].notna().any()

    if use_scores:
        vmin = df[score_col].min()
        vmax = df[score_col].max()
        if abs(vmax - vmin) < 1e-9:
            vmax = vmin + 1e-9
        norm = plt.Normalize(vmin, vmax)
        colormap = plt.cm.get_cmap(cmap or "inferno")

    status_colors = {
            "kept": "forestgreen",
            "removed": "lightcoral",
    }

    for _, row in df.iterrows():
        if status_col is not None and status_col in df.columns:
            rect_color = status_colors.get(row[status_col], color)
        elif use_scores:
            rect_color = colormap(norm(row[score_col]))
        else:
            rect_color = color

        ax.add_patch(
            Rectangle(
                (row["time_position"], row["pitch_midi"] - 0.4),
                row["duration_quarter"],
                0.8,
                color=rect_color,
                alpha=0.95,
            )
        )

    ax.set_xlim(
        df["time_position"].min(),
        (df["time_position"] + df["duration_quarter"]).max(),
    )
    ax.set_ylim(df["pitch_midi"].min() - 2, df["pitch_midi"].max() + 2)

    ax.set_xlabel("Time (quarter lengths)")
    ax.set_ylabel("Pitch MIDI")
    ax.set_title(title)

    if use_scores:
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=score_col)

    plt.tight_layout()
    plt.show()


# Mozart

mozart_original_df = score_to_dataframe("reduction_scores/Mozart_12.musicxml")
mozart_reduced_df = score_to_dataframe("outputs/reductions/Mozart_12-reduction.musicxml")

mozart_compare_df = classify_original_against_reduced(
    mozart_original_df,
    mozart_reduced_df,
)

plot_piano_roll(
    mozart_compare_df,
    title="Mozart Original Score: Kept vs Removed",
    status_col="status",
)

# Rach

rach_original_df = score_to_dataframe("reduction_scores/rach.musicxml")
rach_reduced_df = score_to_dataframe("outputs/rach-test.musicxml")

rach_compare_df = classify_original_against_reduced(
    rach_original_df,
    rach_reduced_df,
)

plot_piano_roll(
    rach_compare_df,
    title="Rach Original Score: Kept vs Removed",
    status_col="status",
)

# Bach

bach_original_df = score_to_dataframe("reduction_scores/bach_fugue.musicxml")
bach_reduced_df = score_to_dataframe("outputs/bach-test.musicxml")

bach_compare_df = classify_original_against_reduced(
    bach_original_df,
    bach_reduced_df,
)

plot_piano_roll(
    bach_compare_df,
    title="Bach Original Score: Kept vs Removed",
    status_col="status",
)