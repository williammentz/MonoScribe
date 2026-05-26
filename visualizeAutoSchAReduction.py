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
                            }
                        )

    return pd.DataFrame(rows)


def plot_piano_roll(df, title, color="darkorange", cmap=None, score_col=None):
    fig, ax = plt.subplots(figsize=(14, 6))

    use_scores = score_col is not None and score_col in df.columns and df[score_col].notna().any()

    if use_scores:
        vmin = df[score_col].min()
        vmax = df[score_col].max()
        if abs(vmax - vmin) < 1e-9:
            vmax = vmin + 1e-9
        norm = plt.Normalize(vmin, vmax)
        colormap = plt.cm.get_cmap(cmap or "inferno")

    for _, row in df.iterrows():
        rect_color = (
            colormap(norm(row[score_col]))
            if use_scores
            else color
        )

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

mozart_original_df = score_to_dataframe("reduction_scores/mozart_12.musicxml")
mozart_reduced_df = score_to_dataframe("outputs/mozart-test.musicxml")

plot_piano_roll(
    mozart_original_df,
    title="Mozart Original Piano Roll",
    color="steelblue",
)

plot_piano_roll(
    mozart_reduced_df,
    title="Mozart Reduced Piano Roll",
    color="darkorange",
)