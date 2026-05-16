import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# Mozart Data
mozart_df = pd.read_csv('outputs/test.csv')

mReduced = mozart_df[mozart_df['layer_2_score'] > 0.55]
mRemoved = mozart_df[mozart_df['layer_2_score'] <= 0.55]

fig, ax = plt.subplots(figsize=(14, 6))

for _, row in mozart_df.iterrows():
    ax.add_patch(
        Rectangle(
            (row["time_position"], row["pitch_midi"] - 0.4),
            row["duration_seconds"],
            0.8,
            color=plt.cm.inferno(row["layer_2_score"]),
            alpha=0.95,
        )
    )

ax.set_xlim(mozart_df["time_position"].min(), (mozart_df["time_position"] + mozart_df["duration_seconds"]).max())
ax.set_ylim(mozart_df["pitch_midi"].min() - 2, mozart_df["pitch_midi"].max() + 2)

ax.set_xlabel("Time")
ax.set_ylabel("Pitch MIDI")
ax.set_title("Mozart AutoSchA Full Heatmap")

sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, 1))
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Structural score")

plt.show()


fig, ax = plt.subplots(figsize=(14, 6))

for _, row in mReduced.iterrows():
    ax.add_patch(
        Rectangle(
            (row["time_position"], row["pitch_midi"] - 0.4),
            row["duration_seconds"],
            0.8,
            color=plt.cm.inferno(row["layer_2_score"]),
            alpha=0.95,
        )
    )

ax.set_xlim(mReduced["time_position"].min(), (mReduced["time_position"] + mReduced["duration_seconds"]).max())
ax.set_ylim(mReduced["pitch_midi"].min() - 2, mReduced["pitch_midi"].max() + 2)

ax.set_xlabel("Time")
ax.set_ylabel("Pitch MIDI")
ax.set_title("Mozart AutoSchA Reduced Heatmap")

sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, 1))
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Structural score")

plt.show()

mozartSummary = mozart_df.groupby('pitch_name')['layer_2_score'].agg(['mean']).sort_values('mean', ascending = False)