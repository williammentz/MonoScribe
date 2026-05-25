import csv
import music21 as m21
import joblib
from symbolic_texture_dataset.descriptors import Pitches, Onsets, Slices
from symbolic_texture_dataset.dataset import FEATURE_NAMES, PIECES, get_samples

def get_beat_duration_from_score(score):
    """Extract beat duration from the score level, where time signatures live."""
    time_sigs = score.recurse().getElementsByClass(m21.meter.TimeSignature)
    if not time_sigs:
        raise ValueError("No time signature found in score.")
    return time_sigs[0].beatDuration.quarterLength


def extract_descriptors_from_musicxml(xml_path, output_tsv):
    score = m21.converter.parse(xml_path)
    chordified = score.chordify()

    beat_duration = get_beat_duration_from_score(score)

    meta = ["m-measure_number", "m-length_beat", "m-beat_duration"]
    descriptors_list = []

    for measure in chordified.getElementsByClass('Measure'):
        mn = measure.number
        print(f"Measure {mn}", end='\r')

        length_quarter = measure.duration.quarterLength
        if length_quarter == 0:
            continue

        pitches = Pitches(measure, length=length_quarter, beat_duration=beat_duration)
        onsets  = Onsets(measure, length=length_quarter, beat_duration=beat_duration)
        slices  = Slices(measure, length=length_quarter, beat_duration=beat_duration)

        d = {}
        d['m-measure_number'] = mn
        d['m-length_beat']    = pitches.length_beat
        d['m-beat_duration']  = beat_duration

        d['has_even_meter']   = int(slices.has_even_meter())
        d['n_notes']          = pitches.n_notes()
        d['n_pitches']        = pitches.n_pitches()
        d['n_onsets']         = onsets.n_onsets()
        d['n_slices']         = slices.n_slices()
        d['n_pitchclasses']   = pitches.n_pitchclasses()
        d['novelty']          = pitches.novelty()
        d['prop_silence']     = slices.prop_silence()
        d['longest_silence']  = slices.longest_silence()

        composed = {
            'pitch':             pitches.stats_pitches(),
            'duration':          pitches.stats_durations(),
            'regularity':        onsets.stats_regularity(),
            'width':             slices.stats_width(),
            'n_voices_per_onset': onsets.stats_n_voices(),
            'n_voices_per_beat': slices.stats_n_voices(),
            'n_gaps_per_onset':  onsets.stats_n_gaps(),
            'n_gaps_per_beat':   slices.stats_n_gaps(),
            'harmonicity':       slices.stats_harmonicity()
        }
        for name, values in composed.items():
            if None in values:
                for suffix in ['avg', 'std', 'min', 'max', 'med']:
                    d[f"{name}_{suffix}"] = None
            else:
                for suffix, v in zip(['avg', 'std', 'min', 'max', 'med'], values):
                    d[f"{name}_{suffix}"] = float(v)

        hi_3_6, hi_4_5, hi_8 = onsets.harmonic_intervals()
        d['harmonic_intervals_3_6'] = hi_3_6
        d['harmonic_intervals_4_5'] = hi_4_5
        d['harmonic_intervals_8']   = hi_8
        d['melodic_intervals_2']    = slices.melodic_intervals("s", onsets.onset_dict)
        d['melodic_intervals_3']    = slices.melodic_intervals("t", onsets.onset_dict)
        d['melodic_intervals_4_5']  = slices.melodic_intervals("f", onsets.onset_dict)
        d['melodic_intervals_8']    = slices.melodic_intervals("o", onsets.onset_dict)
        d['repetitions']            = slices.melodic_intervals("u", onsets.onset_dict)

        descriptors_list.append(d)

    with open(output_tsv, 'w', newline='\n') as f:
        writer = csv.DictWriter(f, fieldnames=meta + FEATURE_NAMES, delimiter='\t')
        writer.writeheader()
        for d in descriptors_list:
            writer.writerow(d)

    print(f"\nSaved descriptors to {output_tsv}")
    return descriptors_list


if __name__ == "__main__":
    scoreName = 'Mozart_12'
    xml_path = f"reduction_scores/{scoreName}.musicxml"

    # Step 1: extract descriptors once
    descriptors_list = extract_descriptors_from_musicxml(xml_path, f"symbolic_texture_dataset/predictedDescriptors/{scoreName}.tsv")

    # Step 2: build feature matrix once, keeping a filtered descriptor list to match
    filtered_descriptors = [
        d for d in descriptors_list
        if all(d.get(key) is not None for key in FEATURE_NAMES)
    ]
    x_new = [
        [float(d[key]) for key in FEATURE_NAMES]
        for d in filtered_descriptors
    ]

    # Step 3: predict all textural elements and collect results
    elements = ['M', 'H', 'S', 'h', 'p', 'o', 'h+', 'p+', 's', 't', 'b', 'r', '_', ',']
    results = {d['m-measure_number']: {} for d in filtered_descriptors}

    for element in elements:
        model = joblib.load(f"symbolic_texture_dataset/saved_models/{element}.joblib")
        y_pred = model.predict(x_new)
        for d, label in zip(filtered_descriptors, y_pred):
            results[d['m-measure_number']][element] = label

    # Step 4: write full annotation TSV
    with open(f"symbolic_texture_dataset/predictedAnnotations/{scoreName}.tsv", 'w', newline='\n') as f:
        writer = csv.DictWriter(f, fieldnames=['mn'] + elements, delimiter='\t')
        writer.writeheader()
        for mn, labels in sorted(results.items()):
            writer.writerow({'mn': mn, **labels})

    print(f"Annotation saved to predictedAnnotations/{scoreName}.tsv")