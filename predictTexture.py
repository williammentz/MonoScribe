import os
import csv
import music21 as m21
import joblib
import pandas as pd

from symbolic_texture_dataset.descriptors import Pitches, Onsets, Slices
from symbolic_texture_dataset.dataset import FEATURE_NAMES, PIECES, get_samples
from symbolic_texture_dataset.predictScoreTexture import get_beat_duration_from_score, extract_descriptors_from_musicxml

scoreName = 'Mozart_12'
xml_path = f"reduction_scores/{scoreName}.musicxml"

def annotateScore(xml_path):
    
    scoreName = xml_path.replace('reduction_scores/', '').replace('.musicxml', '')
    output_path = f"symbolic_texture_dataset/predictedDescriptors/{scoreName}.tsv"

    # Write score descriptors to file
    descriptors_list = extract_descriptors_from_musicxml(xml_path, output_path)

    filtered_descriptors = [
        d for d in descriptors_list
        if all(d.get(key) is not None for key in FEATURE_NAMES)
    ]
    x_new = [
        [float(d[key]) for key in FEATURE_NAMES]
        for d in filtered_descriptors
    ]

    elements = ['M', 'H', 'S', 'h', 'p', 'o', 'h+', 'p+', 's', 't', 'b', 'r', '_', ',']
    results = {d['m-measure_number']: {} for d in filtered_descriptors}

    # Run element prediction models
    for element in elements:
        model = joblib.load(f"symbolic_texture_dataset/saved_models/{element}.joblib")
        y_pred = model.predict(x_new)
        for d, label in zip(filtered_descriptors, y_pred):
            results[d['m-measure_number']][element] = label

    # Step 4: write full annotation TSV

    if os.path.exists('symbolic_texture_dataset/predictedAnnotations'):
        pass
    else:
        os.makedirs('symbolic_texture_dataset/predictedAnnotations')

    with open(f"symbolic_texture_dataset/predictedAnnotations/{scoreName}.tsv", 'w', newline='\n') as f:
        writer = csv.DictWriter(f, fieldnames=['mn'] + elements, delimiter='\t')
        writer.writeheader()
        for mn, labels in sorted(results.items()):
            writer.writerow({'mn': mn, **labels})

    print(f"Annotation saved to predictedAnnotations/{scoreName}.tsv")



if __name__ == '__main__':
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

    if os.path.exists('symbolic_texture_dataset/predictedAnnotations'):
        pass
    else:
        os.makedirs('symbolic_texture_dataset/predictedAnnotations')

    with open(f"symbolic_texture_dataset/predictedAnnotations/{scoreName}.tsv", 'w', newline='\n') as f:
        writer = csv.DictWriter(f, fieldnames=['mn'] + elements, delimiter='\t')
        writer.writeheader()
        for mn, labels in sorted(results.items()):
            writer.writerow({'mn': mn, **labels})

    print(f"Annotation saved to predictedAnnotations/{scoreName}.tsv")

    # Step 5: combine important textural elements and density information
    annot = pd.read_csv(f'symbolic_texture_dataset/predictedAnnotations/{scoreName}.tsv', sep = '\t')
    descr = pd.read_csv(f'symbolic_texture_dataset/predictedDescriptors/{scoreName}.tsv', sep = '\t')[['m-measure_number', 'n_voices_per_beat_med']].rename(columns = {'n_voices_per_beat_med': 'estimatedDensity'})

    df = pd.merge(annot, descr, left_on = 'mn', right_on = 'm-measure_number', how = 'left').drop('m-measure_number', axis = 1)
    print(df)