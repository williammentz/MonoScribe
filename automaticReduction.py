import argparse
import sys
from pathlib import Path
import pandas as pd

from scoreStructure import scorer
from reductionAutoSchA import build_reduction
from predictTexture import annotateScore
from symbolic_texture_dataset.predictScoreTexture import classifyTexture
from reduction import load_score_as_lanes, reduce_score
from music21 import stream

projRoot = Path(__file__).resolve().parent
outputDir = projRoot / "outputs/inference"
sys.path.insert(0, str(projRoot / "AutoSchA"))

if __name__ == "__main__":

    # User input
    parser = argparse.ArgumentParser(
        description = "Run AutoSchA-based GNN reduction on a MusicXML score."
    )
    parser.add_argument("--xml", required = True, help = "Path to input MusicXML file")
    parser.add_argument("--instrument", required = True, help = "Instrument for performance of reduced piece.")

    args = parser.parse_args()

    # Assign AutoSchA score to piece
    scorer(args.xml)

    json_path = 'outputs/inference/' + args.xml.replace('reduction_scores/', '').replace('.musicxml', '') + '.json'
    autoscha_path = 'outputs/autoscha_reductions/' + args.xml.replace('reduction_scores/', '').replace('.musicxml', '-reduction.musicxml')

    # AutoSchA Reducer (Over Whole Piece)
    build_reduction(json_path, layer = 2, output_xml = autoscha_path)

    # Annotate Score Textural Elements
    annotateScore(xml_path = args.xml)

    score = args.xml.replace('reduction_scores/', '').replace('.musicxml', '')

    # Take the new annotations/descriptors and predcit the texture of each measure of the piece
    descriptors = pd.read_csv(f'symbolic_texture_dataset/predictedDescriptors/{score}.tsv', delimiter = '\t')
    elements = pd.read_csv(f'symbolic_texture_dataset/predictedAnnotations/{score}.tsv', delimiter = '\t')

    result = classifyTexture(descriptors, elements)
    print(result.texture_type.value_counts())
    print(result.head(n = 10))

    sch_score, upper_lane, lower_lane = load_score_as_lanes(str(autoscha_path))
    final_part = reduce_score(sch_score, upper_lane, lower_lane, instrument = args.instrument)

    final_score = stream.Score()
    final_score.insert(0, final_part)

    final_xml_path = f"outputs/reductions/{args.xml.replace("reduction_scores/", "").replace(".musicxml", "")}-final-{args.instrument}-reduction.musicxml"
    final_score.write("musicxml", fp=str(final_xml_path))

    print(f"Final reduction written to {final_xml_path}")