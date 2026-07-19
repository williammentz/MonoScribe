import argparse
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd

from scoreStructure import scorer
from predictTextureElements import annotateScore
from symbolic_texture_dataset.predictScoreTexture import classifyTexture
from reduceHomophonic import load_score_as_lanes, reduce_score
# NEED TO ADD REDUCEGRAPH HERE ONCE FINALIZED
from music21 import stream

projRoot = Path(__file__).resolve().parent
outputDir = projRoot / "outputs/inference"
sys.path.insert(0, str(projRoot / "AutoSchA"))

def remove_grace_notes(input_path, output_path = None):
    """
    Removes all grace notes from a MusicXML file
    """
    
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_no_grace{input_path.suffix}")
    else:
        output_path = Path(output_path)

    tree = ET.parse(input_path)
    root = tree.getroot()

    def local_name(tag):
        return tag.split("}", 1)[-1]  # handles XML namespaces cleanly

    removed = 0

    # Walk every parent and inspect its direct children
    for parent in root.iter():
        for child in list(parent):
            if local_name(child.tag) == "note":
                has_grace = any(local_name(grandchild.tag) == "grace" for grandchild in child)
                if has_grace:
                    parent.remove(child)
                    removed += 1

    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return str(output_path), removed

if __name__ == "__main__":

    # User input
    parser = argparse.ArgumentParser(
        description = "Run AutoSchA-based GNN reduction on a MusicXML score."
    )
    parser.add_argument("--xml", required = True, help = "Path to input MusicXML file")
    parser.add_argument("--instrument", required = True, help = "Instrument for performance of reduced piece.")

    args = parser.parse_args()

    score = args.xml.replace('reduction_scores/', '').replace('.musicxml', '')
    clean_score = f'outputs/clean_scores/{score}.musicxml'

    remove_grace_notes(args.xml, clean_score)

    # Annotate Score Textural Elements
    annotateScore(xml_path = clean_score)

    # Take the new annotations/descriptors and predcit the texture of each measure of the piece
    descriptors = pd.read_csv(f'symbolic_texture_dataset/predictedDescriptors/{score}.tsv', delimiter = '\t')
    elements = pd.read_csv(f'symbolic_texture_dataset/predictedAnnotations/{score}.tsv', delimiter = '\t')

    result = classifyTexture(descriptors, elements)
    # print(result.texture_type.value_counts())
    # print(result.head(n = 10))
    result.to_csv(f'outputs/textures/{args.xml.replace('reduction_scores/', '').replace('.musicxml', '-textures')}.csv', index=False)

    # Assign AutoSchA score to piece
    scorer(clean_score)
    # scorer(args.xml)

    json_path = 'outputs/inference/' + args.xml.replace('reduction_scores/', '').replace('.musicxml', '') + '.json'
    autoscha_path = 'outputs/autoscha_reductions/' + args.xml.replace('reduction_scores/', '').replace('.musicxml', '-reduction.musicxml')

    # Homophonic Rules-based reduction
    sch_score, upper_lane, lower_lane = load_score_as_lanes(str(autoscha_path))
    final_part = reduce_score(sch_score, upper_lane, lower_lane, instrument = args.instrument)

    final_score = stream.Score()
    final_score.insert(0, final_part)

    final_xml_path = f"outputs/reductions/{args.xml.replace("reduction_scores/", "").replace(".musicxml", "")}-final-{args.instrument}-reduction.musicxml"
    final_score.write("musicxml", fp=str(final_xml_path))

    print(f"Final reduction written to {final_xml_path}")