import argparse
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd

from scoreStructure import scorer
# from predictTextureElements import annotateScore
# from symbolic_texture_dataset.predictScoreTexture import classifyTexture
# from reduceHomophonic import load_score_as_lanes, reduce_score
# from reduceGraph import graph_reducer
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
    parser.add_argument("--piece", required = True) # format: '--piece Bach_880.musicxml' // So long as the file is in `reduction_scores/`
    parser.add_argument("--instrument", default = 'piano')

    parser.add_argument('--utility', type = float, default = 0.7)
    parser.add_argument('--continuity', type = float, default = 0.2) # Higher <=> More pitch continuity between nodes
    parser.add_argument('--density', type = float, default = 0.4) # Higher <=> Denser
    parser.add_argument('--contour', type = float, default = 0.5) # Higher <=> More contour within each node
    parser.add_argument('--method', default = 'measure') # Horizontal slicing method, default: method

    args = parser.parse_args()

    score = args.piece.replace('.musicxml', '')
    clean_score = f'outputs/clean_scores/{score}.musicxml'

    remove_grace_notes('reduction_scores/' + args.piece, clean_score)

    # Assign AutoSchA score to piece
    scorer(clean_score)

    json_path = 'outputs/inference/' + score + '.json'

    # Graph-based reducer
    # graph_reducer(args)

    # print(f"Final reduction written.")