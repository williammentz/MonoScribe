import argparse
import sys
from pathlib import Path

from scoreStructure import scorer
from reductionAutoSchA import build_reduction

projRoot = Path(__file__).resolve().parent
outputDir = projRoot / "outputs/inference"
sys.path.insert(0, str(projRoot / "AutoSchA"))

if __name__ == "__main__":

    # User input

    parser = argparse.ArgumentParser(
        description = "Run AutoSchA-based GNN reduction on a MusicXML score."
    )
    parser.add_argument("--xml", required = True, help = "Path to input MusicXML file")
    #parser.add_argument("--checkpoint", required = True, help = "Path to trained .pt checkpoint")
    # parser.add_argument(
    #     "--output-prefix",
    #     required = False,
    #     default = None,
    #     help = "Prefix for output files, e.g. outputs/fuguel_structural"
    # )
    # parser.add_argument(
    #     "--layer",
    #     type = int,
    #     default = 2,
    #     help = "Which layer to use as the default reduction mask (1-based)"
    # )

    args = parser.parse_args()

    # Assign AutoSchA score to piece
    scorer(args.xml)

    json_path = 'outputs/inference/' + args.xml.replace('reduction_scores/', '').replace('.musicxml', '') + '.json'
    reduction_xml = 'outputs/reductions/' + args.xml.replace('reduction_scores/', '').replace('.musicxml', '-reduction.musicxml')

    # AutoSchA Reducer (Over Whole Piece)
    build_reduction(json_path, layer = 2, output_xml = reduction_xml)