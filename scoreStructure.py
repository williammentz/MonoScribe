import argparse
import csv
import json
import sys
from pathlib import Path

import music21
import torch
from torch_geometric.data import HeteroData

projRoot = Path(__file__).resolve().parent
sys.path.insert(0, str(projRoot / "AutoSchA"))

from AutoSchA.model.gnn import GroupMat
from AutoSchA.utils.data_processing import HeterGraph, EnharmonicError
from AutoSchA.utils.config import DEVICE, NUM_FEAT, EMB_DIM, HIDDEN_DIM, NUM_CLASS
from AutoSchA.pyScoreParser.musicxml_parser.mxp import MusicXMLDocument

"""
To run the scorer, run the following:

uv run scoreStructure.py \
  --xml reduction_scores/Mozart_12.musicxml \
  --checkpoint AutoSchA/runs/base_model_epoch3.pt \
  --output-prefix outputs/test \
  --layer 2
"""

def build_hetero_data_from_xml(xml_path: str):
    """
    Reuses the repo's existing preprocessing logic so inference matches training.
    Returns:
        hetero_data: PyG HeteroData object
        pyscoreparser_notes: original note objects in node order
        analyzed_key: music21 key estimate used for scale-degree features
    """
    xml_path = Path(xml_path)

    helper = HeterGraph(
        root="processed/inference_tmp",
        train_names=[],
        mode="inference",
    )

    xml_doc = MusicXMLDocument(str(xml_path))
    pyscoreparser_notes = xml_doc.get_notes()

    music21_score = music21.converter.parse(str(xml_path))
    analyzed_key = music21_score.analyze("key")

    hetero_data = HeteroData()
    hetero_data, notes_graph = helper.process_file_nodes(
        hetero_data,
        pyscoreparser_notes,
        analyzed_key,
    )
    hetero_data = helper.process_file_edges(
        hetero_data,
        notes_graph,
        pyscoreparser_notes,
    )

    return hetero_data, pyscoreparser_notes, analyzed_key


def load_model(checkpoint_path: str):
    model = GroupMat(NUM_FEAT, EMB_DIM, HIDDEN_DIM, NUM_CLASS, DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def run_inference(model, hetero_data):
    hetero_data = hetero_data.to(DEVICE)
    with torch.no_grad():
        x, _, _, all_scores = model(
            hetero_data,
            cluster_matrices=[],
            inference=True,
        )

    # Convert to plain CPU tensors for downstream use
    cpu_scores = [scores.detach().cpu().flatten() for scores in all_scores]
    masks = [(scores >= model.min_score) for scores in cpu_scores]

    return x, cpu_scores, masks


def note_to_dict(note, idx, layer_scores, layer_masks):
    """
    Minimal note metadata for downstream reduction/debugging.
    Adjust fields if your pyScoreParser Note class exposes more useful info.
    """
    state = getattr(note, "state_fixed", None)

    return {
        "note_index": idx,
        "pitch_name": note.pitch[0],
        "pitch_midi": note.pitch[1],
        "time_position": note.state_fixed.time_position,
        "duration_seconds": note.note_duration.seconds,
        "voice": getattr(note, "voice", None),
        "layer_scores": [float(s[idx]) for s in layer_scores],
        "layer_masks": [bool(m[idx]) for m in layer_masks],
    }


def write_outputs(
    output_prefix: str,
    notes,
    analyzed_key,
    layer_scores,
    layer_masks,
    selected_layer: int,
):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")

    note_rows = [
        note_to_dict(note, i, layer_scores, layer_masks)
        for i, note in enumerate(notes)
    ]

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "analyzed_key": str(analyzed_key),
                "num_notes": len(notes),
                "num_layers": len(layer_scores),
                "selected_layer": selected_layer,
                "threshold": 0.5,
                "notes": note_rows,
            },
            f,
            indent=2,
        )

    # CSV
    fieldnames = [
        "note_index",
        'measure',
        "pitch_name",
        "pitch_midi",
        "time_position",
        "duration_seconds",
        "voice",
    ]
    for i in range(len(layer_scores)):
        fieldnames.append(f"layer_{i+1}_score")
        fieldnames.append(f"layer_{i+1}_mask")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, note in enumerate(notes):
            row = {
                "note_index": i,
                "pitch_name": note.pitch[0],
                "pitch_midi": note.pitch[1],
                "time_position": note.state_fixed.time_position,
                "duration_seconds": note.note_duration.seconds,
                "voice": getattr(note, "voice", None),
            }
            for layer_idx in range(len(layer_scores)):
                row[f"layer_{layer_idx+1}_score"] = float(layer_scores[layer_idx][i])
                row[f"layer_{layer_idx+1}_mask"] = bool(layer_masks[layer_idx][i])
            writer.writerow(row)

    selected_mask = layer_masks[selected_layer - 1]
    selected_indices = [i for i, keep in enumerate(selected_mask.tolist()) if keep]

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "analyzed_key": str(analyzed_key),
                "selected_layer": selected_layer,
                "threshold": 0.5,
                "num_notes": len(notes),
                "num_structural_notes": len(selected_indices),
                "structural_note_indices": selected_indices,
            },
            f,
            indent=2,
        )

    return json_path, csv_path, summary_path


def main():
    parser = argparse.ArgumentParser(
        description="Run AutoSchA-style GNN inference on a new MusicXML score."
    )
    parser.add_argument("--xml", required=True, help="Path to input MusicXML file")
    parser.add_argument("--checkpoint", required=True, help="Path to trained .pt checkpoint")
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Prefix for output files, e.g. outputs/fugue1_structural",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=2,
        help="Which layer to use as the default reduction mask (1-based)",
    )

    args = parser.parse_args()

    try:
        hetero_data, notes, analyzed_key = build_hetero_data_from_xml(args.xml)
    except EnharmonicError as e:
        raise RuntimeError(f"Failed during feature extraction: {e}") from e

    model = load_model(args.checkpoint)
    _, layer_scores, layer_masks = run_inference(model, hetero_data)

    if args.layer < 1 or args.layer > len(layer_scores):
        raise ValueError(
            f"--layer must be between 1 and {len(layer_scores)}, got {args.layer}"
        )

    json_path, csv_path, summary_path = write_outputs(
        args.output_prefix,
        notes,
        analyzed_key,
        layer_scores,
        layer_masks,
        selected_layer=args.layer,
    )

    print(f"Done.")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()