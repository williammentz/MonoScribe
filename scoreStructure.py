import argparse
import csv
import json
import sys
from pathlib import Path

import music21
import torch
from torch_geometric.data import HeteroData

projRoot = Path(__file__).resolve().parent
outputDir = projRoot / "outputs/inference"
sys.path.insert(0, str(projRoot / "AutoSchA"))

from AutoSchA.model.gnn import GroupMat
from AutoSchA.utils.data_processing import HeterGraph, EnharmonicError
from AutoSchA.utils.config import DEVICE, NUM_FEAT, EMB_DIM, HIDDEN_DIM, NUM_CLASS
from AutoSchA.utils.data_processing import HeterGraph, EnharmonicError, load_training_notes

"""
To run the scorer, run the following:

uv run scoreStructure.py \
  --xml reduction_scores/Mozart_12.musicxml \
  --checkpoint AutoSchA/runs/base_model_epoch3.pt \
  --output-prefix outputs/mozart_new \ (unnecessary)
  --layer 2
"""

def build_hetero_data_from_xml(xml_path: str):
    xml_path = Path(xml_path)

    helper = HeterGraph(
        root="AutoSchA/processed_new_scores/inference_tmp",
        train_names=[],
        mode="inference",
    )

    # Partitura for note loading
    notes, score = load_training_notes(str(xml_path))

    # music21 for key analysis
    music21_score = music21.converter.parse(str(xml_path))
    analyzed_key = music21_score.analyze("key")

    hetero_data = HeteroData()
    hetero_data, notes_graph = helper.process_file_nodes(
        hetero_data,
        notes,
        analyzed_key,
        score,
    )
    hetero_data = helper.process_file_edges(
        hetero_data,
        notes_graph,
        notes,
    )

    # Return notes (TrainingNote list) and score (partitura) instead of xml_doc
    return hetero_data, notes, analyzed_key, score


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


def infer_time_signature(quarter_length):
    if float(quarter_length).is_integer():
        return f"{int(quarter_length)}/4"
    return "4/4"


def extract_measure_metadata(xml_path):
    score = music21.converter.parse(str(xml_path))
    measures = {}
    for part in score.parts:
        current_time_signature = None

        for m in part.getElementsByClass(music21.stream.Measure):
            if m.number is None:
                continue

            measure_number = int(m.number)
            time_signature = m.timeSignature or m.getContextByClass(
                music21.meter.TimeSignature
            )

            if time_signature is not None:
                current_time_signature = time_signature.ratioString

            if m.barDuration is not None:
                quarter_length = float(m.barDuration.quarterLength)
            else:
                quarter_length = float(m.duration.quarterLength)

            if measure_number not in measures:
                measures[measure_number] = {
                    "number": measure_number,
                    "quarter_length": quarter_length,
                    "time_signature": current_time_signature,
                }
            elif measures[measure_number]["time_signature"] is None:
                measures[measure_number]["time_signature"] = current_time_signature

    ordered_measures = [
        measures[number]
        for number in sorted(measures)
    ]

    first_time_signature = next(
        (
            m["time_signature"]
            for m in ordered_measures
            if m["time_signature"] is not None
        ),
        None,
    )

    for m in ordered_measures:
        if m["time_signature"] is None:
            m["time_signature"] = first_time_signature or infer_time_signature(
                m["quarter_length"]
            )

    return ordered_measures


def extract_symbolic_notes(notes, score):
    """Build symbolic note dicts from TrainingNote objects."""
    import partitura as pt

    # Get measure start times from partitura
    measure_starts = {}
    for part in score.parts:
        for m in part.iter_all(pt.score.Measure):
            if m.number is not None and m.number not in measure_starts:
                measure_starts[m.number] = m.start.t

    # Get quarter duration for offset calculation
    from AutoSchA.utils.data_processing import compute_quarter_duration
    qd = compute_quarter_duration(score, "")  # xml_path not needed if score loaded

    symbolic_notes = []
    for n in notes:
        # Find which measure this note belongs to
        measure_number = None
        for mnum in sorted(measure_starts.keys(), reverse=True):
            if n.onset >= measure_starts[mnum]:
                measure_number = mnum
                break
        if measure_number is None:
            measure_number = min(measure_starts.keys()) if measure_starts else 1

        measure_start = measure_starts.get(measure_number, 0)
        measure_offset = (n.onset - measure_start) / qd

        symbolic_notes.append({
            "pitch_name": n.pitch_name,
            "pitch_midi": n.midi_pitch,
            "measure": measure_number,
            "measure_offset": measure_offset,
            "quarter_length": n.duration_ql,
            "time_position": n.onset,
            "duration_seconds": n.duration,  # raw divisions, not seconds
            "staff": None,  # not tracked in TrainingNote
        })

    return symbolic_notes


def note_to_dict(note, idx, layer_scores, layer_masks, symbolic_note):
    """
    Export symbolic note representation suitable for reconstruction.
    """

    return {
        "note_index": idx,

        "pitch_name": symbolic_note["pitch_name"],
        "pitch_midi": symbolic_note["pitch_midi"],

        "measure": symbolic_note["measure"],

        "measure_offset": symbolic_note["measure_offset"],

        "quarter_length": symbolic_note["quarter_length"],

        "time_position": symbolic_note["time_position"],
        "duration_seconds": symbolic_note["duration_seconds"],
        "staff": symbolic_note["staff"],

        "voice": note.voice,

        "layer_scores": [
            float(s[idx]) for s in layer_scores
        ],

        "layer_masks": [
            bool(m[idx]) for m in layer_masks
        ],
    }


def write_outputs(
    output_prefix: str,
    notes,
    symbolic_notes,
    analyzed_key,
    layer_scores,
    layer_masks,
    selected_layer: int,
    measure_metadata,
    source_xml,
):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")

    note_rows = [
        note_to_dict(
            note,
            i,
            layer_scores,
            layer_masks,
            symbolic_notes[i],
        )
        for i, note in enumerate(notes)
    ]

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "analyzed_key": str(analyzed_key),
                "source_xml": str(source_xml),
                "num_notes": len(notes),
                "num_layers": len(layer_scores),
                "selected_layer": selected_layer,
                "threshold": 0.5,
                "measures": measure_metadata,
                "notes": note_rows,
            },
            f,
            indent=2,
        )

    # CSV
    fieldnames = [
        "note_index",
        'measure',
        'measure_offset',
        'quarter_length',
        "time_position",
        "duration_seconds",
        "pitch_name",
        "pitch_midi",
        "staff",
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
                'measure': symbolic_notes[i]['measure'],
                'measure_offset': symbolic_notes[i]['measure_offset'],
                'quarter_length': symbolic_notes[i]['quarter_length'],
                "time_position": symbolic_notes[i]["time_position"],
                "duration_seconds": symbolic_notes[i]["duration_seconds"],
                "pitch_name": notes[i].pitch_name,
                "pitch_midi": notes[i].midi_pitch,
                "staff": symbolic_notes[i]["staff"],
                "voice": notes[i].voice
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
        required=False,
        default=None,
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
        hetero_data, notes, analyzed_key, score = build_hetero_data_from_xml(args.xml)
    except EnharmonicError as e:
        raise RuntimeError(f"Failed during feature extraction: {e}") from e

    model = load_model(args.checkpoint)
    _, layer_scores, layer_masks = run_inference(model, hetero_data)

    symbolic_notes = extract_symbolic_notes(notes, score)
    measure_metadata = extract_measure_metadata(args.xml)

    if args.layer < 1 or args.layer > len(layer_scores):
        raise ValueError(
            f"--layer must be between 1 and {len(layer_scores)}, got {args.layer}"
        )

    output_name = args.output_prefix or Path(args.xml).stem
    output_prefix = outputDir / output_name

    json_path, csv_path, summary_path = write_outputs(
        output_prefix,
        notes,
        symbolic_notes,
        analyzed_key,
        layer_scores,
        layer_masks,
        selected_layer=args.layer,
        measure_metadata=measure_metadata,
        source_xml=args.xml,
    )

    print(f"Done.")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")

def scorer(xml):

    try:
        hetero_data, notes, analyzed_key, score = build_hetero_data_from_xml(xml)
    except EnharmonicError as e:
        raise RuntimeError(f"Failed during feature extraction: {e}") from e

    layer = 2

    model = load_model('AutoSchA/runs/base_model_epoch2.pt')
    _, layer_scores, layer_masks = run_inference(model, hetero_data)

    symbolic_notes = extract_symbolic_notes(notes, score)
    measure_metadata = extract_measure_metadata(xml)

    if layer < 1 or layer > len(layer_scores):
        raise ValueError(
            f"--layer must be between 1 and {len(layer_scores)}, got {layer}"
        )

    output_name = Path(xml).stem
    output_prefix = outputDir / output_name

    json_path, csv_path, summary_path = write_outputs(
        output_prefix,
        notes,
        symbolic_notes,
        analyzed_key,
        layer_scores,
        layer_masks,
        selected_layer=2,
        measure_metadata=measure_metadata,
        source_xml=xml,
    )
    print(f"Piece scored.")


if __name__ == "__main__":
    main()
