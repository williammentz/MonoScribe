# Automatic-Music-Reduction

## Setup

This repository uses multiple submodules. To clone this repository with the required submodules, run the following:

```bash
git clone --recurse-submodules https://github.com/williammentz/Automatic-Music-Reduction.git
cd Automatic-Music-Reduction
```

To load the submodules on a cloned repository missing the submodules, run:

```bash
git submodule update --init --recursive
```

## Train AutoSchA Model
Run `train.py` from the `Automatic-Music-Reduction` directory.

## Begin Reducing Scores
To run the automatic reduction, add a music score (.musicxml) into the `reduction_scores` folder and run the following from the project directory:

```bash
uv run automaticReduction.py --xml reduction_scores/{score_file}
```

For example, for the file `Mozart_12.musicxml`, run:
``` bash
uv run automaticReduction.py --xml reduction_scores/Mozart_12.musicxml
```

A final reduction file will be written to `Automatic-Music-Reduction/outputs/reductions/`.