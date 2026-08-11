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
To run the automatic reduction, add a music score (.musicxml) into the `reduction_scores` folder and run the following from the project directory. All example code with be using the `uv` environment:

```bash
uv run automaticReduction.py --{score_name.musicxml}
```

For example, for the file `Mozart_12.musicxml`, run:
``` bash
uv run automaticReduction.py --piece Mozart_12.musicxml
```

We also have `utility`, `density`, `contour`, and `continuity` as parameters, so altering them is available to the user, as well. For example:

``` bash
uv run automaticReduction.py --piece Mozart_12.musicxml --utility 0.3 --density 0.7 --contour 0.6 --continuity 0.1
```

To interweave the melody and accompanimental material, include `interweave`:

``` bash
uv run automaticReduction.py --piece Mozart_12.musicxml --utility 0.3 --density 0.7 --contour 0.6 --continuity 0.1 --interweave
```

A final reduction file will be written to `Automatic-Music-Reduction/outputs/reductions/`.

## Annotating Scores
To see the original score with the highlighted reduction, run:
``` bash
uv run automaticReduction.py --piece Mozart_12.musicxml --annotate
```