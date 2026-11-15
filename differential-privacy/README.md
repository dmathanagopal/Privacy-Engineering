# Differential Privacy

This project implements and evaluates mechanisms used in differential privacy, with experiments covering the Laplace mechanism and the Exponential mechanism.

## Included experiments

The `src/` directory contains scripts for:

- Laplace mechanism queries at multiple epsilon values.
- Empirical indistinguishability diagnostics.
- Distortion analysis for noisy outputs.
- Exponential mechanism design and sampling.
- Helper functions shared by the experiments.

The `results/` directory contains selected text-based experiment outputs. Large raw datasets and generated CSV samples are intentionally excluded.

## Setup

The scripts use Python. Install the dependencies required by the scripts:

```bash
pip install numpy pandas
```

## Data

The Adult dataset is used in some experiments. Obtain the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult). Do not commit raw source data unless its redistribution terms have been checked. Review each script's expected input filename before running it.

## Interpreting the experiments

Empirical indistinguishability checks and finite-sample output frequencies are diagnostics, not formal proofs of differential privacy. Privacy claims depend on the mechanism, adjacency definition, sensitivity, and assumptions documented by each experiment.

## Reproducibility

Review the constants and random seeds in each script before comparing runs. Epsilon controls the privacy-utility trade-off; the reported results are experimental observations under the selected setup, not universal guarantees.
