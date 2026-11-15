# Data Anonymization

This project explores k-anonymity for relational and graph-derived data, focusing on generalization, minimum-cost degree anonymization, and utility loss.

## Components

### 1. Generalization and precision analysis

The accompanying analysis considers equivalence classes, k-anonymity, 2-minimal generalization, and precision under domain generalization hierarchies. The worked examples and health-table analysis are described in the original technical analysis; the repository focuses on the executable components below.

### 2. Degree anonymization for graph data

`src/degree_anonymization.py`:

1. Generates random simple undirected graphs.
2. Computes descending degree sequences.
3. Uses dynamic programming to find a minimum-cost k-anonymous degree sequence, allowing only degree increases.
4. Checks k-anonymity.
5. Applies the Erdős–Gallai test to check whether the resulting sequence is graphical.

The optimization groups consecutive sorted degrees and raises each degree in a group to the group's largest degree. The cost is the sum of the increases.

Run:

```bash
python src/degree_anonymization.py
```

The experiments use a fixed Python random seed of `2`, with graph sizes of 6, 10, and 20 vertices.

**Reproducibility note:** The accompanying analysis report contains sequences generated with a different implementation/random-number generator. The Python version is reproducible within Python, but its generated graphs and degree sequences need not match those earlier values.

**Algorithm limitation:** Graphicality is checked after the DP optimization, not enforced as a constraint during optimization. A sequence can therefore satisfy k-anonymity and still fail to represent a simple graph; the `n=6, k=3` case demonstrates this.

### 3. Record-specific k-anonymity for the UCI Adult dataset

`src/adult_k_anonymity.py` searches a finite set of global hierarchy configurations for four quasi-identifiers:

- Age
- Education
- Marital status
- Race

Each record receives a reproducible k requirement between 3 and 10 (seed `42`). An equivalence class is valid when its size is at least the maximum k requirement among its records.

The search evaluates the hierarchy-level combinations, prefers fewer fully suppressed attributes, and minimizes normalized distortion among feasible candidates. It first permits at most one fully suppressed quasi-identifier and uses a two-suppression fallback if needed.

#### Data setup

1. Download the UCI Adult dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult).
2. Place the training data file named `adult.data` in the repository root or update `INPUT_FILE` in the script to point to its location.
3. Install dependencies:

```bash
pip install pandas
```

4. Run from the repository root:

```bash
python data-anonymization/src/adult_k_anonymity.py
```

The script writes `adult_anonymized.csv` to the current working directory. The generated file is excluded from Git by `.gitignore`; the source dataset is also not included.

#### Metrics

Normalized distortion is the average of each selected hierarchy level divided by that attribute's maximum hierarchy depth. Precision is `1 - distortion`. A successful run reports zero k-anonymity violations.

## Results and interpretation

The original analysis recorded distortion `0.541667`, precision `0.458333`, and zero k-anonymity violations for its reported run. These values should be treated as reference results from that run, not guaranteed outputs across changed input files or code revisions.

The graph results and graphicality checks are printed by the script. See `results/README.md`.

## Limitations

- The degree-anonymization DP optimizes k-anonymity cost but does not incorporate graphicality into the optimization.
- The Adult-data heuristic uses one global hierarchy level per quasi-identifier, rather than optimizing each record independently.
- The per-record k values are randomly generated for experimentation; they are not inferred from real individuals.
- k-anonymity reduces certain linkage risks but does not by itself prevent all forms of disclosure or inference.
