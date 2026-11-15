"""
Problem 3 - k-Anonymity Heuristic for the UCI Adult Dataset


QIs:
    age, education, marital-status, race

k:
    Each record receives a reproducible k in [3, 10] using seed 42.

Output:
    adult_anonymized.csv
"""

import os
import random
from itertools import product

import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "adult.data"
OUTPUT_FILE = "adult_anonymized.csv"

RANDOM_SEED = 42
K_MIN = 3
K_MAX = 10

QIS = ["age", "education", "marital-status", "race"]

MAX_LEVEL = {
    "age": 3,
    "education": 3,
    "marital-status": 2,
    "race": 2,
}

# At most ONE QI is allowed to reach the top/suppression level
# in the preferred solution.
PREFERRED_SUPPRESSION_BUDGET = 1

# If no feasible solution exists with one suppression, the program
# will search with this fallback budget rather than silently failing.
FALLBACK_SUPPRESSION_BUDGET = 2


# ============================================================
# Adult dataset columns
# ============================================================

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


# ============================================================
# Generalization hierarchies
# ============================================================

EDU_L1 = {
    "Preschool": "Primary/Middle",
    "1st-4th": "Primary/Middle",
    "5th-6th": "Primary/Middle",
    "7th-8th": "Primary/Middle",

    "9th": "Secondary/HS",
    "10th": "Secondary/HS",
    "11th": "Secondary/HS",
    "12th": "Secondary/HS",
    "HS-grad": "Secondary/HS",

    "Some-college": "Higher-Ed",
    "Assoc-acdm": "Higher-Ed",
    "Assoc-voc": "Higher-Ed",
    "Bachelors": "Higher-Ed",

    "Masters": "Post-Grad",
    "Prof-school": "Post-Grad",
    "Doctorate": "Post-Grad",
}

EDU_L2 = {
    "Primary/Middle": "Non-Graduate",
    "Secondary/HS": "Non-Graduate",
    "Higher-Ed": "Higher-Education",
    "Post-Grad": "Higher-Education",
}

MARRIED = {
    "Married-civ-spouse",
    "Married-spouse-absent",
    "Married-AF-spouse",
}

NOT_MARRIED = {
    "Never-married",
    "Divorced",
    "Separated",
    "Widowed",
}

NON_WHITE = {
    "Black",
    "Asian-Pac-Islander",
    "Amer-Indian-Eskimo",
    "Other",
}


# ============================================================
# Load data
# ============================================================

def load_adult(path=INPUT_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} was not found. Put adult.data in the same "
            f"folder as this Python file."
        )

    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        skipinitialspace=True,
        na_values="?",
        keep_default_na=True,
    )

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype("string").str.strip()

    return df


# ============================================================
# Hierarchy functions
# ============================================================

def age_generalize(value, level):
    # Missing values are generalized to the top of the hierarchy.
    if pd.isna(value):
        return "*"

    age = int(value)

    if level == 0:
        return str(age)

    if level == 1:
        low = (age // 5) * 5
        return f"{low}-{low + 4}"

    if level == 2:
        low = (age // 10) * 10
        return f"{low}-{low + 9}"

    return "*"


def education_generalize(value, level):
    if pd.isna(value):
        return "*"

    value = str(value).strip()

    if level == 0:
        return value

    if level == 1:
        return EDU_L1.get(value, "*")

    if level == 2:
        l1 = EDU_L1.get(value, "*")
        return EDU_L2.get(l1, "*")

    return "*"


def marital_generalize(value, level):
    if pd.isna(value):
        return "*"

    value = str(value).strip()

    if level == 0:
        return value

    if level == 1:
        if value in MARRIED:
            return "Married"
        if value in NOT_MARRIED:
            return "Not-Married"
        return "*"

    return "*"


def race_generalize(value, level):
    if pd.isna(value):
        return "*"

    value = str(value).strip()

    if level == 0:
        return value

    if level == 1:
        if value == "White":
            return "White"
        if value in NON_WHITE:
            return "Non-White"
        return "*"

    return "*"


def generalize(qi, value, level):
    if qi == "age":
        return age_generalize(value, level)
    if qi == "education":
        return education_generalize(value, level)
    if qi == "marital-status":
        return marital_generalize(value, level)
    if qi == "race":
        return race_generalize(value, level)

    raise ValueError(qi)


# ============================================================
# Evaluate a hierarchy-level configuration
# ============================================================

def apply_levels(df, levels):
    """
    Apply one GLOBAL hierarchy level per QI.

    This is intentional: a hierarchy-level configuration gives
    a consistent, easy-to-explain anonymization for the entire
    released dataset.
    """
    result = pd.DataFrame(index=df.index)

    for qi in QIS:
        result[qi] = [
            generalize(qi, df.at[i, qi], levels[qi])
            for i in df.index
        ]

    return result


def build_equivalence_classes(generalized):
    groups = generalized.groupby(
        QIS,
        dropna=False,
        sort=False,
    ).groups

    return list(groups.values())


def k_violation(generalized, record_k):
    """
    For every equivalence class:
        class size >= max(k_i in that class)

    A violation is the number of records needed to reach
    the required class size.
    """
    total = 0

    for indices in build_equivalence_classes(generalized):
        required_k = max(record_k[i] for i in indices)
        size = len(indices)

        if size < required_k:
            total += required_k - size

    return total


def satisfies_k_anonymity(generalized, record_k):
    return k_violation(generalized, record_k) == 0


# ============================================================
# Distortion
# ============================================================

def distortion(levels):
    """
    Average normalized hierarchy depth across the four QIs.
    """
    total = sum(
        levels[qi] / MAX_LEVEL[qi]
        for qi in QIS
    )

    return total / len(QIS)


def suppression_count(levels):
    return sum(
        levels[qi] == MAX_LEVEL[qi]
        for qi in QIS
    )


# ============================================================
# Candidate hierarchy configurations
# ============================================================

def all_level_vectors(max_suppressions):
    """
    Generate all global hierarchy configurations.

    Example:
        age:             0..3
        education:       0..3
        marital-status:  0..2
        race:            0..2

    A candidate is kept only if it fully suppresses at most
    max_suppressions QIs.
    """
    ranges = [
        range(MAX_LEVEL[qi] + 1)
        for qi in QIS
    ]

    for values in product(*ranges):
        levels = dict(zip(QIS, values))

        if suppression_count(levels) <= max_suppressions:
            yield levels


def candidate_key(levels, violation):
    """
    Selection priority:
      1. Must satisfy k-anonymity.
      2. Fewer full suppressions.
      3. Lower distortion.
      4. Less total hierarchy movement.
      5. Deterministic QI-level tie break.
    """
    return (
        violation != 0,
        suppression_count(levels),
        distortion(levels),
        sum(levels.values()),
        tuple(levels[qi] for qi in QIS),
    )


def find_best_solution(df, record_k, suppression_budget):
    """
    Search the finite hierarchy space and return the feasible
    configuration with minimum distortion.

    The search is small:
        4 * 4 * 3 * 3 = 144 total configurations.

    Therefore this is practical even for the full Adult dataset.
    """
    best = None
    best_key = None

    candidates = list(all_level_vectors(suppression_budget))

    for levels in candidates:
        generalized = apply_levels(df, levels)
        violation = k_violation(generalized, record_k)

        if violation != 0:
            continue

        key = candidate_key(levels, violation)

        if best is None or key < best_key:
            best = {
                "levels": levels.copy(),
                "generalized": generalized,
                "violation": violation,
                "distortion": distortion(levels),
            }
            best_key = key

    return best


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 72)
    print("Problem 3 - k-Anonymity Heuristic")
    print("=" * 72)

    df = load_adult()

    print(f"Input records: {len(df)}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"k range: {K_MIN}-{K_MAX}")

    # Each record gets a different k value.
    rng = random.Random(RANDOM_SEED)
    record_k = {
        i: rng.randint(K_MIN, K_MAX)
        for i in df.index
    }

    print()
    print("Solutioning...")
    print()

    best = find_best_solution(
        df,
        record_k,
        PREFERRED_SUPPRESSION_BUDGET,
    )

    used_budget = PREFERRED_SUPPRESSION_BUDGET

    # If no one-suppression solution exists, use a clearly reported
    # fallback instead of silently producing an invalid dataset.
    if best is None:
        print(
            "No feasible solution was found with only one fully "
            "suppressed QI."
        )
        print("Trying a two-suppression fallback...")

        best = find_best_solution(
            df,
            record_k,
            FALLBACK_SUPPRESSION_BUDGET,
        )

        used_budget = FALLBACK_SUPPRESSION_BUDGET

    if best is None:
        raise RuntimeError(
            "No feasible hierarchy configuration was found."
        )

    levels = best["levels"]
    generalized = best["generalized"]

    final_violation = k_violation(
        generalized,
        record_k,
    )

    final_distortion = distortion(levels)
    final_precision = 1.0 - final_distortion

    # Add the record-specific k to the released dataset so the
    # result is easy to audit.
    output = df.copy()

    for qi in QIS:
        output[qi] = generalized[qi]

    output["k_req"] = [
        record_k[i]
        for i in df.index
    ]

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("FINAL RESULTS")
    print("=" * 72)

    print()
    print("Final hierarchy levels:")
    for qi in QIS:
        level = levels[qi]
        max_level = MAX_LEVEL[qi]

        if level == max_level:
            status = "SUPPRESSED (*)"
        else:
            status = "generalized"

        print(
            f"{qi}: level {level}/{max_level} - {status}"
        )

    print()
    print(f"Distortion: {final_distortion:.6f}")
    print(f"Precision:  {final_precision:.6f}")
    print(f"k-anonymity violations: {final_violation}")
    print(f"Output file: {OUTPUT_FILE}")

    if final_violation == 0:
        print()
        print(
            "SUCCESS: k-anonymity requirement is satisfied for "
            "all equivalence classes."
        )

    # if suppression_count(levels) <= 1:
    #     print()
    #     print(
    #         # "SUCCESS: The final solution suppresses at most ONE QI."
    #     )
    # else:
    #     print()
    #     print(
    #         "NOTE: A one-suppression solution was not feasible under "
    #         "the selected hierarchies and record-specific k values."
    #     )

    print()
    print("Generalization hierarchies:")
    print("age: exact -> 5-year range -> 10-year range -> *")
    print("education: exact -> 4 groups -> 2 groups -> *")
    print("marital-status: exact -> Married/Not-Married -> *")
    print("race: exact -> White/Non-White -> *")


if __name__ == "__main__":
    main()
