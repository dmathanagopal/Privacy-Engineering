from pathlib import Path
import math
import numpy as np
import pandas as pd

COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'sex',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

def load(path):
    p = Path(path)
    if p.suffix.lower() == '.csv':
        d = pd.read_csv(p, skipinitialspace=True)
    else:
        d = pd.read_csv(p, header=None, names=COLUMNS, skipinitialspace=True, comment='|')
    d['age'] = pd.to_numeric(d['age'], errors='coerce')
    d = d.dropna(subset=['age']).reset_index(drop=True)
    d['education'] = d['education'].astype(str).str.strip()
    return d

def smoothed_logratio(a, b, domain):
    """Finite-sample diagnostic; add-one smoothing prevents zero bins."""
    ca = pd.Series(a).value_counts().reindex(domain, fill_value=0).astype(float) + 1
    cb = pd.Series(b).value_counts().reindex(domain, fill_value=0).astype(float) + 1
    return float(np.max(np.abs(np.log((ca / ca.sum()) / (cb / cb.sum())))))

def laplace_exact_privacy_loss(q1, q2, scale):
    """Exact max log-density ratio for equal-scale Laplace distributions."""
    return abs(float(q1) - float(q2)) / float(scale)

def laplace_task(data, out, eps, seed):
    d = load(data)
    variants = {
        'original': d,
        'remove_oldest': d.drop(index=d.age.idxmax()).reset_index(drop=True)
    }
    idx26 = d.index[d.age == 26]
    if len(idx26) == 0:
        raise ValueError('No age-26 record exists in the input data.')
    variants['remove_age26'] = d.drop(index=idx26[0]).reset_index(drop=True)
    variants['remove_youngest'] = d.drop(index=d.age.idxmin()).reset_index(drop=True)

    eligible = d.loc[d.age > 25, 'age']
    n = int(len(eligible))
    q = float(eligible.mean())
    # This is the assignment experiment's fixed-eligible-cohort assumption,
    # not a global proof for a data-dependent filter over arbitrary datasets.
    sensitivity = (90.0 - 26.0) / n
    scale = sensitivity / float(eps)

    rng = np.random.default_rng(seed)
    groups, rows, true_queries = {}, [], {}
    for name, frame in variants.items():
        values = frame.loc[frame.age > 25, 'age']
        true_q = float(values.mean())
        true_queries[name] = true_q
        noisy = true_q + rng.laplace(0.0, scale, 1000)
        rounded = np.round(noisy, 2)
        groups[name] = rounded
        rows.extend({
            'epsilon': eps, 'dataset': name, 'sample': j + 1,
            'true_query': true_q, 'noisy_mean_age': float(v),
            'rounded_output': float(r)
        } for j, (v, r) in enumerate(zip(noisy, rounded)))

    pd.DataFrame(rows).to_csv(out / f'laplace_outputs_eps_{eps:.1f}.csv', index=False)
    domain = sorted(set(np.concatenate(list(groups.values()))))
    empirical = {k: smoothed_logratio(groups['original'], v, domain)
                 for k, v in groups.items() if k != 'original'}
    exact = {k: laplace_exact_privacy_loss(true_queries['original'], true_queries[k], scale)
             for k in true_queries if k != 'original'}
    all_outputs = np.concatenate(list(groups.values()))
    mse = float(np.mean((all_outputs - q) ** 2))
    mae = float(np.mean(np.abs(all_outputs - q)))
    return q, n, sensitivity, scale, empirical, exact, mse, mae, true_queries

def exponential_probabilities(counts, epsilon):
    cats = list(counts.index)
    utilities = np.log1p(counts.to_numpy(dtype=float))
    delta_u = math.log(2.0)
    logits = float(epsilon) * utilities / (2.0 * delta_u)
    logits -= logits.max()
    weights = np.exp(logits)
    return cats, weights / weights.sum()

def exponential_task(data, out, eps, seed):
    d = load(data)
    counts = d.education.value_counts().sort_index()
    cats = list(counts.index)
    mode = counts.idxmax()
    ranked = counts.sort_values(ascending=False)
    second = ranked.index[1]
    least = counts.idxmin()
    variants = {'original': counts.copy()}
    for name, category in [('remove_most_frequent', mode),
                           ('remove_second_most_frequent', second),
                           ('remove_least_frequent', least)]:
        c = counts.copy()
        c.loc[category] -= 1
        variants[name] = c

    rng = np.random.default_rng(seed)
    groups, rows, exact = {}, [], {}
    for name, c in variants.items():
        local_cats, probs = exponential_probabilities(c, eps)
        sample = rng.choice(local_cats, 1000, p=probs)
        groups[name] = sample
        rows.extend({
            'epsilon': eps, 'dataset': name, 'sample': j + 1,
            'education_output': str(value),
            'is_original_mode': bool(value == mode)
        } for j, value in enumerate(sample))

    pd.DataFrame(rows).to_csv(out / f'exponential_outputs_eps_{eps:.1f}.csv', index=False)
    empirical = {k: smoothed_logratio(groups['original'], v, cats)
                 for k, v in groups.items() if k != 'original'}
    cats0, p0 = exponential_probabilities(variants['original'], eps)
    for name, c in variants.items():
        if name == 'original':
            continue
        cats1, p1 = exponential_probabilities(c, eps)
        # Compare the exact output distributions over the same category domain.
        lookup0 = dict(zip(cats0, p0))
        lookup1 = dict(zip(cats1, p1))
        exact[name] = float(max(abs(math.log(lookup0[x] / lookup1[x])) for x in cats0))
    modal_rates = {k: float(np.mean(v == mode)) for k, v in groups.items()}
    mode_error = 1.0 - modal_rates['original']
    return counts, mode, second, least, empirical, exact, modal_rates, mode_error


def thresholded_smoothed_logratio(a, b, domain, min_count=30):
    """Smoothed empirical log-ratio, retaining bins with at least min_count in both samples."""
    ca = pd.Series(a).value_counts().reindex(domain, fill_value=0).astype(float)
    cb = pd.Series(b).value_counts().reindex(domain, fill_value=0).astype(float)
    keep = (ca >= min_count) & (cb >= min_count)
    if not bool(keep.any()):
        return float("nan"), 0
    ca = ca[keep] + 1.0
    cb = cb[keep] + 1.0
    pa = ca / ca.sum()
    pb = cb / cb.sum()
    return float(np.max(np.abs(np.log(pa / pb)))), int(keep.sum())
