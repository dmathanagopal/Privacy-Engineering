#!/usr/bin/env python3
"""CS528 Assignment 2 Q21: run the Laplace experiment at epsilon=0.5."""
from pathlib import Path
from dp_assignment_helpers import laplace_task
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
DATA=ROOT/'adult.data.txt'
q,n,sensitivity,scale,empirical,exact,mse,mae,true_queries=laplace_task(DATA,OUT,0.5, 5282026)
lines=[
    "Q21 | Laplace mechanism | epsilon=0.5",
    f"Eligible records (age > 25): {n}",
    f"Original mean age: {q:.9f}",
    f"Fixed-cohort sensitivity assumption: {sensitivity:.10f}",
    f"Laplace scale: {scale:.10f}",
    f"MSE across 4,000 outputs: {mse:.8f}",
    f"MAE across 4,000 outputs: {mae:.8f}",
    "Exact distribution-based privacy losses:"
]
for name,value in exact.items():
    lines.append(f"{name}: {value:.9f}; limit=0.5; pass={value <= 0.5 + 1e-12}")
lines.append("Caution: the age > 25 filter is data-dependent; this observed-n sensitivity is not a global sensitivity proof.")
text="\n".join(lines)
(OUT/'Q21_results.txt').write_text(text,encoding='utf-8')
print(text)
