#!/usr/bin/env python3
"""CS528 Assignment 2 Q26: run the Exponential Mechanism at epsilon=0.5."""
from pathlib import Path
import math
from dp_assignment_helpers import exponential_task
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
DATA=ROOT/'adult.data.txt'
counts,mode,second,least,empirical,exact,modal,distortion=exponential_task(DATA,OUT,0.5,5282027)
lines=[
    "Q26 | Exponential Mechanism | epsilon=0.5",
    f"Utility sensitivity delta_u=ln(2)={math.log(2):.8f}",
    f"Most frequent category: {mode} ({int(counts[mode])})",
    f"Second most frequent: {second} ({int(counts[second])})",
    f"Least frequent: {least} ({int(counts[least])})",
    f"Original mode-selection rate: {modal['original']:.6f}",
    f"Observed distortion: {distortion:.6f}",
    "Exact distribution-based privacy losses:"
]
for name,value in exact.items():
    lines.append(f"{name}: {value:.9f}; limit=0.5; pass={value <= 0.5 + 1e-12}")
text="\n".join(lines)
(OUT/'Q26_results.txt').write_text(text,encoding='utf-8')
print(text)
