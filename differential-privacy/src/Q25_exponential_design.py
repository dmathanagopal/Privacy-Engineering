#!/usr/bin/env python3
"""Q25: print the Exponential Mechanism utility and dataset category counts."""
from pathlib import Path
from dp_assignment_helpers import load
import math
ROOT=Path(__file__).resolve().parent
d=load(ROOT/'adult.data.txt'); c=d.education.value_counts()
lines=["Q25 | Exponential Mechanism design",f"delta_u=ln(2)={math.log(2):.8f}",f"Most frequent: {c.idxmax()} ({int(c.max())})",f"Second most frequent: {c.sort_values(ascending=False).index[1]} ({int(c.sort_values(ascending=False).iloc[1])})",f"Least frequent: {c.idxmin()} ({int(c.min())})","Utility: u(D,r)=ln(1+count_D(r))","Probability proportional to exp(epsilon*u(D,r)/(2*delta_u))."]
text="\n".join(lines); (ROOT/'results'/'Q25_results.txt').write_text(text,encoding='utf-8'); print(text)
