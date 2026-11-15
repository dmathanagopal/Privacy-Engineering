#!/usr/bin/env python3
"""Q27: exact and sample-based Exponential Mechanism checks at epsilon=0.5."""
from pathlib import Path
import pandas as pd
from dp_assignment_helpers import exponential_task, thresholded_smoothed_logratio
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
c,mode,second,least,emp,exact,modal,dist=exponential_task(ROOT/'adult.data.txt',OUT,0.5,5282027)
df=pd.read_csv(OUT/'exponential_outputs_eps_0.5.csv')
lines=["Q27 | Exponential mechanism indistinguishability | epsilon=0.5",
       "Comparison | exact max log probability ratio | limit | exact result | empirical ratio (min 30/category)"]
for name in ('remove_most_frequent','remove_second_most_frequent','remove_least_frequent'):
    a=df[df.dataset=='original'].education_output; b=df[df.dataset==name].education_output
    diag,bins=thresholded_smoothed_logratio(a,b,sorted(set(a)|set(b)),30)
    lines.append(f"{name} | {exact[name]:.9f} | 0.500000 | {'Pass' if exact[name] <= .5 else 'Fail'} | {diag:.6f} ({bins} categories)")
lines.append("The second sample-based ratio may exceed epsilon in this finite run; this diagnostic is not a formal DP proof.")
text="\n".join(lines); (OUT/'Q27_results.txt').write_text(text,encoding='utf-8'); print(text)
