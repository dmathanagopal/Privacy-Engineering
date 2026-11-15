#!/usr/bin/env python3
"""Q28: Exponential Mechanism experiment and validation at epsilon=1.0."""
from pathlib import Path
import pandas as pd
from dp_assignment_helpers import exponential_task, thresholded_smoothed_logratio
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
c,mode,second,least,emp,exact,modal,dist=exponential_task(ROOT/'adult.data.txt',OUT,1.0,5282027)
df=pd.read_csv(OUT/'exponential_outputs_eps_1.0.csv')
lines=["Q28 | Exponential mechanism | epsilon=1.0",f"Original mode: {mode} ({int(c[mode])})",
       f"Original-mode selection rate: {modal['original']:.6f}",f"Observed distortion: {dist:.6f}",
       "Comparison | exact max log probability ratio | limit | exact result | empirical ratio (min 30/category)"]
for name in ('remove_most_frequent','remove_second_most_frequent','remove_least_frequent'):
    a=df[df.dataset=='original'].education_output; b=df[df.dataset==name].education_output
    diag,bins=thresholded_smoothed_logratio(a,b,sorted(set(a)|set(b)),30)
    lines.append(f"{name} | {exact[name]:.9f} | 1.000000 | {'Pass' if exact[name] <= 1.0 else 'Fail'} | {diag:.6f} ({bins} categories)")
lines.append("Empirical ratios are sample diagnostics only; exact ratios compare the mechanism distributions.")
text="\n".join(lines); (OUT/'Q28_results.txt').write_text(text,encoding='utf-8'); print(text)
