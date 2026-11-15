#!/usr/bin/env python3
"""Q22: exact and sample-based Laplace indistinguishability checks at epsilon=0.5."""
from pathlib import Path
import pandas as pd
from dp_assignment_helpers import laplace_task, thresholded_smoothed_logratio
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
q,n,sens,scale,emp,exact,mse,mae,trueq=laplace_task(ROOT/'adult.data.txt',OUT,0.5,5282026)
df=pd.read_csv(OUT/'laplace_outputs_eps_0.5.csv')
lines=["Q22 | Laplace indistinguishability | epsilon=0.5",
       "Comparison | exact max log-density ratio | epsilon limit | exact result | empirical ratio (min 30/bin)"]
for name in ('remove_oldest','remove_age26','remove_youngest'):
    a=df[df.dataset=='original'].rounded_output; b=df[df.dataset==name].rounded_output
    diag,bins=thresholded_smoothed_logratio(a,b,sorted(set(a)|set(b)),30)
    lines.append(f"{name} | {exact[name]:.9f} | 0.500000 | {'Pass' if exact[name] <= .5 else 'Fail'} | {diag:.6f} ({bins} bins)")
lines.append("Empirical ratios are diagnostics only; exact ratios compare the mechanism distributions for these query values.")
text="\n".join(lines); (OUT/'Q22_results.txt').write_text(text,encoding='utf-8'); print(text)
