#!/usr/bin/env python3
"""Q23: Laplace experiment and validation at epsilon=1.0."""
from pathlib import Path
import pandas as pd
from dp_assignment_helpers import laplace_task, thresholded_smoothed_logratio
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
q,n,sensitivity,scale,emp,exact,mse,mae,trueq=laplace_task(ROOT/'adult.data.txt',OUT,1.0,5282026)
df=pd.read_csv(OUT/'laplace_outputs_eps_1.0.csv')
lines=["Q23 | Laplace mechanism and indistinguishability | epsilon=1.0",
       f"Eligible records (age > 25): {n}",f"Original mean age: {q:.9f}",
       f"Fixed-cohort sensitivity assumption: {sensitivity:.10f}",f"Laplace scale: {scale:.10f}",
       f"MSE across 4,000 outputs: {mse:.8f}",f"MAE across 4,000 outputs: {mae:.8f}",
       "Comparison | exact max log-density ratio | limit | exact result | empirical ratio (min 30/bin)"]
for name in ('remove_oldest','remove_age26','remove_youngest'):
    a=df[df.dataset=='original'].rounded_output; b=df[df.dataset==name].rounded_output
    diag,bins=thresholded_smoothed_logratio(a,b,sorted(set(a)|set(b)),30)
    lines.append(f"{name} | {exact[name]:.9f} | 1.000000 | {'Pass' if exact[name] <= 1.0 else 'Fail'} | {diag:.6f} ({bins} bins)")
lines.append("Caution: age > 25 is a data-dependent filter; observed-n sensitivity is not a global sensitivity proof.")
text="\n".join(lines); (OUT/'Q23_results.txt').write_text(text,encoding='utf-8'); print(text)
