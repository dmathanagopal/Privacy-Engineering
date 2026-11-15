#!/usr/bin/env python3
"""Q29: compare Exponential Mechanism mode-error distortion from output CSVs."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results'
lines=["Q29 | Exponential Mechanism distortion comparison","epsilon | original-mode selection rate | mode-error distortion"]
for eps in (0.5,1.0):
    df=pd.read_csv(OUT/f'exponential_outputs_eps_{eps:.1f}.csv')
    original=df[df.dataset=='original']
    rate=float(original.is_original_mode.mean())
    lines.append(f"{eps:.1f} | {rate:.6f} | {1-rate:.6f}")
text="\n".join(lines); (OUT/'Q29_results.txt').write_text(text,encoding='utf-8'); print(text)
