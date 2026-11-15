#!/usr/bin/env python3
"""Q24: compare Laplace distortion using the rounded released outputs."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
lines=["Q24 | Laplace distortion comparison","epsilon | MSE | MAE"]
for eps in (0.5,1.0):
    df=pd.read_csv(OUT/f'laplace_outputs_eps_{eps:.1f}.csv')
    truth=float(df.loc[df.dataset=='original','true_query'].iloc[0])
    values=df['rounded_output'].astype(float)
    mse=float(((values-truth)**2).mean())
    mae=float((values-truth).abs().mean())
    lines.append(f"{eps:.1f} | {mse:.8f} | {mae:.8f}")
text="\n".join(lines)
(OUT/'Q24_results.txt').write_text(text,encoding='utf-8')
print(text)
