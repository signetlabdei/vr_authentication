import numpy as np
from scipy.stats import binomtest
import pandas as pd


AUTH = False

if (AUTH):
    users = 2
else:
    users = 30
samples = 24
confidence = 0.95
method = 'wilson' # Wilson or exact (Clopper-Pearson) or wilsoncc (with continuity correction)

voting_data = pd.read_csv('./id_results/voting_effect_MOV_TRAFFIC_NO_HEIGHT_ET.csv')

for row in voting_data.iterrows():
    n_samples = users * (samples - row[1][2] + 1)

    result = binomtest(k=int(row[1][3] * n_samples), n=n_samples)
    interval = result.proportion_ci(confidence_level = confidence, method=method)

    print(row[1][2], row[1][3], row[1][3] - interval.low, interval.high - row[1][3])
