"""
Used to shuffle Narrative data originally.
IMDB data came from Kaggle.
"""

import pandas as pd
df = pd.read_csv('instrument_clean.csv',
                     names=['', "text_raw", "instrument", "text_clean", "text_freq"], skiprows=[0])
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('instrument_clean.csv')

