import pandas as pd
import numpy as np

try:
    rec_algo_df_base = pd.read_csv('RecAlgo.csv')
except FileNotFoundError:
    exit()

all_run_averages = []
DATA_ORDER = [
    'DBLP.graph', 'citeseer.graph', 'human.graph', 'twitch.graph',
    'wordnet.graph', 'yeast.graph', 'youtube.graph'
]

for i in range(1, 6):
    file_name = f'topx_{i}.csv'

    try:
        top_df = pd.read_csv(file_name)
    except FileNotFoundError:
        continue

    merged_df = pd.merge(rec_algo_df_base, top_df, on=['file_name', 'data'], how='inner')
    
    if merged_df.empty:
        continue

    denominator = merged_df['rec_time'] + merged_df['time_feat']
    denominator.replace(0, np.nan, inplace=True)
    
    merged_df['new_metric'] = merged_df['rec_emb'] / denominator
    
    current_averages = merged_df.groupby('data')['new_metric'].mean()
    all_run_averages.append(current_averages)

if all_run_averages:
    final_average = pd.concat(all_run_averages, axis=1).mean(axis=1)
    
    ordered_final_average = final_average.reindex(DATA_ORDER)
    
    final_list = [round(val, 3) if pd.notna(val) else None for val in ordered_final_average]
    
    print(final_list)