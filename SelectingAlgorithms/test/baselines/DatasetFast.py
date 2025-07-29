import pandas as pd
import numpy as np
from collections import Counter

def find_and_average_matching_eps(row, most_common_strategies):
    try:
        target_strategy = most_common_strategies.loc[row['data']]
        target_tuple = (target_strategy['filter1'], target_strategy['order1'], target_strategy['enumerate1'])
    except KeyError:
        return np.nan

    matching_eps_values = []
    for x in range(1, 61):
        try:
            current_strategy_tuple = (row[f'filter{x}'], row[f'order{x}'], row[f'enumerate{x}'])
            if current_strategy_tuple == target_tuple:
                matching_eps_values.append(row[f'eps{x}'])
        except KeyError:
            break
            
    if matching_eps_values:
        return np.mean(matching_eps_values)
    else:
        return np.nan

try:
    DATA_df_base = pd.read_csv('All_data.csv')
except FileNotFoundError:
    exit()

all_run_averages = []
DATA_ORDER = [
    'DBLP.graph', 'citeseer.graph', 'human.graph', 'twitch.graph',
    'wordnet.graph', 'yeast.graph', 'youtube.graph'
]

for i in range(1, 6):
    file_name = f'topx_{i}.csv'
    train_file_name = f'train_{i}.csv'
    
    try:
        train_df = pd.read_csv(train_file_name)
    except FileNotFoundError:
        continue

    results = []
    if not train_df.empty:
        for data_group, group_df in train_df.groupby("data"):
            if group_df.empty: continue
            selected_columns = group_df[["filter1", "order1", "enumerate1"]]
            combinations = [tuple(row) for row in selected_columns.values]
            if not combinations: continue
            counter = Counter(combinations)
            most_common_combination, count = counter.most_common(1)[0]
            results.append({"data": data_group, "combination": most_common_combination, "count": count})
    if not results:
        continue
        
    result_df = pd.DataFrame(results)
    result_df[['filter1', 'order1', 'enumerate1']] = pd.DataFrame(result_df['combination'].tolist(), index=result_df.index)
    result_df.set_index('data', inplace=True)
    most_common_strategies = result_df[['filter1', 'order1', 'enumerate1']]
   
    try:
        top_df = pd.read_csv(file_name)
        valid_keys = set(zip(top_df['file_name'], top_df['data']))
    except FileNotFoundError:
        continue
        
    DATA_df_base['key'] = list(zip(DATA_df_base['file_name'], DATA_df_base['data']))
    current_DATA_subset = DATA_df_base[DATA_df_base['key'].isin(valid_keys)].copy()
    if current_DATA_subset.empty:
        continue
    
    current_DATA_subset['target_eps_avg'] = current_DATA_subset.apply(
        find_and_average_matching_eps, axis=1, most_common_strategies=most_common_strategies
    )
    
    current_averages = current_DATA_subset.groupby('data')['target_eps_avg'].mean()
    all_run_averages.append(current_averages)

if all_run_averages:
    final_average = pd.concat(all_run_averages, axis=1).mean(axis=1)
    ordered_final_average = final_average.reindex(DATA_ORDER)
    
    final_list = [round(val, 3) if pd.notna(val) else None for val in ordered_final_average]
    
    print(final_list)