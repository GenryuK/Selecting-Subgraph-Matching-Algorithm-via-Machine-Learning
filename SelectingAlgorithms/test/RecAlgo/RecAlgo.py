import pandas as pd

df = pd.read_csv('All_data.csv')

recalgo = []

th_low_cand = df['feature13'].quantile(0.1)
th_up_cand = df['feature13'].quantile(0.9)
th_low_label = df['feature12'].quantile(0.1)
th_up_label = df['feature12'].quantile(0.9)
th_low_tree_width = df['feature6'].quantile(0.1)
th_low_density = df['feature10'].quantile(0.1)
th_up_density = df['feature10'].quantile(0.9)


for _, row in df.iterrows():
        query_name = row['file_name']
        data_name = row['data']
        label_num = row['feature12']
        density_d = row['feature10']
        candidate_size = row['feature13']
        tree_width = row['feature6']

        
        rec_filter = "DPiso"
        if label_num >= th_up_label:
                rec_filter = "NLF"
        elif label_num <= th_low_label:
                rec_filter = "VEQ"

        rec_order = "RM"
        if density_d <= th_low_density and candidate_size <= th_low_cand:
                rec_order = "GQL"

        rec_enumeration = "VEQ"
        if tree_width <= th_low_tree_width:
                rec_enumeration = "KSS"
        if candidate_size >= th_up_cand:
                rec_enumeration = "KSS"

        for i in range(1, 61):
            filter_col = f'filter{i}'
            order_col = f'order{i}'
            enumerate_col = f'enumerate{i}'
            
            if (row[filter_col] == rec_filter and
                row[order_col] == rec_order and
                row[enumerate_col] == rec_enumeration):
                
                target_eps = row[f'eps{i}']
                target_emb = row[f'emb{i}']
                target_time = row[f'time{i}']
                found_index = i
        
                break

        
        row_data = [data_name, query_name, target_eps, target_emb, target_time, rec_filter, rec_order, rec_enumeration]
        recalgo.append(row_data)

results = pd.DataFrame(recalgo, columns=['data', 'file_name', 'rec_eps', 'rec_emb', 'rec_time', 'rec_filter', 'rec_order', 'rec_enumeration'])

results.to_csv('RecAlgo.csv', index=False)