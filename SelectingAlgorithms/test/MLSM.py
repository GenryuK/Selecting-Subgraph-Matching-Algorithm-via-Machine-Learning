# 

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import os

import warnings
from pandas.errors import PerformanceWarning

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

data = pd.read_csv('All_data.csv')
data.columns = [''.join(col) if isinstance(col, tuple) else col for col in data.columns]
column_to_modify = [f"eps{i+1}" for i in range(60)]
data.loc[:, column_to_modify] = data[column_to_modify].where(data[column_to_modify] >= 0, 0) 

# data_s is used when you test another query set. (e.g., train Induced query and test Star query, data = INDUCED.csv, data_s = STAR.csv)
data_s = pd.read_csv('All_data.csv')
data_s.columns = [''.join(col) if isinstance(col, tuple) else col for col in data_s.columns]
column_to_modify_s = [f"eps{i+1}" for i in range(60)]
data_s.loc[:, column_to_modify_s] = data_s[column_to_modify_s].where(data_s[column_to_modify_s] >= 0, 0)

# Remove queries where any algorithms can't find embeddings in time.
data = data[data["eps1"] != 0.0].reset_index(drop=True)
data_s = data_s[data_s["eps1"] != 0.0].reset_index(drop=True)

data_cs = data[data["data"] == "citeseer.graph"].reset_index(drop=True)
data_db = data[data["data"] == "DBLP.graph"].reset_index(drop=True)
data_hm = data[data["data"] == "human.graph"].reset_index(drop=True)
data_tw = data[data["data"] == "twitch.graph"].reset_index(drop=True)
data_wn = data[data["data"] == "wordnet.graph"].reset_index(drop=True)
data_ys = data[data["data"] == "yeast.graph"].reset_index(drop=True)
data_yt = data[data["data"] == "youtube.graph"].reset_index(drop=True)

attribute_combinations = {}
attribute_combinations_inv = {}
attribute_num = 1               

for idx, row in data.iterrows():
    for i in range(60):
        combination = (row["filter" + str(i+1)], row["order" + str(i+1)], row["enumerate" + str(i+1)])
        if combination not in attribute_combinations.values():
            attribute_combinations[attribute_num] = combination
            attribute_combinations_inv[combination] = attribute_num
            attribute_num += 1

query_data_pairs = data[['file_name', 'data']].assign(file_name=data['file_name'].str.strip())

# function for TopX
def top_n_dataset(data, num_attributes):
    for i in range(len(attribute_combinations)):
        data["output_feature" + str(i+1)] = 0
    for idx, row in data.iterrows():
        for i in range(num_attributes):
            combination = (row["filter" + str(i+1)], row["order" + str(i+1)], row["enumerate" + str(i+1)])
            data.at[idx, "output_feature" + str(attribute_combinations_inv[combination])] = 1

    X = data[[f"feature{i+1}" for i in range(13)]]
    y = data[['data']+['file_name']+[f"output_feature{i+1}" for i in range(len(attribute_combinations))]]

    return X, y

# function for IncY
def inc_x_dataset(data, permitted_increase_rate):
    for i in range(len(attribute_combinations)):
        data["output_feature" + str(i+1)] = 0

    for idx, row in data.iterrows():
        best_eps = row["eps1"]
        for i in range(60):
            if row["eps" + str(i+1)] >= best_eps * permitted_increase_rate:
                combination = (row["filter" + str(i+1)], row["order" + str(i+1)], row["enumerate" + str(i+1)])
                data.at[idx, "output_feature" + str(attribute_combinations_inv[combination])] = 1

    X = data[[f"feature{i+1}" for i in range(13)]]
    y = data[['data']+['file_name']+[f"output_feature{i+1}" for i in range(len(attribute_combinations))]]

    return X, y

# function for Weight
def time_weighted_dataset(data):
    for i in range(len(attribute_combinations)):
        data["output_feature" + str(i+1)] = 0.0

    for idx, row in data.iterrows():
        best_eps = row["eps1"]
        for i in range(60):
            combination = (row["filter" + str(i+1)], row["order" + str(i+1)], row["enumerate" + str(i+1)])
            if best_eps == 0:
                data.at[idx, "output_feature" + str(attribute_combinations_inv[combination])] = 0
            else:
                data.at[idx, "output_feature" + str(attribute_combinations_inv[combination])] = row["eps" + str(i+1)] / best_eps


    X = data[[f"feature{i+1}" for i in range(13)]]
    y = data[['data']+['file_name']+[f"output_feature{i+1}" for i in range(len(attribute_combinations))]]

    return X, y


# Here, you can select our model.
task = "topx"
# For topx, num_attributes = [1, 2, 3, 4, 5, 6, 7]
# For incy, num_attributes = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
num_attributes = 1

def get_data(task, num):
    if task == "topx":
        X_cs, y_cs = top_n_dataset(data_cs, num)
        X_db, y_db = top_n_dataset(data_db, num)
        X_hm, y_hm = top_n_dataset(data_hm, num)
        X_tw, y_tw = top_n_dataset(data_tw, num)
        X_wn, y_wn = top_n_dataset(data_wn, num)
        X_ys, y_ys = top_n_dataset(data_ys, num)
        X_yt, y_yt = top_n_dataset(data_yt, num)

    elif task == "incy":
        X_cs, y_cs = inc_x_dataset(data_cs, num)
        X_db, y_db = inc_x_dataset(data_db, num)
        X_hm, y_hm = inc_x_dataset(data_hm, num)
        X_tw, y_tw = inc_x_dataset(data_tw, num)
        X_wn, y_wn = inc_x_dataset(data_wn, num)
        X_ys, y_ys = inc_x_dataset(data_ys, num)
        X_yt, y_yt = inc_x_dataset(data_yt, num)

    elif task == "weight":
        X_cs, y_cs = time_weighted_dataset(data_cs)
        X_db, y_db = time_weighted_dataset(data_db)
        X_hm, y_hm = time_weighted_dataset(data_hm)
        X_tw, y_tw = time_weighted_dataset(data_tw)
        X_wn, y_wn = time_weighted_dataset(data_wn)
        X_ys, y_ys = time_weighted_dataset(data_ys)
        X_yt, y_yt = time_weighted_dataset(data_yt)

    return X_cs, y_cs, X_db, y_db, X_hm, y_hm, X_tw, y_tw, X_wn, y_wn, X_ys, y_ys, X_yt, y_yt

def get_data_s(task, num):
  if task == "topx":
    X_s, y_s = top_n_dataset(data_s, num)

  elif task == "incy":
    X_s, y_s = inc_x_dataset(data_s, num)

  elif task == "weight":
    X_s, y_s = time_weighted_dataset(data_s)

  return X_s, y_s



X_cs, y_cs, X_db, y_db, X_hm, y_hm, X_tw, y_tw, X_wn, y_wn, X_ys, y_ys, X_yt, y_yt = get_data(task, num_attributes)
X_s, y_s = get_data_s(task, num_attributes)


# Here, you can select features. 
# RecAlgo features are ['feature6', 'feature10', 'feature12', 'feature13']
X_cs = data_cs[['data', 'file_name', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13']]
X_db = data_db[['data', 'file_name', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13']]
X_hm = data_hm[['data', 'file_name', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13']]
X_tw = data_tw[['data', 'file_name', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13']]
X_wn = data_wn[['data', 'file_name', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13']]
X_ys = data_ys[['data', 'file_name', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13']]
X_yt = data_yt[['data', 'file_name', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13']]

X_s = data_s[['data', 'file_name', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13']]

def normalize_row(row):
    
    row_sum = row.sum()  
    if row_sum == 0:
        row_sum = 1e-10  
    return row / row_sum  


y_str_cs = y_cs.iloc[:, :2]  
y_numeric_cs = y_cs.iloc[:, 2:] 
y_str_db = y_db.iloc[:, :2]
y_numeric_db = y_db.iloc[:, 2:]
y_str_hm = y_hm.iloc[:, :2]
y_numeric_hm = y_hm.iloc[:, 2:]
y_str_tw = y_tw.iloc[:, :2]
y_numeric_tw = y_tw.iloc[:, 2:]
y_str_wn = y_wn.iloc[:, :2]
y_numeric_wn = y_wn.iloc[:, 2:]
y_str_ys = y_ys.iloc[:, :2]
y_numeric_ys = y_ys.iloc[:, 2:]
y_str_yt = y_yt.iloc[:, :2]
y_numeric_yt = y_yt.iloc[:, 2:]

y_str_s = y_s.iloc[:, :2]
y_numeric_s = y_s.iloc[:, 2:]

y_normalized_numeric_cs = y_numeric_cs.apply(normalize_row, axis=1)
y_normalized_numeric_db = y_numeric_db.apply(normalize_row, axis=1)
y_normalized_numeric_hm = y_numeric_hm.apply(normalize_row, axis=1)
y_normalized_numeric_tw = y_numeric_tw.apply(normalize_row, axis=1)
y_normalized_numeric_wn = y_numeric_wn.apply(normalize_row, axis=1)
y_normalized_numeric_ys = y_numeric_ys.apply(normalize_row, axis=1)
y_normalized_numeric_yt = y_numeric_yt.apply(normalize_row, axis=1)

y_normalized_numeric_s = y_numeric_s.apply(normalize_row, axis=1)

y_cs = pd.concat([y_str_cs, y_normalized_numeric_cs], axis=1)
y_db = pd.concat([y_str_db, y_normalized_numeric_db], axis=1)
y_hm = pd.concat([y_str_hm, y_normalized_numeric_hm], axis=1)
y_tw = pd.concat([y_str_tw, y_normalized_numeric_tw], axis=1)
y_wn = pd.concat([y_str_wn, y_normalized_numeric_wn], axis=1)
y_ys = pd.concat([y_str_ys, y_normalized_numeric_ys], axis=1)
y_yt = pd.concat([y_str_yt, y_normalized_numeric_yt], axis=1)

y_s = pd.concat([y_str_s, y_normalized_numeric_s], axis=1)

# for seed number, use 1, 2, 3, 4, 5
seed_num = 1
indices_cs = np.array(range(X_cs.shape[0]))
X_train_cs, X_test_cs, y_train_cs, y_test_cs, indices_train_cs, indices_test_cs = train_test_split(
    X_cs, y_cs, indices_cs, test_size=0.2, random_state=seed_num
)
X_train_cs, X_val_cs, y_train_cs, y_val_cs = train_test_split(
    X_train_cs, y_train_cs, test_size=0.25, random_state=seed_num
)
X_train_idx_cs = X_train_cs
#===============================
indices_db = np.array(range(X_db.shape[0]))
X_train_db, X_test_db, y_train_db, y_test_db, indices_train_db, indices_test_db = train_test_split(
    X_db, y_db, indices_db, test_size=0.2, random_state=seed_num
)
X_train_db, X_val_db, y_train_db, y_val_db = train_test_split(
    X_train_db, y_train_db, test_size=0.25, random_state=seed_num
)
X_train_idx_db = X_train_db
#===============================
indices_hm = np.array(range(X_hm.shape[0]))
X_train_hm, X_test_hm, y_train_hm, y_test_hm, indices_train_hm, indices_test_hm = train_test_split(
    X_hm, y_hm, indices_hm, test_size=0.2, random_state=seed_num
)
X_train_hm, X_val_hm, y_train_hm, y_val_hm = train_test_split(
    X_train_hm, y_train_hm, test_size=0.25, random_state=seed_num
)
X_train_idx_hm = X_train_hm
#===============================
indices_tw = np.array(range(X_tw.shape[0]))
X_train_tw, X_test_tw, y_train_tw, y_test_tw, indices_train_tw, indices_test_tw = train_test_split(
    X_tw, y_tw, indices_tw, test_size=0.2, random_state=seed_num
)
X_train_tw, X_val_tw, y_train_tw, y_val_tw = train_test_split(
    X_train_tw, y_train_tw, test_size=0.25, random_state=seed_num
)
X_train_idx_tw = X_train_tw
#===============================
indices_wn = np.array(range(X_wn.shape[0]))
X_train_wn, X_test_wn, y_train_wn, y_test_wn, indices_train_wn, indices_test_wn = train_test_split(
    X_wn, y_wn, indices_wn, test_size=0.2, random_state=seed_num
)
X_train_wn, X_val_wn, y_train_wn, y_val_wn = train_test_split(
    X_train_wn, y_train_wn, test_size=0.25, random_state=seed_num
)
X_train_idx_wn = X_train_wn
#===============================
indices_ys = np.array(range(X_ys.shape[0]))
X_train_ys, X_test_ys, y_train_ys, y_test_ys, indices_train_ys, indices_test_ys = train_test_split(
    X_ys, y_ys, indices_ys, test_size=0.2, random_state=seed_num
)
X_train_ys, X_val_ys, y_train_ys, y_val_ys = train_test_split(
    X_train_ys, y_train_ys, test_size=0.25, random_state=seed_num
)
X_train_idx_ys = X_train_ys
#===============================
indices_yt = np.array(range(X_yt.shape[0]))
X_train_yt, X_test_yt, y_train_yt, y_test_yt, indices_train_yt, indices_test_yt = train_test_split(
    X_yt, y_yt, indices_yt, test_size=0.2, random_state=seed_num
)
X_train_yt, X_val_yt, y_train_yt, y_val_yt = train_test_split(
    X_train_yt, y_train_yt, test_size=0.25, random_state=seed_num
)
X_train_idx_yt = X_train_yt


# When you use different query set from training set to test query set, you can change these lists.
X_train_list = [X_train_cs, X_train_db, X_train_hm, X_train_tw, X_train_wn, X_train_ys, X_train_yt]
X_test_list = [X_test_cs, X_test_db, X_test_hm, X_test_tw, X_test_wn, X_test_ys, X_test_yt]
X_val_list = [X_val_cs, X_val_db, X_val_hm, X_val_tw, X_val_wn, X_val_ys, X_val_yt]
y_train_list = [y_train_cs, y_train_db, y_train_hm, y_train_tw, y_train_wn, y_train_ys, y_train_yt]
y_test_list = [y_test_cs, y_test_db, y_test_hm, y_test_tw, y_test_wn, y_test_ys, y_test_yt]
y_val_list = [y_val_cs, y_val_db, y_val_hm, y_val_tw, y_val_wn, y_val_ys, y_val_yt]
X_train_idx = [X_train_idx_cs, X_train_idx_db, X_train_idx_hm, X_train_idx_tw, X_train_idx_wn, X_train_idx_ys, X_train_idx_yt]

X_train = pd.concat(X_train_list, ignore_index=True)
X_test = pd.concat(X_test_list, ignore_index=True)
X_val = pd.concat(X_val_list, ignore_index=True)
y_train = pd.concat(y_train_list, ignore_index=True)
y_test = pd.concat(y_test_list, ignore_index=True)
y_val = pd.concat(y_val_list, ignore_index=True)
X_train_idx = pd.concat(X_train_idx, ignore_index=True)

y_test_idx = y_test[['file_name', 'data']].assign(file_name=y_test['file_name'].str.strip())

y_test_idx = y_test_idx.merge(query_data_pairs.reset_index(), on=["file_name", "data"], how="left")

y_test_idx.rename(columns={"index": "idx"}, inplace=True)

X_train = X_train.drop(columns=["file_name", "data"], errors="ignore")
X_val = X_val.drop(columns=["file_name", "data"], errors="ignore")
X_test = X_test.drop(columns=["file_name", "data"], errors="ignore")
y_train = y_train.drop(columns=["file_name", "data"], errors="ignore")
y_val = y_val.drop(columns=["file_name", "data"], errors="ignore")
y_test = y_test.drop(columns=["file_name", "data"], errors="ignore")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

y_test_pos = y_test
y_train_pos = y_train
y_val_pos = y_val

y_train_pos_tensor = torch.tensor(y_train_pos.values, dtype=torch.float32).to(device)
y_val_pos_tensor = torch.tensor(y_val_pos.values, dtype=torch.float32).to(device)
y_test_pos_tensor = torch.tensor(y_test_pos.values, dtype=torch.float32).to(device)


def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate, seed=1):
        super(MLPModel, self).__init__()
        set_seed(seed) 

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)  

        
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  
        x = self.fc2(x)
        return x
    
input_size = X_train.shape[1]
output_size = len(attribute_combinations)

# Hyperparameters are decided based on Hyparameter tuning.
dropout_rate = 0.1
num_layer = 1
hidden_size = 128
learning_rate = 0.05
wd = 1e-5

model = MLPModel(input_size, hidden_size, output_size, dropout_rate)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
criterion = nn.CrossEntropyLoss(reduction='none')

early_stopping_patience = 20
best_val_loss = float('inf')
epochs_without_improvement = 0

train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []


torch.set_num_threads(1)

epochs = 10000
epoch_losses = []  
alpha = 1.0

start = time.process_time()
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    output = model(X_train_tensor)

    loss_posi = criterion(output, y_train_pos_tensor)
    loss_posi.mean()
    loss = loss_posi
    loss = loss.mean()
    train_losses.append(loss.item())

    predicted_classes = torch.argmax(output, dim=1)
    topk_classes = torch.topk(y_train_pos_tensor, int(5), dim=1)[1]
    train_accuracy = torch.any(predicted_classes.unsqueeze(1) == topk_classes, dim=1).cpu().float().mean()
    train_accuracies.append(train_accuracy)

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_output = model(X_val_tensor)
        pos_val_loss = criterion(val_output, y_val_pos_tensor)
        pos_val_loss = pos_val_loss.mean()
        val_loss = pos_val_loss
        val_loss = val_loss.mean()
        valid_losses.append(val_loss.item())

        predicted_classes = torch.argmax(val_output, dim=1)
        topk_classes = torch.topk(y_val_pos_tensor, int(5), dim=1)[1]
        valid_accuracy = torch.any(predicted_classes.unsqueeze(1) == topk_classes, dim=1).cpu().float().mean()
        valid_accuracies.append(valid_accuracy)

    log_line = (f"Epoch [{epoch+1}/{epochs}], "
                f"Train Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy.item():.4f}, "
                f"Val Loss: {val_loss.item():.4f}, Val Accuracy: {valid_accuracy.item():.4f}")
    print(log_line)
    print("-------------------", valid_accuracy)

    max_valid_acc = valid_accuracy

    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= early_stopping_patience:
        stop_line = "Early stopping triggered. Max Validation Accuracy: {:.4f}".format(max_valid_acc.item())
        print(stop_line)
        break

end = time.process_time()
training_time = end - start
print(f"Training time: {training_time:.2f} seconds")



torch.set_num_threads(1)
model.eval()
with torch.no_grad():
    start_time = time.process_time()

    test_output = model(X_test_tensor)
    test_predictions = torch.argmax(test_output, dim=1)

    end_time = time.process_time()
    prediction_time = end_time - start_time

    print('pred timeb is ', prediction_time)


pred_labels = [attribute_combinations[label+1] for label in test_predictions.cpu().numpy()]

test_results = pd.DataFrame({
    'idx': y_test_idx["idx"].values,  
    'file_name': query_data_pairs.loc[y_test_idx["idx"], 'file_name'].values, 
    'data': query_data_pairs.loc[y_test_idx["idx"], 'data'].values,  
    'filter': [pred_labels[i][0] for i in range(len(pred_labels))],  
    'order': [pred_labels[i][1] for i in range(len(pred_labels))],  
    'enumerate': [pred_labels[i][2] for i in range(len(pred_labels))], 
    'prediction_time': round(prediction_time, 5),  
})

for index, row in test_results.iterrows(): 
    idx = row["idx"]  
    flag = 0
    for i in range(60):
        combination = (row["filter"], row["order"], row["enumerate"])
        if combination == (data.at[idx, "filter" + str(i+1)],
                           data.at[idx, "order" + str(i+1)],
                           data.at[idx, "enumerate" + str(i+1)]):
            test_results.at[index, "time_feat"] = (
                data.at[idx, "time_sec"]
            )
            test_results.at[index, "eps"] = data.at[idx, "eps" + str(i+1)]
            test_results.at[index, "emb"] = data.at[idx, "emb" + str(i+1)]
            test_results.at[index, "pred_eps"] = (
                data.at[idx, "emb" + str(i+1)] /
                (data.at[idx, "time" + str(i+1)] + row["prediction_time"] +
                 + data.at[idx, "time_sec"])
            )
            test_results.at[index, "feat_time"] = data.at[idx, "time_sec"]
            test_results.at[index, "filter1"] = data.at[idx, "filter1"]
            test_results.at[index, "order1"] = data.at[idx, "order1"]
            test_results.at[index, "enumerate1"] = data.at[idx, "enumerate1"]

            test_results.at[index, "filter2"] = data.at[idx, "filter2"]
            test_results.at[index, "order2"] = data.at[idx, "order2"]
            test_results.at[index, "enumerate2"] = data.at[idx, "enumerate2"]

            test_results.at[index, "filter3"] = data.at[idx, "filter3"]
            test_results.at[index, "order3"] = data.at[idx, "order3"]
            test_results.at[index, "enumerate3"] = data.at[idx, "enumerate3"]

            test_results.at[index, "filter4"] = data.at[idx, "filter4"]
            test_results.at[index, "order4"] = data.at[idx, "order4"]
            test_results.at[index, "enumerate4"] = data.at[idx, "enumerate4"]

            test_results.at[index, "filter5"] = data.at[idx, "filter5"]
            test_results.at[index, "order5"] = data.at[idx, "order5"]
            test_results.at[index, "enumerate5"] = data.at[idx, "enumerate5"]

            test_results.at[index, "filter6"] = data.at[idx, "filter6"]
            test_results.at[index, "order6"] = data.at[idx, "order6"]
            test_results.at[index, "enumerate6"] = data.at[idx, "enumerate6"]

            test_results.at[index, "filter7"] = data.at[idx, "filter7"]
            test_results.at[index, "order7"] = data.at[idx, "order7"]
            test_results.at[index, "enumerate7"] = data.at[idx, "enumerate7"]

            test_results.at[index, "filter8"] = data.at[idx, "filter8"]
            test_results.at[index, "order8"] = data.at[idx, "order8"]
            test_results.at[index, "enumerate8"] = data.at[idx, "enumerate8"]

            test_results.at[index, "filter9"] = data.at[idx, "filter9"]
            test_results.at[index, "order9"] = data.at[idx, "order9"]
            test_results.at[index, "enumerate9"] = data.at[idx, "enumerate9"]

            test_results.at[index, "filter10"] = data.at[idx, "filter10"]
            test_results.at[index, "order10"] = data.at[idx, "order10"]
            test_results.at[index, "enumerate10"] = data.at[idx, "enumerate10"]

            test_results.at[index, "filter15"] = data.at[idx, "filter15"]
            test_results.at[index, "order15"] = data.at[idx, "order15"]
            test_results.at[index, "enumerate15"] = data.at[idx, "enumerate15"]
            flag = 1

test_results.to_csv(f'topx_{seed_num}.csv', index=False, encoding='utf-8')

train_source = pd.merge(
    data,
    X_train_idx,
    on=["file_name", "data"]
)

train_source.to_csv(f'train_{seed_num}.csv', index=False, encoding='utf-8')

