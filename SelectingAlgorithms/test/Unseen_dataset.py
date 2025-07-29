import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import os
from operator import neg

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

data = pd.read_csv('All_data.csv')
data.columns = [''.join(col) if isinstance(col, tuple) else col for col in data.columns]
column_to_modify = [f"eps{i+1}" for i in range(60)]
data.loc[:, column_to_modify] = data[column_to_modify].where(data[column_to_modify] >= 0, 0)

data = data[data["eps1"] != 0.0].reset_index(drop=True)

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


def inc_x_dataset(data, permitted_increase_rate):
    for i in range(len(attribute_combinations)):
        data["output_feature" + str(i+1)] = 0

    for idx, row in data.iterrows():
        best_eps = row["eps1"]
        for i in range(60):
            if row["eps" + str(i+1)] >= best_eps * permitted_increase_rate:
                combination = (row["filter" + str(i+1)], row["order" + str(i+1)], row["enumerate" + str(i+1)])
                data.at[idx, "output_feature" + str(attribute_combinations_inv[combination])] = 1

    X = data[[f"feature{i+1}" for i in range(13)]]#feature number
    y = data[['data']+['file_name']+[f"output_feature{i+1}" for i in range(len(attribute_combinations))]]

    return X, y


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

task = "topx"
num_attributes = 1

def get_data(task, num):
    
    if task == "topx":
        X, y = top_n_dataset(data, num)
    elif task == "incy":
        X, y = inc_x_dataset(data, num)

    elif task == "wei":
        X, y = time_weighted_dataset(data)
    return X, y

X, y = get_data(task, num_attributes)

X = data[['data', 'file_name', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13']]

def normalize_row(row):
    row_sum = row.sum() 
    if row_sum == 0:
        row_sum = 1e-10 
    return row / row_sum  


y_str = y.iloc[:, :2] 
y_numeric = y.iloc[:, 2:]

y_normalized_numeric = y_numeric.apply(normalize_row, axis=1)

y = pd.concat([y_str, y_normalized_numeric], axis=1)

com_y = y

seed_num = 1
test_mask = X["data"].str.startswith("DBLP")
X_test = X[test_mask]
y_test = com_y[test_mask]

X_remain = X[~test_mask]
y_remain = com_y[~test_mask]

X_train, X_val, y_train, y_val = train_test_split(
    X_remain, y_remain, test_size=0.2, random_state=seed_num
)

X_train_idx = X_train
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

class MLPModel1Layer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate, seed=1):
        super(MLPModel1Layer, self).__init__()
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
    
uni1 = 32
dp = 0.4
wd = 1e-4
lr = 0.05
al = 1
input_size = X_train.shape[1]
output_size = len(attribute_combinations)
model = MLPModel1Layer(input_size, uni1, output_size, dp) 

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
criterion = nn.CrossEntropyLoss(reduction='none')

early_stopping_patience = 20
best_val_loss = float('inf')
patience_counter = 0
epochs = 10000

train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []
epoch_losses = [] 

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

test_results.to_csv(f'db{seed_num}.csv', index=False, encoding='utf-8')

train_source = pd.merge(
    data,
    X_train_idx,
    on=["file_name", "data"]
)

train_source.to_csv(f'training_DBLP{seed_num}.csv', index=False, encoding='utf-8')