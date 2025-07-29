import networkx as nx
import math
import numpy as np
import time
import os
import csv
import subprocess
import re
import sys


def read_graph(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':  
                vertex_id = int(parts[1])
                G.add_node(vertex_id)
            elif parts[0] == 'e':  
                source, target = int(parts[1]), int(parts[2])
                G.add_edge(source, target)
            elif parts[0] == 't':
                vertex_num = int(parts[1])
                edge_num = int(parts[2])
    return G, vertex_num, edge_num

def read_labels(file_path):
    label_counts = {}
    total_nodes = 0
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':  
                vertex_id = int(parts[1])
                label = int(parts[2])
                total_nodes += 1
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1
    label_ratio = {label: count / total_nodes for label, count in label_counts.items()}
    return label_ratio

def calculate_label_ratio_product(data_ratio, query_ratio):
    total = 0
    for label in data_ratio.keys():
        if label in query_ratio:
            total += data_ratio[label] * query_ratio[label]
    return total


def label_num(file_path):
    max_label_id = 0
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v' and len(parts) >= 3:
                label_id = parts[2]
                label_id = int(label_id)
                if label_id >= max_label_id:
                    max_label_id = label_id
    return max_label_id

#------------------------------------------------------------
def generate_args(binary, *params):
    arguments = [binary]
    arguments.extend(list(params))
    return arguments

def execute_binary(args, timeout=100000):
    try:
        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        rc = result.returncode
        std_output = result.stdout
        std_error = result.stderr
    except subprocess.TimeoutExpired:
        rc = -1000
        std_output = b''
        std_error = b'Timeout'

    return rc, std_output, std_error

def extract_values(output):
    output_str = output.decode('utf-8')
    
    fil_time = re.search(r'Filtering time \(seconds\): ([0-9]*\.[0-9]+)', output_str)
    candidate_size = re.search(r'candidate size: ([0-9]*\.[0-9]+)', output_str)
    
    fil_time = float(fil_time.group(1)) if fil_time else None
    candidate_size = float(candidate_size.group(1)) if candidate_size else None

    return fil_time, candidate_size

#------------------------------------------------------------


data = "DBLP.graph"
data_label_ratio = read_labels(data)

filter_type = 'LDF'

target_dirs = ['query_s4', 'query_s8', 'query_s16', 'query_s32', 'query_s64', 'query_s96', 'query_s128']

output_file = 'features/query_features.csv'

input_binary_path = sys.argv[1]
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
input_data_graph_path = '{0}/data_graph/DBLP.graph'.format(dir_path)

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'data', 'file_name', 'time_sec', 
        'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8',
        'feature9', 'feature10', 'feature11', 'feature12', 'feature13'
    ])

    for target_dir in target_dirs:
        if not os.path.exists(target_dir):
            continue
        
        for file_name in os.listdir(target_dir):
            if file_name.endswith('.graph') and file_name != "DBLP.graph":
                query = os.path.join(target_dir, file_name)
                
                start_time = time.time()
                
                
                G, vertex, edge = read_graph(query)
                density = 2*edge/vertex
                average_degree = sum(dict(G.degree()).values()) / len(G)
                
                query_label_ratio = read_labels(query)
                freq_label = calculate_label_ratio_product(data_label_ratio, query_label_ratio)

                tree_width, _ = nx.approximation.treewidth_min_fill_in(G)
                diameter = nx.diameter(G)
                core_num = nx.core_number(G)
                max_core_num = max(core_num.values())
                # poi_ent =time.time()
                # print(f"poi_ent: {poi_ent - start_time:.2f} seconds")
                #poi_t = time.time()

                #G_d, vertex_d, edge_d = read_graph(data)
                #core_num_d = nx.core_number(G_d)
                #max_core_num_d = max(core_num_d.values())
                #density_d = 2*edge_d/vertex_d
                #label_d = label_num(data)
                # poi_ent =time.time()
                # print(f"poi_ent: {poi_ent - start_time:.2f} seconds")
                
                end_time = time.time()
                execution_time = end_time - start_time

                query_graph_path = os.path.join(dir_path, 'query_graph_DBLP_s', target_dir, file_name)

                execution_args = generate_args(input_binary_path, '-d', input_data_graph_path, '-q', query_graph_path, '-filter', filter_type)

                rc, std_output, std_error = execute_binary(execution_args)

                fil_time, candidate_size = extract_values(std_output)

                total_time = fil_time + execution_time
                print(f"File: {fil_time}, Total Time: {total_time:.2f} seconds, Candidate Size: {candidate_size}")

                writer.writerow([
                    data, file_name, total_time, average_degree, density, vertex, edge,
                    freq_label, tree_width, diameter, max_core_num, 317080, 6.6, 113, 14, candidate_size
                ])