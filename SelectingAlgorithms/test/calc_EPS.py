import sys
import os
import glob
import re
import subprocess
import pandas as pd



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
    
    time_match = re.search(r'Total time \(seconds\): ([0-9]*\.[0-9]+)', output_str)
    embeddings_match = re.search(r'#Embeddings: (\d+)', output_str)
    eps_match = re.search(r'EPS : ([0-9]*\.[0-9]+)', output_str)

    time = float(time_match.group(1)) if time_match else None
    embeddings = int(embeddings_match.group(1)) if embeddings_match else None
    eps = float(eps_match.group(1)) if eps_match else None

    return time, embeddings, eps


filter_type_list = ['LDF', 'NLF', 'DPiso', 'VEQ', 'PL']
order_type_list = ['GQL', 'RI', 'RM']
engine_type_list = ['VEQ', 'KSS', 'EXPLORE', 'LFTJ']

dir_path = os.path.dirname(os.path.realpath(__file__))
query_folder_path = '{0}/query_graph_DBLP/query_128'.format(dir_path) 
query_graph_path_list = glob.glob('{0}/*.graph'.format(query_folder_path))
query_graph_path_list = sorted(query_graph_path_list)

print(query_folder_path)

input_data_graph_path = '{0}/data_graph/DBLP.graph'.format(dir_path)
all_execution_data = []  

if __name__ == '__main__':
    input_binary_path = sys.argv[1]
    if not os.path.isfile(input_binary_path):
        print('The binary {0} does not exist.'.format(input_binary_path))
        exit(-1)

    output_dir = os.path.join(dir_path, 'EPS/DBLP/query_128')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for query_graph_path in query_graph_path_list:
        query_graph_filename = os.path.basename(query_graph_path)

        execution_times_with_plans = []

        for filter_type in filter_type_list:
            for order_type in order_type_list:
                for engine_type in engine_type_list:
                    print(filter_type, order_type, engine_type)

                    execution_args = generate_args(input_binary_path, '-d', input_data_graph_path, '-q', query_graph_path, '-filter', filter_type, '-order', order_type, '-engine', engine_type)
                    
                    
                    rc, std_output, std_error = execute_binary(execution_args, timeout=100000)
                    if rc == 0:
                        time, embeddings, eps = extract_values(std_output)
                    elif rc == -1000:
                        time, embeddings, eps = 700, 0, 0 
                    else:
                        time, embeddings, eps = 1000, 0, 0

                    execution_times_with_plans.append((time, embeddings, eps, filter_type, order_type, engine_type))
        
        execution_times_with_plans.sort(key=lambda x: x[2], reverse=True)  
        row_data = ['DBLP.graph', query_graph_filename]
       
        for time, embeddings, eps, filter_type, order_type, engine_type in execution_times_with_plans:
            row_data.extend([eps, embeddings, time, filter_type, order_type, engine_type])

        all_execution_data.append(row_data)
        print(row_data)
    
    csv_output_path = os.path.join(output_dir, 'EPS_DBLP_128.csv')  
    df = pd.DataFrame(all_execution_data)
    df.to_csv(csv_output_path, index=False, header=False) 
    print(f"All execution times saved to {csv_output_path}")
