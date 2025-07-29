import random
import os


input_file = 'data_graph/DBLP.graph'
output_dir = 'query_s/query_DBLP'
num_queries = 200
vertex_num = 96
max_attempts = 1000 

os.makedirs(output_dir, exist_ok=True)

def load_graph(file_path):
    vertices = {}
    edges = set() 
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertex_id = int(parts[1])
                vertex_label = int(parts[2])
                vertices[vertex_id] = {'label': vertex_label, 'edges': []}
            elif parts[0] == 'e':
                source, target = sorted([int(parts[1]), int(parts[2])])  
                if (source, target) not in edges:  
                    edges.add((source, target))
                    vertices[source]['edges'].append(target)
                    vertices[target]['edges'].append(source)
    
    return vertices, list(edges) 


def random_walk(vertices, vertex_id, length):
    for attempt in range(max_attempts):
        start_node = random.choice(vertex_id)
        walk = [start_node]
        print(start_node)
        checkmate = []
        current_id = -1
        query_edges = []
        
        while len(walk) < length:
            current = walk[current_id]
            neighbors = vertices[current]['edges']
            filtered_ne = [n for n in neighbors if n not in walk]
            unwalked_ne = [n for n in filtered_ne if n not in checkmate]
            if unwalked_ne:
              if len(walk) + len(unwalked_ne) > length:
                for add in range(length - len(walk)):
                  walk.append(unwalked_ne[add])
                  
                  query_edges.append((current, unwalked_ne[add]))
              else:
                for ne in range(len(unwalked_ne)):
                  walk.append(unwalked_ne[ne])
                  
                  query_edges.append((current, unwalked_ne[ne]))
                current_id = -1
                
            else:
                checkmate.append(walk[-1])
                if len(walk) == 1:
                  current_id = -1
                else:
                  current_id = current_id - 1
                if walk[current_id] == start_node:
                    break
        if len(walk) == length:
          vertex_mapping = {orig_id: i for i, orig_id in enumerate(walk)}
          query_vertices = [(i, vertices[orig_id]['label']) for i, orig_id in enumerate(walk)]
          query_edges_mapped = [(vertex_mapping[src], vertex_mapping[dst]) for src, dst in query_edges]
          print("CLEAR!")
          return walk, query_vertices, query_edges_mapped
    raise ValueError(f"Random walk failed to find a valid path after {max_attempts} attempts.")

def create_query_graph(vertices, walk, query_id, query_vertices, query_edges):
    with open(f'{output_dir}/query_s{vertex_num}/query_{vertex_num}_s{query_id}.graph', 'w') as f:
        f.write(f't {len(query_vertices)} {len(query_edges)}\n')
        for vid, label in query_vertices:
            degree = len([e for e in query_edges if e[0] == vid or e[1] == vid])
            f.write(f'v {vid} {label} {degree}\n')
        for src, dst in query_edges:
            f.write(f'e {src} {dst}\n')

vertices, edges = load_graph(input_file)
vertex_ids = list(vertices.keys())
for i in range(num_queries):
  try:
    walk, query_vertices, query_edges = random_walk(vertices, vertex_ids, vertex_num)
    create_query_graph(vertices, walk, i, query_vertices, query_edges)
  except ValueError as e:
    print(e)
    print(f"Failed to generate query graph {i}. Skipping this query.")