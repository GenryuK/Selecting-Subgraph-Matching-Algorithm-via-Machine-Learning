import random
import os


input_file = 'data_graph/DBLP.graph'
output_dir = 'query_graph_DBLP'
num_queries = 200
vertex_num = 128
max_attempts = 1000 


os.makedirs(output_dir, exist_ok=True)

'''
# vertices
{label, edges[num_edge]}
ex. {'label': 2, 'edges': [1319, 1091, 1108]}
'''
def load_graph(file_path):
    vertices = {}
    edges = []
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
                source = int(parts[1])
                target = int(parts[2])
                vertices[source]['edges'].append(target)
                vertices[target]['edges'].append(source)
                edges.append((source, target))
    return vertices, edges

def random_walk(vertices, vertex_id, length):
    for attempt in range(max_attempts):
        start_node = random.choice(vertex_id)
        walk = [start_node]
        checkmate = []
        current_id = -1
        while len(walk) < length:
            current = walk[current_id]
            neighbors = vertices[current]['edges']
            filtered_ne = [n for n in neighbors if n not in walk]
            unwalked_ne = [n for n in filtered_ne if n not in checkmate]
            if unwalked_ne:
                next_vertex = random.choice(unwalked_ne)
                walk.append(next_vertex)
                current_id = -1
            else:
                checkmate.append(walk[-1])
                current_id = current_id - 1
                if walk[current_id] == start_node:
                    break
        if len(walk) == length:
            print("CLEAR!")
            return walk
    raise ValueError(f"Random walk failed to find a valid path after {max_attempts} attempts.")

def create_query_graph(vertices, walk, query_id):
    vertex_mapping = {orig_id: i for i, orig_id in enumerate(walk)}
    query_vertices = [(i, vertices[orig_id]['label']) for i, orig_id in enumerate(walk)]
    query_edges = [
        (vertex_mapping[src], vertex_mapping[dst])
        for src in walk
        for dst in vertices[src]['edges']
        if dst in walk and vertex_mapping[src] < vertex_mapping[dst]
    ]
    with open(f'{output_dir}/query_{vertex_num}/query_{vertex_num}_{query_id}.graph', 'w') as f:
        f.write(f't {len(query_vertices)} {len(query_edges)}\n')
        for vid, label in query_vertices:
            degree = len([e for e in query_edges if e[0] == vid or e[1] == vid])
            f.write(f'v {vid} {label} {degree}\n')
        for src, dst in query_edges:
            f.write(f'e {src} {dst}\n')


def main():
    vertices, edges = load_graph(input_file)
    vertex_ids = list(vertices.keys())
    for i in range(num_queries):
        # start_vertex = random.choice(vertex_ids)
        try:
            walk = random_walk(vertices, vertex_ids, vertex_num)
            create_query_graph(vertices, walk, i)
        except ValueError as e:
            print(e)
            print(f"Failed to generate query graph {i}. Skipping this query.")

if __name__ == '__main__':
    main()

