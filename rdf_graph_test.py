from rdflib import Graph, URIRef
from rdflib.util import guess_format
from collections import deque

def get_local_name(uri):
    if isinstance(uri, URIRef):
        return uri.split('/')[-1].split('#')[-1]
    return str(uri)

# Load RDF Graph
g = Graph()
g.parse("/mnt/newdrive/data_files/SpaTex/CSZt.nt", format=guess_format("/mnt/newdrive/data_files/SpaTex/CSZt.nt"))
print("Created graph!")

entity1 = URIRef("http://spatex.org/Cuming_County_Nebraska")
# entity2 = URIRef("http://spatex.org/Colfax_County_Nebraska") # 1
entity2 = URIRef("http://spatex.org/The_State_of_Missouri")    # 2
# entity2 = URIRef("http://spatex.org/Shelby_County_Iowa")    # 3
# entity2 = URIRef("http://spatex.org/Kane_County_Illinois")    # 13

def find_path_bfs_uri(graph, start, end):
    visited = set()
    queue = deque()
    
    # Each item: (current_node, path_so_far)
    queue.append((start, []))
    visited.add(start)
    
    while queue:
        current, path = queue.popleft()
        
        if current == end:
            return path

        # Expand outgoing edges
        for _, p, o in graph.triples((current, None, None)):
            if isinstance(o, URIRef) and o not in visited:
                visited.add(o)
                queue.append((o, path + [(current, p, o)]))
    
    return None

# Run search
path = find_path_bfs_uri(g, entity1, entity2)

# Print result
if path:
    print(f"Path found ({len(path)}):")
    for s, p, o in path:
        print(f"{get_local_name(s)} --[{get_local_name(p)}]--> {get_local_name(o)}")
else:
    print("No path found between the entities.")
