from core.base_index import BaseIndex
from core.base_cg import BaseCG
from index.faiss_index import FAISSSubIndex

from utils import INDEX_TYPE
import rdf_utils

from rdflib.util import guess_format
from rdflib import Graph, URIRef
from collections import deque

class RDFGraphIndex(BaseIndex):
    def __init__(self, data_path, cg_module: BaseCG):
        # set properties
        super().__init__(data_path, INDEX_TYPE.RDF_GRAPH)
        # build the graph
        self.graph = Graph()
        self.graph.parse(data_path, format=guess_format(data_path))
        # set the node label index
        self.node_index = FAISSSubIndex(data_input=self.graph)
        # set the context generator
        self.cg_module = cg_module

    def printPath(self, paths):
        # Print result
        if paths:
            print(f"Path found (length {len(paths)}):")
            for s, p, o in paths:
                print(f"{rdf_utils.get_local_name(s)} --[{rdf_utils.get_local_name(p)}]--> {rdf_utils.get_local_name(o)}")
            print("\n")
        else:
            print("Empty path.")

    def find_path_bfs_uri(self, startLabel:str , endLabel: str) -> deque:
        queue = deque()
        visited = set()

        # Each item in queue: (current_node, path_so_far)
        queue.append((startLabel, []))

        while queue:
            current, path = queue.popleft()

            if current in visited:
                continue
            visited.add(current)

            if current == endLabel:
                return path

            for _, p, o in self.graph.triples((current, None, None)):
                if isinstance(o, URIRef):
                    queue.append((o, path + [(current, p, o)]))

        return None

    def _get_paths(self, pairs) -> "list[(str,str)]":
        '''
        returns all shortest paths between the pairs of entities in the list
        '''
        all_paths = []
        for start, end in pairs:
            path = self.find_path_bfs_uri(start, end)
            if path:
                all_paths.append(path)
            else:
                # no paths, try loking in reverse
                path = self.find_path_bfs_uri(end, start)
                if path:
                    all_paths.append(path)
        return all_paths

    def retrieveK(self, text: str, k: int):
        raise NotImplementedError("RDFGraphIndex uses graph traversal, not embedding retrieval. Use FAISSIndex for retrieveK.")

    def generateContext(self, referenced_entities):
        # find start/end points in graph
        pairs = self.node_index.find_start_end_pairs(referenced_entities)

        # get the shortest paths between the pairs
        paths = self._get_paths(pairs)

        # generate context from paths
        return self.cg_module.generateContext(paths)
