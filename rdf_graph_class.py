from rdflib.util import guess_format
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, SKOS
from sentence_transformers import SentenceTransformer
import faiss
from collections import deque


composition_table_8dir = {
    ('north', 'north'): 'north',
    ('north', 'northeast'): 'northeast',
    ('north', 'east'): 'northeast',
    ('north', 'southeast'): 'east',
    ('north', 'south'): 'north',
    ('north', 'southwest'): 'west',
    ('north', 'west'): 'northwest',
    ('north', 'northwest'): 'northwest',

    ('northeast', 'north'): 'northeast',
    ('northeast', 'northeast'): 'northeast',
    ('northeast', 'east'): 'east',
    ('northeast', 'southeast'): 'southeast',
    ('northeast', 'south'): 'east',
    ('northeast', 'southwest'): 'north',
    ('northeast', 'west'): 'north',
    ('northeast', 'northwest'): 'north',

    ('east', 'north'): 'northeast',
    ('east', 'northeast'): 'east',
    ('east', 'east'): 'east',
    ('east', 'southeast'): 'southeast',
    ('east', 'south'): 'southeast',
    ('east', 'southwest'): 'southeast',
    ('east', 'west'): 'north',
    ('east', 'northwest'): 'north',

    ('southeast', 'north'): 'east',
    ('southeast', 'northeast'): 'southeast',
    ('southeast', 'east'): 'southeast',
    ('southeast', 'southeast'): 'southeast',
    ('southeast', 'south'): 'south',
    ('southeast', 'southwest'): 'south',
    ('southeast', 'west'): 'east',
    ('southeast', 'northwest'): 'east',

    ('south', 'north'): 'south',
    ('south', 'northeast'): 'east',
    ('south', 'east'): 'southeast',
    ('south', 'southeast'): 'south',
    ('south', 'south'): 'south',
    ('south', 'southwest'): 'south',
    ('south', 'west'): 'southwest',
    ('south', 'northwest'): 'west',

    ('southwest', 'north'): 'west',
    ('southwest', 'northeast'): 'south',
    ('southwest', 'east'): 'south',
    ('southwest', 'southeast'): 'south',
    ('southwest', 'south'): 'southwest',
    ('southwest', 'southwest'): 'southwest',
    ('southwest', 'west'): 'west',
    ('southwest', 'northwest'): 'west',

    ('west', 'north'): 'northwest',
    ('west', 'northeast'): 'north',
    ('west', 'east'): 'north',
    ('west', 'southeast'): 'east',
    ('west', 'south'): 'southwest',
    ('west', 'southwest'): 'west',
    ('west', 'west'): 'west',
    ('west', 'northwest'): 'northwest',

    ('northwest', 'north'): 'northwest',
    ('northwest', 'northeast'): 'north',
    ('northwest', 'east'): 'north',
    ('northwest', 'southeast'): 'east',
    ('northwest', 'south'): 'west',
    ('northwest', 'southwest'): 'west',
    ('northwest', 'west'): 'northwest',
    ('northwest', 'northwest'): 'northwest',
}

INVERSE_RELATION = {
    "north":"south",
    "south":"north",
    "east":"west",
    "west":"east",
    "northeast":"southwest",
    "southwest":"northeast",
    "southeast":"northwest",
    "northwest":"southeast",
    
    "adjacent to and north":"adjacent to and south",
    "adjacent to and south":"adjacent to and north",
    "adjacent to and east":"adjacent to and west",
    "adjacent to and west":"adjacent to and east",
    "adjacent to and northeast":"adjacent to and southwest",
    "adjacent to and southwest":"adjacent to and northeast",
    "adjacent to and southeast":"adjacent to and northwest",
    "adjacent to and northwest":"adjacent to and southeast",
    
    "inside":"contains",
    "contains":"inside",
}

class RDFGraphIndexer:
    def __init__(self, graph_path, model_name="all-MiniLM-L6-v2"):
        self.graph = Graph()
        self.graph.parse(graph_path, format=guess_format(graph_path))
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.uri_to_text = {}
        self.uris = []
        
    def _get_text_for_uri(self, uri):
        """Extract text from URI itself if no labels exist (for .nt files)."""
        texts = []
        # Try standard labels first
        for label_prop in [RDFS.label, SKOS.prefLabel]:
            for label in self.graph.objects(uri, label_prop):
                if isinstance(label, Literal):
                    texts.append(str(label))
        
        # Fallback: Use the URI's local name (last path segment)
        if not texts and isinstance(uri, URIRef):
            uri_str = str(uri)
            texts.append(uri_str.split("/")[-1].split("#")[-1])  # Get '66423' from '.../66423'
        
        return " ".join(texts) if texts else None  # Return None if no text at all

    def getInverseRelation(self, relation):
        if relation in  INVERSE_RELATION:
            return INVERSE_RELATION[relation]
        print(f"Inverse relation unknown for relation: {relation}")
        return "<unknown_relation>"

    def build_index(self):
        """Index all URIs with their textual representations."""
        self.uris = []
        texts = []
        
        # Include both subjects and objects that are URIs
        nodes = set(self.graph.subjects()).union(
            {o for o in self.graph.objects() if isinstance(o, URIRef)}
        )
        
        for uri in nodes:
            text = self._get_text_for_uri(uri)
            if text:  # Only index if text exists
                self.uri_to_text[uri] = text
                self.uris.append(uri)
                texts.append(text)

        if not texts:
            raise ValueError("No indexable entities found. Check your RDF data.")
        
        # Compute embeddings
        embeddings = self.model.encode(texts, convert_to_tensor=True).cpu().numpy()
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        print(f"Index built with {len(self.uris)} entities")


    def query_entities(self, query_text, top_k=5, min_score=0.7):
        """Find matching entities with score thresholding and exact match fallback."""
        # First try exact string matching (critical for zipcodes)
        for uri, text in self.uri_to_text.items():
            if query_text.lower() in text.lower():
                return [{
                    "uri": uri,
                    "text": text,
                    "score": 1.0
                }]
        
        # Fallback to semantic search
        query_embedding = self.model.encode(query_text, convert_to_tensor=True)
        query_embedding = query_embedding.cpu().numpy()
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if score >= min_score:  # Only keep good matches
                uri = self.uris[idx]
                results.append({
                    "uri": uri,
                    "text": self.uri_to_text[uri],
                    "score": float(score)
                })
        
        return results
        
    # def find_start_end_pairs(self, ner_entities_dict):
    #     start_end_pairs = []
    #     grounded_entities = []
        
    #     # Ground each entity while preserving original order
    #     for entity_type in ["zipcode", "county", "state"]:  # maintain this order for processing
    #         if entity_type in ner_entities_dict:
    #             for entity in ner_entities_dict[entity_type]:
    #                 matches = self.query_entities(entity, top_k=1)
    #                 if matches:
    #                     grounded_entities.append({
    #                         "uri": matches[0]["uri"],
    #                         "type": entity_type,
    #                         "original_text": entity
    #                     })
        
    #     # # pair ALL with ALL
    #     # for i in range(len(grounded_entities)):
    #     #     for j in range(i + 1, len(grounded_entities)):
    #     #         start_end_pairs.append((
    #     #             grounded_entities[i]["uri"], 
    #     #             grounded_entities[j]["uri"]
    #     #         ))
    #     # Only pair FIRST entity with all others
    #     if len(grounded_entities) > 1:
    #         first_uri = grounded_entities[0]["uri"]
    #         for other_entity in grounded_entities[1:]:
    #             start_end_pairs.append((first_uri, other_entity["uri"]))
        
    #     return start_end_pairs

    def find_start_end_pairs(self, ner_entities_dict):
        start_end_pairs = []
        grounded_entities = []
        
        # First try to use ordered entities if available
        if 'ordered' in ner_entities_dict:
            # Process entities in their original order
            for entity in ner_entities_dict['ordered']:
                matches = self.query_entities(entity['text'], top_k=1)
                if matches:
                    grounded_entities.append({
                        "uri": matches[0]["uri"],
                        "type": entity['type'],
                        "original_text": entity['text']
                    })
        else:
            # Fallback to the old dictionary approach (for backward compatibility)
            for entity_type in ["zipcode", "county", "state"]:  
                if entity_type in ner_entities_dict:
                    for entity in ner_entities_dict[entity_type]:
                        matches = self.query_entities(entity, top_k=1)
                        if matches:
                            grounded_entities.append({
                                "uri": matches[0]["uri"],
                                "type": entity_type,
                                "original_text": entity
                            })
        
        # Only pair FIRST entity with all others (in original order)
        if len(grounded_entities) > 1:
            first_uri = grounded_entities[0]["uri"]
            for other_entity in grounded_entities[1:]:
                start_end_pairs.append((first_uri, other_entity["uri"]))
        
        return start_end_pairs
        
    def get_local_name(self, uri):
        if isinstance(uri, URIRef):
            return uri.split('/')[-1].split('#')[-1].replace("_"," ")
        return str(uri)
    
    def printPath(self, paths):
        # Print result
        if paths:
            print(f"Path found (length {len(paths)}):")
            for s, p, o in paths:
                print(f"{self.get_local_name(s)} --[{self.get_local_name(p)}]--> {self.get_local_name(o)}")
            print("\n")
        else:
            print("Empty path.")

    def getPathsAsContext(self, paths):
        path_string = ""
        counter = 1
        for path in paths:
            for s, p, o in path:
                path_string += f"'{counter}: {self.get_local_name(s)} {self.get_local_name(p)} {self.get_local_name(o)}'\n"
                counter += 1
        return path_string
    
    def getSinglePathAsContext(self, path):
        path_string = ""
        counter = 1
        for s, p, o in path:
            path_string += f"'{counter}: {self.get_local_name(s)} {self.get_local_name(p)} {self.get_local_name(o)}'\n"
            counter += 1
        return path_string
    
    def getCombinedRelation(self, path):
        if not path:
            return ""
        if len(path) > 1:
            # adjacency is usually obsolete after more than 1 hops
            clean_name = self.get_local_name(path[0][1])
            if "adjacent to and " in clean_name:
                clean_name = clean_name.replace("adjacent to and ", "")
            current_relation = clean_name
                    
            for s, p, o in path[1:]:
                # print(f"Current: {current_relation}")
                clean_name = self.get_local_name(p)
                if "adjacent to and " in clean_name:
                    clean_name = clean_name.replace("adjacent to and ", "")
                
                # print(f"({current_relation}, {clean_name}) gives:")
                if current_relation == "inside" or current_relation == "contains" or current_relation == "intersects with":
                    # retain the same
                    current_relation = clean_name
                elif clean_name == "inside" or clean_name == "contains" or clean_name == "intersects with":
                    # retain the same
                    current_relation = current_relation
                else:
                    current_relation = composition_table_8dir[(current_relation, clean_name)]
                # print(f"    {current_relation}")
        else:
            # single hop jump, keep as it is
            current_relation = self.get_local_name(path[0][1])
            
        return current_relation
    
    def find_path_bfs_uri(self, start, end):
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
            for _, p, o in self.graph.triples((current, None, None)):
                if isinstance(o, URIRef) and o not in visited:
                    visited.add(o)
                    queue.append((o, path + [(current, p, o)]))
        
        return None


# Example Usage
# if __name__ == "__main__":
#     graph_indexer = RDFGraphIndexer("/mnt/newdrive/data_files/SpaTex/CSZt.nt")
#     print("Graph loaded")
#     graph_indexer.build_index()
    
#     # ner_entities = {'zipcode': ['66423', '66834'], 'state': [], 'county': []}
#     ner_entities = {'zipcode': ['57363'], 'state': [], 'county': ['Sanborn County South Dakota']}
#     pairs = graph_indexer.find_start_end_pairs(ner_entities)
    
#     for start, end in pairs:
#         print(f"Start: {start}\nEnd: {end}\n")