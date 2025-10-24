
from core.base_index import BaseIndex
from utils import INDEX_TYPE

from rdflib.namespace import RDFS, SKOS
from rdflib import URIRef, Graph, Literal
from sentence_transformers import SentenceTransformer
import faiss

class FAISSIndex(BaseIndex):
    '''
    Uses FAISS to index the node labels of a given graph and 
    offers an interface for similarity matching between text and node labels
    '''
    def __init__(self, data_input, model_id="all-MiniLM-L6-v2"):
        # set properties
        super().__init__(data_input, INDEX_TYPE.FAISS)
        # model
        self.model = SentenceTransformer(model_id)
        # index the data
        if isinstance(data_input, Graph):
            self.indexGraph(data_input)
            self.mode = "graph"
        elif isinstance(data_input, list):
            self.indexFewShotExamples(data_input)
            self.mode = "fewshot"
        else:
            raise ValueError("Unsupported data input for FAISS index: ", type(data_input))
        
    def indexFewShotExamples(self, examples: "list[list[dict[str, str]]]"):
        """
        Index few-shot examples by encoding each full (user, assistant) pair
        as a single semantic unit.
        """
        self.examples = examples
        texts = []

        # Each example is a conversation list, e.g. [{"role": "user", ...}, {"role": "assistant", ...}]
        for convo in examples:
            # Concatenate all message contents for that few-shot pair
            convo_text = " ".join([f"{msg['role']}: {msg['content']}" for msg in convo])
            texts.append(convo_text)

        # Compute embeddings (1 per conversation)
        embeddings = self.model.encode(texts, convert_to_tensor=True).cpu().numpy()

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        print(f"Index for few-shot examples built with {len(self.examples)} conversations")

    
    def indexGraph(self, graph: Graph):
        # model
        """Index all URIs with their textual representations."""
        self.uris = []
        self.uri_to_text = {}
        texts = []
        
        # Include both subjects and objects that are URIs
        nodes = set(graph.subjects()).union(
            {o for o in graph.objects() if isinstance(o, URIRef)}
        )
        
        for uri in nodes:
            text = self._get_text_for_uri(graph, uri)
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
        print(f"Index for node labels built with {len(self.uris)} entities")
    

    def _get_text_for_uri(self, graph, uri):
        """Extract text from URI itself if no labels exist (for .nt files)."""
        texts = []
        # Try standard labels first
        for label_prop in [RDFS.label, SKOS.prefLabel]:
            for label in graph.objects(uri, label_prop):
                if isinstance(label, Literal):
                    texts.append(str(label))
        
        # Fallback: Use the URI's local name (last path segment)
        if not texts and isinstance(uri, URIRef):
            uri_str = str(uri)
            texts.append(uri_str.split("/")[-1].split("#")[-1])  # Get '66423' from '.../66423'
        
        return " ".join(texts) if texts else None  # Return None if no text at all
    
    def retrieveK(self, text: str, k: int) -> "list[dict[str, str]]":
        query_embedding = self.model.encode(text, convert_to_tensor=True).cpu().numpy()
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        results = []
        
        if self.mode == "graph":
            for idx, score in zip(indices[0], distances[0]):
                uri = self.uris[idx]
                results.append({
                    "uri": uri,
                    "text": self.uri_to_text[uri],
                    "score": float(score)
                })
        else:  # few-shot
            for idx, score in zip(indices[0], distances[0]):
                results.append({
                    "messages": self.examples[idx],
                    "score": float(score)
                })
        
        return results



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
    

    def generateContext(self, paths: "list[list[(str,str,str)]]") -> str:
        pass