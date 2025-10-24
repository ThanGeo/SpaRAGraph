from rdflib.util import guess_format
from rdflib import Graph, URIRef

def get_local_name(uri):
    if isinstance(uri, URIRef):
        return uri.split('/')[-1].split('#')[-1].replace("_"," ")
    return str(uri)


def formatRDFtoNL(subject: str, predicate: str, object: str) -> str:
        context = ""
        if "contains" in predicate or "intersects with" in predicate:
            context += f"{subject} {predicate} {object}.\n"
        else:
            context += f"{subject} is {predicate} of {object}.\n"
            
        return context