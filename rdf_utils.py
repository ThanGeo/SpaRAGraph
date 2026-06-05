import re
from urllib.parse import unquote

from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, SKOS
from rdflib.util import guess_format


def local_name_to_text(raw: str) -> str:
    """Convert a URI local name to natural language text.

    Handles the three common naming conventions found in .nt files:
      - Underscores       (SpaTex):   Cuming_County_Nebraska → "Cuming County Nebraska"
      - CamelCase         (OWL/DBpedia): containsPlace → "contains Place"
      - URL-encoded chars (Wikidata): New%20York → "New York"
    """
    text = unquote(raw)                                        # %20 → space, %27 → ' etc.
    text = text.replace("_", " ")                             # SpaTex underscore style
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)         # camelCase → camel Case
    text = re.sub(r'([A-Z]{2,})([A-Z][a-z])', r'\1 \2', text)  # ABCDef → ABC Def
    text = re.sub(r'\s+', ' ', text).strip()                  # collapse spaces
    return text


def get_local_name(uri) -> str:
    """Extract and humanize the local name from a URI (or return a Literal as-is)."""
    if isinstance(uri, URIRef):
        raw = str(uri).split('/')[-1].split('#')[-1]
        return local_name_to_text(raw)
    return str(uri)


def uri_label(graph: Graph, uri) -> str:
    """Best available text for a URI: RDFS/SKOS label first, local name as fallback.

    Priority:
      1. rdfs:label
      2. skos:prefLabel
      3. local_name_to_text applied to the URI's local segment
    """
    if isinstance(uri, Literal):
        return local_name_to_text(str(uri))
    for prop in (RDFS.label, SKOS.prefLabel):
        for label in graph.objects(uri, prop):
            if isinstance(label, Literal):
                return str(label)
    return get_local_name(uri)


def formatRDFtoNL(subject: str, predicate: str, object: str) -> str:
    if "contains" in predicate or "intersects with" in predicate:
        return f"{subject} {predicate} {object}."
    return f"{subject} is {predicate} of {object}."
